"""Geometric anisotropy by profile likelihood over the major-axis azimuth.

Plan 03 costed three ways to give a covariance a rotation and chose this one:
scan the angle on a grid, fit everything else at each angle, keep the best, and
read the likelihood-vs-angle curve as the diagnostic of whether the angle was
identifiable at all.  A flat curve is a *result* — "no axis is preferred" — not
a failure, and neither alternative produces it for free.

Two deliberate departures from the plan as written, both forced by what this
package actually is:

**No sklearn.**  The plan fits an ARD Matern with
``sklearn.gaussian_process``; sklearn is not a dependency here and adding it
would mean a GP estimator stack alongside the gstools one, which is a much
larger change than "one anisotropy method".  The likelihood is computed
directly instead — a Cholesky of the covariance matrix, which is plan 03's own
Alternative B machinery used *only to fit the anisotropy*, not as a replacement
estimator.  The method is unchanged; only the arithmetic behind the objective
is.  The covariance itself is built through :func:`cpt_geostat.covariance.build_model`,
so the azimuth convention and the gstools ``anis``/``angles`` inversion stay in
the one place that is already tested for them.

**The scan runs 0-180, not 0-90.**  Plan 03's argument for the half interval is
correct *for the parameterisation it assumed*: unordered ARD length scales make
``(theta, l1, l2)`` and ``(theta+90, l2, l1)`` the same model, so [0, 90) names
each ellipse exactly once.  This module uses the package's own ordered form —
``aniso_ratio = major/minor >= 1`` with the azimuth naming the **major** axis —
in which that degeneracy does not exist, and the ellipse with its long axis at
30 deg is a genuinely different model from one at 120 deg.  In the ordered
parameterisation the complete non-degenerate interval is [0, 180).  Scanning
only [0, 90) here would silently forbid half the possible bearings.

What gets profiled
------------------

At each azimuth, the mean and the total variance come out in closed form (GLS
mean, then the maximum-likelihood variance scale), so the numerical search is
over three parameters only: the major range, the ratio, and the share of the
variance that is nugget.  That keeps a fit to roughly a second on the largest
IJmuiden unit and makes the simulation-based null below affordable.

Is the anisotropy real?
-----------------------

:func:`null_lr_threshold` answers it the only way that survives scrutiny.  The
likelihood-ratio statistic against the isotropic fit has no usable chi-square
distribution here: under the null of isotropy the azimuth is not a parameter at
all — it is unidentified — which is Davies' problem, and the textbook
``chi2(2)`` p-value is simply wrong.  So the null is *simulated*: isotropic
fields with the fitted isotropic covariance, at the same CPT locations, refitted
the same way.  That is expensive and honest, and it is opt-in for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from ..covariance import GrfConfig, build_model, range_to_len_scale
from .variogram import MIN_CPT_FOR_DIRECTIONAL, bin_edges, fit_variogram, unit_block

#: Azimuth grid step, degrees.  Coarse on purpose — :func:`fit_anisotropy`
#: refines around the winner, so the effective resolution is ``STEP / 4``.
AZIMUTH_STEP_DEG = 10.0

#: Largest major/minor ratio the search will entertain.  Beyond this the minor
#: axis is far finer than any CPT spacing on this project's sites, so the fit
#: would be extrapolating rather than measuring.
MAX_RATIO = 10.0

#: Smallest share of the total variance the nugget may take, and the largest.
#: Neither bound is reachable in a sane fit; they exist to keep the correlation
#: matrix conditioned.
_ETA_BOUNDS = (1e-4, 0.999)


@dataclass(frozen=True)
class AnisotropyFit:
    """A fitted anisotropic covariance, and the evidence for its anisotropy."""

    azimuth_deg: float          # major axis, degrees clockwise from north
    ratio: float                # major / minor, >= 1
    range_major_km: float       # practical range along the major axis
    sill: float
    nugget: float
    loglik: float
    loglik_isotropic: float
    range_isotropic_km: float
    #: The isotropic fit's own variance split.  Carried because the null in
    #: :func:`null_lr_threshold` must simulate *the isotropic model that was
    #: actually fitted* — nugget included.  Simulating the total variance as
    #: structured makes the null fields far smoother than the data, spurious
    #: axes far easier to find, and the threshold uselessly conservative.
    sill_isotropic: float = float("nan")
    nugget_isotropic: float = float("nan")
    n_cpt: int = 0
    model: str = "matern25"
    #: The diagnostic plan 03 wanted for free: profile log-likelihood by azimuth.
    azimuths: np.ndarray = field(default_factory=lambda: np.empty(0))
    loglik_curve: np.ndarray = field(default_factory=lambda: np.empty(0))
    fit_failed: bool = False
    #: Ratio sitting on :data:`MAX_RATIO`.  The likelihood will happily shrink a
    #: minor axis it cannot see to nothing, so this is reported the same way
    #: :class:`cpt_geostat.models.variogram.VariogramFit` reports ``at_range_bound``:
    #: the number is a bound, not a measurement.
    at_ratio_bound: bool = False
    #: Shortest separation between CPTs holding this unit, km.
    min_separation_km: float = float("nan")

    @property
    def range_minor_km(self) -> float:
        return self.range_major_km / self.ratio

    @property
    def minor_resolved(self) -> bool:
        """Is the minor range longer than the closest pair of CPTs?

        If it is not, the fit is describing structure finer than anything the
        survey sampled, and the ratio is an extrapolation however good the
        likelihood looks.  Synthetic unit 3 is exactly this case by design.
        """
        if not np.isfinite(self.min_separation_km):
            return True
        return self.range_minor_km > self.min_separation_km

    @property
    def lr_stat(self) -> float:
        """``2 * (loglik - loglik_isotropic)`` — the evidence for an axis.

        Compared against :func:`null_lr_threshold`, never against a chi-square:
        the azimuth is unidentified under the null, so the usual asymptotics do
        not apply.
        """
        return 2.0 * (self.loglik - self.loglik_isotropic)

    @property
    def curve_contrast(self) -> float:
        """Spread of the profile curve, in log-likelihood units.

        A flat curve means no azimuth is preferred — the honest outcome on a
        site with no fabric, and distinguishable from a peaked curve that merely
        sits at a low ratio.
        """
        if self.loglik_curve.size < 2:
            return float("nan")
        return float(np.nanmax(self.loglik_curve) - np.nanmin(self.loglik_curve))

    def to_grf_config(self) -> GrfConfig:
        """The fit as the package's covariance configuration.

        The point of the whole exercise: this goes straight to
        :func:`cpt_geostat.covariance.build_model` and therefore to the estimators,
        with no convention handling at the call site.
        """
        return GrfConfig(
            sill=self.sill,
            range_km=self.range_major_km,
            aniso_ratio=self.ratio,
            aniso_angle_deg=self.azimuth_deg,
            model=self.model,
        )

    def describe(self) -> str:
        out = (
            f"ratio {self.ratio:.2f} at {self.azimuth_deg:.0f} deg, "
            f"ranges {self.range_major_km:.2f}/{self.range_minor_km:.2f} km, "
            f"LR {self.lr_stat:.1f}"
        )
        if self.at_ratio_bound:
            out += " [ratio at bound]"
        if not self.minor_resolved:
            out += " [minor axis finer than the CPT spacing]"
        return out


# --------------------------------------------------------------------------
# the likelihood
# --------------------------------------------------------------------------

def _correlation(x, y, cfg: GrfConfig, eta: float) -> np.ndarray:
    """Unit-variance correlation matrix: ``(1 - eta) * Corr(h) + eta * I``.

    Built through :func:`build_model` so the azimuth reaches gstools by the one
    route this package has tested.  ``cov_spatial`` — not ``cov`` — because only
    the spatial form applies the rotation and the axis scaling; passing scalar
    distances to ``cov`` would silently fit an isotropic model at every angle
    and produce a perfectly flat, perfectly meaningless profile curve.
    """
    model = build_model(GrfConfig(**{**cfg.__dict__, "sill": 1.0}), nugget=0.0)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    corr = np.asarray(
        model.cov_spatial(np.stack([dx.ravel(), dy.ravel()]))
    ).reshape(dx.shape)
    out = (1.0 - eta) * corr
    out[np.diag_indices_from(out)] = 1.0  # (1 - eta) * 1 + eta
    return out


def _profile_loglik(x, y, v, cfg: GrfConfig, eta: float):
    """``(loglik, mean, total_variance)`` with the mean and scale profiled out.

    For ``K = s2 * C`` with an unknown constant mean, both the GLS mean and the
    maximum-likelihood ``s2`` have closed forms, so the numerical search never
    has to carry them.  This is what makes a 27-point azimuth scan cheap enough
    to run per unit and 50 times over for the null.
    """
    n = v.size
    C = _correlation(x, y, cfg, eta)
    try:
        cho = cho_factor(C, lower=True)
    except np.linalg.LinAlgError:
        return -np.inf, float("nan"), float("nan")

    ones = np.ones(n)
    Ci_1 = cho_solve(cho, ones)
    denom = float(ones @ Ci_1)
    if not np.isfinite(denom) or denom <= 0:
        return -np.inf, float("nan"), float("nan")
    mu = float(v @ Ci_1) / denom

    r = v - mu
    s2 = float(r @ cho_solve(cho, r)) / n
    if not np.isfinite(s2) or s2 <= 0:
        return -np.inf, float("nan"), float("nan")

    logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
    ll = -0.5 * (n * np.log(2.0 * np.pi * s2) + logdet + n)
    return float(ll), mu, s2


def _fit_at_azimuth(x, y, v, azimuth, model, bounds, start):
    """Best ``(range, ratio, eta)`` at one fixed azimuth.

    Returns ``(loglik, range_km, ratio, eta, s2)``.  ``ratio`` is bounded below
    at 1.0 so the azimuth always names the *major* axis; without that the
    optimiser could return a sub-unit ratio and a bearing 90 degrees from the
    one it means.
    """
    (r_lo, r_hi), (a_lo, a_hi) = bounds

    def objective(p):
        rng, ratio, eta = np.exp(p[0]), 1.0 + np.exp(p[1]), _sigmoid(p[2])
        if not (r_lo <= rng <= r_hi and a_lo <= ratio <= a_hi):
            return 1e6
        cfg = GrfConfig(sill=1.0, range_km=rng, aniso_ratio=ratio,
                        aniso_angle_deg=azimuth, model=model)
        ll, _, _ = _profile_loglik(x, y, v, cfg, eta)
        return -ll if np.isfinite(ll) else 1e6

    p0 = np.array([np.log(start[0]), np.log(max(start[1] - 1.0, 1e-3)), _logit(start[2])])
    res = minimize(objective, p0, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-4})
    rng, ratio, eta = np.exp(res.x[0]), 1.0 + np.exp(res.x[1]), _sigmoid(res.x[2])
    rng = float(np.clip(rng, r_lo, r_hi))
    ratio = float(np.clip(ratio, a_lo, a_hi))
    cfg = GrfConfig(sill=1.0, range_km=rng, aniso_ratio=ratio,
                    aniso_angle_deg=azimuth, model=model)
    ll, _, s2 = _profile_loglik(x, y, v, cfg, eta)
    return ll, rng, ratio, eta, s2


def _sigmoid(t):
    return _ETA_BOUNDS[0] + (_ETA_BOUNDS[1] - _ETA_BOUNDS[0]) / (1.0 + np.exp(-t))


def _logit(e):
    lo, hi = _ETA_BOUNDS
    p = np.clip((e - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    return float(np.log(p / (1.0 - p)))


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------

def fit_anisotropy(
    x, y, v,
    model: str = "matern25",
    step_deg: float = AZIMUTH_STEP_DEG,
    refine: bool = True,
    max_ratio: float = MAX_RATIO,
) -> AnisotropyFit:
    """Profile the likelihood over the major-axis azimuth and keep the best.

    The scan covers **[0, 180)** — see the module docstring; the half interval in
    plan 03 belongs to a parameterisation this package does not use.  ``refine``
    re-scans at a quarter of ``step_deg`` around the winner, which costs nine
    more fits and takes the effective resolution to 2.5 degrees by default.

    The isotropic fit is computed with the *same* likelihood rather than taken
    from :func:`cpt_geostat.models.variogram.fit_variogram`, because a likelihood
    ratio between a likelihood and a least-squares variogram fit would not be a
    likelihood ratio at all.  The variogram fit is still used, but only to start
    the optimiser somewhere sensible.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    n = v.size
    if n < 3:
        raise ValueError(f"need at least 3 locations to fit anisotropy, got {n}")

    max_lag = float(bin_edges(x, y)[-1])
    r_bounds = (max_lag * 1e-3, max_lag)
    bounds = (r_bounds, (1.0, float(max_ratio)))

    # Isotropic reference, under the same likelihood — and multi-started,
    # because (range, nugget) is a *ridge*, not a basin.  A short range with
    # almost no nugget and a long range with a large one explain the same data
    # to within a fraction of a log-likelihood, so a single start seeded from
    # the variogram fit lands in whichever basin it began in and stays there.
    # That is not a cosmetic difference: the isotropic fit is what
    # `null_lr_threshold` simulates from, so the wrong basin means the wrong
    # null.  Twelve small starts cost well under a second and settle it.
    iso = fit_variogram(x, y, v, model=model)
    total = iso.sill + iso.nugget
    seeds = [
        (r, e)
        for r in (float(np.clip(iso.range_km, *r_bounds)), 0.15 * max_lag,
                  0.4 * max_lag, 0.8 * max_lag)
        for e in (0.05, 0.5, 0.85)
    ]
    ll_iso, rng_iso, _, eta_iso, s2_iso = max(
        (_fit_at_azimuth(x, y, v, 0.0, model, (r_bounds, (1.0, 1.0)), (r, 1.0, e))
         for r, e in seeds),
        key=lambda out: out[0],
    )

    # The azimuth scan starts from the basin the isotropic fit chose — and from
    # several ratios within it.  This is not optional polish: the isotropic
    # reference is multi-started, so searching the anisotropic model *less*
    # thoroughly would depress its likelihood and shrink every LR statistic
    # towards zero.  The comparison has to be like for like or the null is being
    # given a handicap the alternative does not get.
    starts = [(float(np.clip(rng_iso, *r_bounds)), r, float(eta_iso))
              for r in (1.5, 3.0, 6.0)]  # starts[0] scans; all three refine

    def best_at(az, seeds=None):
        return max((_fit_at_azimuth(x, y, v, az, model, bounds, s)
                    for s in (seeds if seeds is not None else starts[:1])),
                   key=lambda out: out[0])

    azimuths = np.arange(0.0, 180.0, float(step_deg))
    results = [best_at(az) for az in azimuths]
    curve = np.array([r[0] for r in results], dtype=float)

    best = int(np.nanargmax(curve))
    if refine:
        fine_step = float(step_deg) / 4.0
        fine = np.arange(azimuths[best] - step_deg, azimuths[best] + step_deg + 1e-9,
                         fine_step)
        fine = np.mod(fine, 180.0)
        extra = [best_at(az) for az in fine]
        azimuths = np.concatenate([azimuths, fine])
        curve = np.concatenate([curve, [e[0] for e in extra]])
        results = results + extra
        order = np.argsort(azimuths)
        azimuths, curve = azimuths[order], curve[order]
        results = [results[i] for i in order]
        best = int(np.nanargmax(curve))

    # Multi-start at the winner only.  The scan itself needs to *rank* azimuths,
    # which one start does adequately; the reported statistic needs to be a fair
    # match for the multi-started isotropic reference, or every LR is depressed
    # towards zero.  Paying for both costs about a fifth more, not three times.
    results[best] = max([results[best], best_at(float(azimuths[best]), starts)],
                        key=lambda out: out[0])
    curve[best] = results[best][0]

    ll, rng, ratio, eta, s2 = results[best]
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    min_sep = float(d[d > 0].min()) if np.any(d > 0) else float("nan")
    return AnisotropyFit(
        azimuth_deg=float(azimuths[best]),
        ratio=float(ratio),
        range_major_km=float(rng),
        sill=float(s2 * (1.0 - eta)),
        nugget=float(s2 * eta),
        loglik=float(ll),
        loglik_isotropic=float(ll_iso),
        range_isotropic_km=float(rng_iso),
        sill_isotropic=float(s2_iso * (1.0 - eta_iso)),
        nugget_isotropic=float(s2_iso * eta_iso),
        n_cpt=int(n),
        model=model,
        azimuths=azimuths,
        loglik_curve=curve,
        fit_failed=not np.isfinite(ll),
        at_ratio_bound=bool(ratio >= 0.99 * max_ratio),
        min_separation_km=min_sep,
    )


def fit_unit_anisotropy(ds, unit_id: str, **kwargs) -> Optional[AnisotropyFit]:
    """:func:`fit_anisotropy` on one unit's per-CPT depth-averages.

    Returns ``None`` below :data:`cpt_geostat.models.variogram.MIN_CPT_FOR_DIRECTIONAL`
    rather than a fit, for the reason that constant exists: under about 30 CPTs
    the azimuth is sampling noise, and a number returned there would be used.
    """
    block = unit_block(ds, unit_id)
    if len(block) < MIN_CPT_FOR_DIRECTIONAL:
        return None
    return fit_anisotropy(
        block["x"].to_numpy(), block["y"].to_numpy(),
        block["log_Q_mean"].to_numpy(), **kwargs
    )


# --------------------------------------------------------------------------
# is it real?
# --------------------------------------------------------------------------

def simulate_isotropic(x, y, fit: AnisotropyFit, rng) -> np.ndarray:
    """One draw from the *fitted isotropic* model — structured part plus nugget.

    Split out so the nugget is testable rather than merely intended.  The null
    once simulated the total variance as structured and added no nugget at all,
    which produces fields far smoother than the data: a spurious axis then fits
    them much better and the threshold comes back inflated, in one measured case
    by a factor of nine.  The failure is invisible in the returned statistic —
    the null simply looks conservative — so it is pinned here on the field
    itself, where it is a matter of arithmetic rather than of sampling.
    """
    import gstools as gs

    sill = fit.sill_isotropic if np.isfinite(fit.sill_isotropic) else fit.sill
    nugget = fit.nugget_isotropic if np.isfinite(fit.nugget_isotropic) else fit.nugget
    model = build_model(
        GrfConfig(sill=max(float(sill), 1e-12), range_km=fit.range_isotropic_km,
                  model=fit.model)
    )
    srf = gs.SRF(model, seed=int(rng.integers(1 << 31)))
    field = np.asarray(srf((x, y), mesh_type="unstructured"), dtype=float)
    return field + rng.normal(0.0, np.sqrt(max(float(nugget), 0.0)), field.size)


def null_lr_threshold(x, y, fit: AnisotropyFit, n_sim: int = 50,
                      quantile: float = 0.95, seed: Optional[int] = None,
                      **kwargs) -> dict:
    """What likelihood ratio isotropic ground produces at these locations.

    The identifiability test, done by simulation because the asymptotics are not
    available: under the null the azimuth is unidentified (Davies' problem), so
    ``lr_stat`` has no chi-square distribution and comparing it to one would
    manufacture significance.

    Fields are drawn from the *fitted isotropic* covariance at the *same* CPT
    positions and refitted by the same routine, so the null inherits the survey
    geometry — which is the thing that actually generates spurious axes, a
    gappy or elongated layout preferring a bearing all by itself.

    Returns the threshold, the simulated statistics, and ``exceeds``.
    """
    import gstools as gs

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    stats = []
    for _ in range(int(n_sim)):
        sim = simulate_isotropic(x, y, fit, rng)
        try:
            stats.append(fit_anisotropy(x, y, sim, model=fit.model, **kwargs).lr_stat)
        except (ValueError, np.linalg.LinAlgError):
            continue
    stats = np.array(stats, dtype=float)
    stats = stats[np.isfinite(stats)]
    threshold = float(np.quantile(stats, quantile)) if stats.size else float("nan")
    return {
        "threshold": threshold,
        "quantile": float(quantile),
        "n_sim": int(stats.size),
        "lr_stat": fit.lr_stat,
        "exceeds": bool(fit.lr_stat > threshold) if np.isfinite(threshold) else False,
        "null_lr": stats,
    }


# --------------------------------------------------------------------------
# handing it to the estimators
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AnisotropyDecision:
    """Whether one unit's kriging gets an axis, and why."""

    unit_id: str
    use_anisotropy: bool
    reason: str
    fit: Optional[AnisotropyFit] = None

    @property
    def covariance(self):
        """What to hand the estimator: a covariance outright, or ``"fit"``.

        Falling back to the string rather than to an isotropic ``GrfConfig``
        keeps the rejected units on **exactly** the path the isotropic runs
        already take, so a comparison between the two estimators is a comparison
        of the units that changed and nothing else.
        """
        if not self.use_anisotropy or self.fit is None:
            return "fit"
        return (self.fit.to_grf_config(), self.fit.nugget)


def decide_anisotropy(ds, n_sim: int = 30, seed: Optional[int] = None,
                      **kwargs) -> dict:
    """Per unit: fit an axis, then decide whether it may be believed.

    Four gates, in increasing order of cost, each of which has a synthetic unit
    behind it:

    1. **Under the directional gate.**  Below
       :data:`cpt_geostat.models.variogram.MIN_CPT_FOR_DIRECTIONAL` an azimuth is
       sampling noise (unit 5).
    2. **Ratio on its bound.**  The likelihood shrinking a minor axis it cannot
       see to nothing (unit 3).
    3. **Minor axis finer than the CPT spacing.**  Structure the survey never
       sampled, so the ratio is extrapolation (unit 3 again).
    4. **Does not beat the isotropic null.**  The one that catches unit 6, which
       is isotropic by construction and fits ``ratio 9.35 at 80 deg`` — a number
       indistinguishable at a glance from a real channel.

    A unit that fails any gate keeps the isotropic covariance.  The point is not
    to be conservative for its own sake: an axis fitted to noise rotates the
    prediction *and* narrows its variance along a direction chosen by the noise,
    which is worse than isotropy rather than merely no better.

    ``n_sim = 0`` skips gate 4 and is recorded as ``"null not tested"`` in the
    reason, so a run that took the shortcut says so in its own report.
    """
    out = {}
    for uid in ds.unit_ids:
        block = unit_block(ds, uid)
        if len(block) < MIN_CPT_FOR_DIRECTIONAL:
            out[uid] = AnisotropyDecision(
                uid, False, f"under the directional gate (n={len(block)})"
            )
            continue
        fit = fit_anisotropy(block["x"].to_numpy(), block["y"].to_numpy(),
                             block["log_Q_mean"].to_numpy(), **kwargs)
        if fit.fit_failed:
            out[uid] = AnisotropyDecision(uid, False, "fit failed", fit)
        elif fit.at_ratio_bound:
            out[uid] = AnisotropyDecision(uid, False, "ratio on its bound", fit)
        elif not fit.minor_resolved:
            out[uid] = AnisotropyDecision(
                uid, False, "minor axis finer than the CPT spacing", fit
            )
        elif n_sim:
            null = null_lr_threshold(block["x"].to_numpy(), block["y"].to_numpy(),
                                     fit, n_sim=n_sim, seed=seed, **kwargs)
            if null["exceeds"]:
                out[uid] = AnisotropyDecision(
                    uid, True,
                    f"LR {fit.lr_stat:.1f} over null p95 {null['threshold']:.1f}", fit
                )
            else:
                out[uid] = AnisotropyDecision(
                    uid, False,
                    f"LR {fit.lr_stat:.1f} under null p95 {null['threshold']:.1f}", fit
                )
        else:
            out[uid] = AnisotropyDecision(uid, True, "null not tested", fit)
    return out


def anisotropic_kriging_factory(ds, method: str = "OK", decisions: Optional[dict] = None,
                                n_sim: int = 30, seed: Optional[int] = None,
                                model: str = "matern25", **kwargs):
    """``unit_id -> estimator``, anisotropic only where the axis survives the gates.

    Nothing in :mod:`cpt_geostat.models.kriging` changes: the estimators already accept
    a covariance given outright, which is the path the synthetic truth-covariance
    reference uses.  This supplies a *fitted* anisotropic covariance by the same
    route.

    ``decisions`` is accepted so a caller that has already paid for the null
    (they are the expensive part) can reuse it rather than refit — the CLI does,
    to report the table it prints and the estimator it runs from one decision.
    """
    from .kriging import KRIGING_METHODS

    try:
        cls = KRIGING_METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown kriging method {method!r}; have {sorted(KRIGING_METHODS)}"
        ) from None

    if decisions is None:
        decisions = decide_anisotropy(ds, n_sim=n_sim, seed=seed, **kwargs)

    def factory(unit_id):
        decision = decisions.get(unit_id)
        cov = decision.covariance if decision is not None else "fit"
        return cls(covariance=cov, model=model)

    return factory


def decision_table(decisions: dict):
    """The decisions as a frame, for printing and for the run record."""
    import pandas as pd

    rows = []
    for uid, d in decisions.items():
        row = {"unit_id": uid, "anisotropic": d.use_anisotropy, "reason": d.reason}
        if d.fit is not None:
            row.update({
                "n_cpt": d.fit.n_cpt, "azimuth_deg": d.fit.azimuth_deg,
                "ratio": d.fit.ratio, "range_major_km": d.fit.range_major_km,
                "range_minor_km": d.fit.range_minor_km, "lr_stat": d.fit.lr_stat,
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("unit_id") if rows else pd.DataFrame()


def anisotropy_table(ds, n_sim: int = 0, seed: Optional[int] = None, **kwargs):
    """Per unit: the fitted axis, and optionally whether it beats an isotropic null.

    ``n_sim = 0`` skips the null entirely and returns the fits alone, which is
    the cheap mode.  Anything above about 30 simulations per unit is minutes of
    work, so it is never the default.
    """
    import pandas as pd

    rows = []
    for uid in ds.unit_ids:
        fit = fit_unit_anisotropy(ds, uid, **kwargs)
        if fit is None:
            continue
        row = {
            "unit_id": uid, "n_cpt": fit.n_cpt, "azimuth_deg": fit.azimuth_deg,
            "ratio": fit.ratio, "range_major_km": fit.range_major_km,
            "range_minor_km": fit.range_minor_km, "nugget": fit.nugget,
            "lr_stat": fit.lr_stat, "curve_contrast": fit.curve_contrast,
            "at_ratio_bound": fit.at_ratio_bound, "minor_resolved": fit.minor_resolved,
        }
        if n_sim:
            block = unit_block(ds, uid)
            null = null_lr_threshold(block["x"].to_numpy(), block["y"].to_numpy(),
                                     fit, n_sim=n_sim, seed=seed, **kwargs)
            row["null_lr_p95"] = null["threshold"]
            row["anisotropic"] = null["exceeds"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("unit_id") if rows else pd.DataFrame()
