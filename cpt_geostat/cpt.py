"""CPT normalisation and its inverse — ``Qtn`` back to ``qt``.

The model works in ``log(Qtn)`` because that is the variable whose statistics
behave: normalisation strips the depth trend that would otherwise dominate every
variogram, and the log makes the within-unit distribution roughly symmetric.
The price is that a prediction comes back in units nobody specifies a foundation
against.  This module pays it back, and it is the *only* place in the package
that knows about stresses and unit weights.

The normalisation, and the one that matters
-------------------------------------------

Two definitions are in circulation and they are not interchangeable::

    Qt   = (qt - sigma_v0) / sigma_v0_eff                       Robertson (1990)
    Qtn  = ((qt - sigma_v0) / pa) * (pa / sigma_v0_eff) ** n     Robertson (2009)

``Qtn`` reduces to ``Qt`` exactly when ``n = 1`` — ``pa`` cancels — so ``Qt`` is
the ``n = 1`` member of the family rather than a different quantity.  Away from
``n = 1`` the two differ by ``(pa / sigma_v0_eff) ** (n - 1)``, which at 10 m
below seabed is a factor of about 3.  Inverting with the wrong ``n`` therefore
does not produce a slightly wrong ``qt``; it produces a badly wrong one, and
wrong in a depth-dependent way that looks like a trend.

**The IJmuiden export is ``n = 1``.**  Its ``Qt`` column reproduces to a median
0.5% from ``(qt - sigma_v0) / sigma_v0_eff`` with ``sigma_v0_eff = 10.1 * z``
kPa, and is nowhere near the ``n = 0.5`` form.  :func:`infer_gamma_eff` is how
that was established and is the check to run on any new export before trusting
its label: a column named ``Qtn`` in a deliverable is not evidence of which
formula produced it.

Stresses are referenced to the seabed
-------------------------------------

Water depth does not appear anywhere below, and that is not an omission.  A
piezocone measures ``u2`` as a gauge pressure and ``qc`` against the ambient
water pressure, so ``qt = qc + u2 * (1 - a)`` is already referenced to the
seabed; ``sigma_v0`` must be too, or the subtraction mixes datums.  The water
column cancels out of ``sigma_v0_eff`` in any case, since it adds equally to
total stress and pore pressure.  Depths here are metres below seabed, positive
down — the project's convention.

Everything is in kPa
--------------------

``qt`` in kPa, unit weights in kN/m^3 (numerically kPa/m), ``pa = 100`` kPa.
Raw CPT exports are usually MPa — the IJmuiden one is — so a factor of 1000
belongs at the edge, in the project's preparation script, not in here.

What this module deliberately does not do
-----------------------------------------

``n`` is an input, not something inferred.  Robertson (2009) solves for it
iteratively from the soil behaviour type index ``Ic``, which needs sleeve
friction; the model carries neither ``fs`` nor ``Ic`` past
:mod:`cpt_geostat.contract`, so an ``Ic``-driven ``n`` cannot be reconstructed from a
prediction.  A single ``n`` per unit, taken from whatever produced the input
column, is the honest option and keeps the map monotone — which is what makes
:func:`qt_from_log_qtn` able to transform an uncertainty band exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

#: Atmospheric reference pressure, kPa.  Cancels entirely when ``n = 1``.
PA_KPA = 100.0

#: Standard gravity as geotechnics rounds it, m/s^2.
G = 9.81

#: Seawater density, t/m^3.  Fresh water is 1.0.
RHO_SEAWATER = 1.025


@dataclass(frozen=True)
class StressProfile:
    """A constant-gradient in-situ stress profile below the seabed.

    Densities in t/m^3 (= g/cm^3), which is how a soil density is quoted, and
    unit weights come out in kN/m^3.  Constant with depth is a coarse model of a
    real soil column, but it is the right coarseness for this purpose: the
    normalisation only needs ``sigma_v0_eff`` to within a few percent, and the
    alternative — a unit weight correlated from ``qt`` and integrated down the
    hole — reintroduces the reading-by-reading dependence the depth-averaged
    model has already given up.

    Use :meth:`from_gradients` when the source of the ``Qtn`` column states unit
    weights rather than densities, or when :func:`infer_gamma_eff` has recovered
    the gradient a spreadsheet actually used.
    """

    rho_soil: float = 2.0
    rho_water: float = RHO_SEAWATER
    g: float = G

    @property
    def gamma_sat(self) -> float:
        """Total (saturated) unit weight, kN/m^3."""
        return self.rho_soil * self.g

    @property
    def gamma_w(self) -> float:
        """Water unit weight, kN/m^3."""
        return self.rho_water * self.g

    @property
    def gamma_eff(self) -> float:
        """Submerged (buoyant) unit weight, kN/m^3 — the gradient that matters.

        ``sigma_v0_eff`` is the denominator of the normalisation, so a 3% error
        here is a 3% error in every ``Qtn``, whereas the same error in
        ``gamma_sat`` alone moves ``qt`` by well under 1% (``sigma_v0`` is a
        small fraction of ``qt`` at any depth a CPT reaches).
        """
        return self.gamma_sat - self.gamma_w

    @classmethod
    def from_gradients(cls, gamma_sat: float, gamma_w: float = RHO_SEAWATER * G) -> "StressProfile":
        """Build from unit weights (kN/m^3) instead of densities."""
        return cls(rho_soil=gamma_sat / G, rho_water=gamma_w / G, g=G)

    def sigma_v0(self, z):
        """Total vertical stress at ``z`` m below seabed, kPa."""
        return self.gamma_sat * np.asarray(z, dtype=float)

    def u0(self, z):
        """Hydrostatic pore pressure *above seabed ambient*, kPa."""
        return self.gamma_w * np.asarray(z, dtype=float)

    def sigma_v0_eff(self, z):
        """Vertical effective stress, kPa — ``sigma_v0 - u0``."""
        return self.gamma_eff * np.asarray(z, dtype=float)

    def describe(self) -> str:
        """One line, for a run log or a figure caption.

        Leads with ``gamma_eff`` because that is the number the normalisation
        divides by; ``gamma_w`` is left implicit rather than making the line too
        long to fit above a figure.
        """
        return f"gamma_eff {self.gamma_eff:.2f} kN/m3 (gamma_sat {self.gamma_sat:.2f})"

    def to_metadata(self) -> dict:
        """Unit weights as JSON, for a project's ``prepare_metadata.json``.

        Unit weights rather than densities: they are what the arithmetic uses,
        and writing ``rho_soil`` would leave a reader to guess which ``g``.
        ``gamma_eff`` is redundant and written anyway — it is the one number
        that matters, and a reader should not have to subtract to find it.
        """
        return {
            "gamma_sat_kn_m3": round(self.gamma_sat, 4),
            "gamma_w_kn_m3": round(self.gamma_w, 4),
            "gamma_eff_kn_m3": round(self.gamma_eff, 4),
        }


#: What this project assumes absent better information: 2.0 t/m^3 soil, seawater.
DEFAULT_PROFILE = StressProfile()


def _stresses(z, profile: StressProfile, seabed_ok: bool = False):
    """``(sigma_v0, sigma_v0_eff)`` with invalid depths masked to nan.

    The seabed is asymmetric between the two directions, and the mask follows
    that rather than picking one rule for both:

    * **Normalising** at ``z = 0`` divides by an effective stress of zero, so it
      is undefined — nan.  That beats raising (a real trace starts at 0.01 m and
      one leading zero should not kill a hole) and beats inf, which would
      propagate silently into an average.
    * **De-normalising** at ``z = 0`` is perfectly defined: both stresses vanish,
      so ``qt = 0`` whatever ``Qtn`` says.  Degenerate, but it is what the
      constant-gradient model states, and it is the honest value to draw at the
      top of a unit that outcrops at the seabed.

    Negative depths are nan either way — above the seabed there is no soil.
    """
    z = np.asarray(z, dtype=float)
    valid = z >= 0.0 if seabed_ok else z > 0.0
    sv = np.where(valid, profile.sigma_v0(z), np.nan)
    sve = np.where(valid, profile.sigma_v0_eff(z), np.nan)
    return sv, sve


def qtn_from_qt(qt, z, n: float = 1.0, profile: StressProfile = DEFAULT_PROFILE,
                pa: float = PA_KPA):
    """Normalise: cone resistance (kPa) at depth ``z`` (m below seabed) to ``Qtn``.

    ``n = 1`` gives Robertson's ``Qt`` exactly.  Included mostly so the inverse
    can be round-tripped against it in tests, and so a project can re-derive the
    normalisation from ``qc``/``u2`` rather than inheriting a column it cannot
    account for.
    """
    qt = np.asarray(qt, dtype=float)
    sv, sve = _stresses(z, profile)
    return (qt - sv) / pa * (pa / sve) ** n


def qt_from_qtn(qtn, z, n: float = 1.0, profile: StressProfile = DEFAULT_PROFILE,
                pa: float = PA_KPA):
    """De-normalise: ``Qtn`` at depth ``z`` (m below seabed) back to ``qt`` in kPa.

    ``qt = sigma_v0 + Qtn * pa ** (1 - n) * sigma_v0_eff ** n``, which for
    ``n = 1`` is just ``sigma_v0 + Qtn * sigma_v0_eff``.

    A constant ``Qtn`` through a unit therefore comes back as a ``qt`` rising
    *linearly* with depth — the depth trend the normalisation removed, put back.
    That is the whole content of the transform, and it is why a per-unit
    prediction is enough to draw a ``qt`` profile: the model supplies the level,
    the stress profile supplies the gradient.
    """
    qtn = np.asarray(qtn, dtype=float)
    sv, sve = _stresses(z, profile, seabed_ok=True)
    return sv + qtn * pa ** (1.0 - n) * sve**n


def qt_from_log_qtn(log_qtn, z, n: float = 1.0, profile: StressProfile = DEFAULT_PROFILE,
                    pa: float = PA_KPA):
    """``qt`` in kPa from the model's own variable, ``log(Qtn)``.

    The one to use on estimator output, including on band edges.  Both ``exp``
    and :func:`qt_from_qtn` are strictly increasing in their argument, so the
    composite map is monotone and quantiles pass straight through it: applying
    this to ``pred`` and to ``pred ± 1.96 * sd`` gives the median ``qt`` and an
    exact 95% interval on ``qt``, no delta method and no lognormal correction.

    What does *not* survive the transform is the mean.  ``exp(pred)`` is a
    median because ``pred`` is a mean of logs, and adding ``sigma_v0`` to a
    median leaves a median.  A mean ``qt`` would need the lognormal correction
    and is not what the profile figures show.
    """
    return qt_from_qtn(np.exp(np.asarray(log_qtn, dtype=float)), z, n=n, profile=profile, pa=pa)


#: The key a project's ``prepare_metadata.json`` records its normalisation under.
METADATA_KEY = "normalisation"

#: Written into the metadata so the file states the formula, not just its inputs.
NORMALISATION_FORM = "Qtn = ((qt - sigma_v0) / pa) * (pa / sigma_v0_eff) ** n"


def normalisation_metadata(profile: StressProfile, n, basis: str = "",
                           pa: float = PA_KPA, n_sd=0.0) -> dict:
    """The JSON block a preparation step writes so nobody re-derives this.

    The stress exponent and the unit weights are properties of *the column that
    was supplied*, not of the model, and they are unrecoverable from the
    prepared tables — ``cpt_samples`` keeps ``Qtn`` and drops ``qt`` and ``fs``.
    Written once at preparation time, they travel with the data.  ``basis``
    should say where they came from: fitted, quoted in a report, or assumed.

    ``n`` may be a single number or a mapping of ``unit_id -> n``, which is how
    a variable exponent is recorded — one value per unit, from
    :func:`soil_behaviour_index` averaged over the unit's readings.  ``n_sd``
    records how well pinned it is, in the same form; the natural source is the
    scatter of the per-reading ``n`` within the unit.
    """
    block = {
        "form": NORMALISATION_FORM,
        "n": _exponent_json(n),
        "pa_kpa": float(pa),
        "qt_units": "kPa",
        "depth_datum": "m below seabed, positive down",
        "basis": basis,
        **profile.to_metadata(),
    }
    if np.any(np.asarray(list(_as_mapping(n_sd).values()) or [n_sd], dtype=float) > 0):
        block["n_sd"] = _exponent_json(n_sd)
    return block


def _as_mapping(value) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _exponent_json(value):
    """A scalar exponent, or one per unit, in a form ``json`` will accept."""
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    return float(value)


def normalisation_from_metadata(meta: dict) -> tuple[StressProfile, object]:
    """``(profile, n)`` back out of a project's metadata.

    Accepts either the whole ``prepare_metadata.json`` or just the
    ``normalisation`` block.  ``n`` comes back as a float, or as a
    ``unit_id -> n`` dict if that is how it was recorded — both are accepted
    everywhere an exponent is taken.  Raises rather than defaulting: a dataset
    whose metadata is silent about the normalisation is one where the exponent
    is a guess, and a guessed exponent is a depth-dependent error in ``qt``.
    """
    block = meta.get(METADATA_KEY, meta)
    try:
        gamma_sat = float(block["gamma_sat_kn_m3"])
        gamma_w = float(block["gamma_w_kn_m3"])
        n = block["n"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"metadata has no usable {METADATA_KEY!r} block "
            f"(need gamma_sat_kn_m3, gamma_w_kn_m3, n); got keys {sorted(block)}"
        ) from exc
    n = {str(k): float(v) for k, v in n.items()} if isinstance(n, Mapping) else float(n)
    return StressProfile.from_gradients(gamma_sat, gamma_w), n


def exponent_uncertainty_from_metadata(meta: dict):
    """``n_sd`` as recorded, or ``0.0`` — a separate accessor on purpose.

    Absent ``n_sd`` means *the exponent is being treated as known*, which is a
    defensible position and the one every earlier run took.  It does not mean
    "unknown, pick something", so it defaults to zero rather than to a guess,
    and it is read separately from :func:`normalisation_from_metadata` so that
    adding it cannot change what an existing caller gets back.
    """
    block = meta.get(METADATA_KEY, meta)
    value = block.get("n_sd", 0.0) if isinstance(block, Mapping) else 0.0
    return {str(k): float(v) for k, v in value.items()} if isinstance(value, Mapping) \
        else float(value)


def log_stress_ratio(z, profile: StressProfile = DEFAULT_PROFILE, pa: float = PA_KPA):
    """``log(sigma_v0_eff / pa)`` — the lever the stress exponent acts through.

    Everything ``n`` does, it does through this one number.  ``qt - sigma_v0``
    scales as ``exp(n * L)``, so two exponents that differ by ``dn`` disagree by
    ``exp(dn * L)`` — nothing at all where ``L = 0``, and more the further from
    it in either direction.

    ``L = 0`` at ``sigma_v0_eff = pa``, which on IJmuiden's gradient is 9.9 m
    below seabed.  That is the **pivot**: at 10 m the choice of exponent is
    irrelevant, at 0.5 m it is worth a factor of 3, and at 60 m a factor of 1.8
    the other way.  Shallow ground is where an exponent argument actually
    matters, which is the opposite of most people's intuition.
    """
    _, sve = _stresses(z, profile, seabed_ok=True)
    with np.errstate(divide="ignore"):
        return np.log(sve / pa)


def exponent_log_sd(z, n_sd, profile: StressProfile = DEFAULT_PROFILE, pa: float = PA_KPA):
    """Log-space sd added by uncertainty in ``n``: ``|n_sd * L|``.

    Comparable, term for term, with the model's own ``log_Q_sd`` — which is what
    makes it possible to say whether arguing about the exponent is worth the
    breath.  On IJmuiden, ``n_sd = 0.1`` adds 0.30 at 0.5 m depth, against a
    between-hole ``log_Q_sd`` of 0.31-0.71: at the seabed the exponent is as
    large a source of doubt as the spatial model.  By 10 m it adds nothing, and
    at 60 m about 0.18.
    """
    L = log_stress_ratio(z, profile, pa)
    n_sd = np.asarray(n_sd, dtype=float)
    # -inf at the seabed, where the scale is zero and qt is deterministic:
    # no spread to add.  nan above it stays nan.
    return np.where(np.isfinite(L), np.abs(n_sd * np.where(np.isfinite(L), L, 0.0)),
                    np.where(np.isnan(L), np.nan, 0.0))


def qt_scale(z, n=1.0, profile: StressProfile = DEFAULT_PROFILE, pa: float = PA_KPA):
    """``c(z)`` in ``qt = sigma_v0 + c(z) * Qtn`` — kPa per unit of ``Qtn``.

    Isolated because it carries the whole reason the uncertainty transforms
    cleanly: at fixed depth the de-normalisation is **affine** in ``Qtn``.  The
    stress exponent sets ``c``; it does not put ``Qtn`` inside a power.  So a
    lognormal ``Qtn`` becomes a shifted, scaled lognormal ``qt``, and its
    moments and quantiles both follow in closed form — for any ``n``, not just
    for ``n = 1``.
    """
    _, sve = _stresses(z, profile, seabed_ok=True)
    return pa ** (1.0 - n) * sve**n


def qt_quantile(log_mean, log_sd, z, q: float = 0.5, n=1.0, n_sd=0.0,
                profile: StressProfile = DEFAULT_PROFILE, pa: float = PA_KPA):
    """The ``q``-quantile of ``qt`` (kPa) from a Gaussian ``log(Qtn)``.

    The exact route, and the one to prefer.  ``exp`` and the affine map are both
    strictly increasing, so the ``q``-quantile of ``qt`` is the transform of the
    ``q``-quantile of ``log(Qtn)`` — no approximation anywhere, and ``q = 0.5``
    gives the median.

    ``n_sd`` widens the interval for not knowing the exponent (see
    :func:`total_log_sd`); it leaves the median where it was, because the
    exponent enters through a lognormal factor whose median is one.
    """
    log_mean = np.asarray(log_mean, dtype=float)
    sd = total_log_sd(log_sd, z, n_sd=n_sd, profile=profile, pa=pa)
    sv, _ = _stresses(z, profile, seabed_ok=True)
    return sv + qt_scale(z, n=n, profile=profile, pa=pa) * np.exp(
        log_mean + float(norm.ppf(q)) * sd
    )


def total_log_sd(log_sd, z, n_sd=0.0, profile: StressProfile = DEFAULT_PROFILE,
                 pa: float = PA_KPA):
    """The model's log-space sd and the exponent's, combined in quadrature.

    A Gaussian ``n`` keeps everything closed form, which is the whole reason to
    treat the exponent this way rather than by simulation.  Writing
    ``L = log(sigma_v0_eff / pa)``::

        qt - sigma_v0 = pa * exp(n * L) * Qtn

    so if ``n ~ N(n_mean, n_sd**2)`` and ``log Qtn ~ N(mu, sd**2)``
    independently, the exponent contributes a *lognormal factor* and the product
    of two lognormals is lognormal::

        log(qt - sigma_v0) ~ N(log(pa) + n_mean * L + mu,  sd**2 + (n_sd * L)**2)

    So ``qt`` stays a shifted lognormal, every closed form below still holds, and
    the only change is a wider sd.  **The independence is the assumption to
    watch**: Robertson's ``n`` is a function of ``Ic``, which is itself a
    function of ``Qtn``, so in truth ``n`` and ``Qtn`` are negatively dependent
    within a soil.  Treating them as independent is conservative for the spread
    and is what makes the interval quotable without simulation; a project that
    needs the dependence should sample instead.

    The Gaussian also ignores Robertson's ``n <= 1`` cap.  For a unit whose
    fitted ``n`` sits at the cap the upper half of the assumed spread is
    unphysical, and the honest reading of the resulting band is that its wide
    side is wider than it needs to be.
    """
    log_sd = np.asarray(log_sd, dtype=float)
    extra = exponent_log_sd(z, n_sd, profile=profile, pa=pa)
    return np.sqrt(log_sd**2 + extra**2)


def qt_moments(log_mean, log_sd, z, n=1.0, n_sd=0.0,
               profile: StressProfile = DEFAULT_PROFILE, pa: float = PA_KPA) -> dict:
    """Median, mean, sd and cv of ``qt`` (kPa) from a Gaussian ``log(Qtn)``.

    ``log_mean`` and ``log_sd`` are what the estimators produce — a prediction
    and *one of* the three sds (``sd_latent``, ``sd_obs``, ``sd_reading``); which
    one is a question about which uncertainty is being asked for, and this
    function will not choose.  See :mod:`cpt_geostat.models.profile`.

    Three things to notice before quoting the ``sd``:

    * **The median is not the mean.**  ``exp(log_mean)`` is a median, so
      ``qt_median`` is one too; the mean is larger by ``exp(sd**2 / 2)`` on the
      net-resistance part, which at ``sd = 0.6`` is 20%.
    * **``qt +/- sd`` is not an interval.**  ``qt`` is lognormal-plus-a-constant
      and strongly right-skewed at these sds, so a symmetric band is wrong in
      both directions at once: it overstates how low ``qt`` can go and
      understates how high.  Use :func:`qt_quantile`, and treat the ``sd`` as
      what it is — a moment, for propagating into a downstream calculation that
      wants one.
    * **The scatter is depth-dependent.**  ``sd`` scales with ``c(z)``, so it
      grows down a unit even where the model's log-space uncertainty is
      constant.  ``cv`` is the stable summary, and the cv of the *net*
      resistance, ``sqrt(exp(sd**2) - 1)``, does not depend on depth at all.

    Returns a dict of arrays: ``qt_median``, ``qt_mean``, ``qt_sd``, ``qt_cv``.
    """
    log_mean = np.asarray(log_mean, dtype=float)
    # One sd from here on: the model's, widened for not knowing the exponent.
    # The two are the same kind of quantity — a log-space spread — which is why
    # they combine in quadrature rather than needing a second machinery.
    sd = total_log_sd(log_sd, z, n_sd=n_sd, profile=profile, pa=pa)
    sv, _ = _stresses(z, profile, seabed_ok=True)
    c = qt_scale(z, n=n, profile=profile, pa=pa)

    net_median = np.exp(log_mean)
    net_mean = np.exp(log_mean + sd**2 / 2.0)
    # expm1 rather than exp(.)-1: at the small sds a well-held unit reaches,
    # the subtraction is where the precision goes.
    net_sd = net_mean * np.sqrt(np.expm1(sd**2))

    qt_mean = sv + c * net_mean
    qt_sd = c * net_sd  # the shift by sigma_v0 moves the mean, never the spread
    with np.errstate(invalid="ignore", divide="ignore"):
        # At the seabed everything is zero and the cv is 0/0 — genuinely
        # undefined, and nan says so rather than a warning saying it twice.
        qt_cv = qt_sd / qt_mean
    return {
        "qt_median": sv + c * net_median,
        "qt_mean": qt_mean,
        "qt_sd": qt_sd,
        "qt_cv": qt_cv,
    }


#: Robertson's cap.  Above 1.0 the normalisation would over-correct for
#: overburden; clays sit at the cap and clean sands near 0.5.
N_MAX = 1.0


def robertson_n(ic, sigma_v0_eff, pa: float = PA_KPA):
    """Stress exponent from soil behaviour type index — Robertson (2009).

    ``n = 0.381 * Ic + 0.05 * (sigma_v0_eff / pa) - 0.15``, capped at
    :data:`N_MAX`.  Clean sands come out near 0.5 and clays at the cap, which is
    the physical content: granular resistance scales with the square root of
    confining stress, cohesive resistance in proportion to it.

    No floor is applied.  Values below 0.5 do occur in clean sand at low stress
    and are the formula's answer, not an error to clip away.
    """
    ic = np.asarray(ic, dtype=float)
    sve = np.asarray(sigma_v0_eff, dtype=float)
    return np.minimum(0.381 * ic + 0.05 * (sve / pa) - 0.15, N_MAX)


def soil_behaviour_index(qt, fs, z, profile: StressProfile = DEFAULT_PROFILE,
                         pa: float = PA_KPA, tol: float = 0.01,
                         max_iter: int = 50) -> dict:
    """Solve Robertson (2009) for ``Ic``, ``n`` and ``Qtn`` together.

    ``n`` depends on ``Ic``, ``Ic`` depends on ``Qtn``, and ``Qtn`` depends on
    ``n`` — so the three are found by iteration, starting from ``n = 1`` (i.e.
    ``Qtn = Qt``) and stopping when ``n`` moves by less than ``tol``::

        Fr  = 100 * fs / (qt - sigma_v0)
        Ic  = sqrt((3.47 - log10 Qtn)**2 + (log10 Fr + 1.22)**2)
        n   = min(0.381 * Ic + 0.05 * sigma_v0_eff / pa - 0.15, 1.0)

    **This needs sleeve friction, so it belongs at preparation time.**  The
    dataset contract keeps ``Qtn`` and drops ``qt`` and ``fs``, so an ``Ic``
    cannot be recovered downstream from a prediction — which is the whole reason
    :func:`qt_from_qtn` takes ``n`` as an input.  A project that wants a variable
    exponent runs this over its raw export, averages ``n`` per unit, and records
    the result in its metadata.

    Readings where the net resistance or the friction ratio is non-positive
    (``qt`` below the overburden, or a zero sleeve) have no ``Ic`` and come back
    nan rather than as a fabricated soil type.

    Returns ``n``, ``ic``, ``qtn``, ``fr``, and the convergence report:
    ``iterations``, ``converged``, ``n_unconverged`` and ``max_step``.  The
    count matters more than the flag — the fixed point is approached
    geometrically at about 0.83 per step, so on IJmuiden's 521k readings 99.99%
    are settled within ten iterations while a handful of extreme ``Fr`` take
    thirty, and a bare ``converged=False`` would condemn the whole solve for
    them.  A silently truncated iteration is still a wrong exponent dressed as a
    right one, so the numbers are reported rather than the loop being trusted.
    """
    qt = np.asarray(qt, dtype=float)
    fs = np.asarray(fs, dtype=float)
    sv, sve = _stresses(z, profile)

    net = qt - sv
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = 100.0 * fs / net
    usable = np.isfinite(net) & (net > 0) & np.isfinite(fr) & (fr > 0) & (sve > 0)

    n = np.ones(np.broadcast(qt, fs, sve).shape, dtype=float)
    iterations, max_step, unconverged = 0, 0.0, 0
    for iterations in range(1, max_iter + 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            qtn = net / pa * (pa / sve) ** n
            ic = np.sqrt((3.47 - np.log10(qtn)) ** 2 + (np.log10(fr) + 1.22) ** 2)
            n_new = robertson_n(ic, sve, pa=pa)
        n_new = np.where(usable & np.isfinite(n_new), n_new, np.nan)
        step = np.abs(n_new - n)
        n = np.where(np.isfinite(n_new), n_new, n)
        max_step = float(np.nanmax(step)) if np.isfinite(step).any() else 0.0
        unconverged = int(np.nansum(step > tol))
        if max_step < tol:
            break

    blank = np.where(usable, 1.0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        qtn = net / pa * (pa / sve) ** n
        ic = np.sqrt((3.47 - np.log10(qtn)) ** 2 + (np.log10(fr) + 1.22) ** 2)
    return {
        "n": n * blank,
        "ic": ic * blank,
        "qtn": qtn * blank,
        "fr": fr * blank,
        "iterations": iterations,
        "converged": bool(max_step < tol),
        "n_unconverged": unconverged,
        "max_step": max_step,
        "n_usable": int(usable.sum()),
    }


def infer_gamma_eff(qt, z, qtn, n: float = 1.0, gamma_sat: float = DEFAULT_PROFILE.gamma_sat,
                    pa: float = PA_KPA) -> dict:
    """Recover the effective-stress gradient a supplied ``Qtn`` column implies.

    A forensic tool, for the situation this project actually met: a deliverable
    carries ``qt`` *and* a normalised column, and the memo does not say which
    formula, unit weights or stress exponent produced the second from the first.
    Rather than assume, solve for the gradient that reproduces the column and
    see whether the answer is a plausible soil.

    With ``n`` and ``gamma_sat`` fixed, ``qt - sigma_v0 = Qtn * pa ** (1 - n) *
    (gamma_eff * z) ** n`` is linear in ``gamma_eff ** n``, so the fit is one
    least-squares ratio.  ``gamma_sat`` is passed rather than fitted because it
    is barely identifiable — ``sigma_v0`` is under 1% of ``qt`` over most of a
    hole, so the data cannot pin it, and pretending otherwise would report a
    precise number for the one parameter that does not matter.

    Returns ``gamma_eff``, ``n``, ``gamma_sat``, and the ratio of the
    reconstructed column to the supplied one summarised as ``ratio_median`` and
    ``ratio_p05``/``ratio_p95``.  A median near 1.0 with a tight spread means the
    assumed ``n`` is right; a median far from 1.0 *and* a spread that widens with
    depth means it is not, and no gradient will fix that.
    """
    qt = np.asarray(qt, dtype=float)
    z = np.asarray(z, dtype=float)
    qtn = np.asarray(qtn, dtype=float)
    ok = np.isfinite(qt) & np.isfinite(z) & np.isfinite(qtn) & (z > 0) & (qtn > 0)
    if not ok.any():
        raise ValueError("no finite readings with z > 0 and Qtn > 0 to fit")
    qt, z, qtn = qt[ok], z[ok], qtn[ok]

    # (qt - sigma_v0) = k * (Qtn * pa**(1-n) * z**n), least squares through the
    # origin, so k = gamma_eff**n.
    target = qt - gamma_sat * z
    basis = qtn * pa ** (1.0 - n) * z**n
    k = float(basis @ target / (basis @ basis))
    if k <= 0:
        raise ValueError(f"fit gives a non-positive gradient (k={k:.4g}); check n and the units")
    gamma_eff = k ** (1.0 / n)

    fitted = StressProfile.from_gradients(gamma_sat, gamma_sat - gamma_eff)
    ratio = qtn_from_qt(qt, z, n=n, profile=fitted, pa=pa) / qtn
    p05, p95 = np.percentile(ratio, [5, 95])
    return {
        "gamma_eff": gamma_eff,
        "n": float(n),
        "gamma_sat": float(gamma_sat),
        "n_readings": int(ok.sum()),
        "ratio_median": float(np.median(ratio)),
        "ratio_p05": float(p05),
        "ratio_p95": float(p95),
    }
