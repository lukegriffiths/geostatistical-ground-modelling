"""Empirical variograms and model fitting.

Fitting three free parameters to a scatter of 20-120 points is unstable, and it
fails quietly: an unconstrained fit returns a length scale of 1700 km rather
than raising.  Three constraints, each removing a specific way of failing:

**Bins resolve the short lags.**  The nugget is only identifiable from pairs
closer together than the correlation range, and on a turbine grid there are
almost none — which is why ``config.yaml`` puts four CPTs 60 m apart.  Uniform
bins average those six pairs into a 600 m bin and the nugget fits to zero; a
first bin of 200 m recovers it.  The cluster is the only thing separating the
nugget from a short range, and binning is what lets it do its job.

**Lags are capped at half the maximum separation.**  The usual rule, and here it
is also the bound on what can be claimed: a range longer than the largest fitted
lag is not identifiable, so ``len_scale`` is bounded to match and the fit records
whether it ended up against that bound.

**The total sill is fixed to the sample variance.**  It is the one quantity
these data estimate well, and pinning it stops sill, nugget and range from
trading off freely — the failure that produced a 22 km range on unit 4.  The
cost is that the split between structure and noise has to be paid for out of a
known total, which is the honest constraint.

None of this rescues a unit whose structure is finer than the CPT spacing.  Unit
3's minor axis is 1 km against a 1.3 km grid, so an isotropic variogram sees
noise and fits nearly all nugget.  **That is the correct answer**, not a fitting
failure, and :attr:`VariogramFit.resolved` says so rather than leaving the
reader to infer it from a suspicious number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gstools as gs
import numpy as np
import pandas as pd

from ..covariance import MODELS, GrfConfig, len_scale_to_range, range_to_len_scale
from ..geometry import azimuth_unit_vector

#: Bin edges below this are spaced by hand to catch clustered pairs (km).
_NEAR_EDGES = (0.0, 0.2, 0.5, 1.0)
_N_FAR_BINS = 10
_MAX_LAG_FRACTION = 0.5

#: Sector azimuths for a directional variogram, degrees CW from north.
#: Four sectors at 45 deg spacing span every axis once — see
#: :func:`directional_variogram` on why the range is 0-180 and not 0-360.
DEFAULT_SECTORS = (0.0, 45.0, 90.0, 135.0)

#: Half-width of each sector.  22.5 deg makes the four sectors tile 0-180
#: exactly: wide enough to hold pairs, narrow enough that a ratio-4 anisotropy
#: is not averaged away by mixing the major and minor axes into one estimate.
DEFAULT_SECTOR_TOL_DEG = 22.5

#: Below this many pairs a sector's variogram is drawn but not believed.
MIN_SECTOR_PAIRS = 30

#: Lag bins for a *directional* estimate.  Deliberately far coarser than the
#: omnidirectional binning: splitting the pairs four ways leaves each sector
#: with a quarter of them, so keeping the same ~13 bins puts 4-9 pairs in each
#: and produces a curve that is pure sampling noise — which then reads as
#: dramatic anisotropy.  Six uniform bins is the coarsest useful compromise.
DIRECTIONAL_N_BINS = 6


@dataclass
class VariogramFit:
    """A fitted model plus what it is safe to say about it."""

    model: object  # gstools CovModel, nugget included
    sill: float
    range_km: float  # practical
    nugget: float
    sample_var: float
    n_pairs: int
    n_short_pairs: int  # pairs inside the first bin — what pins the nugget
    max_lag_km: float
    at_range_bound: bool
    nugget_floored: bool = False
    fit_failed: bool = False  # optimiser gave up; fell back to pure nugget

    lags: np.ndarray = field(default_factory=lambda: np.empty(0))
    gamma: np.ndarray = field(default_factory=lambda: np.empty(0))
    counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def structured_fraction(self) -> float:
        """Share of the total variance the model calls structure, not noise."""
        total = self.sill + self.nugget
        return float(self.sill / total) if total > 0 else float("nan")

    @property
    def resolved(self) -> bool:
        """Is there structure here that kriging can actually exploit?

        False when the fit is essentially pure nugget, or when the range ran to
        the bound — in both cases kriging will reproduce the unit mean and the
        honest report is that the layout cannot see this unit's structure.
        """
        return bool(
            self.structured_fraction > 0.1
            and not self.at_range_bound
            and self.n_pairs > 0
            and not self.fit_failed
        )

    def why_not_resolved(self) -> Optional[str]:
        if self.resolved:
            return None
        if self.n_pairs == 0:
            return "no pairs"
        if self.fit_failed:
            return "variogram fit did not converge — treated as pure nugget"
        if self.at_range_bound:
            return f"range ran to the {self.max_lag_km:.1f} km bound (trend or too few lags)"
        return (
            f"nearly pure nugget ({self.structured_fraction:.0%} structured) — "
            f"structure is finer than the CPT spacing"
        )


def bin_edges(x, y, max_lag_fraction: float = _MAX_LAG_FRACTION, n_far: int = _N_FAR_BINS):
    """Short lags resolved by hand, then uniform to half the maximum separation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    max_lag = float(d.max()) * max_lag_fraction
    near = np.array([e for e in _NEAR_EDGES if e < max_lag])
    if near.size < 2:
        return np.linspace(0.0, max(max_lag, 1e-6), n_far + 1)
    return np.concatenate([near, np.linspace(near[-1] * 1.5, max_lag, n_far)])


def empirical_variogram(x, y, v, edges=None):
    """``(lags, gamma, counts)`` — omnidirectional, empty bins dropped."""
    if edges is None:
        edges = bin_edges(x, y)
    lags, gamma, counts = gs.vario_estimate(
        (np.asarray(x, dtype=float), np.asarray(y, dtype=float)),
        np.asarray(v, dtype=float),
        bin_edges=edges,
        return_counts=True,
    )
    keep = counts > 0
    return lags[keep], gamma[keep], counts[keep]


def directional_variogram(
    x, y, v,
    azimuths=DEFAULT_SECTORS,
    tol_deg: float = DEFAULT_SECTOR_TOL_DEG,
    bandwidth_km: Optional[float] = None,
    edges=None,
):
    """``{azimuth: (lags, gamma, counts)}`` — one estimate per azimuth sector.

    The diagnostic an isotropic fit cannot provide.  Unit 3's minor axis is
    finer than the CPT spacing, so an omnidirectional variogram correctly sees
    noise and fits nearly all nugget — but the structure *is* there, off-axis,
    and separating the lags by bearing is the only way to see it.

    **A variogram direction is an axis, not a bearing.**  Pairs are unordered,
    so 70 deg and 250 deg select exactly the same pairs and return exactly the
    same estimate.  Sectors therefore span 0-180 (as
    :func:`cpt_geostat.geometry.pair_distances` already folds them), and asking for
    both halves would double the figure without adding an estimate.

    ``bandwidth_km`` caps the perpendicular offset a pair may have from the
    sector axis.  Without it a wide sector at a long lag admits pairs that are
    nearly perpendicular to the direction being estimated, which blurs exactly
    the contrast the figure exists to show.  ``None`` lets gstools decide.

    Azimuths reach gstools through :func:`cpt_geostat.geometry.azimuth_unit_vector`
    and nowhere else: gstools wants a direction vector in the ``(x, y)`` frame,
    that helper is the package's single definition of one, and a sign slip or a
    from-east mix-up here rotates every recovered axis by 90 deg while still
    producing a plausible-looking figure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    if edges is None:
        # Uniform and coarse, rather than the hand-spaced short-lag edges used
        # omnidirectionally.  Those exist to pin the *nugget*, which a single
        # sector never holds enough close pairs to do; a directional figure is
        # asking about range and bearing, and it needs pairs per bin instead.
        edges = np.linspace(0.0, float(bin_edges(x, y)[-1]), DIRECTIONAL_N_BINS + 1)

    kwargs = {} if bandwidth_km is None else {"bandwidth": float(bandwidth_km)}
    out = {}
    for az in azimuths:
        lags, gamma, counts = gs.vario_estimate(
            (x, y), v,
            bin_edges=edges,
            direction=[list(azimuth_unit_vector(az))],
            angles_tol=np.deg2rad(tol_deg),
            return_counts=True,
            **kwargs,
        )
        lags = np.asarray(lags).ravel()
        gamma = np.asarray(gamma).ravel()
        counts = np.asarray(counts).ravel()
        keep = counts > 0
        out[float(az)] = (lags[keep], gamma[keep], counts[keep])
    return out


#: Floor the nugget at gamma(h_min) only when h_min is this small a fraction of
#: the fitted range.  Beyond it, gamma(h_min) is mostly structure, not nugget.
_NUGGET_FLOOR_LAG_FRACTION = 0.1


def _nugget_floor(lags, gamma, range_km: float) -> float:
    """A lower bound on the nugget from the shortest resolved lag, or 0.

    An optimiser handed a variogram with no pairs below the correlation range
    will happily return a nugget of exactly zero, and kriging then interpolates
    noisy depth-averages as though they were exact.  But ``gamma(h)`` is nugget
    *plus* whatever structure has accumulated by ``h``, so it is only a usable
    nugget estimate when ``h`` is far inside the range — on this layout that
    means the four clustered CPTs at 60 m, and nothing else.

    Where the shortest available lag is not short enough the floor declines to
    act, leaving a zero nugget visible as a zero rather than papering over it
    with a number that is really a range estimate in disguise.
    """
    if not lags.size or not np.isfinite(range_km) or range_km <= 0:
        return 0.0
    if lags[0] > _NUGGET_FLOOR_LAG_FRACTION * range_km:
        return 0.0
    return float(max(gamma[0], 0.0))


def fit_variogram(
    x, y, v, model: str = "matern25", fix_sill: bool = True, nugget_floor: bool = True
) -> VariogramFit:
    """Fit an isotropic ``model`` to the empirical variogram of ``v`` at ``(x, y)``.

    ``fix_sill`` pins ``sill + nugget`` to the sample variance.  Turning it off
    is supported for diagnostics but is not the default, because an unconstrained
    three-parameter fit on this many points does not fail loudly.

    ``nugget_floor`` applies :func:`_nugget_floor`.  It changes nothing on a unit
    without genuinely short-lag pairs, so it is safe to leave on.
    """
    if model not in MODELS:
        raise ValueError(f"unknown covariance model {model!r}; have {sorted(MODELS)}")
    v = np.asarray(v, dtype=float)
    edges = bin_edges(x, y)
    lags, gamma, counts = empirical_variogram(x, y, v, edges)
    sample_var = float(np.var(v, ddof=1)) if v.size > 1 else float("nan")
    max_lag = float(edges[-1])

    cov = MODELS[model](dim=2)
    # A range beyond the longest fitted lag is not identifiable; bound len_scale
    # so the optimiser cannot claim one, and record when it sits on the bound.
    max_len = float(range_to_len_scale(GrfConfig(range_km=max_lag, model=model)))
    cov.set_arg_bounds(len_scale=[1e-3, max_len])

    fit_failed = False
    if lags.size >= 3:
        kwargs = {"sill": sample_var} if (fix_sill and np.isfinite(sample_var)) else {}
        try:
            cov.fit_variogram(lags, gamma, nugget=True, **kwargs)
        except RuntimeError:
            # The optimiser gave up.  This is rare but real — a near-flat
            # residual variogram (universal kriging on a unit whose drift
            # absorbs almost everything) has no curvature for it to latch on
            # to.  Falling back to pure nugget says "no structure I can fit",
            # which is the truth; raising would kill the whole CV run for one
            # fold, and a bare `except` here would hide genuine bugs.
            fit_failed = True
            cov.var = 0.0
            cov.nugget = sample_var if np.isfinite(sample_var) else 0.0
            cov.len_scale = max_len
    else:
        # Not enough bins to fit anything: call it all noise, which is what a
        # constant-mean fallback amounts to, rather than inventing a range.
        cov.var = 0.0
        cov.nugget = sample_var if np.isfinite(sample_var) else 0.0
        cov.len_scale = max_len

    range_km = len_scale_to_range(cov.len_scale, model, "practical", 2)
    floor = _nugget_floor(lags, gamma, range_km) if nugget_floor else 0.0
    if floor > cov.nugget:
        # Take it out of the sill, so the total variance still matches the data.
        cov.var = max(float(cov.var) - (floor - float(cov.nugget)), 0.0)
        cov.nugget = floor

    return VariogramFit(
        model=cov,
        sill=float(cov.var),
        range_km=range_km,
        nugget=float(cov.nugget),
        nugget_floored=bool(floor > 0),
        fit_failed=fit_failed,
        sample_var=sample_var,
        n_pairs=int(counts.sum()),
        n_short_pairs=int(counts[0]) if counts.size else 0,
        max_lag_km=max_lag,
        at_range_bound=bool(cov.len_scale >= 0.98 * max_len),
        lags=lags,
        gamma=gamma,
        counts=counts,
    )


#: Fewest CPTs a variogram is attempted from at all.
MIN_CPT_FOR_FIT = 3

#: Fewest CPTs before a *directional* variogram is worth estimating.
#: Plan 02's estimability tier: anisotropy needs pairs spread over ~4 azimuth
#: sectors and ~10 lag bins, which is roughly n >= 30 (435 pairs, ~11 per bin).
#: Below it the sector curves are noise and inviting a reader to compare them
#: is worse than not drawing them.
MIN_CPT_FOR_DIRECTIONAL = 30


def unit_block(ds, unit_id: str) -> pd.DataFrame:
    """The ``unit_summary`` rows for one unit — the model input for that unit."""
    return ds.unit_summary[ds.unit_summary["unit_id"] == unit_id]


def fit_unit_variogram(ds, unit_id: str, **kwargs) -> Optional[VariogramFit]:
    """Fit one unit's variogram from ``ds.unit_summary``, or ``None`` if too thin.

    One definition of "the variogram for this unit", shared by the CLI's summary
    table and by the figures.  Two copies of the same three lines would be free
    to disagree about the threshold or the value column, and the figure would
    then illustrate a different fit from the one in the table.
    """
    block = unit_block(ds, unit_id)
    if len(block) < MIN_CPT_FOR_FIT:
        return None
    return fit_variogram(
        block["x"].to_numpy(), block["y"].to_numpy(), block["log_Q_mean"].to_numpy(), **kwargs
    )
