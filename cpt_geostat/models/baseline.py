"""The no-spatial-structure baseline: one mean and one variance per unit.

This is the model a ground investigation report already contains implicitly —
"unit 3 has a log Qtn of 1.9 ± 0.3" — written down as an estimator so that
everything spatial has something to be measured against.  It is deliberately
the weakest possible model, and on units where a GP cannot beat it, *it is the
answer*: unit 6 (nugget > sill) and unit 5 (short range, sparse coverage) are in
the contrast set precisely so that outcome shows up rather than being assumed
away.  On IJmuiden it is also the mandated fallback for the ten units held at
too few CPTs to fit anything directional.

Three different "average and std" live in this module and they answer three
different questions.  Conflating them is the easiest mistake here, so each has
its own column and its own name:

``log_Q_mean`` / ``log_Q_sd`` (across CPTs)
    Mean and sd of the per-CPT depth-averages, **one weight per CPT**.  This is
    the estimator: the distribution of the unit's value at an unvisited
    location.  It is what a GP or kriging prediction gets compared against, and
    the sd is the number a spatial model has to reduce to earn its keep.

``within_sd`` (within a CPT, pooled)
    Depth-to-depth scatter about a single hole's own average.  Not spatial at
    all — it is the texture of the trace — and it is *not* reduced by adding
    more CPTs.

``reading_sd`` (over every reading)
    Scatter of a single 2 cm reading about the unit's overall mean, pooled over
    the whole site.  This is what to quote for "what Qtn will the cone see in
    this unit", and it is the only one of the three that weights a CPT by how
    thick the unit is there.

The first is computed from ``unit_summary``, so it runs unchanged on real data;
the third is computed from ``cpt_samples`` directly rather than by adding the
first two in quadrature, which would double-count the depth-averaging error.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import SpatialEstimator, check_coords

# Quantiles quoted for Qtn.  Deciles rather than 95% bounds: with 20-120 CPTs
# per unit the tails are not resolved, and a P10-P90 range does not pretend
# otherwise.
_P_LOW, _P_HIGH = 0.10, 0.90
_Z_LOW, _Z_HIGH = float(norm.ppf(_P_LOW)), float(norm.ppf(_P_HIGH))


class UnitMeanEstimator(SpatialEstimator):
    """Constant mean over the site; all residual scatter called noise.

    The model is ``y_i = m + e_i`` with ``e_i`` iid.  Because it has no
    covariance function, it cannot separate spatial structure from measurement
    noise and does not pretend to: everything about the constant is attributed
    to the nugget, and :attr:`params_` reports the sill as exactly zero and the
    range as *not identifiable* rather than as a large number.

    That attribution is what makes the two variances come out right:

    * :meth:`predict` — the latent field is the constant ``m``, known only to
      within its standard error ``s / sqrt(n)``, so the latent sd *shrinks* with
      more CPTs.  It is not a claim that the field is flat to that precision;
      it is the flat model's own uncertainty about where its one level sits.
    * :meth:`predict_observation` — adds ``noise_var_ = s**2`` back, giving
      ``s * sqrt(1 + 1/n)``, the textbook predictive sd for a new draw.  This is
      the one cross-validation scores, and it is honest: on a unit with no
      exploitable structure it is also very nearly optimal.

    ``fallback_var`` covers the case that a single CPT holds the unit, where the
    sample variance does not exist.  Supplying a pooled residual variance from
    the other units keeps the estimator usable and, more importantly, keeps it
    *wide*; leaving it ``None`` yields ``nan``, which fails loudly downstream.
    """

    def __init__(self, fallback_var: Optional[float] = None, ddof: int = 1):
        self.fallback_var = fallback_var
        self.ddof = ddof

    def fit(self, X, y) -> "UnitMeanEstimator":
        y = np.asarray(y, dtype=float).ravel()
        if X is not None:
            X = check_coords(X)
            if len(X) != y.size:
                raise ValueError(f"X has {len(X)} rows but y has {y.size}")
        finite = np.isfinite(y)
        y = y[finite]
        if y.size == 0:
            raise ValueError("no finite observations to fit")

        self.n_ = int(y.size)
        self.mean_ = float(y.mean())
        if self.n_ > self.ddof:
            self.residual_var_ = float(y.var(ddof=self.ddof))
        elif self.fallback_var is not None:
            self.residual_var_ = float(self.fallback_var)
        else:
            self.residual_var_ = float("nan")
        self.mean_var_ = self.residual_var_ / self.n_
        return self

    def predict(self, X, return_std: bool = False):
        """Latent field: the fitted constant, with its standard error."""
        n = len(check_coords(X))
        mean = np.full(n, self.mean_)
        if not return_std:
            return mean
        return mean, np.full(n, math.sqrt(self.mean_var_))

    @property
    def noise_var_(self) -> float:
        return self.residual_var_

    @property
    def params_(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_,
            "sill": 0.0,  # asserted, not fitted: the model has no spatial term
            "range_km": None,  # not identifiable
            "aniso_ratio": None,
            "aniso_angle_deg": None,
            "nugget": self.residual_var_,
            "n": self.n_,
        }

    def __repr__(self) -> str:
        if not hasattr(self, "mean_"):
            return "UnitMeanEstimator(unfitted)"
        return (
            f"UnitMeanEstimator(n={self.n_}, mean={self.mean_:.3f}, "
            f"sd={math.sqrt(self.residual_var_):.3f})"
        )


# --------------------------------------------------------------------------- #
# per-unit fitting and reporting
# --------------------------------------------------------------------------- #


def _unit_summary_of(ds) -> pd.DataFrame:
    """Accept a :class:`~cpt_geostat.contract.schema.Dataset` or a bare dataframe."""
    return ds if isinstance(ds, pd.DataFrame) else ds.unit_summary


def _unit_ids_of(ds, summary: pd.DataFrame):
    if not isinstance(ds, pd.DataFrame):
        return [u for u in ds.unit_ids if u in set(summary["unit_id"])]
    return list(summary["unit_id"].drop_duplicates())


def pooled_residual_var(summary: pd.DataFrame, ddof: int = 1) -> float:
    """Residual variance of ``log_Q_mean`` about its own unit mean, pooled.

    The fallback for units too thinly held to have a variance of their own.  It
    is pooled *within* unit — units differ in level by design, so pooling the
    raw values instead would inflate it by the between-unit spread.
    """
    dev = summary["log_Q_mean"] - summary.groupby("unit_id")["log_Q_mean"].transform("mean")
    n, k = len(summary), summary["unit_id"].nunique()
    if n - k <= 0:
        return float("nan")
    return float((dev**2).sum() / (n - k)) if ddof else float((dev**2).mean())


def fit_unit_baselines(ds, fallback_var: Optional[float] = None) -> Dict[str, UnitMeanEstimator]:
    """One :class:`UnitMeanEstimator` per unit, keyed by ``unit_id``.

    ``fallback_var`` defaults to :func:`pooled_residual_var` over all units, so a
    unit held at one CPT gets a wide interval instead of a ``nan``.
    """
    summary = _unit_summary_of(ds)
    if fallback_var is None:
        fallback_var = pooled_residual_var(summary)

    out = {}
    for uid in _unit_ids_of(ds, summary):
        block = summary[summary["unit_id"] == uid]
        X = block[["x", "y"]].to_numpy(dtype=float)
        out[uid] = UnitMeanEstimator(fallback_var=fallback_var).fit(X, block["log_Q_mean"])
    return out


def baseline_factory(ds, fallback_var: Optional[float] = None):
    """``unit_id -> fresh unfitted UnitMeanEstimator``, for cross-validation.

    Cross-validation needs an *unfitted* estimator per fold, not the fitted ones
    from :func:`fit_unit_baselines`.  The pooled fallback is computed once from
    the whole dataset rather than per fold: it is a property of the site, and
    recomputing it inside each fold would leak nothing but cost n times more.
    """
    summary = _unit_summary_of(ds)
    if fallback_var is None:
        fallback_var = pooled_residual_var(summary)
    return lambda unit_id: UnitMeanEstimator(fallback_var=fallback_var)


def _pooled_within_sd(block: pd.DataFrame) -> float:
    """Depth-to-depth sd pooled across CPTs, weighted by ``n_samples - 1``."""
    sd = block["log_Q_sd"].to_numpy(dtype=float)
    dof = block["n_samples"].to_numpy(dtype=float) - 1.0
    ok = np.isfinite(sd) & (dof > 0)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.sum(dof[ok] * sd[ok] ** 2) / np.sum(dof[ok])))


def _reading_stats(samples: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """Mean and sd of ``log(Qtn)`` over every reading, per unit."""
    if samples is None or not len(samples):
        return {}
    lq = np.log(samples["Qtn"].to_numpy(dtype=float))
    grouped = (
        samples.assign(_lq=lq)
        .groupby("unit_id", sort=False)["_lq"]
        .agg(reading_mean="mean", reading_sd="std", n_readings="size")
    )
    return grouped.to_dict("index")


def unit_baseline_table(ds, fallback_var: Optional[float] = None) -> pd.DataFrame:
    """The per-unit average-and-std report, one row per unit.

    Columns, in the order they should be read:

    ``n_cpt``, ``cpt_fraction``
        How many CPTs hold the unit, and what fraction of the site's CPTs that
        is.  Everything to the right is conditional on presence — these are
        statistics *of the unit where it exists*, which is the only thing the
        data supports and is not the same as a statistic of the site.
    ``log_Q_mean``, ``log_Q_sd``, ``se_mean``
        The estimator.  ``se_mean = log_Q_sd / sqrt(n_cpt)`` is the uncertainty
        on the level; ``log_Q_sd`` is the sd of a *new location*, which is
        roughly ``sqrt(n_cpt)`` times larger and is the number that matters for
        design.
    ``within_sd``
        Depth-to-depth scatter inside one hole, pooled.  Not reducible by
        drilling more CPTs.
    ``reading_mean``, ``reading_sd``, ``n_readings``
        The pooled single-reading distribution, thickness-weighted.
    ``Qtn_median``, ``Qtn_p10``, ``Qtn_p90``
        The reading-level distribution back in Qtn.  ``exp`` of a mean of logs
        is a **median**, not a mean — labelled as such so it does not get
        quoted as an average.

    Works on real data: only ``unit_summary`` is required, and the reading-level
    columns are dropped rather than faked when ``cpt_samples`` is unavailable.
    """
    summary = _unit_summary_of(ds)
    samples = None if isinstance(ds, pd.DataFrame) else getattr(ds, "samples", None)
    reading = _reading_stats(samples)
    models = fit_unit_baselines(ds, fallback_var=fallback_var)
    n_cpt_total = summary["cpt_id"].nunique()

    rows = []
    for uid, model in models.items():
        block = summary[summary["unit_id"] == uid]
        sd = math.sqrt(model.residual_var_)
        row = {
            "unit_id": uid,
            "n_cpt": model.n_,
            "cpt_fraction": model.n_ / n_cpt_total if n_cpt_total else float("nan"),
            "log_Q_mean": model.mean_,
            "log_Q_sd": sd,
            "se_mean": math.sqrt(model.mean_var_),
            "within_sd": _pooled_within_sd(block),
            "thickness_mean_m": float(block["thickness_m"].mean()),
        }
        if uid in reading:
            r = reading[uid]
            row.update(
                reading_mean=float(r["reading_mean"]),
                reading_sd=float(r["reading_sd"]),
                n_readings=int(r["n_readings"]),
                Qtn_median=math.exp(r["reading_mean"]),
                Qtn_p10=math.exp(r["reading_mean"] + _Z_LOW * r["reading_sd"]),
                Qtn_p90=math.exp(r["reading_mean"] + _Z_HIGH * r["reading_sd"]),
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("unit_id")


# The truth comparison (synthetic only) lives in :mod:`cpt_geostat.synthetic.truth`;
# nothing in this package imports from the synthetic side.

