"""Turning per-unit predictions back into a Qtn-versus-depth profile.

The estimators predict **one number per unit per location** — a depth-average of
``log Qtn``.  This module puts that number back beside the thing it describes:
the actual CPT trace.  Done leave-one-out it is the most direct validation the
project has, because it asks *"if this hole had never been drilled, what would
we have said, and how wrong would we have been?"*

Three nested uncertainties, and conflating them is the easy mistake here — they
are the same three quantities :mod:`cpt_geostat.models.baseline` separates, carried
through to a prediction:

``sd_latent``
    Where the unit's **true** value is at this location.  Shrinks as more CPTs
    are drilled; this is what a spatial model reduces.
``sd_obs``
    What a new **depth-average** here would be: latent plus the nugget and the
    depth-averaging error.  This is what cross-validation scores, because a
    held-out ``log_Q_mean`` is an observation.
``sd_reading``
    Where an individual 2 cm **reading** will fall: adds the within-unit
    depth-to-depth scatter.  This is the only one comparable with the raw trace,
    and on IJmuiden it is much the widest — ``within_sd`` runs 0.45-0.56 against
    a between-hole ``log_Q_sd`` of 0.31-0.71, so depth texture inside one hole
    is as large as the variation between holes.  Quoting the narrow band against
    a raw trace would claim a precision the model does not have.

**The reading band slightly over-counts and the honest response is to measure
it.**  ``sd_obs`` already contains the depth-averaging error, which is itself
produced by the within-unit scatter, so ``sd_obs**2 + within_sd**2`` counts that
part twice.  The averaging error is the smaller term, but rather than argue a
correction, :func:`profile_coverage` reports the *realised* coverage and the
figures print it — the same treatment every other calibration claim here gets.

Predictions are joined to readings on ``(cpt_id, unit_id)`` and never through
``layers``.  ``cpt_samples`` already labels every reading with its unit, and
going through the layer table would break on the 19 IJmuiden holes where a unit
re-enters lower down: its collapsed layer row spans first-top to last-base and
claims depth the unit does not occupy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..cpt import (
    DEFAULT_PROFILE,
    StressProfile,
    qt_from_qtn,
    qt_moments,
    qt_quantile,
    total_log_sd,
)
from .baseline import unit_baseline_table

#: Columns a cv table must supply — the output of :func:`cpt_geostat.validate.cv.loo_by_unit`.
_CV_COLUMNS = ["cpt_id", "unit_id", "pred", "sd_latent", "sd_obs"]

#: Band name -> the sd it is built from, shared by the log and ``qt`` tables.
_BANDS = {"mean": "sd_obs", "read": "sd_reading"}


def _select_model(cv: pd.DataFrame, model: Optional[str]) -> pd.DataFrame:
    """One model's rows.  Ambiguity raises rather than picking silently."""
    if "model" not in cv.columns:
        return cv
    names = list(pd.unique(cv["model"]))
    if not names:
        # No folds produced a prediction at all — a legitimate state on a site
        # where every unit is too thinly held.  Every reading then comes back
        # nan and the figures show gaps, which is the same graceful degradation
        # `loo_predict` already chooses over raising.
        return cv
    if model is None:
        if len(names) != 1:
            raise ValueError(
                f"cv holds {len(names)} models {names}; pass model= to choose one — "
                f"a profile drawn from a silently picked estimator is unattributable"
            )
        model = names[0]
    if model not in names:
        raise ValueError(f"no model {model!r} in cv table; have {names}")
    return cv[cv["model"] == model]


def _within_sd_by_unit(ds, within_sd) -> pd.Series:
    """Pooled within-unit sd per unit.

    Pooled across CPTs rather than taken from the hole's own ``log_Q_sd``: the
    profile predicts this location *as if unvisited*, and its own depth-to-depth
    scatter is exactly what would not be known.
    """
    if within_sd is not None:
        return pd.Series(within_sd, name="within_sd")
    table = unit_baseline_table(ds)
    return table["within_sd"].rename("within_sd")


def reading_predictions(
    ds,
    cv: pd.DataFrame,
    model: Optional[str] = None,
    within_sd=None,
    level: float = 0.95,
) -> pd.DataFrame:
    """One row per reading: the actual value beside what the model predicted.

    ``cv`` is a :func:`cpt_geostat.validate.cv.loo_by_unit` table, so the predictions
    are leave-one-out and the whole spatial half is reused rather than refitted.

    Returns ``cpt_id, z, unit_id, Qtn, log_Q`` (the data) alongside ``pred``,
    the three sds, ``Qtn_median`` and the band edges in both log and Qtn units,
    and ``resid_z`` — the standardised residual against the reading band, which
    is what :func:`profile_coverage` and the calibration figure consume.

    ``exp(pred)`` is a **median**, not a mean: it is the exponential of a mean of
    logs.  It is named ``Qtn_median`` for the same reason the baseline table
    names it that.  The band edges transform exactly, since ``exp`` is monotone
    and quantiles are equivariant under a monotone map — which is why the
    interval is built in log space and exponentiated rather than approximated
    with a delta method.
    """
    missing = [c for c in _CV_COLUMNS if c not in cv.columns]
    if missing:
        raise ValueError(f"cv table is missing required column(s): {missing}")

    block = _select_model(cv, model)[_CV_COLUMNS]
    samples = ds.samples
    if not len(samples):
        raise ValueError("dataset has no cpt_samples; nothing to profile")

    out = samples[["cpt_id", "z", "unit_id", "Qtn"]].copy()
    out["log_Q"] = np.log(out["Qtn"].to_numpy(dtype=float))
    # Left join: a reading whose unit had no fittable fold keeps its actual
    # value and gets nan predictions, so the gap survives to the figure.
    out = out.merge(block, on=["cpt_id", "unit_id"], how="left")

    w = _within_sd_by_unit(ds, within_sd)
    out["within_sd"] = out["unit_id"].map(w).astype(float)
    # A unit held at one CPT has no pooled within-unit sd; treat it as zero
    # extra spread rather than nan, which would erase an otherwise usable
    # unit-mean prediction from the figure.
    within = np.nan_to_num(out["within_sd"].to_numpy(dtype=float), nan=0.0)
    out["sd_reading"] = np.sqrt(out["sd_obs"].to_numpy(dtype=float) ** 2 + within**2)

    z = float(norm.ppf(0.5 + level / 2.0))
    out["Qtn_median"] = np.exp(out["pred"])
    for name, sd in _BANDS.items():
        lo = out["pred"] - z * out[sd]
        hi = out["pred"] + z * out[sd]
        out[f"log_lo_{name}"] = lo
        out[f"log_hi_{name}"] = hi
        out[f"Qtn_lo_{name}"] = np.exp(lo)
        out[f"Qtn_hi_{name}"] = np.exp(hi)

    out["resid_z"] = (out["log_Q"] - out["pred"]) / out["sd_reading"]
    out.attrs["level"] = level
    out.attrs["model"] = model or _model_name(cv)
    return out.sort_values(["cpt_id", "z"]).reset_index(drop=True)


def _model_name(cv: pd.DataFrame) -> str:
    """A label for the figure title; never indexes an empty table."""
    if "model" in cv.columns:
        names = list(pd.unique(cv["model"]))
        if names:
            return str(names[0])
    return "model"


def _by_unit(value, unit_ids: pd.Series, name: str):
    """A scalar, or one value per unit resolved against ``unit_id``.

    A variable stress exponent is per *unit*, not per reading: it comes from a
    soil type, and the model's whole geometry is one soil type per unit.  A unit
    with no value raises rather than falling back to a default — a silently
    defaulted exponent on one unit would put a depth-dependent bias into that
    unit alone, which looks exactly like a real contrast between soils.
    """
    if not isinstance(value, (Mapping, pd.Series)):
        return float(value)
    lookup = value if isinstance(value, pd.Series) else pd.Series(value, dtype=float)
    out = unit_ids.map(lookup).astype(float)
    if out.isna().any():
        missing = sorted(set(unit_ids[out.isna()].astype(str)))
        raise ValueError(f"no {name} given for unit(s) {missing}")
    return out.to_numpy()


def _exponent_label(n, n_sd) -> str:
    """How the exponent reads in a caption: one number, or a range per unit."""
    if isinstance(n, (Mapping, pd.Series)):
        values = np.asarray(list(dict(n).values()), dtype=float)
        label = f"n {values.min():.2f}-{values.max():.2f} by unit"
    else:
        label = f"n = {float(n):g}"
    spread = np.asarray(list(dict(n_sd).values()) if isinstance(n_sd, (Mapping, pd.Series))
                        else [n_sd], dtype=float)
    if np.any(spread > 0):
        label += (f" ± {spread[0]:g}" if spread.size == 1
                  else f" ± {spread.min():.2f}-{spread.max():.2f}")
    return label


def qt_readings(
    readings: pd.DataFrame,
    profile: StressProfile = DEFAULT_PROFILE,
    n=1.0,
    n_sd=0.0,
    level: Optional[float] = None,
) -> pd.DataFrame:
    """The reading table again, in ``qt`` (kPa) — the units a design uses.

    Takes the output of :func:`reading_predictions` and adds ``qt`` (the reading
    itself, de-normalised), ``qt_median``, and per band the interval edges and a
    ``qt_sd``/``qt_cv``.

    ``n`` and ``n_sd`` may each be a number or a ``unit_id -> value`` mapping.
    With ``n_sd = 0`` — the exponent treated as known — the edges come out
    *identical* to the log-space edges pushed through the monotone map, so a
    band that was 95% in ``log(Qtn)`` is exactly 95% in ``qt`` and the coverage
    :func:`profile_coverage` reports carries over untouched.

    **With ``n_sd > 0`` that equality is deliberately broken**, and the honest
    reading of the wider band is not "the model got worse".  The exponent
    uncertainty is uncertainty about *the transform*, not about the ground: the
    ``qt`` column on the same row is itself conditional on the central ``n``, so
    the measured trace moves with the band rather than staying put to be scored
    against it.  Realised coverage against that trace will therefore run above
    nominal, and ``attrs['coverage_transfers']`` is set False to say so.

    ``qt_sd`` is offered for propagating into something downstream that wants a
    variance, not for drawing ``qt_median +/- qt_sd``: ``qt`` is lognormal plus a
    constant, so its band is asymmetric and the symmetric one is wrong on both
    sides.  The columns are named so the mistake is at least visible.

    ``level`` defaults to whatever :func:`reading_predictions` used, carried on
    ``readings.attrs``, so the two tables cannot silently disagree about what
    their intervals mean.
    """
    for col in ("z", "pred", "Qtn"):
        if col not in readings.columns:
            raise ValueError(f"not a reading_predictions table: missing {col!r}")
    if "unit_id" not in readings.columns:
        raise ValueError("not a reading_predictions table: missing 'unit_id'")

    out = readings.copy()
    z = out["z"].to_numpy(dtype=float)
    level = float(level if level is not None else readings.attrs.get("level", 0.95))
    kw = {
        "n": _by_unit(n, out["unit_id"], "n"),
        "n_sd": _by_unit(n_sd, out["unit_id"], "n_sd"),
        "profile": profile,
    }
    pred = out["pred"].to_numpy(dtype=float)
    tail = (1.0 - level) / 2.0

    # The measurement carries no exponent *uncertainty* — it is one number under
    # the central n — so it is de-normalised with n alone.
    out["qt"] = qt_from_qtn(out["Qtn"].to_numpy(dtype=float), z, n=kw["n"], profile=profile)
    out["qt_median"] = qt_quantile(pred, 0.0, z, q=0.5, **kw)

    for name, sd_col in _BANDS.items():
        sd = out[sd_col].to_numpy(dtype=float)
        # Rebuilt from (pred, sd, level) rather than by transforming the stored
        # log edges.  With n_sd = 0 the two agree to the last bit; with n_sd > 0
        # the stored edges are the wrong width, and with a `level` that differs
        # from the one the log table used they would be the wrong width *and*
        # mislabelled.
        out[f"qt_lo_{name}"] = qt_quantile(pred, sd, z, q=tail, **kw)
        out[f"qt_hi_{name}"] = qt_quantile(pred, sd, z, q=1.0 - tail, **kw)
        m = qt_moments(pred, sd, z, **kw)
        out[f"qt_sd_{name}"] = m["qt_sd"]
        out[f"qt_cv_{name}"] = m["qt_cv"]

    out.attrs = dict(readings.attrs)
    out.attrs["level"] = level
    out.attrs["stress_profile"] = profile.describe()
    out.attrs["n"] = n
    out.attrs["n_sd"] = n_sd
    out.attrs["exponent"] = _exponent_label(n, n_sd)
    out.attrs["coverage_transfers"] = not np.any(np.asarray(kw["n_sd"], dtype=float) > 0)
    return out


def qt_by_unit(
    cv: pd.DataFrame,
    layers: pd.DataFrame,
    model: Optional[str] = None,
    profile: StressProfile = DEFAULT_PROFILE,
    n=1.0,
    n_sd=0.0,
    sd_column: str = "sd_obs",
    level: float = 0.95,
) -> pd.DataFrame:
    """Per-unit predictions as ``qt`` at the top, middle and base of each layer.

    The model predicts **one number per unit per location**, but ``qt`` is not a
    number — it is a line.  A constant ``Qtn`` through a unit de-normalises to a
    ``qt`` rising linearly with depth, so the honest summary of a unit-level
    prediction in ``qt`` is its value at the layer's top, mid-depth and base,
    which is what this returns: three rows per unit occurrence, long format.

    Take the depths from ``intervals`` rather than ``layers`` where the two
    differ.  A unit that re-enters lower down (19 of the IJmuiden holes) has a
    single collapsed ``layers`` row spanning first-top to last-base, whose
    mid-depth can sit in a different unit entirely; ``intervals`` keeps the runs
    separate.  Either frame works — both carry ``cpt_id``, ``unit_id``,
    ``z_top``, ``z_bot`` — and the choice is the caller's because only the
    caller knows whether the collapse matters for what they are drawing.

    ``sd_column`` picks which uncertainty is being quoted, and there is no
    universally right answer: ``sd_latent`` for where the unit's true average
    is, ``sd_obs`` (the default, and what cross-validation scores) for what a
    new CPT here would measure.  For a single reading use
    :func:`reading_predictions` and :func:`qt_readings`, which carry the
    within-unit scatter this table has no depth resolution to represent.

    ``n`` and ``n_sd`` take the same forms as in :func:`qt_readings` — a number,
    or one per unit.  Exponent uncertainty bites hardest at the *top* of a
    shallow unit and vanishes where the effective stress passes ``pa``, so it
    changes the three rows by different amounts and is worth reading down the
    ``where`` column rather than as a single inflation factor.
    """
    block = _select_model(cv, model)
    for col in ("cpt_id", "unit_id", "pred", sd_column):
        if col not in block.columns:
            raise ValueError(f"cv table is missing required column {col!r}")
    for col in ("cpt_id", "unit_id", "z_top", "z_bot"):
        if col not in layers.columns:
            raise ValueError(f"layer table is missing required column {col!r}")

    keep = ["cpt_id", "unit_id", "z_top", "z_bot"]
    joined = layers[keep].merge(
        block[["cpt_id", "unit_id", "pred", sd_column]], on=["cpt_id", "unit_id"], how="inner"
    )

    z_at = {
        "top": joined["z_top"].to_numpy(dtype=float),
        "mid": (joined["z_top"].to_numpy(dtype=float) + joined["z_bot"].to_numpy(dtype=float)) / 2,
        "bot": joined["z_bot"].to_numpy(dtype=float),
    }
    pred = joined["pred"].to_numpy(dtype=float)
    sd = joined[sd_column].to_numpy(dtype=float)
    tail = (1.0 - level) / 2.0
    kw = {
        "n": _by_unit(n, joined["unit_id"], "n"),
        "n_sd": _by_unit(n_sd, joined["unit_id"], "n_sd"),
        "profile": profile,
    }

    rows = []
    for where, z in z_at.items():
        part = joined[["cpt_id", "unit_id"]].copy()
        part["where"] = where
        part["z"] = z
        part["log_Q_pred"] = pred
        part["log_Q_sd"] = sd
        part["log_Q_sd_total"] = total_log_sd(sd, z, n_sd=kw["n_sd"], profile=profile)
        for name, values in qt_moments(pred, sd, z, **kw).items():
            part[name] = values
        for name, q in (("qt_lo", tail), ("qt_hi", 1.0 - tail)):
            part[name] = qt_quantile(pred, sd, z, q=q, **kw)
        rows.append(part)

    order = pd.CategoricalDtype(["top", "mid", "bot"], ordered=True)
    out = pd.concat(rows, ignore_index=True)
    out["where"] = out["where"].astype(order)
    out = out.sort_values(["cpt_id", "z", "where"]).reset_index(drop=True)
    out.attrs["level"] = level
    out.attrs["sd_column"] = sd_column
    out.attrs["stress_profile"] = profile.describe()
    out.attrs["n"] = n
    out.attrs["n_sd"] = n_sd
    out.attrs["exponent"] = _exponent_label(n, n_sd)
    return out


def profile_coverage(readings: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Per unit: what fraction of real readings actually fell inside each band.

    The measurement that keeps the reading band honest.  It is built from a
    variance decomposition that knowingly double-counts the depth-averaging
    error, so whether it is right is an *observable*, not a claim — and a
    systematic miss is a finding to report rather than a number to tune.

    ``coverage_mean`` is included for contrast and is expected to be far below
    nominal: it is the band for a depth-*average*, being scored against
    individual readings, which carry the within-unit scatter it excludes.
    """
    rows = []
    for uid, g in readings.groupby("unit_id", sort=False):
        # How many readings exist and how many could be predicted are different
        # numbers, and a unit that was measured but never predicted must still
        # report its readings rather than vanishing as a zero.
        n_readings = int(len(g))
        g = g[g["pred"].notna()]
        if not len(g):
            rows.append({"unit_id": uid, "n_readings": n_readings, "n_predicted": 0,
                         "coverage_read": float("nan"), "coverage_mean": float("nan"),
                         "mssr_read": float("nan"), "within_sd": float("nan")})
            continue
        inside_read = (g["log_Q"] >= g["log_lo_read"]) & (g["log_Q"] <= g["log_hi_read"])
        inside_mean = (g["log_Q"] >= g["log_lo_mean"]) & (g["log_Q"] <= g["log_hi_mean"])
        rows.append({
            "unit_id": uid,
            "n_readings": n_readings,
            "n_predicted": int(len(g)),
            "coverage_read": float(inside_read.mean()),
            "coverage_mean": float(inside_mean.mean()),
            # Mean squared standardised residual: 1.0 means the reading band is
            # exactly the right width, above 1 means too narrow.
            "mssr_read": float(np.mean(g["resid_z"].to_numpy(dtype=float) ** 2)),
            "within_sd": float(g["within_sd"].iloc[0]),
        })
    out = pd.DataFrame(rows).set_index("unit_id")
    out.attrs["level"] = level
    return out
