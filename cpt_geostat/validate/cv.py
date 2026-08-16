"""Leave-one-CPT-out cross-validation, per unit.

A prediction plotted against the value it was fitted on is not a test — for the
constant-mean baseline it is not even a weak one, since the fitted value *is*
the mean of the points being plotted.  Every prediction here is therefore made
without the point it is scored against.

Folds are per unit, over that unit's present CPTs, because the estimators are
per unit: holding out a CPT means holding out one unit's value at it, not the
whole hole.

**Both variances are carried, on every row.**  ``sd_latent`` is the estimator's
uncertainty about the field; ``sd_obs`` adds ``noise_var_``.  Which one is right
depends on what the prediction is being compared with, and the two targets are
in the same table:

=====================  ============================  ==========
Compare against        Column                        Use
=====================  ============================  ==========
the held-out reading   ``observed`` (``log_Q_mean``) ``sd_obs``
the field that made it ``latent`` (``log_Q_field``)  ``sd_latent``
=====================  ============================  ==========

Pairing them the other way is the quiet failure in plan 02's risk C: scoring
against a noisy observation using the latent sd guarantees under-coverage, and
scoring against the latent field using the observation sd guarantees the
opposite.  Carrying both columns means the pairing is chosen at the call site
and is visible there.

``log_Q_true`` is deliberately *not* offered as a target.  It is
``log_Q_field`` plus that CPT's own nugget draw, which is independent of
everything else by construction — no estimator can predict it, and scoring
against it would penalise a perfect model by exactly the nugget.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..models.base import SpatialEstimator
from .metrics import summarise

EstimatorFactory = Callable[[], SpatialEstimator]


def loo_predict(factory: EstimatorFactory, X, y):
    """Leave-one-out ``(pred, sd_latent, sd_obs)``, refitting on each fold.

    Refitting every fold rather than fitting once and dropping a row is the
    honest default; it is also what makes the baseline's leave-one-out ``r2``
    land slightly below zero rather than exactly at it.  Folds that cannot be
    fitted — a unit held at a single CPT leaves nothing behind — come back
    ``nan`` rather than raising, so one thin unit does not take the run down.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    pred = np.full(n, np.nan)
    sd_latent = np.full(n, np.nan)
    sd_obs = np.full(n, np.nan)

    for i in range(n):
        keep = np.arange(n) != i
        try:
            model = factory().fit(X[keep], y[keep])
        except ValueError:
            continue
        mean, sd = model.predict(X[i : i + 1], return_std=True)
        pred[i] = mean[0]
        sd_latent[i] = sd[0]
        sd_obs[i] = np.sqrt(sd[0] ** 2 + model.noise_var_)

    return pred, sd_latent, sd_obs


def loo_by_unit(
    ds,
    factory: Callable[[str], SpatialEstimator],
    model_name: str,
    unit_ids: Optional[list] = None,
) -> pd.DataFrame:
    """Run :func:`loo_predict` per unit and stack the folds into one table.

    ``factory`` takes a ``unit_id`` and returns a fresh, unfitted estimator, so
    per-unit settings (a pooled fallback variance, later a fitted covariance)
    are chosen by the caller rather than guessed here.

    Returns one row per (CPT, unit) with ``model``, ``pred``, ``sd_latent``,
    ``sd_obs``, ``observed`` and — synthetic only — ``latent``.
    """
    summary = ds.unit_summary
    unit_ids = list(unit_ids or ds.unit_ids)

    frames = []
    for uid in unit_ids:
        block = summary[summary["unit_id"] == uid]
        if not len(block):
            continue
        X = block[["x", "y"]].to_numpy(dtype=float)
        make = lambda uid=uid: factory(uid)  # noqa: E731 — bind uid per iteration
        pred, sd_latent, sd_obs = loo_predict(make, X, block["log_Q_mean"])
        frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "cpt_id": block["cpt_id"].to_numpy(),
                    "unit_id": uid,
                    "x": X[:, 0],
                    "y": X[:, 1],
                    "observed": block["log_Q_mean"].to_numpy(dtype=float),
                    "pred": pred,
                    "sd_latent": sd_latent,
                    "sd_obs": sd_obs,
                }
            )
        )

    if not frames:
        raise ValueError(f"no unit_summary rows for any of {unit_ids}")
    out = pd.concat(frames, ignore_index=True)

    values = getattr(ds, "unit_values", None)
    if values is not None and "log_Q_field" in values:
        out = out.merge(
            values[["cpt_id", "unit_id", "log_Q_field"]].rename(columns={"log_Q_field": "latent"}),
            on=["cpt_id", "unit_id"],
            how="left",
        )
    return out


TARGETS = {
    # target column -> the sd that belongs with it
    "observed": "sd_obs",
    "latent": "sd_latent",
}


def target_columns(target: str):
    """``(value_column, sd_column)`` for a target, or raise.

    The pairing lives here so no call site has to remember it.
    """
    try:
        return target, TARGETS[target]
    except KeyError:
        raise ValueError(f"unknown target {target!r}; use one of {sorted(TARGETS)}") from None


def score_by_unit(cv: pd.DataFrame, target: str = "observed", level: float = 0.95) -> pd.DataFrame:
    """One row of metrics per (model, unit), from a :func:`loo_by_unit` table."""
    value_col, sd_col = target_columns(target)
    if value_col not in cv.columns:
        raise ValueError(f"cv table has no {value_col!r} column for target={target!r}")

    rows = []
    for (name, uid), sub in cv.groupby(["model", "unit_id"], sort=False):
        rows.append(
            {"model": name, "unit_id": uid,
             **summarise(sub[value_col], sub["pred"], sub[sd_col], level)}
        )
    return pd.DataFrame(rows).set_index(["model", "unit_id"])
