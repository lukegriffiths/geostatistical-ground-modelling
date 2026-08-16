"""Scoring.

The subset needed to read a cross plot: how far off the predictions are, and
whether the error bars are the right size.  ``nlpd``, ``crps`` and ``pit`` land
with the full harness.

Two of these are about accuracy (:func:`rmse`, :func:`bias`) and two are about
*calibration* (:func:`coverage`, :func:`mssr`), and the pair matters more than
either alone.  A model can be inaccurate and perfectly calibrated — which is the
correct outcome on a unit with no exploitable spatial structure, and the report
must not read it as failure.  The reverse, accurate but overconfident, is the
one that hurts: it is what puts a foundation design outside an interval that was
never wide enough.

Every metric drops non-finite pairs and reports over what remains, so a unit
where some folds could not be fitted still scores on the folds that could.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import norm


def _clean(y_true, y_pred, sd=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if sd is None:
        return y_true[ok], y_pred[ok], None
    sd = np.asarray(sd, dtype=float)
    ok &= np.isfinite(sd) & (sd > 0)
    return y_true[ok], y_pred[ok], sd[ok]


def rmse(y_true, y_pred) -> float:
    y_true, y_pred, _ = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean signed error, ``pred - true``.  Positive means over-prediction."""
    y_true, y_pred, _ = _clean(y_true, y_pred)
    return float(np.mean(y_pred - y_true)) if y_true.size else float("nan")


def r2(y_true, y_pred) -> float:
    """Fraction of variance explained, against the *held-out* mean.

    Under leave-one-out a constant-mean model scores slightly **below zero**,
    not at zero: each fold predicts with a mean that excludes the point it is
    being scored on.  That small negative number is the correct reference line
    for every spatial estimator, and it is not a bug to be zeroed out.
    """
    y_true, y_pred, _ = _clean(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def coverage(y_true, y_pred, sd, level: float = 0.95) -> float:
    """Fraction of truths inside the central ``level`` predictive interval."""
    y_true, y_pred, sd = _clean(y_true, y_pred, sd)
    if not y_true.size:
        return float("nan")
    z = float(norm.ppf(0.5 + level / 2.0))
    return float(np.mean(np.abs(y_true - y_pred) <= z * sd))


def mssr(y_true, y_pred, sd) -> float:
    """Mean squared standardised residual.  1.0 is calibrated.

    More useful than coverage at small n, where coverage moves in steps of
    ``1/n`` and cannot distinguish mildly from wildly overconfident.  Above 1 is
    overconfident — intervals too narrow.
    """
    y_true, y_pred, sd = _clean(y_true, y_pred, sd)
    if not y_true.size:
        return float("nan")
    return float(np.mean(((y_true - y_pred) / sd) ** 2))


def summarise(y_true, y_pred, sd, level: float = 0.95) -> Dict[str, float]:
    """All of the above plus ``n``, for one unit and one model."""
    y_clean, _, _ = _clean(y_true, y_pred, sd)
    return {
        "n": int(y_clean.size),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        f"coverage{int(level * 100)}": coverage(y_true, y_pred, sd, level),
        "mssr": mssr(y_true, y_pred, sd),
    }
