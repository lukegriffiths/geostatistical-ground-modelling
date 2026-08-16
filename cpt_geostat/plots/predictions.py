"""Prediction cross plots — predicted against true, one panel per unit.

Reading one of these:

* **The 1:1 line is the target**, not the regression line through the points.  A
  cloud that is tight but parallel-and-offset is a biased model, and a
  least-squares line through it would hide exactly that.
* **A flat cloud means the model found no structure.**  The constant-mean
  baseline is flat *by construction* — it predicts the same number everywhere —
  so its panel is a horizontal band and its ``r2`` sits just below zero.  That
  is the reference every spatial estimator has to tilt away from, and the panel
  says so on its face rather than looking like a broken plot.
* **Error bars are ±1.96 sd of the predictive distribution**, so about 19 in 20
  should touch the 1:1 line.  Far more than that and the model is
  under-confident; far fewer and it is overconfident, which is the direction
  that matters.  The count is reported as ``cov95`` and paired with ``mssr``,
  which at these sample sizes resolves the difference between mildly and wildly
  overconfident that coverage cannot.

The target and the error bar are chosen together, by :func:`cpt_geostat.validate.cv.
target_columns`: against the held-out **observation** the bar includes the
nugget and the depth-averaging error; against the **latent** field it does not.
Mixing them is plan 02's risk C and it makes a correct model look miscalibrated.

Follows the Part B contract: takes a ``Dataset`` and a dataframe, returns a
Figure, touches no files.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm

from ..validate.cv import target_columns
from ..validate.metrics import summarise
from .style import figure_suptitle, unit_colours, unit_grid, unit_label

# Enough to separate the estimators plan 02 adds; cycled if ever exceeded.
_MARKERS = ["o", "s", "^", "D", "v", "P"]
_MODEL_COLOURS = ["#264653", "#E76F51", "#2A9D8F", "#6A4C93", "#F4A261", "#1D3557"]

_TARGET_LABEL = {
    "observed": "observed log Qtn (held-out depth-average)",
    "latent": "true latent log Qtn (mu + trend + GRF)",
}


def plot_prediction_vs_truth(
    ds,
    cv: pd.DataFrame,
    target: str = "observed",
    unit_ids: Optional[Sequence[str]] = None,
    level: float = 0.95,
) -> Figure:
    """Predicted (y) against true (x), per unit, from leave-one-out folds.

    ``cv`` is the table from :func:`cpt_geostat.validate.cv.loo_by_unit`.  If it holds
    more than one ``model`` they are drawn in the same panel, so the comparison
    is per unit rather than per figure.

    ``target="latent"`` needs ``truth_points.csv`` and raises without it — the
    alternative, silently falling back to the observed column, would relabel the
    axis and quietly change what the error bars mean.
    """
    value_col, sd_col = target_columns(target)
    if value_col not in cv.columns or cv[value_col].isna().all():
        raise ValueError(
            f"cv table has no usable {value_col!r} column; "
            f"target={target!r} needs truth_points.csv (synthetic runs only)"
        )

    unit_ids = [u for u in (unit_ids or ds.unit_ids) if u in set(cv["unit_id"])]
    models = list(pd.unique(cv["model"]))
    fig, axes = unit_grid(len(unit_ids), size=(4.4, 4.2))
    per_unit = unit_colours(ds, unit_ids)
    z = float(norm.ppf(0.5 + level / 2.0))

    for ax, uid in zip(axes, unit_ids):
        block = cv[cv["unit_id"] == uid]
        lo, hi = _square_limits(block, value_col, sd_col, z)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="0.35", zorder=1)

        notes = []
        for k, name in enumerate(models):
            sub = block[block["model"] == name]
            if not len(sub):
                continue
            # One model: use the unit's own colour, so the panel still reads as
            # "this unit" alongside every other figure in the set.
            colour = per_unit[uid] if len(models) == 1 else _MODEL_COLOURS[k % len(_MODEL_COLOURS)]
            ax.errorbar(
                sub[value_col], sub["pred"], yerr=z * sub[sd_col],
                fmt=_MARKERS[k % len(_MARKERS)], ms=4.5, lw=0, elinewidth=0.7,
                ecolor="0.55", alpha=0.9, color=colour, mec="k", mew=0.3,
                zorder=2 + k, label=name,
            )
            notes.append((name, summarise(sub[value_col], sub["pred"], sub[sd_col], level)))

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(_TARGET_LABEL[target], fontsize=8)
        ax.set_ylabel("predicted log Qtn", fontsize=8)
        ax.set_title(unit_label(ds, uid), fontsize=9)
        ax.text(
            0.03, 0.97, "\n".join(_annotate(n, s, level) for n, s in notes),
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="0.7", alpha=0.85),
        )
        # Notes go bottom-left and the legend bottom-right, so a panel carrying
        # both stays readable.
        footnotes = []
        for name, s in notes:
            sub = block[block["model"] == name]
            prefix = "" if len(models) == 1 else f"{name}: "
            if _is_constant(sub):
                footnotes.append((f"{prefix}flat by construction", "0.4"))
            text, colour = _calibration_note(s)
            if text:
                footnotes.append((f"{prefix}{text}", colour))
        for j, (text, colour) in enumerate(footnotes):
            ax.text(0.03, 0.03 + 0.05 * (len(footnotes) - 1 - j), text,
                    transform=ax.transAxes, va="bottom", ha="left", fontsize=6.5,
                    color=colour, style="italic")
        if len(models) > 1:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

    figure_suptitle(
        fig,
        f"Leave-one-out prediction vs. {'truth' if target == 'latent' else 'observation'} "
        f"— error bars ±{level:.0%} predictive interval",
    )
    return fig


def _annotate(name: str, s: dict, level: float) -> str:
    cov = s[f"coverage{int(level * 100)}"]
    return (
        f"{name}  n={s['n']}\n"
        f"  rmse {s['rmse']:.3f}  bias {s['bias']:+.3f}\n"
        f"  r2 {s['r2']:+.2f}  cov {cov:.0%}  mssr {s['mssr']:.2f}"
    )


def _square_limits(block: pd.DataFrame, value_col: str, sd_col: str, z: float):
    """Identical limits on both axes, so the 1:1 line is at 45 degrees.

    A cross plot with independently scaled axes puts the 1:1 line at an
    arbitrary angle and makes any model look like it tracks truth.
    """
    vals = np.concatenate(
        [
            block[value_col].to_numpy(dtype=float),
            block["pred"].to_numpy(dtype=float) - z * block[sd_col].to_numpy(dtype=float),
            block["pred"].to_numpy(dtype=float) + z * block[sd_col].to_numpy(dtype=float),
        ]
    )
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.06 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _calibration_note(s: dict):
    """Flag intervals that are the wrong size, with the direction that matters.

    Overconfident is the dangerous direction — it is what puts a design outside
    an interval that was never wide enough — so it is flagged in red and from a
    lower threshold than under-confidence.  A constant-mean model scored against
    the *latent* field lands here at ``mssr`` in the tens: it attributes every
    bit of spatial variation to noise, so its belief about the field is a flat
    line held to its own standard error.  That is a true statement about the
    model, not a plotting fault, and the panel should say which.
    """
    m = s["mssr"]
    if not np.isfinite(m):
        return "", "0.4"
    if m > 2.0:
        return f"overconfident: intervals ~{np.sqrt(m):.0f}x too narrow", "#B23A2E"
    if m < 0.5:
        return f"under-confident: intervals ~{1 / np.sqrt(m):.0f}x too wide", "#2A6F97"
    return "", "0.4"


def _is_constant(block: pd.DataFrame, slack: float = 2.0) -> bool:
    """Does the model predict essentially the same value everywhere?

    Leave-one-out moves a constant model's prediction between folds by exactly
    ``range(y) / (n - 1)`` — small, but not zero, and at n = 13 it is a quarter
    of an sd.  A tolerance stated as a fraction of ``sd(y)`` therefore flags the
    big units and misses the small ones, which is precisely backwards: the small
    units are where a reader most needs telling that the panel is flat by
    construction rather than by fit.

    So the comparison is against that known wobble, with ``slack`` to spare.  A
    model that tracks truth exactly spreads by ``range(y)``, which is ``n - 1``
    times larger; one explaining even 20% of the variance still clears the bar
    at n = 13.
    """
    pred = block["pred"].to_numpy(dtype=float)
    obs = block["observed"].to_numpy(dtype=float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    n = pred.size
    if n < 3:
        return False
    obs_range = float(obs.max() - obs.min())
    if not obs_range:
        return False
    return float(pred.max() - pred.min()) * (n - 1) <= slack * obs_range
