"""Maps of what a fitted model says — the companion to the truth maps.

:func:`cpt_geostat.synthetic.plots.plot_unit_truth_panel` maps the ground the
generator actually made.  This maps what an estimator believes, in the same
style and — on a synthetic run — on the same grid, so the two can be compared by
flipping between figures rather than by eye across different rasters.

One figure per model, one **row per unit**, four columns:

``predicted``   the median surface, ``exp`` of the kriged mean in log space
``latent sd``   how well the field itself is known — what more drilling reduces
``lower``/``upper``  the two ends of the interval, as maps in their own right

The bounds are the reason to draw all four together.  A mean map alone invites
being read as *the* answer; putting the interval either side of it on **the same
colour scale** turns "how uncertain is this" from a second figure you have to
hold in your head into something you can see in one glance across a row.

Nothing is masked by presence, because presence is not modelled yet.  A unit
held at four holes still gets a surface across the whole site — so read the row,
not the first panel: where the mean map is inventing, the sd panel is brightest
and the lower and upper maps diverge.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import norm

from ..models.field import field_raster, predict_field
from .style import figure_suptitle, map_axes, unit_label

#: Shared by the three value columns; the sd column gets its own.
_VALUE_CMAP = "magma"
_SD_CMAP = "viridis"


def shared_colour_limits(ds, fields_by_model: dict, level: float = 0.95) -> dict:
    """``{"value": {unit: (lo, hi)}, "sd": {...}}`` spanning every model.

    Two scales, and the split is the point:

    * **value** covers the predicted surface *and both bounds*, over every
      model.  Scaling the mean to itself would clip the bounds off the ends of
      the colourbar, and scaling each panel separately would make a wide
      interval look identical to a narrow one — which is exactly the thing the
      four-column row exists to show.
    * **sd** is a different quantity and gets its own range.

    Shared **across models, per unit** — not across units.  Units differ in
    level by design (unit 2 sits near 1.0 and unit 6 near 2.4), so one scale for
    all of them would flatten every panel to a single hue.
    """
    z = float(norm.ppf(0.5 + level / 2.0))
    out = {"value": {}, "sd": {}}
    for per_unit in fields_by_model.values():
        for uid, pred in per_unit.items():
            if not pred.fitted or not np.isfinite(pred.mean).any():
                continue
            lo = float(np.nanmin(pred.mean - z * pred.sd))
            hi = float(np.nanmax(pred.mean + z * pred.sd))
            _widen(out["value"], uid, lo, hi)
            _widen(out["sd"], uid, float(np.nanmin(pred.sd)), float(np.nanmax(pred.sd)))

    # The observations belong on the same scale as the surfaces meant to honour
    # them, so a CPT the model failed to match stays visible rather than
    # clipping to the end of the bar.
    summary = ds.unit_summary
    for uid in list(out["value"]):
        block = summary[summary["unit_id"] == uid]
        if len(block):
            _widen(out["value"], uid,
                   float(block["log_Q_mean"].min()), float(block["log_Q_mean"].max()))
    return out


def _widen(store: dict, uid: str, lo: float, hi: float) -> None:
    if uid in store:
        lo = min(lo, store[uid][0])
        hi = max(hi, store[uid][1])
    store[uid] = (lo, hi)


def _colour_limits(lo: float, hi: float, kind: str):
    """``(vmin, vmax, is_uniform)`` — never a negative standard deviation.

    A unit whose variogram fits pure nugget gets a *constant* field, and
    matplotlib needs a range.  Padding it symmetrically is the obvious thing and
    it is wrong for the sd column: it produces a colourbar running below zero,
    which is not a value a standard deviation can take and reads as a broken
    figure rather than as "kriging found nothing here".
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0, False
    if hi > lo:
        return lo, hi, False

    span = max(abs(lo) * 0.1, 1e-3)
    if kind == "sd":
        return max(lo - span, 0.0), lo + span, True
    return lo - span, lo + span, True


def plot_prediction_map(
    ds,
    factory=None,
    model_name: str = "model",
    raster=None,
    unit_ids: Optional[Sequence[str]] = None,
    fields: Optional[dict] = None,
    limits: Optional[dict] = None,
    level: float = 0.95,
) -> Figure:
    """One model: a row per unit, four columns across the site.

    ``factory`` is the ``unit_id -> unfitted estimator`` callable the CV harness
    uses, so any estimator drops in unchanged.  Fitted **in-sample**: this is the
    model's best statement given everything known, not a held-out test.

    The CPTs holding the unit are overlaid on the predicted panel, **coloured by
    their own observed value on the same scale**.  A marker that clashes with
    the background means the surface is not honouring its own data — the cheap
    check the truth panels use, drawn the same way.  On the other three columns
    they are hollow: there, where the data is matters more than what it said.

    ``fields`` accepts a precomputed ``{unit_id: FieldPrediction}`` so a caller
    drawing several models does not evaluate each twice, and ``limits`` takes
    the output of :func:`shared_colour_limits` so colours mean the same thing in
    every model's figure.  Left at ``None`` each row scales to itself.
    """
    if factory is None and fields is None:
        raise ValueError("pass either a factory to fit, or precomputed fields")

    raster = raster if raster is not None else field_raster(ds)
    unit_ids = list(unit_ids or ds.unit_ids)
    extent = raster.extent
    z = float(norm.ppf(0.5 + level / 2.0))

    fig, axes = plt.subplots(
        len(unit_ids), 4, figsize=(14.4, 3.35 * len(unit_ids)),
        constrained_layout=True, squeeze=False,
    )
    headers = ("predicted", "latent sd", f"lower {level:.0%}", f"upper {level:.0%}")

    for row, uid in zip(axes, unit_ids):
        pred = fields[uid] if fields is not None else predict_field(ds, uid, factory, raster)
        block = ds.unit_summary[ds.unit_summary["unit_id"] == uid]

        if not pred.fitted:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
            row[0].set_ylabel(f"{unit_label(ds, uid)}\n{pred.n_cpt} CPTs", fontsize=8)
            row[1].text(0.5, 0.5, f"not fitted — {pred.note}", transform=row[1].transAxes,
                        ha="center", va="center", fontsize=9, color="#B00020", wrap=True)
            continue

        lower, upper = pred.mean - z * pred.sd, pred.mean + z * pred.sd
        v_lo, v_hi = _limit_for(limits, "value", uid, [lower, upper], block)
        s_lo, s_hi = _limit_for(limits, "sd", uid, [pred.sd], None)
        v_lo, v_hi, _ = _colour_limits(v_lo, v_hi, "value")
        s_lo, s_hi, sd_uniform = _colour_limits(s_lo, s_hi, "sd")

        panels = (
            (pred.mean, _VALUE_CMAP, v_lo, v_hi),
            (pred.sd, _SD_CMAP, s_lo, s_hi),
            (lower, _VALUE_CMAP, v_lo, v_hi),
            (upper, _VALUE_CMAP, v_lo, v_hi),
        )
        for col, (ax, (field, cmap, lo, hi)) in enumerate(zip(row, panels)):
            im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
            # One bar for the sd, one for the three value panels it cannot be
            # compared with; hanging it off the last value column keeps the row
            # from carrying four near-identical colourbars.
            if col in (1, 3):
                fig.colorbar(
                    im, ax=ax, shrink=0.88,
                    label=r"latent sd of log $Q_{tn}$" if col == 1
                    else r"log $Q_{tn}$ (shared: predicted, lower, upper)",
                )
            if len(block):
                if col == 0:
                    ax.scatter(block["x"], block["y"], c=block["log_Q_mean"], s=20,
                               cmap=cmap, vmin=lo, vmax=hi, edgecolor="k", lw=0.4, zorder=3)
                else:
                    ax.scatter(block["x"], block["y"], s=11, facecolor="none",
                               edgecolor="w", lw=0.6, zorder=3)
            map_axes(ax, extent)
            if col:
                ax.set_ylabel("")
            # Axis captions only on the outside edge: repeated down 23 rows
            # they are pure noise, and the tick numbers carry the information.
            if uid != unit_ids[-1]:
                ax.set_xlabel("")

        if sd_uniform:
            # A flat sd is a result, not a rendering accident: the variogram
            # came back pure nugget, so kriging has collapsed onto the baseline
            # and has no spatial information to offer for this unit.
            row[1].text(0.5, 0.06, "uniform — no spatial structure resolved",
                        transform=row[1].transAxes, ha="center", va="bottom", fontsize=6.5,
                        color="w",
                        bbox=dict(boxstyle="round,pad=0.25", fc="0.25", ec="none", alpha=0.7))

        row[0].set_ylabel(
            f"{unit_label(ds, uid)}\n{pred.n_cpt} CPTs ({pred.cpt_fraction:.0%})", fontsize=8
        )

    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=11)

    figure_suptitle(
        fig,
        f"Predicted field — {model_name}   ·   fitted in-sample, no presence mask   ·   "
        f"predicted / lower / upper share one colour scale per unit",
    )
    return fig


def _limit_for(limits, key, uid, arrays, block):
    """Shared limits when supplied, else the row's own range."""
    if limits is not None and uid in limits.get(key, {}):
        return limits[key][uid]
    lo = min(float(np.nanmin(a)) for a in arrays)
    hi = max(float(np.nanmax(a)) for a in arrays)
    if block is not None and len(block):
        lo = min(lo, float(block["log_Q_mean"].min()))
        hi = max(hi, float(block["log_Q_mean"].max()))
    return lo, hi
