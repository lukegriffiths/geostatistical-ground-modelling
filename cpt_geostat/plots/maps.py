"""B2 — presence, value and thickness maps.

These run unchanged on real data: they read ``unit_summary`` and ``layers`` only.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

from .style import figure_suptitle, map_axes, site_extent, unit_colours, unit_grid, unit_label


def plot_presence_map(ds, unit_ids: Optional[Sequence[str]] = None) -> Figure:
    """Filled where the unit is present, hollow where absent — one panel per unit.

    Shows the channel and patch geometry exactly as the estimator sees it: not as
    a field, but as a scatter of presences with everything else a hole.
    """
    unit_ids = list(unit_ids or ds.unit_ids)
    colours = unit_colours(ds, unit_ids)
    extent = site_extent(ds)
    fig, axes = unit_grid(len(unit_ids))
    lay = ds.layout

    for ax, uid in zip(axes, unit_ids):
        present = set(ds.layers.loc[ds.layers["unit_id"] == uid, "cpt_id"])
        mask = lay["cpt_id"].isin(present).to_numpy()
        ax.scatter(lay.loc[~mask, "x"], lay.loc[~mask, "y"], s=22,
                   facecolor="none", edgecolor="#9e9e9e", lw=0.7)
        ax.scatter(lay.loc[mask, "x"], lay.loc[mask, "y"], s=30,
                   facecolor=colours[uid], edgecolor="k", lw=0.4)
        frac = mask.mean()
        map_axes(ax, extent, f"{unit_label(ds, uid)}\n{mask.sum()}/{len(lay)} CPTs ({frac:.0%})")

    figure_suptitle(fig, "Presence at CPTs — filled = present, hollow = absent")
    return fig


def plot_value_map(ds, unit_ids: Optional[Sequence[str]] = None, robust: bool = True) -> Figure:
    """CPTs coloured by depth-averaged ``log Qtn``; one colourbar per unit.

    Scales are per unit deliberately — a shared scale across units would be
    dominated by the between-unit mean offsets and hide the within-unit spatial
    structure, which is the thing being modelled.
    """
    unit_ids = list(unit_ids or ds.unit_ids)
    extent = site_extent(ds)
    fig, axes = unit_grid(len(unit_ids))
    lay = ds.layout

    for ax, uid in zip(axes, unit_ids):
        sub = ds.unit_summary[ds.unit_summary["unit_id"] == uid]
        ax.scatter(lay["x"], lay["y"], s=12, facecolor="none", edgecolor="#dddddd", lw=0.5)
        if len(sub):
            v = sub["log_Q_mean"].to_numpy()
            vmin, vmax = (np.percentile(v, [2, 98]) if robust and len(v) > 10 else (v.min(), v.max()))
            sc = ax.scatter(sub["x"], sub["y"], c=v, s=42, cmap="viridis",
                            vmin=vmin, vmax=vmax, edgecolor="k", lw=0.4)
            fig.colorbar(sc, ax=ax, label=r"depth-avg log $Q_{tn}$", shrink=0.85)
        map_axes(ax, extent, f"{unit_label(ds, uid)}  (n={len(sub)})")

    figure_suptitle(fig, r"Depth-averaged log $Q_{tn}$ per unit — the model input")
    return fig


def plot_thickness_map(ds, unit_ids: Optional[Sequence[str]] = None) -> Figure:
    """Retained thickness per unit, colour and size encoded."""
    unit_ids = list(unit_ids or ds.unit_ids)
    extent = site_extent(ds)
    fig, axes = unit_grid(len(unit_ids))
    lay = ds.layout
    layers = ds.layers.merge(lay[["cpt_id", "x", "y"]], on="cpt_id", how="left")

    for ax, uid in zip(axes, unit_ids):
        sub = layers[layers["unit_id"] == uid]
        ax.scatter(lay["x"], lay["y"], s=10, facecolor="none", edgecolor="#dddddd", lw=0.5)
        if len(sub):
            t = sub["thickness_m"].to_numpy()
            sizes = 12 + 90 * (t - t.min()) / (np.ptp(t) or 1.0)
            sc = ax.scatter(sub["x"], sub["y"], c=t, s=sizes, cmap="YlGnBu",
                            edgecolor="k", lw=0.4)
            fig.colorbar(sc, ax=ax, label="thickness (m)", shrink=0.85)
            title = f"{unit_label(ds, uid)}\nmean {t.mean():.1f} m, max {t.max():.1f} m"
        else:
            title = f"{unit_label(ds, uid)} — not present"
        map_axes(ax, extent, title)

    figure_suptitle(fig, "Retained thickness per unit")
    return fig


def plot_layout(ds) -> Figure:
    """CPT layout, with the short-lag cluster called out.

    Worth its own figure because the cluster and the thinned corner are design
    decisions that the variogram and the sensitivity sweep both depend on.
    """
    import matplotlib.pyplot as plt

    extent = site_extent(ds)
    fig, ax = plt.subplots(figsize=(6.5, 6.2), constrained_layout=True)
    lay = ds.layout
    kinds = lay["kind"] if "kind" in lay else np.full(len(lay), "grid")
    for kind, colour, size, label in [
        ("grid", "#264653", 26, "grid"),
        ("cluster", "#E76F51", 46, "short-lag cluster"),
    ]:
        sel = kinds == kind
        if sel.any():
            ax.scatter(lay.loc[sel, "x"], lay.loc[sel, "y"], s=size, c=colour,
                       edgecolor="k", lw=0.4, label=f"{label} (n={int(sel.sum())})")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    map_axes(ax, extent, f"CPT layout — {len(lay)} locations")
    return fig
