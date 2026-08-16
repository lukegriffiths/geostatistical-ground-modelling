"""B2 — fence / section plots.

This is how a geotechnical reviewer will want to check the stratigraphy: unit
intervals stacked against chainage along a transect, rather than as a map.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .style import unit_colours, unit_label


def project_to_section(layout: pd.DataFrame, p0, p1, corridor_km: float) -> pd.DataFrame:
    """CPTs within ``corridor_km`` of the segment ``p0 -> p1``.

    Returns the layout with ``chainage_km`` (distance along the transect from
    ``p0``) and ``offset_km`` (signed perpendicular distance), sorted by chainage.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    axis = p1 - p0
    length = float(np.hypot(*axis))
    if length == 0:
        raise ValueError("section endpoints coincide")
    unit = axis / length

    rel = layout[["x", "y"]].to_numpy() - p0
    chainage = rel @ unit
    offset = rel[:, 0] * -unit[1] + rel[:, 1] * unit[0]

    out = layout.assign(chainage_km=chainage, offset_km=offset)
    keep = (np.abs(offset) <= corridor_km) & (chainage >= 0) & (chainage <= length)
    return out[keep].sort_values("chainage_km").reset_index(drop=True)


def plot_section(ds, p0, p1, corridor_km: float = 1.2, name: str = "",
                 unit_ids: Optional[Sequence[str]] = None) -> Figure:
    """Stacked unit intervals against chainage, with a location inset.

    Each CPT is drawn as a column of unit intervals.  Adjacent columns are joined
    only where both share the unit, so a channel pinching out shows as a break
    rather than as a spurious continuous layer.
    """
    unit_ids = list(unit_ids or ds.unit_ids)
    colours = unit_colours(ds, unit_ids)
    sel = project_to_section(ds.layout, p0, p1, corridor_km)
    if sel.empty:
        raise ValueError(f"no CPTs within {corridor_km} km of section {name or 'unnamed'}")

    layers = ds.layers.merge(sel[["cpt_id", "chainage_km"]], on="cpt_id", how="inner")

    fig = plt.figure(figsize=(14.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.19])
    ax = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])

    spacing = np.diff(np.sort(sel["chainage_km"].to_numpy()))
    median_gap = float(np.median(spacing)) if len(spacing) else 0.7
    bar_w = max(median_gap * 0.55, 0.08)

    # Connect matching units between adjacent CPTs.
    for uid in unit_ids:
        sub = layers[layers["unit_id"] == uid].sort_values("chainage_km")
        if len(sub) < 2:
            continue
        ch = sub["chainage_km"].to_numpy()
        top = sub["z_top"].to_numpy()
        bot = sub["z_bot"].to_numpy()
        # Only join near neighbours: bridging a long gap draws a confident
        # correlation across ground where nothing was measured.
        gap_limit = median_gap * 2.0
        start = 0
        for i in range(1, len(ch) + 1):
            broken = i == len(ch) or (ch[i] - ch[i - 1]) > gap_limit
            if broken and i - start >= 2:
                s = slice(start, i)
                ax.fill_between(ch[s], top[s], bot[s], color=colours[uid], alpha=0.35, lw=0)
            if broken:
                start = i

    # Per-CPT columns: always correct, even where nothing connects.
    for uid in unit_ids:
        sub = layers[layers["unit_id"] == uid]
        if sub.empty:
            continue
        ax.bar(sub["chainage_km"], height=sub["z_bot"] - sub["z_top"], bottom=sub["z_top"],
               width=bar_w, color=colours[uid], edgecolor="k", lw=0.3,
               label=unit_label(ds, uid))

    for ch in sel["chainage_km"]:
        ax.plot([ch, ch], [0, 0], marker="v", color="k", ms=5, clip_on=False)

    ax.set_xlabel("chainage along section (km)")
    ax.set_ylabel("depth below seabed (m)")
    # Depth increases downwards; set the limits directly rather than inverting,
    # so the seabed stays pinned at 0 regardless of what the data reaches.
    z_max = float(layers["z_bot"].max()) if len(layers) else 1.0
    ax.set_ylim(z_max * 1.03, 0.0)
    ax.grid(axis="y", ls=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=8, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.005), framealpha=0.9)

    az = np.rad2deg(np.arctan2(p1[0] - p0[0], p1[1] - p0[1])) % 360.0
    fig.suptitle(
        f"Section {name or ''} — ({p0[0]:.1f}, {p0[1]:.1f}) to ({p1[0]:.1f}, {p1[1]:.1f}) km, "
        f"azimuth {az:.0f}° CW from N, corridor ±{corridor_km:g} km, {len(sel)} CPTs",
        fontsize=11,
    )

    _draw_location_map(ax_map, ds, sel, p0, p1)
    return fig


def _draw_location_map(ax, ds, sel, p0, p1) -> None:
    """Small map showing where the section runs — a section without one is unreadable."""
    from .style import site_extent

    extent = site_extent(ds)
    ax.scatter(ds.layout["x"], ds.layout["y"], s=4, c="#c4c4c4", lw=0)
    ax.scatter(sel["x"], sel["y"], s=11, c="#E76F51", lw=0)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c="k", lw=1.4)
    ax.scatter([p0[0]], [p0[1]], s=18, c="k", marker="o", zorder=4)
    ax.annotate("0 km", (p0[0], p0[1]), fontsize=6.5, xytext=(3, -9),
                textcoords="offset points")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("location", fontsize=8)


def plot_configured_sections(ds) -> list:
    """One figure per transect in the config; empty list if none are configured."""
    cfg = getattr(ds, "config", None)
    if cfg is None or not cfg.sections:
        return []
    return [
        plot_section(ds, s.p0_km, s.p1_km, corridor_km=s.corridor_km, name=s.name)
        for s in cfg.sections
    ]
