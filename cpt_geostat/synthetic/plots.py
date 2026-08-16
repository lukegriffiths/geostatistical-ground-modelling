"""B1 — truth diagnostics.

Synthetic only.  These panels confirm the *generator* works; if the CPT markers
do not visually match the background field, the sampling step has a bug and
nothing downstream is worth looking at.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..plots.style import (
    add_anisotropy_glyph,
    add_north_azimuth_arrow,
    figure_suptitle,
    map_axes,
    site_extent,
    unit_label,
)


class TruthUnavailable(RuntimeError):
    """Raised when a truth-only diagnostic is asked for on real data."""


def _require_rasters(ds, *kinds):
    if getattr(ds, "raster", None) is None or not getattr(ds, "rasters", None):
        raise TruthUnavailable("truth rasters are not available (real-data run)")
    missing = [k for k in kinds if k not in ds.rasters]
    if missing:
        raise TruthUnavailable(f"truth rasters missing: {missing}")


def plot_unit_truth_panel(ds, unit_id: str) -> Figure:
    """2x2: presence / thickness / trend-only surface / full property surface.

    CPTs are overlaid on every panel, coloured by the value actually sampled
    there, on the same colour scale as the background.  Points that stand out
    against the field mean the sampling step disagrees with the field.
    """
    _require_rasters(ds, "presence_prob", "thickness", "trend", "property")
    extent = site_extent(ds)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    lay = ds.layout
    present = set(ds.layers.loc[ds.layers["unit_id"] == unit_id, "cpt_id"])
    is_present = lay["cpt_id"].isin(present).to_numpy()

    # -- presence -----------------------------------------------------------
    ax = axes[0, 0]
    im = ax.imshow(ds.rasters["presence_prob"][unit_id], origin="lower", extent=extent,
                   cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="P(present)", shrink=0.85)
    ax.scatter(lay.loc[is_present, "x"], lay.loc[is_present, "y"], s=16,
               facecolor="k", edgecolor="w", lw=0.4, label="present", zorder=3)
    ax.scatter(lay.loc[~is_present, "x"], lay.loc[~is_present, "y"], s=16,
               facecolor="none", edgecolor="k", lw=0.6, label="absent", zorder=3)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    map_axes(ax, extent, "Presence probability + realised draws")

    # -- thickness ----------------------------------------------------------
    ax = axes[0, 1]
    thick = ds.rasters["thickness"][unit_id]
    im = ax.imshow(thick, origin="lower", extent=extent, cmap="YlGnBu")
    fig.colorbar(im, ax=ax, label="thickness (m)", shrink=0.85)
    lyr = ds.layers[ds.layers["unit_id"] == unit_id].merge(
        lay[["cpt_id", "x", "y"]], on="cpt_id", how="left"
    )
    if len(lyr):
        ax.scatter(lyr["x"], lyr["y"], c=lyr["thickness_m"], s=26, cmap="YlGnBu",
                   vmin=im.get_clim()[0], vmax=im.get_clim()[1],
                   edgecolor="k", lw=0.4, zorder=3)
    map_axes(ax, extent, "Thickness field + retained thickness at CPTs")

    # -- trend only ---------------------------------------------------------
    # Shares the colour scale with the full property panel below, so "how much of
    # this field is trend" is readable directly.  Auto-scaling a constant field
    # would instead paint a no-trend unit with a full-range colourbar.
    prop = ds.rasters["property"][unit_id]
    trend = ds.rasters["trend"][unit_id]
    clim = (float(np.min(prop)), float(np.max(prop)))

    ax = axes[1, 0]
    im = ax.imshow(trend, origin="lower", extent=extent, cmap="magma",
                   vmin=clim[0], vmax=clim[1])
    fig.colorbar(im, ax=ax, label=r"$\mu$ + trend (log $Q_{tn}$)", shrink=0.85)
    cfg_unit = ds.config.units[unit_id] if getattr(ds, "config", None) else None
    if cfg_unit is not None and cfg_unit.property.trend.grad != 0:
        t = cfg_unit.property.trend
        add_north_azimuth_arrow(ax, t.azimuth_deg, length=0.28 * (extent[1] - extent[0]),
                                colour="w", label=f"{t.azimuth_deg:g}°")
        title = f"Trend only — grad {t.grad:g} / km at {t.azimuth_deg:g}° (CW from N)"
    else:
        title = "Trend only — none (constant mean)"
    map_axes(ax, extent, title)

    # -- full property ------------------------------------------------------
    ax = axes[1, 1]
    im = ax.imshow(prop, origin="lower", extent=extent, cmap="magma",
                   vmin=clim[0], vmax=clim[1])
    fig.colorbar(im, ax=ax, label=r"log $Q_{tn}$", shrink=0.85)
    if ds.unit_values is not None:
        vals = ds.unit_values[ds.unit_values["unit_id"] == unit_id].merge(
            lay[["cpt_id", "x", "y"]], on="cpt_id", how="left"
        )
        # Every CPT is shown, present or not: the field exists everywhere, and a
        # marker that clashes with the background means the sampler is wrong.
        ax.scatter(vals["x"], vals["y"], c=vals["log_Q_true"], s=26, cmap="magma",
                   vmin=clim[0], vmax=clim[1], edgecolor="k", lw=0.4, zorder=3)
    map_axes(ax, extent, r"Full property field ($\mu$ + trend + GRF) + CPT values (all locations)")

    figure_suptitle(fig, f"Truth diagnostics — {unit_label(ds, unit_id)}")
    return fig


def plot_anisotropy_check(ds, unit_ids: Optional[list] = None) -> Figure:
    """Per unit: the GRF component with an ellipse glyph at the stated anisotropy.

    Cheap, and it is the check that catches angle-convention errors — rotation
    sign, and whether the angle is measured from north or from east.  The glyph
    should lie *along* the grain of the field, not across it.
    """
    _require_rasters(ds, "grf")
    if getattr(ds, "config", None) is None:
        raise TruthUnavailable("anisotropy glyphs need the generative config")

    unit_ids = list(unit_ids or ds.unit_ids)
    extent = site_extent(ds)
    ncols = min(3, len(unit_ids))
    nrows = int(np.ceil(len(unit_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, uid in zip(axes, unit_ids):
        grf_cfg = ds.config.units[uid].property.grf
        field = ds.rasters["grf"][uid]
        lim = float(np.max(np.abs(field))) or 1.0
        im = ax.imshow(field, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax, label="GRF (log $Q_{tn}$)", shrink=0.85)
        add_anisotropy_glyph(
            ax,
            range_km=grf_cfg.range_km,
            ratio=grf_cfg.aniso_ratio,
            azimuth_deg=grf_cfg.aniso_angle_deg,
            centre=(0.0, 0.0),
            colour="k",
        )
        iso = grf_cfg.aniso_ratio == 1.0
        subtitle = (
            f"range {grf_cfg.range_km:g} km, isotropic"
            if iso
            else f"range {grf_cfg.range_km:g} km, ratio {grf_cfg.aniso_ratio:g} "
                 f"at {grf_cfg.aniso_angle_deg:g}° (CW from N)"
        )
        map_axes(ax, extent, f"{uid}\n{subtitle}")

    for ax in axes[len(unit_ids):]:
        ax.set_visible(False)
    figure_suptitle(fig, "Anisotropy check — glyph should follow the grain of the field")
    return fig
