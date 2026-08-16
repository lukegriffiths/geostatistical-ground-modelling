"""Shared plotting conventions (B3).

Fixed once, here, and applied by every plotting module:

* equal aspect ratio on all maps;
* kilometre coordinates on map axes, metres on depth axes;
* the unit colour palette comes from the config;
* azimuths are degrees clockwise from north (see :mod:`cpt_geostat.geometry`).

Every public plotting function takes a dataframe or :class:`Dataset` and returns
a :class:`matplotlib.figure.Figure`.  None of them touch the filesystem, so the
harness can call them on real data unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from ..geometry import azimuth_unit_vector

# Fallback palette for real data, where no config supplies colours.
_FALLBACK = [
    "#E9C46A", "#F4A261", "#E76F51", "#2A9D8F",
    "#264653", "#8D6E63", "#6A4C93", "#1D3557",
]


def unit_colours(ds, unit_ids: Optional[Sequence[str]] = None) -> Dict[str, str]:
    unit_ids = list(unit_ids or ds.unit_ids)
    if getattr(ds, "config", None) is not None:
        configured = ds.config.unit_colours
        if all(u in configured for u in unit_ids):
            return {u: configured[u] for u in unit_ids}
    return {u: _FALLBACK[i % len(_FALLBACK)] for i, u in enumerate(unit_ids)}


def unit_label(ds, unit_id: str) -> str:
    cfg = getattr(ds, "config", None)
    if cfg is not None and unit_id in cfg.units and cfg.units[unit_id].name:
        return f"{unit_id} — {cfg.units[unit_id].name}"
    return unit_id


def map_axes(ax, extent=None, title: Optional[str] = None) -> None:
    """Apply the map conventions to one axis."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (km east of centre)")
    ax.set_ylabel("y (km north of centre)")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    if title:
        ax.set_title(title, fontsize=10)


def unit_grid(n_units: int, size=(4.2, 4.0), ncols: int = 3):
    """A panel-per-unit grid, sized so panels stay square-ish."""
    nrows = math.ceil(n_units / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_units:]:
        ax.set_visible(False)
    return fig, axes[:n_units]


def site_extent(ds):
    """Map extent in km, from the raster if present, otherwise from the CPTs."""
    if getattr(ds, "raster", None) is not None:
        return ds.raster.extent
    pad = 0.5
    x, y = ds.layout["x"], ds.layout["y"]
    return (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)


def add_anisotropy_glyph(ax, range_km: float, ratio: float, azimuth_deg: float,
                         centre=(0.0, 0.0), colour: str = "k") -> None:
    """Ellipse showing the stated anisotropy, in map coordinates.

    ``Ellipse.angle`` is degrees counter-clockwise from +x while our azimuths are
    clockwise from north, hence ``90 - azimuth``.  Drawing the glyph on top of
    the field is the cheapest possible check that the two agree: if the ellipse
    is across the grain rather than along it, the convention is inverted.
    """
    ell = Ellipse(
        xy=centre,
        width=2 * range_km,          # major axis, along the azimuth
        height=2 * range_km / ratio,  # minor axis
        angle=90.0 - azimuth_deg,
        facecolor="none",
        edgecolor=colour,
        lw=1.6,
    )
    ax.add_patch(ell)
    ux, uy = azimuth_unit_vector(azimuth_deg)
    ax.plot(
        [centre[0] - range_km * ux, centre[0] + range_km * ux],
        [centre[1] - range_km * uy, centre[1] + range_km * uy],
        color=colour, lw=1.2, ls="--",
    )


def add_north_azimuth_arrow(ax, azimuth_deg: float, length: float = 1.0,
                            centre=(0.0, 0.0), colour: str = "k", label: str = "") -> None:
    """Arrow pointing along a compass azimuth, for trend-direction panels."""
    ux, uy = azimuth_unit_vector(azimuth_deg)
    ax.annotate(
        "",
        xy=(centre[0] + length * ux, centre[1] + length * uy),
        xytext=(centre[0] - length * ux, centre[1] - length * uy),
        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8),
    )
    if label:
        ax.text(centre[0] + length * ux, centre[1] + length * uy, f" {label}",
                color=colour, fontsize=8, ha="left", va="bottom")


def figure_suptitle(fig: Figure, text: str) -> None:
    fig.suptitle(text, fontsize=12)
