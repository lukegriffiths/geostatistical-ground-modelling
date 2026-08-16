"""Site geometry and CPT positions (A2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

_CORNER_SIGNS = {"NE": (1, 1), "NW": (-1, 1), "SE": (1, -1), "SW": (-1, -1)}


def make_layout(cfg: Config) -> pd.DataFrame:
    """Jittered turbine-like grid + a tight cluster, optionally thinned in one corner.

    Returns ``cpt_id, x, y, kind`` with ``kind`` in ``{"grid", "cluster"}``.

    The cluster exists purely so the variogram has a few short-lag pairs; without
    them the nugget and the range trade off against each other freely near the
    origin.
    """
    lay = cfg.layout
    half = cfg.site.size_km / 2.0 - lay.margin_km
    rng = cfg.rng("layout")

    axis = np.linspace(-half, half, lay.grid_n)
    gx, gy = np.meshgrid(axis, axis, indexing="xy")
    x = gx.ravel() + rng.uniform(-lay.jitter_km, lay.jitter_km, gx.size)
    y = gy.ravel() + rng.uniform(-lay.jitter_km, lay.jitter_km, gy.size)
    kind = np.full(x.size, "grid", dtype=object)

    if lay.drop_corner is not None and lay.drop_corner.enabled:
        dc = lay.drop_corner
        try:
            sx, sy = _CORNER_SIGNS[dc.corner.upper()]
        except KeyError:
            raise ValueError(f"drop_corner.corner must be one of {sorted(_CORNER_SIGNS)}")
        cx, cy = sx * half, sy * half
        in_corner = np.hypot(x - cx, y - cy) <= dc.radius_km
        drop = in_corner & (rng.random(x.size) > dc.keep_fraction)
        x, y, kind = x[~drop], y[~drop], kind[~drop]

    if lay.cluster is not None and lay.cluster.n > 0:
        cl = lay.cluster
        cxy = np.asarray(cl.centre_km, dtype=float)
        offs = rng.normal(0.0, cl.spread_km, size=(cl.n, 2))
        x = np.concatenate([x, cxy[0] + offs[:, 0]])
        y = np.concatenate([y, cxy[1] + offs[:, 1]])
        kind = np.concatenate([kind, np.full(cl.n, "cluster", dtype=object)])

    order = np.lexsort((x, y))  # south-to-north, west-to-east: stable, readable ids
    x, y, kind = x[order], y[order], kind[order]

    return pd.DataFrame(
        {
            "cpt_id": [f"CPT{i + 1:03d}" for i in range(x.size)],
            "x": x,
            "y": y,
            "kind": kind,
        }
    )
