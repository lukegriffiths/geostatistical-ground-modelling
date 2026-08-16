"""Rasters, Gaussian random fields and trend surfaces.

All rasters are stored as ``(ny, nx)`` arrays so ``field[iy, ix]`` matches
``imshow(field, origin="lower", extent=raster.extent)`` without transposing at
the plotting site.  gstools returns ``(nx, ny)``; the transpose happens once,
here.
"""

from __future__ import annotations

import gstools as gs
import numpy as np

from ..contract.schema import Raster
from ..covariance import GrfConfig, build_model
from ..geometry import project_on_azimuth
from .config import gstools_seed

__all__ = ["Raster", "grf_raster", "unit_grf", "trend_surface", "property_raster"]


def grf_raster(raster: Raster, cfg: GrfConfig, rng) -> np.ndarray:
    """A single GRF realisation on the raster, shape ``(ny, nx)``."""
    if cfg.sill <= 0:
        return np.zeros(raster.shape)
    srf = gs.SRF(build_model(cfg), seed=gstools_seed(rng))
    return np.asarray(srf.structured((raster.x, raster.y))).T


def unit_grf(raster: Raster, rng, range_km: float, model: str = "gaussian") -> np.ndarray:
    """Unit-variance smooth field — the workhorse for presence and thickness."""
    return grf_raster(
        raster, GrfConfig(sill=1.0, range_km=range_km, model=model), rng
    )


def trend_surface(raster: Raster, grad: float, azimuth_deg: float) -> np.ndarray:
    """``grad`` (per km) applied along a compass azimuth. Shape ``(ny, nx)``."""
    XX, YY = raster.meshgrid()
    return grad * project_on_azimuth(XX, YY, azimuth_deg)


def property_raster(raster: Raster, cfg, rng):
    """``log Q(x, y) = mu + trend + GRF``.

    Returns ``(total, trend_only, grf_only)``, each ``(ny, nx)``.  The nugget is
    *not* included: it is a point-support term added when sampling at CPTs.
    """
    trend = trend_surface(raster, cfg.trend.grad, cfg.trend.azimuth_deg)
    grf = grf_raster(raster, cfg.grf, rng)
    return cfg.mu + trend + grf, cfg.mu + trend, grf
