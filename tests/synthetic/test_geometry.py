"""Convention checks.

The azimuth convention is the plan's flagged "single most likely source of silent
bugs", so it is checked numerically rather than only by eye on the B1 panels.
"""

from __future__ import annotations

import numpy as np

from cpt_geostat.covariance import GrfConfig
from cpt_geostat.synthetic.fields import Raster, grf_raster, trend_surface
from cpt_geostat.geometry import (
    azimuth_to_math_angle,
    azimuth_unit_vector,
    math_angle_to_azimuth,
    project_on_azimuth,
)


def test_azimuth_cardinal_directions():
    for az, expected in [(0, (0, 1)), (90, (1, 0)), (180, (0, -1)), (270, (-1, 0))]:
        ux, uy = azimuth_unit_vector(az)
        assert np.allclose([ux, uy], expected, atol=1e-12), az


def test_projection_is_clockwise_from_north():
    # A point due east projects fully onto azimuth 90 and not at all onto 0.
    assert np.isclose(project_on_azimuth(1.0, 0.0, 90.0), 1.0)
    assert np.isclose(project_on_azimuth(1.0, 0.0, 0.0), 0.0)
    # 45 deg is north-east.
    assert np.isclose(project_on_azimuth(1.0, 1.0, 45.0), np.sqrt(2))


def test_math_angle_roundtrip():
    for az in [0.0, 25.0, 70.0, 115.0, 135.0]:
        assert np.isclose(math_angle_to_azimuth(azimuth_to_math_angle(az)), az % 180.0)


def test_trend_increases_along_its_azimuth():
    raster = Raster.from_site(10.0, 0.5)
    az = 115.0  # east-south-east
    field = trend_surface(raster, grad=0.1, azimuth_deg=az)
    ux, uy = azimuth_unit_vector(az)
    XX, YY = raster.meshgrid()
    # Gradient of the surface must point along the azimuth unit vector.
    gy, gx = np.gradient(field, raster.y, raster.x)
    assert np.allclose(gx, 0.1 * ux, atol=1e-9)
    assert np.allclose(gy, 0.1 * uy, atol=1e-9)


def _directional_variance(field, raster, azimuth_deg, lag_km):
    """Mean squared increment at a fixed lag along a compass azimuth."""
    ux, uy = azimuth_unit_vector(azimuth_deg)
    dx = int(round(lag_km * ux / (raster.x[1] - raster.x[0])))
    dy = int(round(lag_km * uy / (raster.y[1] - raster.y[0])))
    if dx == 0 and dy == 0:
        raise ValueError("lag smaller than one cell")
    a = field[max(dy, 0):field.shape[0] + min(dy, 0), max(dx, 0):field.shape[1] + min(dx, 0)]
    b = field[max(-dy, 0):field.shape[0] + min(-dy, 0), max(-dx, 0):field.shape[1] + min(-dx, 0)]
    return float(np.mean((a - b) ** 2))


def test_anisotropy_major_axis_follows_its_azimuth():
    """A ratio-6 field must decorrelate slowest along `aniso_angle_deg`.

    This is the check that catches a rotation-sign error or a from-north /
    from-east mix-up in the gstools bridge — both of which produce a field that
    looks anisotropic and is simply pointing the wrong way.
    """
    raster = Raster.from_site(20.0, 0.1)
    for az in [0.0, 45.0, 70.0, 135.0]:
        cfg = GrfConfig(sill=1.0, range_km=6.0, aniso_ratio=6.0, aniso_angle_deg=az,
                        model="gaussian")
        field = grf_raster(raster, cfg, np.random.default_rng(7))
        # Lag well inside the major range and well outside the minor one.
        along = _directional_variance(field, raster, az, lag_km=1.0)
        across = _directional_variance(field, raster, az + 90.0, lag_km=1.0)
        assert along < 0.3 * across, f"az={az}: along={along:.3f} across={across:.3f}"
