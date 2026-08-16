"""Coordinate and azimuth conventions — the single source of truth.

AZIMUTH CONVENTION (used for *both* trend direction and anisotropy angle)
------------------------------------------------------------------------
Azimuths are **degrees clockwise from north**:

    0   deg -> +y  (north)
    90  deg -> +x  (east)
    180 deg -> -y  (south)
    270 deg -> -x  (west)

The unit vector along azimuth ``a`` is therefore ``(sin a, cos a)`` in ``(x, y)``,
NOT ``(cos a, sin a)``.  Every projection, rotation and third-party call in this
package goes through the helpers below so the convention is applied exactly once.

Coordinates are kilometres with the origin at the site centre.

gstools uses the mathematical convention: angles are radians counter-clockwise
from the +x axis, and ``len_scale[0]`` lies along that rotated axis.  The bridge
is :func:`azimuth_to_math_angle` (``math = 90 - azimuth``); do not inline it.
"""

from __future__ import annotations

import numpy as np

AZIMUTH_CONVENTION = "degrees clockwise from north (0 = +y = north, 90 = +x = east)"


def azimuth_unit_vector(azimuth_deg: float) -> tuple:
    """Unit vector ``(ux, uy)`` pointing along a compass azimuth."""
    a = np.deg2rad(azimuth_deg)
    return float(np.sin(a)), float(np.cos(a))


def project_on_azimuth(x, y, azimuth_deg: float):
    """Signed distance (km) along ``azimuth_deg``.

    This is the coordinate a linear trend runs on::

        log Q = mu + grad * project_on_azimuth(x, y, azimuth)
    """
    ux, uy = azimuth_unit_vector(azimuth_deg)
    return np.asarray(x) * ux + np.asarray(y) * uy


def project_across_azimuth(x, y, azimuth_deg: float):
    """Signed distance (km) perpendicular to ``azimuth_deg``.

    Positive to the left of the azimuth direction.  Used for channel offsets.
    """
    return project_on_azimuth(x, y, azimuth_deg - 90.0)


def rotate_to_azimuth_frame(x, y, azimuth_deg: float):
    """Return ``(along, across)`` coordinates in the frame of ``azimuth_deg``."""
    return (
        project_on_azimuth(x, y, azimuth_deg),
        project_across_azimuth(x, y, azimuth_deg),
    )


def azimuth_to_math_angle(azimuth_deg: float) -> float:
    """Compass azimuth (deg CW from north) -> radians CCW from +x, for gstools."""
    return float(np.deg2rad(90.0 - azimuth_deg))


def math_angle_to_azimuth(theta_rad: float) -> float:
    """Inverse of :func:`azimuth_to_math_angle`, wrapped to [0, 180)."""
    return float(np.rad2deg(np.pi / 2 - theta_rad) % 180.0)


def pair_distances(layout):
    """All unordered pairs: ``(distance_km, azimuth_deg)``, azimuth CW from north in [0, 180).

    ``layout`` is any frame with ``x``/``y`` columns; runs on real data.
    """
    x = layout["x"].to_numpy()
    y = layout["y"].to_numpy()
    i, j = np.triu_indices(x.size, k=1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    dist = np.hypot(dx, dy)
    az = np.rad2deg(np.arctan2(dx, dy)) % 180.0  # atan2(dx, dy) == CW from north
    return dist, az
