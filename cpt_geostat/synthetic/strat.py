"""Presence fields, thickness fields and stratigraphic assembly (A3-A4)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

from ..geometry import rotate_to_azimuth_frame
from .config import Config, PresenceConfig, ThicknessConfig
from .fields import Raster, unit_grf

# --------------------------------------------------------------------------- #
# A3 — presence
# --------------------------------------------------------------------------- #


def presence_probability(raster: Raster, cfg: PresenceConfig, rng) -> np.ndarray:
    """P(unit present) on the raster, shape ``(ny, nx)``.

    The raster carries the *probability*; the Bernoulli draw happens once, at the
    CPT locations (:func:`sample_presence`).  Drawing per raster cell would give
    salt-and-pepper edges that no sampling could undo.
    """
    if cfg.mode == "everywhere":
        return np.ones(raster.shape)

    if cfg.mode == "channel":
        return _channel_probability(raster, cfg, rng)

    if cfg.mode == "patch":
        return _patch_probability(raster, cfg, rng)

    raise ValueError(f"unknown presence mode {cfg.mode!r}")


def _channel_probability(raster: Raster, cfg: PresenceConfig, rng) -> np.ndarray:
    """Sinuous ribbon: centreline is a sine wave in the channel's own frame.

    Working in the rotated frame is what makes the channel run oblique rather
    than axis-aligned.
    """
    XX, YY = raster.meshgrid()
    along, across = rotate_to_azimuth_frame(XX, YY, cfg.azimuth_deg)

    amplitude = cfg.sinuosity * cfg.wavelength_km / 4.0
    centre = cfg.offset_km + amplitude * np.sin(
        2 * np.pi * along / cfg.wavelength_km + np.deg2rad(cfg.phase_deg)
    )

    half_width = cfg.width_km / 2.0
    softness = max(cfg.edge_softness, 1e-3) * half_width
    # A short-range GRF perturbs the bank position, so edges are ragged rather
    # than merely blurred.
    ragged = softness * unit_grf(raster, rng, cfg.ragged_range_km)
    return expit((half_width - np.abs(across - centre) + ragged) / softness)


def _patch_probability(raster: Raster, cfg: PresenceConfig, rng) -> np.ndarray:
    """Threshold a smooth GRF at the quantile that hits the target coverage."""
    g = unit_grf(raster, rng, cfg.range_km)
    threshold = float(np.quantile(g, 1.0 - cfg.coverage))
    softness = max(cfg.edge_softness, 1e-3) * float(np.std(g))
    return expit((g - threshold) / softness)


def sample_presence(raster: Raster, prob: np.ndarray, layout: pd.DataFrame, rng) -> np.ndarray:
    """Bernoulli draw at each CPT from the interpolated presence probability."""
    p = np.clip(raster.sample(prob, layout["x"], layout["y"]), 0.0, 1.0)
    return rng.random(p.size) < p


# --------------------------------------------------------------------------- #
# A4 — thickness
# --------------------------------------------------------------------------- #


def _logistic_moment_match(cfg: ThicknessConfig):
    """Find ``(a, b)`` with ``t = min + (max-min)*expit(a + b*g)``, ``g ~ N(0,1)``,
    matching the requested mean and sd.

    Gauss-Hermite quadrature, so the solve is deterministic and independent of
    the realisation.  A bounded variable cannot have arbitrary sd; if the target
    is unreachable the closest achievable fit is used and a warning is emitted.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(64)
    weights = weights / weights.sum()
    span = cfg.max_m - cfg.min_m

    def moments(params):
        a, b = params
        t = cfg.min_m + span * expit(a + b * nodes)
        mean = float(weights @ t)
        var = float(weights @ (t - mean) ** 2)
        return mean, np.sqrt(var)

    def residual(params):
        mean, sd = moments(params)
        return [(mean - cfg.mean_m) / span, (sd - cfg.sd_m) / span]

    start = np.log(max((cfg.mean_m - cfg.min_m) / max(cfg.max_m - cfg.mean_m, 1e-9), 1e-9))
    sol = least_squares(residual, x0=[start, 1.0], xtol=1e-12, ftol=1e-12)
    mean, sd = moments(sol.x)
    if abs(sd - cfg.sd_m) > 0.05 * cfg.sd_m:
        warnings.warn(
            f"thickness sd {cfg.sd_m:.2f} m not attainable inside "
            f"[{cfg.min_m}, {cfg.max_m}] m with mean {cfg.mean_m:.2f} m; "
            f"realised sd will be ~{sd:.2f} m",
            stacklevel=3,
        )
    return float(sol.x[0]), float(sol.x[1]), mean, sd


def thickness_field(raster: Raster, cfg: ThicknessConfig, rng) -> np.ndarray:
    """Smooth thickness field (m) bounded in ``[min_m, max_m]``, shape ``(ny, nx)``."""
    g = unit_grf(raster, rng, cfg.range_km)
    g = (g - g.mean()) / (g.std() or 1.0)  # the map below assumes standard normal
    a, b, _, _ = _logistic_moment_match(cfg)
    return cfg.min_m + (cfg.max_m - cfg.min_m) * expit(a + b * g)


# --------------------------------------------------------------------------- #
# A4 — assembly
# --------------------------------------------------------------------------- #


@dataclass
class StratRasters:
    presence_prob: Dict[str, np.ndarray]
    thickness: Dict[str, np.ndarray]


def build_strat_rasters(cfg: Config, raster: Raster) -> StratRasters:
    presence_prob, thickness = {}, {}
    for uid, unit in cfg.units.items():
        presence_prob[uid] = presence_probability(
            raster, unit.presence, cfg.rng("presence", uid)
        )
        thickness[uid] = thickness_field(raster, unit.thickness, cfg.rng("thickness", uid))
    return StratRasters(presence_prob=presence_prob, thickness=thickness)


def assemble(cfg: Config, raster: Raster, layout: pd.DataFrame, strat: StratRasters) -> pd.DataFrame:
    """Walk the units in order at each CPT, accumulating depths.

    Absent units are simply skipped (no row).  Everything is truncated at
    ``site.max_depth_m``, so deep units pinch out where shallow ones are thick —
    a second, *correlated* source of missingness on top of lateral absence.
    A unit whose retained thickness falls below ``sampling.min_thickness_m`` is
    marked absent rather than kept as a sliver.
    """
    n = len(layout)
    z_cursor = np.zeros(n)
    records = []

    for uid in cfg.unit_ids:
        present = sample_presence(
            raster, strat.presence_prob[uid], layout, cfg.rng("presence_draw", uid)
        )
        thick = raster.sample(strat.thickness[uid], layout["x"], layout["y"])

        z_top = z_cursor.copy()
        z_bot = np.minimum(z_top + thick, cfg.site.max_depth_m)
        retained = z_bot - z_top

        keep = present & (retained >= cfg.sampling.min_thickness_m)
        z_cursor = np.where(keep, z_bot, z_cursor)

        idx = np.flatnonzero(keep)
        records.append(
            pd.DataFrame(
                {
                    "cpt_id": layout["cpt_id"].to_numpy()[idx],
                    "unit_id": uid,
                    "z_top": z_top[idx],
                    "z_bot": z_bot[idx],
                    # kept for truth diagnostics, dropped before layers.csv
                    "thickness_target_m": thick[idx],
                    "truncated": (z_top[idx] + thick[idx]) > cfg.site.max_depth_m,
                }
            )
        )

    layers = pd.concat(records, ignore_index=True)
    layers["thickness_m"] = layers["z_bot"] - layers["z_top"]
    unit_order = {u: i for i, u in enumerate(cfg.unit_ids)}
    layers = layers.sort_values(
        ["cpt_id", "unit_id"], key=lambda s: s.map(unit_order) if s.name == "unit_id" else s
    ).reset_index(drop=True)
    return layers
