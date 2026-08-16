"""Property sampling at CPTs and depth-series synthesis (A5 sampling, A6)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..contract.schema import Raster
from .config import Config, WithinUnitConfig


def sample_unit_values(
    cfg: Config, raster: Raster, layout: pd.DataFrame, property_rasters: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """The true ``log Qtn`` unit value at every CPT, with nugget added.

    Produced for *every* CPT regardless of presence: the field exists whether or
    not the unit was deposited there, and holding the values fixed keeps them
    comparable across presence realisations in a sensitivity sweep.

    Returns ``cpt_id, unit_id, log_Q_field, log_Q_true`` where ``log_Q_field`` is
    ``mu + trend + GRF`` and ``log_Q_true`` adds the nugget.
    """
    frames = []
    for uid, unit in cfg.units.items():
        field = raster.sample(property_rasters[uid], layout["x"], layout["y"])
        nugget_sd = np.sqrt(unit.property.nugget)
        rng = cfg.rng("nugget", uid)
        noise = rng.normal(0.0, nugget_sd, size=field.size) if nugget_sd > 0 else 0.0
        frames.append(
            pd.DataFrame(
                {
                    "cpt_id": layout["cpt_id"].to_numpy(),
                    "unit_id": uid,
                    "log_Q_field": field,
                    "log_Q_true": field + noise,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _within_unit_residual(n: int, cfg: WithinUnitConfig, dz_m: float, rng) -> np.ndarray:
    """AR(1) residual with stationary sd ``cfg.sd``.

    Correlated depth-to-depth scatter matters: with ``corr_len_m`` of a few
    centimetres the effective sample size behind a depth-average is far smaller
    than the raw sample count, which is exactly what the averaging step has to
    survive being tested against.
    """
    if n == 0 or cfg.sd <= 0:
        return np.zeros(n)
    if cfg.corr_len_m <= 0:
        return rng.normal(0.0, cfg.sd, size=n)

    rho = float(np.exp(-dz_m / cfg.corr_len_m))
    driving = rng.normal(0.0, cfg.sd * np.sqrt(1.0 - rho**2), size=n)
    driving[0] = rng.normal(0.0, cfg.sd)  # start from the stationary distribution
    return lfilter([1.0], [1.0, -rho], driving)


def make_depth_series(
    cfg: Config, layers: pd.DataFrame, unit_values: pd.DataFrame, layout: pd.DataFrame
) -> pd.DataFrame:
    """One row per depth reading: ``cpt_id, x, y, z, unit_id, Qtn``."""
    dz = cfg.sampling.dz_m
    xy = layout.set_index("cpt_id")[["x", "y"]]
    truth = unit_values.set_index(["cpt_id", "unit_id"])["log_Q_true"]

    chunks = []
    for row in layers.itertuples(index=False):
        n = int(np.floor((row.z_bot - row.z_top) / dz))
        if n <= 0:
            continue
        z = row.z_top + (np.arange(n) + 0.5) * dz
        wcfg = cfg.units[row.unit_id].property.within_unit
        rng = cfg.rng("series", row.cpt_id, row.unit_id)

        resid = _within_unit_residual(n, wcfg, dz, rng)
        drift = wcfg.drift_per_m * (z - z.mean())
        log_q = truth.loc[(row.cpt_id, row.unit_id)] + drift + resid

        chunks.append(
            pd.DataFrame(
                {
                    "cpt_id": row.cpt_id,
                    "x": xy.at[row.cpt_id, "x"],
                    "y": xy.at[row.cpt_id, "y"],
                    "z": z,
                    "unit_id": row.unit_id,
                    "Qtn": np.exp(log_q),
                }
            )
        )

    samples = pd.concat(chunks, ignore_index=True)
    return samples.sort_values(["cpt_id", "z"]).reset_index(drop=True)


# ``summarise_units`` lives in :mod:`cpt_geostat.contract.summarise` — it is part of
# the data contract and runs unchanged on real data.
