"""The truth side — everything that only exists because the generator ran.

Synthetic validates, real applies: the estimators in :mod:`cpt_geostat.models` run on
any data, and this module is where they are held against what the generator
actually put there.  Nothing in ``cpt_geostat.models`` imports from here; the truth
comparison is a consumer of the estimator layer, not part of it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..contract.schema import Raster
from ..covariance import build_model
from ..models.kriging import SimpleKriging
from .config import Config


# --------------------------------------------------------------------------- #
# what the generator delivered (written to truth.yaml)
# --------------------------------------------------------------------------- #


def realised_truth(
    cfg: Config,
    raster: Raster,
    layout: pd.DataFrame,
    layers: pd.DataFrame,
    samples: pd.DataFrame,
    unit_summary: pd.DataFrame,
    unit_values: pd.DataFrame,
    rasters: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Resolved config plus what the realisation actually delivered.

    Requested parameters and realised statistics differ — a finite realisation of
    a field with sill 0.09 does not have sample variance 0.09 — and hyperparameter
    recovery should be scored against numbers that are honest about which is which.
    """
    n_cpt = len(layout)
    per_unit = {}
    for uid in cfg.unit_ids:
        lay_u = layers[layers["unit_id"] == uid]
        sum_u = unit_summary[unit_summary["unit_id"] == uid]
        val_u = unit_values[unit_values["unit_id"] == uid]
        # Resolved unambiguously, so recovery is not scored against a range whose
        # definition depends on which parameterisation the reader assumed.
        model = build_model(cfg.units[uid].property.grf)
        per_unit[uid] = {
            "grf_len_scale_km": float(model.len_scale),
            "grf_range_km_practical": float(model.percentile_scale(0.95)),
            "grf_integral_scale_km": float(model.integral_scale),
            "coverage_raster": float(np.mean(rasters["presence_prob"][uid])),
            "coverage_cpt": float(len(lay_u) / n_cpt),
            "n_cpt_present": int(len(lay_u)),
            "n_truncated": int(lay_u["truncated"].sum()) if len(lay_u) else 0,
            "thickness_mean_m": float(lay_u["thickness_m"].mean()) if len(lay_u) else float("nan"),
            "thickness_sd_m": float(lay_u["thickness_m"].std()) if len(lay_u) else float("nan"),
            "grf_var_raster": float(np.var(rasters["grf"][uid])),
            "log_Q_field_mean": float(val_u["log_Q_field"].mean()),
            "log_Q_field_sd": float(val_u["log_Q_field"].std()),
            "log_Q_mean_observed": float(sum_u["log_Q_mean"].mean()) if len(sum_u) else float("nan"),
            "log_Q_sd_observed": float(sum_u["log_Q_mean"].std()) if len(sum_u) else float("nan"),
            "within_unit_sd_observed": float(sum_u["log_Q_sd"].mean()) if len(sum_u) else float("nan"),
        }

    return {
        "config": cfg.to_dict(),
        "realised": {
            "n_cpt": n_cpt,
            "n_cpt_unit_pairs": int(len(unit_summary)),
            "n_depth_readings": int(len(samples)),
            "units": per_unit,
        },
    }


# --------------------------------------------------------------------------- #
# the truth reference the baseline is compared against
# --------------------------------------------------------------------------- #


def truth_reference_table(ds) -> Optional[pd.DataFrame]:
    """What the generator actually put there.  ``None`` on real data.

    Configured versus realised is the point of having truth at all:

    ``mu``, ``trend_grad``, ``sill``, ``nugget``, ``within_unit_sd``
        *Configured*.  The baseline mean should sit near ``mu`` only where the
        unit is present everywhere; a channel or patch unit samples a biased
        subset of a trended or merely lumpy field, and the gap is a real
        property of the data, not an estimator error.
    ``field_sd``, ``true_sd``, ``obs_noise_sd``, ``nugget_sd``, ``depth_avg_sd``
        *Realised*, and — the load-bearing detail — **restricted to the CPTs
        where the unit is actually present**.  ``field_sd`` is the sd of the
        latent ``mu + trend + GRF``, the most a perfect spatial model could
        strip out of the baseline's spread; ``true_sd`` adds the nugget.
        ``obs_noise_sd`` is the rest — what separates the observed depth-average
        from the latent field — and it splits into the ``nugget_sd`` the config
        asked for and the ``depth_avg_sd`` incurred by averaging a finite,
        autocorrelated trace.  The second is not small: because the within-unit
        scatter is AR(1), it runs far above ``within_sd / sqrt(n_samples)``.
    ``field_sd_all_cpt``
        The same latent sd over *every* CPT, present or not — what
        ``truth.yaml`` records.  It is carried only for contrast and must not be
        compared with ``log_Q_sd``: for a narrow channel the two populations are
        different by a factor of 1.5, because a compact 16% of a lumpy field is
        far less variable than the whole of it.  Comparing across that
        difference is how a baseline gets scored as explaining more variance
        than it contains.
    """
    truth = getattr(ds, "truth", None)
    cfg = getattr(ds, "config", None)
    if not truth or cfg is None:
        return None
    realised = truth.get("realised", {}).get("units", {})
    present = _present_cpt_truth_sd(ds)

    rows = []
    for uid, unit in cfg.units.items():
        p = unit.property
        row = {
            "unit_id": uid,
            "mu": p.mu,
            "trend_grad": p.trend.grad,
            "sill": p.grf.sill,
            "nugget": p.nugget,
            "within_unit_sd": p.within_unit.sd,
            "field_sd_all_cpt": realised.get(uid, {}).get("log_Q_field_sd", float("nan")),
        }
        row.update(present.get(uid, _NAN_SD))
        rows.append(row)
    return pd.DataFrame(rows).set_index("unit_id")


_NAN_SD = {
    "field_sd": float("nan"), "true_sd": float("nan"),
    "obs_noise_sd": float("nan"), "nugget_sd": float("nan"), "depth_avg_sd": float("nan"),
}


def _present_cpt_truth_sd(ds) -> Dict[str, Dict[str, float]]:
    """Latent and noise sds per unit, over the CPTs that unit is present at.

    The noise term is ``log_Q_mean - log_Q_field``, **not**
    ``log_Q_mean - log_Q_true``.  ``log_Q_true`` already carries the nugget, so
    differencing against it captures only the depth-averaging error and leaves
    the total short by exactly the nugget — a factor-of-two understatement of
    the noise floor, on a quantity whose whole job is to say how much of the
    baseline's spread is irreducible.

    Needs ``truth_points.csv``; without it these columns go ``nan`` rather than
    silently falling back to the all-CPT figure.
    """
    values = getattr(ds, "unit_values", None)
    summary = getattr(ds, "unit_summary", None)
    if values is None or summary is None:
        return {}
    present = summary[["cpt_id", "unit_id", "log_Q_mean"]].merge(
        values, on=["cpt_id", "unit_id"], how="inner"
    )
    out = {}
    for uid, g in present.groupby("unit_id"):
        out[uid] = {
            "field_sd": float(g["log_Q_field"].std()),
            "true_sd": float(g["log_Q_true"].std()),
            "obs_noise_sd": float(_rms(g["log_Q_mean"] - g["log_Q_field"])),
            "nugget_sd": float(_rms(g["log_Q_true"] - g["log_Q_field"])),
            "depth_avg_sd": float(_rms(g["log_Q_mean"] - g["log_Q_true"])),
        }
    return out


def _rms(e) -> float:
    """Root-mean-square, not sd: these errors have a known mean of zero.

    Centring them would discard a real bias — the whole point of measuring the
    noise floor is that it is an error about truth, not a spread about itself.
    """
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")


# --------------------------------------------------------------------------- #
# kriging handed the generating covariance
# --------------------------------------------------------------------------- #


def truth_kriging_factory(ds, model: str = "matern25"):
    """``unit_id -> SimpleKriging`` carrying the generating covariance per unit.

    The kriging-with-the-right-covariance reference: it separates *"is kriging
    implemented correctly"* from *"can a variogram be fitted from 30 CPTs"* —
    two questions that a single number confounds, and the second is much harder
    than the first.  Unavailable on real data, and it raises there rather than
    silently falling back to a fit — the two answer different questions and the
    difference between them is the result worth having.

    **The nugget used is the realised observation noise, not ``config.nugget``.**
    The data are depth-averages, so what is uncorrelated between two CPTs is the
    nugget *plus* the depth-averaging error — 1.0x to 2.3x the configured nugget
    depending on the unit.  Handing kriging the configured value would give the
    reference model too little noise and make it overconfident, which would then
    read as a flaw in kriging rather than in what it was told.
    """
    cfg = getattr(ds, "config", None)
    if cfg is None:
        raise ValueError("the truth covariance needs a config; real data has none")
    truth = truth_reference_table(ds)

    def make(unit_id: str) -> SimpleKriging:
        unit = cfg.units[unit_id].property
        nugget = unit.nugget
        if truth is not None and unit_id in truth.index:
            measured = truth.loc[unit_id, "obs_noise_sd"] ** 2
            if np.isfinite(measured):
                nugget = float(measured)
        return SimpleKriging(covariance=(unit.grf, nugget), model=unit.grf.model)

    return make
