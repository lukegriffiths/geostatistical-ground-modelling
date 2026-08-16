"""Depth-averaging — the step that turns readings into model input."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SUMMARY_COLUMNS


def summarise_units(samples: pd.DataFrame, layers: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    """Depth-average ``log Qtn`` per CPT per unit — the actual model input.

    Deliberately computed from ``cpt_samples.csv`` and nothing else, so the same
    function runs unchanged on real data — and projects import it from here
    rather than reimplementing the natural-log-then-average convention.
    """
    lq = np.log(samples["Qtn"].to_numpy())
    grouped = (
        samples.assign(log_Q=lq)
        .groupby(["cpt_id", "unit_id"], sort=False)["log_Q"]
        .agg(log_Q_mean="mean", log_Q_sd="std", n_samples="size")
        .reset_index()
    )
    out = grouped.merge(
        layers[["cpt_id", "unit_id", "thickness_m"]], on=["cpt_id", "unit_id"], how="left"
    ).merge(layout[["cpt_id", "x", "y"]], on="cpt_id", how="left")
    return out[SUMMARY_COLUMNS]
