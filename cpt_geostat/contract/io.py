"""Readers and writers (A7).

The write side is synthetic-only.  The read side is the *real-data entry point*:
supply ``cpt_samples.csv`` and ``layers.csv`` and everything downstream works,
with ``truth.yaml`` absent and truth-dependent diagnostics skipped.  Reading
real data must stay cheap, so the synthetic config is imported only when a
``truth.yaml`` is actually present.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import yaml

from .schema import LAYERS_COLUMNS, SAMPLES_COLUMNS, SUMMARY_COLUMNS, Dataset, Raster
from .summarise import summarise_units

if TYPE_CHECKING:
    from ..synthetic.config import Config


def write_dataset(ds: Dataset, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds.samples[SAMPLES_COLUMNS].to_csv(out / "cpt_samples.csv", index=False, float_format="%.6g")
    ds.layers[LAYERS_COLUMNS].to_csv(out / "layers.csv", index=False, float_format="%.6g")
    ds.unit_summary[SUMMARY_COLUMNS].to_csv(
        out / "unit_summary.csv", index=False, float_format="%.6g"
    )

    if ds.truth is not None:
        (out / "truth.yaml").write_text(
            yaml.safe_dump(ds.truth, sort_keys=False, default_flow_style=False), encoding="utf-8"
        )
    if ds.raster is not None and ds.rasters:
        _write_rasters(ds, out / "truth_fields.npz")
    return out


def _write_rasters(ds: Dataset, path: Path) -> None:
    """Truth rasters + true CPT node values, for the B1 diagnostics.

    Not part of the real-data contract; the B1 panels are skipped without it.
    """
    payload = {"__x": ds.raster.x, "__y": ds.raster.y}
    for kind, per_unit in ds.rasters.items():
        for uid, arr in per_unit.items():
            payload[f"{kind}/{uid}"] = arr.astype(np.float32)
    np.savez_compressed(path, **payload)
    if ds.unit_values is not None:
        ds.unit_values.to_csv(path.with_name("truth_points.csv"), index=False, float_format="%.6g")


def _read_table(src: Path, stem: str) -> Optional[pd.DataFrame]:
    """Read ``stem`` as parquet if present, else csv, else ``None``.

    Parquet is preferred because real exports are large — the IJmuiden site is
    129 MB of csv against 4 MB of parquet — and because it round-trips dtypes.
    """
    parquet = src / f"{stem}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = src / f"{stem}.csv"
    return pd.read_csv(csv) if csv.exists() else None


def read_dataset(in_dir, config: Optional["Config"] = None) -> Dataset:
    """Load a dataset directory.  Works with only the two required tables present.

    Each table may be csv or parquet; parquet wins where both exist.
    """
    src = Path(in_dir)
    samples = _read_table(src, "cpt_samples")
    layers = _read_table(src, "layers")
    if samples is None or layers is None:
        missing = [n for n, t in (("cpt_samples", samples), ("layers", layers)) if t is None]
        raise FileNotFoundError(f"{src}: no .parquet or .csv for {missing}")
    _check_columns(samples, SAMPLES_COLUMNS, "cpt_samples")
    _check_columns(layers, LAYERS_COLUMNS, "layers")

    if "thickness_m" not in layers:
        layers["thickness_m"] = layers["z_bot"] - layers["z_top"]
    layout = (
        samples[["cpt_id", "x", "y"]]
        .drop_duplicates("cpt_id")
        .sort_values("cpt_id")
        .reset_index(drop=True)
    )
    if "kind" not in layout:
        layout["kind"] = "grid"

    unit_summary = _read_table(src, "unit_summary")
    if unit_summary is None:
        unit_summary = summarise_units(samples, layers, layout)

    truth = None
    truth_path = src / "truth.yaml"
    if truth_path.exists():
        truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
        if config is None and isinstance(truth, dict) and "config" in truth:
            from ..synthetic.config import Config

            config = Config.from_dict(truth["config"])

    raster, rasters = None, {}
    fields_path = src / "truth_fields.npz"
    if fields_path.exists():
        raster, rasters = _read_rasters(fields_path)

    unit_values = None
    points_path = src / "truth_points.csv"
    if points_path.exists():
        unit_values = pd.read_csv(points_path)

    return Dataset(
        layout=layout,
        layers=layers,
        samples=samples,
        unit_summary=unit_summary,
        config=config,
        raster=raster,
        rasters=rasters,
        unit_values=unit_values,
        truth=truth,
    )


def _read_rasters(path: Path):
    with np.load(path) as npz:
        raster = Raster(x=npz["__x"], y=npz["__y"])
        rasters: dict = {}
        for key in npz.files:
            if key.startswith("__"):
                continue
            kind, uid = key.split("/", 1)
            rasters.setdefault(kind, {})[uid] = npz[key]
    return raster, rasters


def _check_columns(df: pd.DataFrame, required, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")
