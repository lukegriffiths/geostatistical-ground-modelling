"""Round-trip and the real-data entry point (A7)."""

from __future__ import annotations

import pandas as pd
import pytest

from cpt_geostat.contract import read_dataset


def test_write_produces_the_documented_files(written):
    for name in ["cpt_samples.csv", "layers.csv", "unit_summary.csv", "truth.yaml"]:
        assert (written / name).exists(), name


def test_round_trip_preserves_the_data(dataset, written):
    back = read_dataset(written)
    assert back.is_synthetic
    assert len(back.samples) == len(dataset.samples)
    assert len(back.layers) == len(dataset.layers)
    assert back.config is not None
    assert back.config.seed == dataset.config.seed
    assert back.unit_ids == dataset.unit_ids


def test_truth_config_survives_the_yaml_round_trip(dataset, written):
    back = read_dataset(written)
    for uid, unit in dataset.config.units.items():
        got = back.config.units[uid].property
        assert got.mu == unit.property.mu
        assert got.grf.aniso_angle_deg == unit.property.grf.aniso_angle_deg
        assert got.grf.range_kind == unit.property.grf.range_kind


# --------------------------------------------------------------------------- #
# the real-data path
# --------------------------------------------------------------------------- #


def test_real_data_loads_from_two_csvs_alone(real_data_dir):
    ds = read_dataset(real_data_dir)
    assert not ds.is_synthetic
    assert ds.truth is None
    assert ds.config is None
    assert len(ds.unit_summary) > 0  # recomputed, not read
    assert len(ds.layout) > 0


def test_missing_required_column_is_reported(real_data_dir, tmp_path):
    df = pd.read_csv(real_data_dir / "cpt_samples.csv").drop(columns=["Qtn"])
    df.to_csv(tmp_path / "cpt_samples.csv", index=False)
    pd.read_csv(real_data_dir / "layers.csv").to_csv(tmp_path / "layers.csv", index=False)
    with pytest.raises(ValueError, match="Qtn"):
        read_dataset(tmp_path)
