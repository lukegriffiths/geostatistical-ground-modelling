from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cpt_geostat.contract import LAYERS_COLUMNS, SAMPLES_COLUMNS, write_dataset
from cpt_geostat.synthetic import Config, generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def full_config() -> Config:
    return Config.load(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def small_config(full_config) -> Config:
    """A cheap config: coarser raster, fewer CPTs, 20 cm depth spacing.

    Keeps all six units so the contrast set is still exercised.
    """
    cfg = copy.deepcopy(full_config)
    cfg.site.raster_res_km = 0.25
    cfg.layout.grid_n = 7
    cfg.sampling.dz_m = 0.2
    return cfg


@pytest.fixture(scope="session")
def dataset(small_config):
    return generate(small_config)


@pytest.fixture(scope="session")
def written(dataset, tmp_path_factory):
    """A full synthetic dataset on disk: truth, rasters and all."""
    out = tmp_path_factory.mktemp("synthetic")
    write_dataset(dataset, out)
    return out


@pytest.fixture(scope="session")
def real_data_dir(dataset, tmp_path_factory):
    """Only the two files real data supplies — no truth, no summary, no rasters."""
    out = tmp_path_factory.mktemp("real")
    dataset.samples[SAMPLES_COLUMNS].to_csv(out / "cpt_samples.csv", index=False)
    dataset.layers[LAYERS_COLUMNS].to_csv(out / "layers.csv", index=False)
    return out
