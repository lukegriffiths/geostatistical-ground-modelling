"""Evaluating a fitted estimator over a grid.

The properties worth pinning are the ones the *figures* rely on to stay honest:
the maps are drawn unmasked, so the sd field growing away from the data is what
stops a mean map for a sparse unit being read as knowledge.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.contract.schema import Raster
from cpt_geostat.models import baseline_factory, kriging_factory
from cpt_geostat.models.field import (
    DEFAULT_RES_KM,
    field_raster,
    predict_field,
    predict_fields,
)


@pytest.fixture(scope="module")
def raster(dataset):
    return field_raster(dataset)


# --------------------------------------------------------------------------- #
# Raster.from_bounds
# --------------------------------------------------------------------------- #


def test_from_bounds_covers_the_requested_rectangle():
    r = Raster.from_bounds(-11.6, 11.6, -9.8, 9.8, 0.15)
    xmin, xmax, ymin, ymax = r.extent
    # Tolerance is float noise, not slack: the edge lands within 1e-12 km
    # (a picometre) of the request, which is coverage at any meaningful scale.
    eps = 1e-9
    assert xmin <= -11.6 + eps and xmax >= 11.6 - eps
    assert ymin <= -9.8 + eps and ymax >= 9.8 - eps
    # cells are square and the requested size
    assert r.x[1] - r.x[0] == pytest.approx(0.15)
    assert r.y[1] - r.y[0] == pytest.approx(0.15)


def test_from_bounds_handles_a_non_square_site():
    """The reason it exists: ``from_site`` assumes a centred square, and a real
    survey is only that by coincidence."""
    r = Raster.from_bounds(0.0, 20.0, 0.0, 10.0, 0.5)
    assert r.shape == (20, 40)  # (ny, nx)


@pytest.mark.parametrize(
    "bounds", [(1.0, 1.0, 0.0, 1.0), (0.0, 1.0, 2.0, 1.0)]
)
def test_from_bounds_rejects_empty_bounds(bounds):
    with pytest.raises(ValueError, match="empty bounds"):
        Raster.from_bounds(*bounds, 0.1)


def test_from_bounds_rejects_a_non_positive_resolution():
    with pytest.raises(ValueError, match="res_km"):
        Raster.from_bounds(0.0, 1.0, 0.0, 1.0, 0.0)


# --------------------------------------------------------------------------- #
# choosing the grid
# --------------------------------------------------------------------------- #


def test_a_synthetic_dataset_reuses_its_own_raster(dataset):
    """The pixel-alignment guarantee.

    Prediction maps are meant to be compared with the truth maps by flipping
    between figures; a different grid would make that comparison approximate
    for no reason.
    """
    assert field_raster(dataset) is dataset.raster


def test_real_data_gets_a_grid_built_from_the_cpt_extent(real_data_dir):
    from cpt_geostat.contract import read_dataset

    ds = read_dataset(real_data_dir)
    assert ds.raster is None
    r = field_raster(ds, res_km=0.5)
    xmin, xmax, _, _ = r.extent
    assert xmin < ds.layout["x"].min()  # padded beyond the outermost hole
    assert xmax > ds.layout["x"].max()
    assert r.x[1] - r.x[0] == pytest.approx(0.5)


def test_the_default_resolution_is_used_when_none_is_given(real_data_dir):
    from cpt_geostat.contract import read_dataset

    r = field_raster(read_dataset(real_data_dir))
    assert r.x[1] - r.x[0] == pytest.approx(DEFAULT_RES_KM)


# --------------------------------------------------------------------------- #
# the predicted field
# --------------------------------------------------------------------------- #


def test_the_field_matches_the_raster_shape(dataset, raster):
    p = predict_field(dataset, "unit_1", kriging_factory(dataset, method="OK"), raster)
    assert p.mean.shape == raster.shape
    assert p.sd.shape == raster.shape
    assert np.isfinite(p.mean).all() and np.isfinite(p.sd).all()
    assert p.fitted and p.note is None


def test_the_baseline_field_is_exactly_flat(dataset, raster):
    """What "no spatial model" actually claims, and the thing every other map
    is read against.  Its sd is flat too — it does not know where the data is."""
    p = predict_field(dataset, "unit_1", baseline_factory(dataset), raster)
    assert np.ptp(p.mean) == pytest.approx(0.0, abs=1e-12)
    assert np.ptp(p.sd) == pytest.approx(0.0, abs=1e-12)


def test_kriging_uncertainty_is_not_flat_and_is_lowest_at_the_data(dataset, raster):
    """The property the unmasked maps depend on.

    Nothing is masked by presence, so a mean map for a sparse unit covers ground
    the model knows nothing about.  The sd map is the counterweight — if it were
    flat, the pair of figures would be misleading rather than honest.
    """
    p = predict_field(dataset, "unit_1", kriging_factory(dataset, method="OK"), raster)
    assert np.ptp(p.sd) > 1e-3

    block = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"]
    ix = np.abs(raster.x - block["x"].iloc[0]).argmin()
    iy = np.abs(raster.y - block["y"].iloc[0]).argmin()
    at_data = p.sd[iy, ix]
    # the farthest corner from any CPT in this layout
    assert at_data < np.nanmax(p.sd)


def test_kriging_honours_its_own_data_without_interpolating_it_exactly(dataset, raster):
    """Conditioning is ``exact=False``: a depth-average is a noisy observation,
    so the surface should pass close to it, not through it."""
    block = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"]
    p = predict_field(dataset, "unit_1", kriging_factory(dataset, method="OK"), raster)
    ix = np.abs(raster.x - block["x"].iloc[0]).argmin()
    iy = np.abs(raster.y - block["y"].iloc[0]).argmin()
    observed = float(block["log_Q_mean"].iloc[0])
    assert abs(p.mean[iy, ix] - observed) < 0.5 * block["log_Q_mean"].std()


def test_the_field_is_the_latent_sd_not_the_observation_sd(dataset, raster):
    """A map is a statement about the ground.  Observation noise is a property
    of taking a measurement, and adding it would make every model's map look
    uniformly and misleadingly wide."""
    factory = kriging_factory(dataset, method="OK")
    p = predict_field(dataset, "unit_1", factory, raster)
    block = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"]
    est = factory("unit_1").fit(block[["x", "y"]].to_numpy(), block["log_Q_mean"].to_numpy())
    probe = np.array([[float(raster.x[0]), float(raster.y[0])]])
    _, sd_latent = est.predict(probe, return_std=True)
    assert p.sd[0, 0] == pytest.approx(sd_latent[0])


# --------------------------------------------------------------------------- #
# degenerate units
# --------------------------------------------------------------------------- #


def test_an_unfittable_unit_returns_nan_and_a_reason(dataset, raster):
    """One thin unit out of 23 must not take the whole figure down."""
    import pandas as pd

    one = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"].head(1)
    bare = type("Bare", (), {"unit_summary": pd.concat([one]), "unit_ids": ["unit_1"],
                             "raster": None, "config": None})()
    p = predict_field(bare, "unit_1", kriging_factory(bare, method="OK"), raster)
    assert not p.fitted
    assert np.isnan(p.mean).all() and np.isnan(p.sd).all()
    assert "not fitted" in p.note


def test_a_unit_absent_from_the_summary_is_reported_not_raised(dataset, raster):
    p = predict_field(dataset, "nope", kriging_factory(dataset, method="OK"), raster)
    assert not p.fitted and p.n_cpt == 0
    assert "absent" in p.note


def test_cpt_fraction_reports_presence(dataset, raster):
    p = predict_field(dataset, "unit_3", baseline_factory(dataset), raster)
    assert 0.0 < p.cpt_fraction < 1.0
    assert p.n_cpt < p.n_cpt_total


# --------------------------------------------------------------------------- #
# all units at once
# --------------------------------------------------------------------------- #


def test_predict_fields_covers_every_unit_in_order(dataset, raster):
    out = predict_fields(dataset, baseline_factory(dataset), raster)
    assert list(out) == list(dataset.unit_ids)
    assert all(p.mean.shape == raster.shape for p in out.values())
