"""Generator invariants (Part A)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from cpt_geostat.geometry import pair_distances
from cpt_geostat.synthetic import generate
from cpt_geostat.synthetic.config import Config, ThicknessConfig
from cpt_geostat.synthetic.layout import make_layout
from cpt_geostat.synthetic.strat import _logistic_moment_match


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_reproduces_everything(small_config):
    a = generate(small_config)
    b = generate(small_config)
    pd.testing.assert_frame_equal(a.layout, b.layout)
    pd.testing.assert_frame_equal(a.layers, b.layers)
    pd.testing.assert_frame_equal(a.unit_summary, b.unit_summary)


def test_different_seed_changes_the_realisation(small_config):
    other = copy.deepcopy(small_config)
    other.seed = small_config.seed + 1
    a, b = generate(small_config), generate(other)
    # The corner thinning is itself random, so the CPT count can differ too.
    same_layout = len(a.layout) == len(b.layout) and np.allclose(
        a.layout[["x", "y"]].to_numpy(), b.layout[["x", "y"]].to_numpy()
    )
    assert not same_layout
    assert not np.allclose(
        a.rasters["grf"]["unit_1"], b.rasters["grf"]["unit_1"]
    )


def test_streams_are_named_not_ordered(small_config):
    """Changing one unit must not perturb the others.

    The generator draws from streams keyed by name, so a config edit stays local;
    without that, adding a unit would silently re-roll the whole site and make
    before/after comparisons meaningless.
    """
    edited = copy.deepcopy(small_config)
    edited.units["unit_6"].property.mu += 1.0
    base, new = generate(small_config), generate(edited)
    for uid in ["unit_1", "unit_2", "unit_3", "unit_4", "unit_5"]:
        b = base.unit_values[base.unit_values["unit_id"] == uid]["log_Q_true"].to_numpy()
        n = new.unit_values[new.unit_values["unit_id"] == uid]["log_Q_true"].to_numpy()
        assert np.allclose(b, n), uid


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #


def test_layout_stays_inside_the_site(small_config):
    lay = make_layout(small_config)
    half = small_config.site.size_km / 2
    assert lay["x"].abs().max() <= half
    assert lay["y"].abs().max() <= half
    assert lay["cpt_id"].is_unique


def test_cluster_supplies_short_lag_pairs(small_config):
    lay = make_layout(small_config)
    dist, _ = pair_distances(lay)
    assert (lay["kind"] == "cluster").sum() == small_config.layout.cluster.n
    # Without the cluster the shortest lag is roughly the grid spacing.
    assert dist.min() < 0.5


def test_dropped_corner_is_thinner_than_the_rest(small_config):
    lay = make_layout(small_config)
    half = small_config.site.size_km / 2 - small_config.layout.margin_km
    dc = small_config.layout.drop_corner
    near = np.hypot(lay["x"] - half, lay["y"] - half) <= dc.radius_km
    # NE corner: fewer CPTs there than in the mirrored SW corner.
    mirror = np.hypot(lay["x"] + half, lay["y"] + half) <= dc.radius_km
    assert near.sum() < mirror.sum()


# --------------------------------------------------------------------------- #
# stratigraphy
# --------------------------------------------------------------------------- #


def test_layers_are_contiguous_and_ordered(dataset, small_config):
    order = {u: i for i, u in enumerate(small_config.unit_ids)}
    for cid, grp in dataset.layers.groupby("cpt_id"):
        grp = grp.assign(rank=grp["unit_id"].map(order)).sort_values("rank")
        z = grp[["z_top", "z_bot"]].to_numpy()
        assert np.all(z[:, 1] > z[:, 0]), f"{cid}: non-positive thickness"
        # each unit starts exactly where the previous one ended
        assert np.allclose(z[1:, 0], z[:-1, 1]), f"{cid}: gap or overlap"
        assert z[0, 0] == 0.0, f"{cid}: first unit does not start at the seabed"


def test_nothing_exceeds_max_depth(dataset, small_config):
    assert dataset.layers["z_bot"].max() <= small_config.site.max_depth_m + 1e-9
    assert dataset.samples["z"].max() <= small_config.site.max_depth_m


def test_minimum_retained_thickness_guard(dataset, small_config):
    assert dataset.layers["thickness_m"].min() >= small_config.sampling.min_thickness_m - 1e-9


def test_absent_units_have_no_rows(dataset):
    n_cpt = len(dataset.layout)
    counts = dataset.layers.groupby("unit_id").size()
    assert counts["unit_1"] == n_cpt        # present everywhere
    assert counts["unit_3"] < n_cpt         # channel
    assert counts["unit_5"] < counts["unit_4"]  # narrow channel vs. patch


@pytest.mark.parametrize("unit_id,target", [("unit_4", 0.45), ("unit_6", 0.30)])
def test_patch_coverage_hits_its_target(dataset, unit_id, target):
    realised = dataset.truth["realised"]["units"][unit_id]["coverage_raster"]
    assert abs(realised - target) < 0.06


def test_thickness_moment_match_hits_mean_and_sd():
    cfg = ThicknessConfig(mean_m=6.0, sd_m=2.0, range_km=4.0, min_m=1.0, max_m=11.0)
    _, _, mean, sd = _logistic_moment_match(cfg)
    assert np.isclose(mean, 6.0, atol=1e-6)
    assert np.isclose(sd, 2.0, atol=1e-6)


def test_infeasible_thickness_sd_warns_rather_than_silently_lying():
    # sd of 5 m is impossible for a variable confined to a 4 m window.
    cfg = ThicknessConfig(mean_m=6.0, sd_m=5.0, range_km=4.0, min_m=4.0, max_m=8.0)
    with pytest.warns(UserWarning, match="not attainable"):
        _logistic_moment_match(cfg)


# --------------------------------------------------------------------------- #
# property and depth series
# --------------------------------------------------------------------------- #


def test_depth_series_covers_each_interval(dataset, small_config):
    dz = small_config.sampling.dz_m
    merged = dataset.samples.groupby(["cpt_id", "unit_id"])["z"].agg(["min", "max", "size"])
    layers = dataset.layers.set_index(["cpt_id", "unit_id"])
    for key, row in merged.iterrows():
        z_top, z_bot = layers.loc[key, "z_top"], layers.loc[key, "z_bot"]
        assert row["min"] >= z_top
        assert row["max"] <= z_bot
        assert row["size"] == int(np.floor((z_bot - z_top) / dz))


def test_depth_average_tracks_the_true_unit_value(dataset):
    """Averaging must recover the generating value to within its own noise level."""
    merged = dataset.unit_summary.merge(
        dataset.unit_values, on=["cpt_id", "unit_id"], how="left"
    )
    err = merged["log_Q_mean"] - merged["log_Q_true"]
    assert abs(err.mean()) < 0.02
    assert err.abs().max() < 0.5


def test_qtn_is_positive(dataset):
    assert (dataset.samples["Qtn"] > 0).all()


def test_within_unit_residuals_are_correlated_in_depth(dataset):
    """AR(1) is configured, so consecutive readings must not look independent."""
    trace = dataset.samples[
        (dataset.samples["cpt_id"] == dataset.samples["cpt_id"].iloc[0])
    ].sort_values("z")
    biggest = trace.groupby("unit_id").size().idxmax()
    v = np.log(trace[trace["unit_id"] == biggest]["Qtn"].to_numpy())
    v = v - v.mean()
    lag1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    assert lag1 > 0.1


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"units": {"unit_1": {"presence": {"mode": "everywhere"}}},
                          "site": {"size_km": 10.0, "typo_key": 1}})


def test_aniso_ratio_below_one_is_rejected(small_config):
    bad = copy.deepcopy(small_config)
    bad.units["unit_1"].property.grf.aniso_ratio = 0.25
    with pytest.raises(ValueError, match="aniso_ratio"):
        bad.validate()


def test_thickness_bounds_are_checked(small_config):
    bad = copy.deepcopy(small_config)
    bad.units["unit_1"].thickness.mean_m = 99.0
    with pytest.raises(ValueError, match="mean_m outside"):
        bad.validate()
