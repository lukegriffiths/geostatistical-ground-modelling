"""Assembling per-unit predictions into a depth profile.

The maths that matters here is the variance decomposition — three nested sds
answering three different questions — and the join, which must go through the
readings' own ``unit_id`` rather than through ``layers``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpt_geostat.models import kriging_factory
from cpt_geostat.cpt import StressProfile, qt_from_log_qtn, qtn_from_qt
from cpt_geostat.models.profile import (
    profile_coverage,
    qt_by_unit,
    qt_readings,
    reading_predictions,
)
from cpt_geostat.validate import loo_by_unit


@pytest.fixture(scope="module")
def cv(dataset):
    return loo_by_unit(dataset, kriging_factory(dataset, method="OK"), "OK")


@pytest.fixture(scope="module")
def readings(dataset, cv):
    return reading_predictions(dataset, cv)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def test_one_row_per_reading_and_nothing_invented(dataset, readings):
    assert len(readings) == len(dataset.samples)
    assert set(readings["cpt_id"]) == set(dataset.samples["cpt_id"])
    # the actual data is carried through untouched
    got = readings.sort_values(["cpt_id", "z"])["Qtn"].to_numpy()
    want = dataset.samples.sort_values(["cpt_id", "z"])["Qtn"].to_numpy()
    assert np.allclose(got, want)


def test_every_reading_gets_its_own_units_prediction(dataset, readings):
    """The join is on ``(cpt_id, unit_id)`` from the *readings*.

    Going via ``layers`` would break on a hole where a unit re-enters lower
    down: its collapsed layer row spans first-top to last-base and claims depth
    the unit does not occupy.
    """
    merged = readings.merge(
        dataset.unit_summary[["cpt_id", "unit_id"]].assign(known=True),
        on=["cpt_id", "unit_id"], how="left",
    )
    assert merged["known"].notna().all()
    # a unit's prediction is constant down the hole, however many runs it has
    for (cid, uid), g in readings.groupby(["cpt_id", "unit_id"]):
        assert g["pred"].nunique(dropna=True) <= 1, (cid, uid)


def test_a_re_entrant_unit_gets_the_same_prediction_in_both_runs():
    """Constructed rather than found, so the guarantee is explicit.

    Two runs of unit ``a`` separated by unit ``b``; both runs must carry ``a``'s
    single prediction, and ``b``'s run must not be overwritten by it.
    """
    samples = pd.DataFrame({
        "cpt_id": ["C1"] * 6,
        "x": 0.0, "y": 0.0,
        "z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "unit_id": ["a", "a", "b", "b", "a", "a"],
        "Qtn": [10.0, 12.0, 3.0, 3.5, 11.0, 13.0],
    })
    summary = pd.DataFrame({
        "cpt_id": ["C1", "C1"], "x": [0.0, 0.0], "y": [0.0, 0.0],
        "unit_id": ["a", "b"], "log_Q_mean": [2.4, 1.2], "log_Q_sd": [0.1, 0.1],
        "n_samples": [4, 2], "thickness_m": [4.0, 2.0],
    })
    ds = type("DS", (), {"samples": samples, "unit_summary": summary,
                         "unit_ids": ["a", "b"], "config": None})()
    cv = pd.DataFrame({
        "model": "m", "cpt_id": ["C1", "C1"], "unit_id": ["a", "b"],
        "pred": [2.4, 1.2], "sd_latent": [0.2, 0.2], "sd_obs": [0.3, 0.3],
    })
    out = reading_predictions(ds, cv, within_sd={"a": 0.1, "b": 0.1})

    a = out[out["unit_id"] == "a"]
    b = out[out["unit_id"] == "b"]
    assert len(a) == 4 and len(b) == 2
    assert a["pred"].nunique() == 1 and a["pred"].iloc[0] == pytest.approx(2.4)
    assert b["pred"].iloc[0] == pytest.approx(1.2)


def test_an_unpredicted_unit_yields_nan_not_a_filled_value(dataset, cv):
    """A fold that could not be fitted must leave a gap.

    Interpolating across it would draw a confident line through ground the
    model said nothing about.
    """
    dropped = cv[cv["unit_id"] != "unit_3"]
    out = reading_predictions(dataset, dropped)
    assert out.loc[out["unit_id"] == "unit_3", "pred"].isna().all()
    assert out.loc[out["unit_id"] == "unit_1", "pred"].notna().any()
    # the measured values survive regardless
    assert out.loc[out["unit_id"] == "unit_3", "Qtn"].notna().all()


# --------------------------------------------------------------------------- #
# the variance decomposition
# --------------------------------------------------------------------------- #


def test_the_three_sds_are_strictly_nested(readings):
    """latent < unit-mean < single-reading, everywhere.

    Getting this ordering wrong would quote a depth-average's interval against a
    raw trace, claiming a precision the model does not have.
    """
    g = readings[readings["pred"].notna()]
    assert (g["sd_obs"] >= g["sd_latent"] - 1e-12).all()
    assert (g["sd_reading"] >= g["sd_obs"] - 1e-12).all()
    # and the widening is real wherever there is within-unit scatter
    has_w = g["within_sd"] > 0
    assert (g.loc[has_w, "sd_reading"] > g.loc[has_w, "sd_obs"]).all()


def test_the_reading_sd_adds_within_unit_scatter_in_quadrature(readings):
    g = readings[readings["pred"].notna()]
    assert np.allclose(
        g["sd_reading"], np.hypot(g["sd_obs"], g["within_sd"].fillna(0.0)), rtol=1e-12
    )


def test_within_sd_is_pooled_per_unit_not_the_holes_own_scatter(dataset, readings):
    """At a location predicted as unvisited, its own depth-to-depth scatter is
    exactly what would not be known."""
    for uid, g in readings.groupby("unit_id"):
        assert g["within_sd"].nunique(dropna=False) == 1, uid


# --------------------------------------------------------------------------- #
# back to Qtn
# --------------------------------------------------------------------------- #


def test_the_centre_is_a_median_not_a_mean(readings):
    """``exp`` of a mean of logs is a median.  Quoting it as an average
    overstates the typical value, and the column is named accordingly."""
    g = readings[readings["pred"].notna()]
    assert np.allclose(g["Qtn_median"], np.exp(g["pred"]))


def test_band_edges_transform_exactly_rather_than_by_approximation(readings):
    """``exp`` is monotone, so quantiles are equivariant: the Qtn interval is
    the exponential of the log interval, with no delta-method error."""
    g = readings[readings["pred"].notna()]
    for name in ("mean", "read"):
        assert np.allclose(g[f"Qtn_lo_{name}"], np.exp(g[f"log_lo_{name}"]))
        assert np.allclose(g[f"Qtn_hi_{name}"], np.exp(g[f"log_hi_{name}"]))
        assert (g[f"Qtn_lo_{name}"] < g["Qtn_median"]).all()
        assert (g["Qtn_median"] < g[f"Qtn_hi_{name}"]).all()


def test_the_interval_widens_with_the_requested_level(dataset, cv):
    lo = reading_predictions(dataset, cv, level=0.50)
    hi = reading_predictions(dataset, cv, level=0.99)
    ok = lo["pred"].notna()
    width = lambda t: (t.loc[ok, "log_hi_read"] - t.loc[ok, "log_lo_read"])  # noqa: E731
    assert (width(hi) > width(lo)).all()


# --------------------------------------------------------------------------- #
# coverage — measured, not asserted
# --------------------------------------------------------------------------- #


def test_the_reading_band_is_broadly_the_right_width_on_synthetic(readings):
    """A sanity range on the whole decomposition, not a precision claim.

    The tolerance is wide on purpose, for a reason worth stating rather than
    hiding: **this fixture is the pessimistic case.**  It runs 49 CPTs at
    ``dz = 0.2 m`` against the full config's 117 at 0.02 m, which both degrades
    the spatial prediction and changes the within-unit AR(1) correlation
    (``rho = exp(-dz/corr_len)`` falls from 0.95 to 0.61).  Realised coverage
    here is 0.87-0.95 with mssr 1.1-1.6; on the full run it is 0.94-0.95 with
    mssr 0.95-1.09.

    The sharp check on whether ``within_sd`` is actually being included is
    :func:`test_the_unit_mean_band_under_covers_individual_readings` and the
    nesting test, not this one — on this fixture the two bands are only 6 points
    of coverage apart, so a threshold here would not discriminate.
    """
    cov = profile_coverage(readings)
    untrended = cov.drop(index=["unit_2", "unit_4"], errors="ignore")
    assert untrended["coverage_read"].between(0.85, 0.99).all(), cov["coverage_read"]
    assert untrended["mssr_read"].between(0.7, 1.8).all(), cov["mssr_read"]


def test_the_trended_units_come_out_narrow_and_that_is_the_known_cause(readings):
    """Not a defect in the profile — ordinary kriging assumes a constant mean.

    Unit 2 carries a 0.10/km trend, which OK cannot model, so its errors exceed
    its intervals; the same failure already shows up as mssr 2.29 in the
    cross-validation table.  Asserting it here ties the profile's calibration to
    a cause that is understood, rather than leaving a bad number unexplained.
    Universal kriging is the estimator that fixes it.
    """
    cov = profile_coverage(readings)
    assert cov.loc["unit_2", "mssr_read"] > 1.8
    assert cov.loc["unit_2", "coverage_read"] < 0.85


def test_the_unit_mean_band_under_covers_individual_readings(readings):
    """Not a bug — it is the interval for a depth-*average*, scored against
    single readings, which carry the within-unit scatter it excludes.  If these
    ever matched, the two bands would not be measuring different things."""
    cov = profile_coverage(readings)
    assert (cov["coverage_mean"] < cov["coverage_read"]).all()


def test_coverage_counts_only_predicted_readings(dataset, cv):
    out = reading_predictions(dataset, cv[cv["unit_id"] != "unit_3"])
    cov = profile_coverage(out)
    assert cov.loc["unit_3", "n_predicted"] == 0
    assert np.isnan(cov.loc["unit_3", "coverage_read"])
    assert cov.loc["unit_3", "n_readings"] > 0


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


def test_an_ambiguous_cv_table_raises_rather_than_picking_a_model(dataset, cv):
    both = pd.concat([cv, cv.assign(model="other")], ignore_index=True)
    with pytest.raises(ValueError, match="pass model="):
        reading_predictions(dataset, both)
    assert len(reading_predictions(dataset, both, model="other")) == len(dataset.samples)


def test_an_unknown_model_name_raises(dataset, cv):
    with pytest.raises(ValueError, match="no model"):
        reading_predictions(dataset, cv, model="nope")


def test_a_cv_table_missing_columns_is_reported(dataset, cv):
    with pytest.raises(ValueError, match="missing required column"):
        reading_predictions(dataset, cv.drop(columns=["sd_obs"]))


# --------------------------------------------------------------------------- #
# back to qt
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def profile():
    """IJmuiden's fitted stress profile, as the preparation step records it."""
    return StressProfile.from_gradients(20.1, 10.0)


@pytest.fixture(scope="module")
def in_qt(readings, profile):
    return qt_readings(readings, profile=profile, n=1.0)


def test_the_reading_itself_round_trips(in_qt, profile):
    """``qt`` beside ``Qtn`` is the same reading, not a second estimate."""
    z = in_qt["z"].to_numpy()
    assert qtn_from_qt(in_qt["qt"].to_numpy(), z, n=1.0, profile=profile) == pytest.approx(
        in_qt["Qtn"].to_numpy()
    )


def test_converting_to_qt_leaves_the_coverage_untouched(readings, in_qt):
    """The reason the edges are transformed rather than rebuilt from moments.

    A 95% band in log space is a 95% band in ``qt``, exactly — so the realised
    coverage measured on one is a statement about the other, and the conversion
    cannot quietly improve or ruin a calibration result.
    """
    inside_log = (readings["log_Q"] >= readings["log_lo_read"]) & (
        readings["log_Q"] <= readings["log_hi_read"]
    )
    inside_qt = (in_qt["qt"] >= in_qt["qt_lo_read"]) & (in_qt["qt"] <= in_qt["qt_hi_read"])
    assert (inside_log == inside_qt).all()


def test_the_qt_band_brackets_the_median_asymmetrically(in_qt):
    g = in_qt.dropna(subset=["qt_median"])
    assert (g["qt_lo_read"] < g["qt_median"]).all()
    assert (g["qt_median"] < g["qt_hi_read"]).all()
    # right-skewed: the upper arm is the longer one, everywhere
    assert ((g["qt_hi_read"] - g["qt_median"]) > (g["qt_median"] - g["qt_lo_read"])).all()


def test_qt_grows_down_a_unit_even_where_the_prediction_is_flat(in_qt):
    """The depth trend the normalisation removed, restored by the transform.

    Within one unit at one hole the prediction is a single number, so any
    variation in ``qt_median`` is the stress profile — and it must increase.
    """
    for (cid, uid), g in in_qt.dropna(subset=["qt_median"]).groupby(["cpt_id", "unit_id"]):
        g = g.sort_values("z")
        assert g["pred"].nunique() == 1
        assert np.all(np.diff(g["qt_median"].to_numpy()) > 0), (cid, uid)


def test_the_sd_in_qt_units_grows_with_depth_at_constant_log_sd(in_qt):
    """Why an sd in kPa cannot be quoted per unit without a depth."""
    for _, g in in_qt.dropna(subset=["qt_sd_read"]).groupby(["cpt_id", "unit_id"]):
        g = g.sort_values("z")
        if len(g) < 3:
            continue
        assert g["sd_reading"].nunique() == 1
        assert g["qt_sd_read"].iloc[-1] > g["qt_sd_read"].iloc[0]


def test_qt_readings_rejects_a_table_that_is_not_one(readings, profile):
    with pytest.raises(ValueError, match="missing 'pred'"):
        qt_readings(readings.drop(columns=["pred"]), profile=profile)


def test_qt_by_unit_gives_three_depths_per_layer(dataset, cv, profile):
    table = qt_by_unit(cv, dataset.layers, profile=profile, n=1.0)
    assert set(table["where"]) == {"top", "mid", "bot"}
    assert len(table) == 3 * len(
        dataset.layers.merge(cv[["cpt_id", "unit_id"]], on=["cpt_id", "unit_id"])
    )

    for _, g in table.groupby(["cpt_id", "unit_id"]):
        g = g.set_index("where")
        assert g.loc["top", "z"] < g.loc["mid", "z"] < g.loc["bot", "z"]
        # one prediction, three depths -> a line, not three estimates
        assert g["log_Q_pred"].nunique() == 1
        assert g.loc["top", "qt_median"] < g.loc["bot", "qt_median"]
        # The band has width wherever the stress does.  A unit outcropping at
        # the seabed is the exception and collapses to zero there, which is the
        # model's actual statement about z = 0, not a missing value.
        assert g.loc["bot", "qt_lo"] < g.loc["bot", "qt_median"] < g.loc["bot", "qt_hi"]
        if g.loc["top", "z"] > 0:
            assert g.loc["top", "qt_lo"] < g.loc["top", "qt_median"] < g.loc["top", "qt_hi"]
        else:
            assert (g.loc["top", ["qt_lo", "qt_median", "qt_hi", "qt_sd"]] == 0).all()


def test_qt_by_unit_matches_the_readings_at_the_same_depth(dataset, cv, profile):
    """The unit-level and reading-level routes are one transform, not two."""
    table = qt_by_unit(cv, dataset.layers, profile=profile, n=1.0)
    row = table.dropna(subset=["qt_median"]).iloc[0]
    direct = qt_from_log_qtn(row["log_Q_pred"], row["z"], n=1.0, profile=profile)
    assert row["qt_median"] == pytest.approx(direct)


def test_qt_by_unit_reports_a_missing_layer_column(dataset, cv, profile):
    with pytest.raises(ValueError, match="layer table is missing"):
        qt_by_unit(cv, dataset.layers.drop(columns=["z_bot"]), profile=profile)


# --------------------------------------------------------------------------- #
# a variable exponent
# --------------------------------------------------------------------------- #


def test_a_certain_exponent_still_transforms_the_log_band_exactly(readings, profile):
    """The guarantee the qt suite was built on, kept when n_sd exists.

    Rebuilding the edges from (pred, sd, level) must land on the same numbers as
    pushing the stored log edges through the monotone map — otherwise adding
    exponent uncertainty would have silently changed every earlier figure.
    """
    q = qt_readings(readings, profile=profile, n=1.0)
    z = q["z"].to_numpy()
    for band in ("mean", "read"):
        for side in ("lo", "hi"):
            direct = qt_from_log_qtn(q[f"log_{side}_{band}"].to_numpy(), z, n=1.0,
                                     profile=profile)
            assert q[f"qt_{side}_{band}"].to_numpy() == pytest.approx(direct, nan_ok=True)
    assert q.attrs["coverage_transfers"] is True


def test_a_per_unit_exponent_is_applied_per_unit(readings, profile):
    units = sorted(set(readings["unit_id"]))
    n_map = {u: 0.5 + 0.1 * i for i, u in enumerate(units)}
    q = qt_readings(readings, profile=profile, n=n_map)

    for uid, g in q.groupby("unit_id"):
        one = qt_readings(g, profile=profile, n=n_map[uid])
        assert g["qt"].to_numpy() == pytest.approx(one["qt"].to_numpy(), nan_ok=True)
    assert "by unit" in q.attrs["exponent"]


def test_a_unit_with_no_exponent_raises_rather_than_defaulting(readings, profile):
    """A defaulted exponent on one unit is a depth-dependent bias in that unit
    alone — indistinguishable on a map from a real contrast between soils."""
    n_map = {u: 0.6 for u in sorted(set(readings["unit_id"]))[:-1]}
    with pytest.raises(ValueError, match="no n given for unit"):
        qt_readings(readings, profile=profile, n=n_map)


def test_exponent_uncertainty_widens_the_band_and_says_coverage_no_longer_transfers(
    readings, profile
):
    """The flag matters: the trace is de-normalised with the central n, so a
    band widened for exponent doubt is no longer being scored fairly."""
    narrow = qt_readings(readings, profile=profile, n=0.7)
    wide = qt_readings(readings, profile=profile, n=0.7, n_sd=0.15)

    assert wide.attrs["coverage_transfers"] is False
    assert narrow.attrs["coverage_transfers"] is True
    ok = narrow["qt_median"].notna()
    # the median is untouched; only the band moves
    assert wide.loc[ok, "qt_median"].to_numpy() == pytest.approx(
        narrow.loc[ok, "qt_median"].to_numpy()
    )
    away = ok & (wide["z"] > 20.0)
    if away.any():
        assert (wide.loc[away, "qt_hi_read"] > narrow.loc[away, "qt_hi_read"]).all()
        assert (wide.loc[away, "qt_lo_read"] < narrow.loc[away, "qt_lo_read"]).all()
    # and the reading itself is not re-scaled — it carries no exponent doubt
    assert wide["qt"].to_numpy() == pytest.approx(narrow["qt"].to_numpy(), nan_ok=True)


def test_qt_by_unit_reports_the_total_sd_it_used(dataset, cv, profile):
    table = qt_by_unit(cv, dataset.layers, profile=profile, n=0.7, n_sd=0.12)
    g = table.dropna(subset=["qt_median"])
    assert (g["log_Q_sd_total"] >= g["log_Q_sd"]).all()
    # the inflation is depth-dependent, so the three rows of one layer differ
    spread = g.groupby(["cpt_id", "unit_id"])["log_Q_sd_total"].nunique()
    assert (spread > 1).any()
