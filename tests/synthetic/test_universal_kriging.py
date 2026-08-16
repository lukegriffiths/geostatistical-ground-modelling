"""Universal kriging against the contrast set — does the trend unit get fixed?

Unit 2 exists to isolate trend handling: it carries a 0.10/km gradient at 115°
and nothing else unusual.  Simple and ordinary kriging both assume a constant
mean and come out badly overconfident on it, and that failure is *recorded* in
``tests/synthetic/test_truth_kriging.py`` as the expected behaviour of an
estimator with no trend model.

This file is the other half of that statement: universal kriging is supposed to
fix it, and if it does not, the trend machinery is not doing anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpt_geostat.models import kriging_factory
from cpt_geostat.models.kriging import UniversalKriging
from cpt_geostat.models.variogram import fit_variogram
from cpt_geostat.validate import loo_by_unit, score_by_unit

TRENDED = ("unit_2", "unit_4")  # the two units configured with a real gradient


@pytest.fixture(scope="module")
def scores(dataset):
    """Leave-one-out scores for SK, OK and UK on every unit."""
    cv = pd.concat(
        [loo_by_unit(dataset, kriging_factory(dataset, method=m), m) for m in ("SK", "OK", "UK")],
        ignore_index=True,
    )
    return score_by_unit(cv)


# --------------------------------------------------------------------------- #
# the headline: the trended unit
# --------------------------------------------------------------------------- #


def test_universal_kriging_fixes_unit_2s_calibration(scores):
    """SK and OK are wildly overconfident on the trended unit; UK is not.

    The unmodelled trend shows up as excess error against unchanged intervals,
    so MSSR runs to ~31 and 95% coverage collapses to below a third.  Modelling
    the drift is what recovers both, and the size of the change is the evidence
    that the drift is actually being used.
    """
    mssr = scores["mssr"].unstack(0).loc["unit_2"]
    cover = scores["coverage95"].unstack(0).loc["unit_2"]

    assert mssr["SK"] > 5.0 and mssr["OK"] > 5.0     # broken, as designed
    assert 0.5 < mssr["UK"] < 2.0                     # calibrated
    assert cover["SK"] < 0.5 and cover["OK"] < 0.5
    assert cover["UK"] > 0.85


def test_universal_kriging_also_predicts_unit_2_better_than_the_others(scores):
    rmse = scores["rmse"].unstack(0).loc["unit_2"]
    assert rmse["UK"] < rmse["SK"]
    assert rmse["UK"] < rmse["OK"]


def test_it_wins_on_the_trended_units_and_pays_on_the_others(scores):
    """Three drift parameters are not free.

    On a unit with no trend they buy nothing and cost variance, so UK should
    *lose* there.  An implementation that won everywhere would mean the drift
    was being fitted to noise and rewarded for it.
    """
    rmse = scores["rmse"].unstack(0)
    for uid in TRENDED:
        assert rmse.loc[uid, "UK"] < rmse.loc[uid, "SK"], uid
    for uid in ("unit_1", "unit_3", "unit_5"):
        assert rmse.loc[uid, "UK"] > rmse.loc[uid, "SK"], uid


# --------------------------------------------------------------------------- #
# why it works: the residual variogram
# --------------------------------------------------------------------------- #


def test_the_raw_variogram_of_a_trended_unit_is_unbounded(dataset):
    """The reason universal kriging needs a detrended variogram.

    A trend keeps adding variance as the lag grows, so the fit runs to the
    range bound and puts nearly everything into a long-range structured
    component with a tiny nugget — intervals several times too narrow.
    """
    b = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_2"]
    raw = fit_variogram(b["x"].to_numpy(), b["y"].to_numpy(), b["log_Q_mean"].to_numpy())
    assert raw.at_range_bound
    assert not raw.resolved


def test_detrending_bounds_it_and_shrinks_the_sill(dataset):
    """The same data, with the drift removed first."""
    from cpt_geostat.trend import detrend

    b = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_2"]
    x, y = b["x"].to_numpy(), b["y"].to_numpy()
    raw = fit_variogram(x, y, b["log_Q_mean"].to_numpy())
    resid, _ = detrend(x, y, b["log_Q_mean"].to_numpy())
    det = fit_variogram(x, y, resid)

    assert not det.at_range_bound
    # The trend was being counted as spatial structure; removing it takes that
    # variance out of the sill, which is exactly what UK needs.
    assert det.sample_var < raw.sample_var
    assert det.range_km < raw.range_km


def test_universal_kriging_uses_the_residual_variogram_not_the_raw_one(dataset):
    """Pins the wiring, not just the outcome.

    ``_variogram_values`` is the hook that makes UK work; if a refactor routed
    the raw values through it again, every calibration number above would
    silently regress.
    """
    b = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_2"]
    X = b[["x", "y"]].to_numpy()
    y = b["log_Q_mean"].to_numpy()

    uk = UniversalKriging().fit(X, y)
    assert not uk.variogram_.at_range_bound  # the raw fit *is* at the bound

    # and the values it fitted to really are the residuals
    v = uk._variogram_values(X, y)
    assert np.abs(np.mean(v)) < 1e-9
    assert np.var(v) < np.var(y)


# --------------------------------------------------------------------------- #
# where it breaks down — a finding, not a defect
# --------------------------------------------------------------------------- #


def test_it_degrades_on_the_data_limited_unit(scores):
    """Unit 5 is 22 CPTs in a narrow channel, and UK spends three of its
    degrees of freedom on a drift that geometry cannot pin down.

    The result is a badly calibrated estimator — which is the argument for the
    IJmuiden estimability gate rather than for tuning UK: ten of that site's 23
    units are held at fewer CPTs than this.
    """
    mssr = scores["mssr"].unstack(0).loc["unit_5"]
    assert mssr["UK"] > mssr["SK"]
    assert mssr["UK"] > 3.0  # visibly overconfident, and should be reported so


def test_the_trend_it_recovers_on_unit_2_points_the_right_way(dataset):
    """A drift fitted on the wrong bearing would still fit the data well."""
    b = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_2"]
    uk = UniversalKriging().fit(b[["x", "y"]].to_numpy(), b["log_Q_mean"].to_numpy())
    p = uk.params_
    true = dataset.config.units["unit_2"].property.trend

    assert p["trend_azimuth_deg"] is not None  # identifiable at 0.10/km
    diff = abs(p["trend_azimuth_deg"] - true.azimuth_deg) % 360.0
    assert min(diff, 360.0 - diff) < 20.0
    assert p["trend_grad"] == pytest.approx(true.grad, rel=0.4)
