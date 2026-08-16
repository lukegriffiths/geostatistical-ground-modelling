"""Ordinary and universal kriging, and the gap between them and simple kriging.

The three estimators differ only in what they assume about the mean.  These
tests are organised around making each difference *visible as a number* rather
than asserting all three are roughly the same — a suite that only checked they
agreed would pass just as happily if OK had been wired up as SK.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.covariance import GrfConfig
from cpt_geostat.models import (
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
    kriging_factory,
)


@pytest.fixture
def sample():
    """40 points with a known constant mean and no trend."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, (40, 2))
    y = rng.normal(3.0, 1.0, 40)
    return X, y


TRUE_COV = (GrfConfig(sill=0.8, range_km=2.0), 0.2)


# --------------------------------------------------------------------------- #
# the variance convention — pinned for OK and UK, not extrapolated from SK
# --------------------------------------------------------------------------- #


def test_ordinary_kriging_variance_also_includes_the_nugget(sample):
    """gstools' convention is the same for OK as for SK — but *checked*.

    ``SimpleKriging`` already relies on this and pins it.  Assuming it carries
    over to a different gstools class is exactly the kind of extrapolation that
    the SK nugget bug came from, so OK gets its own assertion.

    Far from data the OK variance tends to ``sill + nugget`` **plus** the cost
    of estimating the unknown mean, so it is bounded below by the SK limit
    rather than equal to it.
    """
    X, y = sample
    ok = OrdinaryKriging(covariance=TRUE_COV).fit(X, y)
    far = np.array([[500.0, 500.0]])
    _, sd_latent = ok.predict(far, return_std=True)
    _, sd_obs = ok.predict_observation(far)

    # latent -> at least the sill; observation -> at least sill + nugget
    assert sd_latent[0] ** 2 > 0.8 - 1e-9
    assert sd_obs[0] ** 2 > 1.0 - 1e-9
    assert sd_obs[0] > sd_latent[0]


def test_universal_kriging_variance_also_includes_the_nugget(sample):
    X, y = sample
    uk = UniversalKriging(covariance=TRUE_COV).fit(X, y)
    at = X[:5]
    _, sd_latent = uk.predict(at, return_std=True)
    _, sd_obs = uk.predict_observation(at)
    assert np.all(sd_obs > sd_latent)
    assert np.allclose(sd_obs**2 - sd_latent**2, 0.2, atol=1e-9)


# --------------------------------------------------------------------------- #
# the OK gap — quantified, not hidden
# --------------------------------------------------------------------------- #


def test_ordinary_kriging_is_strictly_wider_than_simple_kriging(sample):
    """The headline difference, and it must be a real number, not noise.

    Simple kriging is *handed* the mean; ordinary kriging estimates it.  That
    one degree of freedom has to show up as wider intervals everywhere — if it
    does not, OK has been wired up as SK and every calibration number below is
    measuring the wrong estimator.
    """
    X, y = sample
    sk = SimpleKriging(mean=float(y.mean()), covariance=TRUE_COV).fit(X, y)
    ok = OrdinaryKriging(covariance=TRUE_COV).fit(X, y)

    grid = np.random.default_rng(1).uniform(-5, 5, (50, 2))
    _, sd_sk = sk.predict(grid, return_std=True)
    _, sd_ok = ok.predict(grid, return_std=True)

    assert np.all(sd_ok >= sd_sk - 1e-12)
    assert np.mean(sd_ok - sd_sk) > 1e-4  # a real gap, not float noise


def test_the_ok_gap_grows_with_distance_from_the_data(sample):
    """Where the difference comes from: not knowing the level costs most where
    there is no data to pin it down.  Near the CPTs the two nearly agree."""
    X, y = sample
    sk = SimpleKriging(mean=float(y.mean()), covariance=TRUE_COV).fit(X, y)
    ok = OrdinaryKriging(covariance=TRUE_COV).fit(X, y)

    near = X[:10] + 0.05
    far = np.array([[60.0, 60.0], [-60.0, 60.0]])
    gap = lambda P: float(  # noqa: E731
        np.mean(ok.predict(P, return_std=True)[1] - sk.predict(P, return_std=True)[1])
    )
    assert gap(far) > gap(near)


def test_far_from_data_simple_kriging_returns_to_its_given_mean_and_ordinary_does_not(sample):
    """SK is told the mean and reverts to it; OK has only the data's own level.

    They happen to be close here because SK was handed the sample mean — the
    difference that matters is the *variance*, above, not the mean.
    """
    X, y = sample
    sk = SimpleKriging(mean=1.23, covariance=TRUE_COV).fit(X, y)
    far = np.array([[900.0, 900.0]])
    assert sk.predict(far)[0] == pytest.approx(1.23, abs=1e-8)

    ok = OrdinaryKriging(covariance=TRUE_COV).fit(X, y)
    assert ok.predict(far)[0] == pytest.approx(float(y.mean()), rel=0.25)


# --------------------------------------------------------------------------- #
# universal kriging and the trend
# --------------------------------------------------------------------------- #


def test_universal_kriging_follows_a_linear_trend_the_others_flatten(sample):
    """The capability UK adds.  On data with a strong planted trend, SK and OK
    revert to a constant away from the data; UK continues the plane."""
    rng = np.random.default_rng(5)
    X = rng.uniform(-5, 5, (60, 2))
    y = 2.0 + 0.30 * X[:, 0] + rng.normal(0, 0.1, 60)

    probe = np.array([[9.0, 0.0]])  # beyond the data, along the trend
    sk = SimpleKriging(covariance=TRUE_COV).fit(X, y).predict(probe)[0]
    ok = OrdinaryKriging(covariance=TRUE_COV).fit(X, y).predict(probe)[0]
    uk = UniversalKriging(covariance=TRUE_COV).fit(X, y).predict(probe)[0]

    expected = 2.0 + 0.30 * 9.0
    assert uk > ok and uk > sk           # only UK keeps climbing
    assert abs(uk - expected) < abs(ok - expected)


def test_universal_kriging_reports_the_trend_it_found(sample):
    rng = np.random.default_rng(6)
    X = rng.uniform(-6, 6, (60, 2))
    y = 1.0 + 0.2 * X[:, 0] + rng.normal(0, 0.08, 60)
    uk = UniversalKriging(covariance=TRUE_COV).fit(X, y)
    p = uk.params_
    assert p["mean_model"] == "unknown linear drift"
    assert p["trend_grad"] == pytest.approx(0.2, rel=0.15)
    assert p["trend_azimuth_deg"] == pytest.approx(90.0, abs=10)  # +x is east


def test_universal_kriging_declines_to_report_a_bearing_for_a_flat_surface(sample):
    """The same rule anisotropy angles follow at ratio 1: a bearing fitted to
    something statistically flat is a number with no content."""
    X, y = sample  # no trend
    uk = UniversalKriging(covariance=TRUE_COV).fit(X, y)
    assert uk.params_["trend_azimuth_deg"] is None


# --------------------------------------------------------------------------- #
# interface conformance and guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [SimpleKriging, OrdinaryKriging, UniversalKriging])
def test_all_three_conform_to_the_estimator_interface(sample, cls):
    X, y = sample
    est = cls(covariance=TRUE_COV).fit(X, y)
    mean, sd = est.predict(X[:4], return_std=True)
    assert mean.shape == (4,) and sd.shape == (4,)
    assert np.all(np.isfinite(mean)) and np.all(sd >= 0)
    for key in ("sill", "range_km", "nugget", "mean_model", "n"):
        assert key in est.params_, key


@pytest.mark.parametrize(
    "cls,n", [(OrdinaryKriging, 1), (UniversalKriging, 3)]
)
def test_too_few_points_for_the_mean_model_raises(cls, n):
    """UK spends three degrees of freedom on the drift.  Fitting it to three
    CPTs does not fail loudly in gstools — it returns numbers — which is the
    IJmuiden failure mode the estimability gate exists for."""
    X = np.zeros((n, 2))
    with pytest.raises(ValueError, match="degrees of freedom"):
        cls(covariance=TRUE_COV).fit(X, np.arange(float(n)))


@pytest.mark.parametrize("method", ["SK", "OK", "UK"])
def test_factory_builds_each_method_unfitted(dataset, method):
    est = kriging_factory(dataset, covariance="fit", method=method)("unit_1")
    assert not hasattr(est, "cov_")
    assert isinstance(est, {"SK": SimpleKriging, "OK": OrdinaryKriging,
                            "UK": UniversalKriging}[method])


def test_factory_rejects_an_unknown_method(dataset):
    with pytest.raises(ValueError, match="unknown kriging method"):
        kriging_factory(dataset, method="OrdinaryIsh")


def test_a_pure_nugget_covariance_collapses_ordinary_kriging_onto_the_mean(dataset):
    """With no structure to exploit there is nothing for OK to do either."""
    b = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"]
    X, y = b[["x", "y"]].to_numpy(), b["log_Q_mean"].to_numpy()
    est = OrdinaryKriging(covariance=(GrfConfig(sill=0.0, range_km=1.0), 0.09)).fit(X, y)
    assert np.allclose(est.predict(X), y.mean(), atol=1e-6)


@pytest.mark.parametrize("method", ["OK", "UK"])
def test_runs_on_every_unit_and_predicts_finitely(dataset, method):
    from cpt_geostat.validate import loo_by_unit

    cv = loo_by_unit(dataset, kriging_factory(dataset, method=method), method)
    assert cv["pred"].notna().all()
    assert (cv["sd_obs"] > 0).all()
    assert (cv["sd_obs"] >= cv["sd_latent"]).all()
