"""Simple kriging, the variogram fit, and the covariance parameterisation."""

from __future__ import annotations

import numpy as np
import pytest
import gstools as gs

from cpt_geostat.covariance import (
    GrfConfig,
    build_model,
    len_scale_to_range,
    model_params,
    range_factor,
    range_to_len_scale,
)
from cpt_geostat.models import SimpleKriging, kriging_factory
from cpt_geostat.models.variogram import fit_variogram
from cpt_geostat.validate import loo_by_unit


# --------------------------------------------------------------------------- #
# the parameterisation bridge (plan 02, risk A)
# --------------------------------------------------------------------------- #


def test_practical_range_is_not_the_len_scale():
    """The bug this module exists to prevent: 3.74x, not 1x."""
    assert range_factor("matern25", "practical") == pytest.approx(3.74, abs=0.01)
    assert range_factor("matern25", "len_scale") == 1.0


def test_range_conversion_round_trips():
    for model in ("matern25", "matern15", "exponential", "gaussian", "spherical"):
        cfg = GrfConfig(range_km=3.5, model=model, range_kind="practical")
        ls = range_to_len_scale(cfg)
        assert len_scale_to_range(ls, model, "practical") == pytest.approx(3.5, rel=1e-12)


def test_built_model_reaches_5_percent_correlation_at_the_practical_range():
    """The definition, checked against the library rather than against itself."""
    cfg = GrfConfig(sill=1.0, range_km=3.0, model="matern25")
    m = build_model(cfg)
    # correlation() takes a separation; cor() takes h/len_scale, and confusing
    # the two silently rescales every range by 3.74.
    assert m.correlation(3.0) == pytest.approx(0.05, abs=1e-9)


def test_model_params_round_trips_anisotropy_without_rotating_it_90_degrees():
    """gstools stores minor/major and radians CCW from +x; we quote major/minor
    and degrees CW from north.  A sign slip here puts a recovered channel axis
    across the channel, which looks perfectly plausible on a map."""
    cfg = GrfConfig(sill=0.09, range_km=4.0, aniso_ratio=4.0, aniso_angle_deg=70.0)
    p = model_params(build_model(cfg, nugget=0.006))
    assert p["sill"] == pytest.approx(0.09)
    assert p["range_km"] == pytest.approx(4.0, rel=1e-9)
    assert p["aniso_ratio"] == pytest.approx(4.0, rel=1e-9)
    assert p["aniso_angle_deg"] == pytest.approx(70.0, abs=1e-9)
    assert p["nugget"] == pytest.approx(0.006)


def test_isotropic_model_reports_no_anisotropy_angle():
    """An axis is meaningless at ratio 1, and a number there would be read as one."""
    assert model_params(build_model(GrfConfig(aniso_ratio=1.0)))["aniso_angle_deg"] is None


# --------------------------------------------------------------------------- #
# the variance convention (plan 02, risk C) — pinned, not assumed
# --------------------------------------------------------------------------- #


def test_gstools_kriging_variance_includes_the_nugget():
    """Far from any data gstools' kriging variance tends to sill + nugget.

    It is therefore an **observation** variance, not a latent-field one.  Every
    downstream convention in this package depends on that fact, and it is a
    property of the library rather than of our code, so it is pinned here: if a
    gstools upgrade changes it, this fails rather than the calibration numbers
    quietly drifting.
    """
    sill, nugget = 0.8, 0.2
    model = gs.Matern(dim=2, nu=2.5, var=sill, len_scale=1.0, nugget=nugget)
    rng = np.random.default_rng(0)
    pos = rng.uniform(-5, 5, (2, 20))
    k = gs.krige.Simple(model, pos, rng.normal(0, 1, 20), mean=0.0,
                        exact=False, cond_err="nugget")
    _, var_far = k(([500.0], [500.0]), return_var=True)
    assert var_far[0] == pytest.approx(sill + nugget, rel=1e-6)
    assert var_far[0] != pytest.approx(sill, rel=1e-3)


def test_predict_returns_latent_and_predict_observation_adds_the_noise_back():
    """Our two methods sit either side of the library's single number."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-5, 5, (30, 2))
    y = rng.normal(1.5, 0.5, 30)
    cfg = GrfConfig(sill=0.8, range_km=2.0)
    est = SimpleKriging(covariance=(cfg, 0.2)).fit(X, y)

    far = np.array([[500.0, 500.0]])
    _, sd_latent = est.predict(far, return_std=True)
    _, sd_obs = est.predict_observation(far)

    assert sd_latent[0] ** 2 == pytest.approx(0.8, rel=1e-6)   # sill alone
    assert sd_obs[0] ** 2 == pytest.approx(1.0, rel=1e-6)      # sill + nugget
    assert est.noise_var_ == pytest.approx(0.2)


def test_far_from_data_the_prediction_returns_to_the_known_mean():
    rng = np.random.default_rng(2)
    X = rng.uniform(-5, 5, (25, 2))
    est = SimpleKriging(mean=1.23, covariance=(GrfConfig(sill=0.5, range_km=1.0), 0.05))
    est.fit(X, rng.normal(1.23, 0.7, 25))
    assert est.predict(np.array([[900.0, 900.0]]))[0] == pytest.approx(1.23, abs=1e-8)


# --------------------------------------------------------------------------- #
# variogram fitting
# --------------------------------------------------------------------------- #


def test_fitted_sill_plus_nugget_equals_the_sample_variance(dataset):
    """The constraint that stops sill, nugget and range trading off freely."""
    for uid in dataset.unit_ids:
        b = dataset.unit_summary[dataset.unit_summary["unit_id"] == uid]
        if len(b) < 5:
            continue
        vf = fit_variogram(b["x"].to_numpy(), b["y"].to_numpy(), b["log_Q_mean"].to_numpy())
        assert vf.sill + vf.nugget == pytest.approx(vf.sample_var, rel=0.02), uid


def test_the_fitted_range_cannot_exceed_the_longest_fitted_lag(dataset):
    """An unbounded fit returned 1700 km on unit 2.  A range beyond the largest
    lag in the fit is not identifiable, so it is not offered."""
    for uid in dataset.unit_ids:
        b = dataset.unit_summary[dataset.unit_summary["unit_id"] == uid]
        if len(b) < 5:
            continue
        vf = fit_variogram(b["x"].to_numpy(), b["y"].to_numpy(), b["log_Q_mean"].to_numpy())
        assert vf.range_km <= vf.max_lag_km * 1.01, uid


def test_a_trended_unit_is_reported_as_unresolved(dataset):
    """Unit 2 carries a 0.10/km trend, which makes its variogram unbounded: the
    range runs to the bound.  Simple kriging assumes a constant mean, so the
    right output is a flag, not a number."""
    b = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_2"]
    vf = fit_variogram(b["x"].to_numpy(), b["y"].to_numpy(), b["log_Q_mean"].to_numpy())
    assert vf.at_range_bound
    assert not vf.resolved
    assert "bound" in vf.why_not_resolved()

    est = SimpleKriging().fit(b[["x", "y"]].to_numpy(), b["log_Q_mean"])
    assert est.params_["range_km"] is None
    assert "not_identifiable" in est.params_


def test_nugget_floor_declines_when_the_shortest_lag_is_not_short(dataset):
    """gamma(h) is nugget *plus* structure, so it only estimates the nugget when
    h is far inside the range.  Flooring regardless would report a range
    estimate as a nugget."""
    for uid in dataset.unit_ids:
        b = dataset.unit_summary[dataset.unit_summary["unit_id"] == uid]
        if len(b) < 5:
            continue
        vf = fit_variogram(b["x"].to_numpy(), b["y"].to_numpy(), b["log_Q_mean"].to_numpy())
        if vf.nugget_floored:
            assert vf.lags[0] <= 0.1 * vf.range_km, uid


# --------------------------------------------------------------------------- #
# does it actually work
# --------------------------------------------------------------------------- #


def test_a_pure_nugget_covariance_collapses_kriging_onto_the_baseline(dataset):
    """With no structure to exploit there is nothing for kriging to do, and it
    must return the known mean rather than something merely similar to it."""
    b = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"]
    X, y = b[["x", "y"]].to_numpy(), b["log_Q_mean"].to_numpy()
    est = SimpleKriging(covariance=(GrfConfig(sill=0.0, range_km=1.0), 0.09)).fit(X, y)
    assert np.allclose(est.predict(X), y.mean(), atol=1e-8)


def test_fitted_kriging_runs_on_every_unit_and_predicts_finitely(dataset):
    cv = loo_by_unit(dataset, kriging_factory(dataset, covariance="fit"), "SK")
    assert cv["pred"].notna().all()
    assert (cv["sd_obs"] > 0).all()
    assert (cv["sd_obs"] >= cv["sd_latent"]).all()
