"""The shared linear trend surface.

Root-level module, root-level test: ``trend.py`` is a peer of ``geometry.py``
and, like it, exists so that one convention has one implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.geometry import project_on_azimuth
from cpt_geostat.plots.diagnostics import fit_trend_azimuth
from cpt_geostat.trend import detrend, fit_linear_trend


@pytest.fixture
def grid():
    rng = np.random.default_rng(0)
    return rng.uniform(-8, 8, 60), rng.uniform(-8, 8, 60)


# --------------------------------------------------------------------------- #
# the convention bridge
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("azimuth", [0.0, 25.0, 90.0, 115.0, 200.0, 340.0])
@pytest.mark.parametrize("gradient", [0.05, 0.4])
def test_recovers_a_planted_gradient_and_bearing(grid, azimuth, gradient):
    """The round trip that pins the convention.

    A trend built with ``project_on_azimuth`` must come back with the same
    gradient and bearing.  A sign slip or a from-north/from-east mix-up here
    would put a recovered trend on the wrong bearing while still fitting the
    data perfectly — the exact failure mode the azimuth convention exists to
    prevent, and it is invisible on a scatter plot.
    """
    x, y = grid
    v = 1.7 + gradient * project_on_azimuth(x, y, azimuth)
    fit = fit_linear_trend(x, y, v)

    assert fit.gradient == pytest.approx(gradient, rel=1e-10)
    assert fit.azimuth_deg == pytest.approx(azimuth % 360.0, abs=1e-8)
    assert fit.intercept == pytest.approx(1.7, abs=1e-10)


def test_azimuth_wraps_mod_360_not_180(grid):
    """A trend is a direction of *increase*, unlike an anisotropy axis.

    Rising to the north and falling to the north are different surfaces, so the
    two bearings must be 180° apart rather than identified with each other.
    """
    x, y = grid
    up = fit_linear_trend(x, y, project_on_azimuth(x, y, 30.0))
    down = fit_linear_trend(x, y, -project_on_azimuth(x, y, 30.0))
    assert up.azimuth_deg == pytest.approx(30.0, abs=1e-8)
    assert down.azimuth_deg == pytest.approx(210.0, abs=1e-8)


def test_coefficients_match_the_polar_form(grid):
    """``bx, by`` and ``gradient, azimuth`` are the same plane, stated twice."""
    x, y = grid
    fit = fit_linear_trend(x, y, 0.3 * x - 0.2 * y + 4.0)
    a = np.deg2rad(fit.azimuth_deg)
    assert fit.bx == pytest.approx(fit.gradient * np.sin(a), abs=1e-12)
    assert fit.by == pytest.approx(fit.gradient * np.cos(a), abs=1e-12)


# --------------------------------------------------------------------------- #
# predict / detrend — the half the old plotting helper could not do
# --------------------------------------------------------------------------- #


def test_predict_reproduces_the_surface_it_was_fitted_to(grid):
    """The capability ``fit_trend_azimuth`` lacked: it dropped the intercept,
    so it could describe a trend but not add one back at a new location."""
    x, y = grid
    v = 2.5 + 0.12 * x - 0.07 * y
    fit = fit_linear_trend(x, y, v)
    assert np.allclose(fit.predict(x, y), v, atol=1e-10)
    # and at points it never saw
    assert fit.predict(100.0, -50.0) == pytest.approx(2.5 + 0.12 * 100.0 - 0.07 * -50.0)


def test_detrend_leaves_no_trend_behind(grid):
    x, y = grid
    rng = np.random.default_rng(3)
    v = 2.0 + 0.3 * x + rng.normal(0, 0.05, x.size)
    resid, fit = detrend(x, y, v)
    again = fit_linear_trend(x, y, resid)
    assert again.gradient < 1e-12
    assert np.allclose(resid + fit.predict(x, y), v, atol=1e-12)


# --------------------------------------------------------------------------- #
# identifiability — a bearing fitted to a flat surface has no content
# --------------------------------------------------------------------------- #


def test_a_real_gradient_is_identifiable(grid):
    x, y = grid
    rng = np.random.default_rng(1)
    v = 1.0 + 0.25 * project_on_azimuth(x, y, 70.0) + rng.normal(0, 0.15, x.size)
    assert fit_linear_trend(x, y, v).gradient_is_identifiable


def test_pure_noise_is_not_identifiable(grid):
    """Flat plus noise still fits *some* plane; the flag is what stops that
    number being reported as a recovered bearing."""
    x, y = grid
    rng = np.random.default_rng(2)
    fit = fit_linear_trend(x, y, rng.normal(0.0, 0.3, x.size))
    assert fit.gradient > 0  # OLS always returns something
    assert not fit.gradient_is_identifiable


# --------------------------------------------------------------------------- #
# degenerate input — nan, not a least-norm answer dressed as a fit
# --------------------------------------------------------------------------- #


def test_too_few_points_is_undetermined():
    fit = fit_linear_trend([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])
    assert not np.isfinite(fit.gradient)
    assert not fit.gradient_is_identifiable


def test_collinear_layout_is_undetermined():
    """Every CPT on one line determines no unique plane.  ``lstsq`` would return
    the minimum-norm solution, which is a choice the data does not support."""
    t = np.linspace(-5, 5, 20)
    fit = fit_linear_trend(t, 2 * t, 3.0 + 0.5 * t)
    assert not np.isfinite(fit.gradient)


def test_non_finite_values_are_dropped_not_propagated(grid):
    x, y = grid
    v = 1.0 + 0.2 * x
    v = v.copy()
    v[:5] = np.nan
    fit = fit_linear_trend(x, y, v)
    assert fit.n == x.size - 5
    assert fit.gradient == pytest.approx(0.2, rel=1e-9)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        fit_linear_trend([0.0, 1.0, 2.0], [0.0, 1.0], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# the plotting wrapper
# --------------------------------------------------------------------------- #


def test_plot_helper_delegates_rather_than_reimplementing(dataset):
    """``fit_trend_azimuth`` is now a view onto ``fit_linear_trend``.

    Two implementations of one convention is how they drift apart; this asserts
    there is only one.
    """
    sub = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_2"]
    grad, az = fit_trend_azimuth(sub)
    fit = fit_linear_trend(sub["x"], sub["y"], sub["log_Q_mean"])
    assert grad == pytest.approx(fit.gradient, rel=1e-12)
    assert az == pytest.approx(fit.azimuth_deg, rel=1e-12)


def test_plot_helper_still_recovers_the_planted_trend(dataset):
    """The behaviour the wrapper had before it was rewired, unchanged."""
    sub = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_2"]
    grad, azimuth = fit_trend_azimuth(sub)
    true = dataset.config.units["unit_2"].property.trend
    assert abs(grad - true.grad) < 0.02
    assert min(abs(azimuth - true.azimuth_deg), 360 - abs(azimuth - true.azimuth_deg)) < 15
