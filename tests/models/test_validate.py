"""Cross-validation, metrics and the prediction cross plots."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cpt_geostat.models import UnitMeanEstimator, baseline_factory  # noqa: E402
from cpt_geostat.plots import plot_prediction_vs_truth  # noqa: E402
from cpt_geostat.plots.predictions import _is_constant  # noqa: E402
from cpt_geostat.validate import coverage, loo_by_unit, loo_predict, mssr, r2, rmse, score_by_unit  # noqa: E402
from cpt_geostat.validate.cv import target_columns  # noqa: E402


# --------------------------------------------------------------------------- #
# metrics, against cases with a known answer
# --------------------------------------------------------------------------- #


def test_rmse_and_r2_against_hand_computed_values():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.0, 2.5])
    assert rmse(y, pred) == pytest.approx(np.sqrt((0.25 + 0 + 0.25) / 3))
    assert r2(y, pred) == pytest.approx(1 - 0.5 / 2.0)


def test_coverage_and_mssr_on_an_exactly_calibrated_sample():
    """Residuals drawn at exactly ±1 sd: mssr is 1 and everything is inside 95%."""
    y = np.array([1.0, -1.0, 1.0, -1.0])
    pred = np.zeros(4)
    sd = np.ones(4)
    assert mssr(y, pred, sd) == pytest.approx(1.0)
    assert coverage(y, pred, sd, level=0.95) == pytest.approx(1.0)
    # halving the stated sd doubles the standardised residual: mssr goes to 4
    assert mssr(y, pred, sd / 2) == pytest.approx(4.0)
    assert coverage(y, pred, sd / 2, 0.95) == pytest.approx(0.0)


def test_metrics_ignore_non_finite_pairs():
    y = np.array([1.0, 2.0, np.nan])
    pred = np.array([1.0, 2.0, 5.0])
    assert rmse(y, pred) == pytest.approx(0.0)
    assert coverage(y, pred, np.array([1.0, 1.0, np.nan])) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# leave-one-out
# --------------------------------------------------------------------------- #


def test_loo_prediction_excludes_the_held_out_point():
    """For a constant mean the analytic answer is ``(n*m - y_i) / (n - 1)``.

    If a fold ever leaked its own point the prediction would equal the full-sample
    mean instead, which at n = 5 is a 25% difference — but on a real dataset it
    would look like an unusually good model rather than a bug.
    """
    y = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    X = np.zeros((5, 2))
    pred, _, _ = loo_predict(UnitMeanEstimator, X, y)
    expected = (y.sum() - y) / (y.size - 1)
    assert np.allclose(pred, expected)
    assert not np.allclose(pred, y.mean())


def test_loo_r2_of_a_constant_model_hits_its_closed_form(dataset):
    """The reference line every spatial estimator has to clear.

    A held-out constant-mean fold predicts ``m_-i``, and ``y_i - m_-i`` is
    exactly ``n/(n-1)`` times ``y_i - m``, so the leave-one-out ``r2`` is
    ``1 - (n/(n-1))**2`` for every unit regardless of the data.  It is *always*
    negative — never zero — and it shrinks towards zero as n grows.  Anything
    that lands above it has explained something real.
    """
    cv = loo_by_unit(dataset, baseline_factory(dataset), "baseline")
    scores = score_by_unit(cv, target="observed").droplevel("model")
    for uid, row in scores.iterrows():
        n = row["n"]
        assert row["r2"] == pytest.approx(1.0 - (n / (n - 1)) ** 2, rel=1e-9), uid
    assert (scores["r2"] < 0).all()


def test_loo_carries_both_variances_and_they_differ_by_the_noise(dataset):
    cv = loo_by_unit(dataset, baseline_factory(dataset), "baseline")
    assert (cv["sd_obs"] > cv["sd_latent"]).all()
    # for the constant model the latent sd is the standard error, so it shrinks
    # with the number of CPTs while the observation sd does not
    per_unit = cv.groupby("unit_id").agg(n=("pred", "size"), lat=("sd_latent", "first"),
                                         obs=("sd_obs", "first"))
    assert per_unit.loc["unit_1", "lat"] < per_unit.loc["unit_5", "lat"]
    assert per_unit.loc["unit_1", "obs"] / per_unit.loc["unit_1", "lat"] > 5


def test_baseline_is_calibrated_against_observations_and_not_against_the_field(dataset):
    """The headline result, and the reason both targets exist.

    Scored on what it predicts — a held-out observation — the constant model is
    calibrated: it dumps all spatial variation into the noise term and the
    intervals come out right.  Scored against the latent field the same
    intervals are wrong by an order of magnitude, because a flat field held to
    its own standard error is a false claim.  A harness that reported only one
    of these numbers would be misleading whichever one it chose.
    """
    cv = loo_by_unit(dataset, baseline_factory(dataset), "baseline")
    observed = score_by_unit(cv, target="observed").droplevel("model")
    latent = score_by_unit(cv, target="latent").droplevel("model")

    # Units 1 and 2 are present at every CPT, so mssr is estimated from enough
    # folds to be stable.  The thin units are not pinned tightly: mssr at n = 5
    # has a sampling sd of its own near 0.6, and a tight bound there would be
    # asserting the seed, not the estimator.
    assert observed.loc[["unit_1", "unit_2"], "mssr"].between(0.8, 1.25).all()
    assert observed.loc[["unit_1", "unit_2"], "coverage95"].between(0.88, 1.0).all()
    assert (observed["mssr"] < 3.0).all()

    # Against the latent field the same intervals are out badly everywhere, and
    # by more than an order of magnitude on the units that carry real structure.
    # The gap is narrowest on unit 6 — nugget above sill, so there is less field
    # for a flat model to be wrong about, which is the unit behaving as designed.
    assert (latent["mssr"] > 5).all()
    assert (latent.loc[["unit_1", "unit_2"], "mssr"] > 20).all()
    assert latent["mssr"].idxmin() == "unit_6"


def test_target_columns_pairs_each_target_with_its_own_sd():
    assert target_columns("observed") == ("observed", "sd_obs")
    assert target_columns("latent") == ("latent", "sd_latent")
    with pytest.raises(ValueError):
        target_columns("truth")


def test_a_unit_at_one_cpt_yields_nan_rather_than_raising():
    """Leave-one-out on a single point leaves nothing to fit."""
    pred, sd_latent, sd_obs = loo_predict(UnitMeanEstimator, np.zeros((1, 2)), np.array([2.0]))
    assert np.isnan(pred).all() and np.isnan(sd_obs).all()


# --------------------------------------------------------------------------- #
# the cross plot
# --------------------------------------------------------------------------- #


def test_cross_plot_returns_a_figure_with_one_panel_per_unit(dataset):
    cv = loo_by_unit(dataset, baseline_factory(dataset), "baseline")
    fig = plot_prediction_vs_truth(dataset, cv)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == len(dataset.unit_ids)
    plt.close(fig)


def test_cross_plot_axes_are_square_so_the_one_to_one_line_is_at_45_degrees(dataset):
    """Independently scaled axes put the 1:1 line at an arbitrary angle and make
    any model look like it tracks truth."""
    cv = loo_by_unit(dataset, baseline_factory(dataset), "baseline")
    fig = plot_prediction_vs_truth(dataset, cv)
    for ax in (a for a in fig.axes if a.get_visible()):
        assert ax.get_xlim() == pytest.approx(ax.get_ylim())
    plt.close(fig)


def test_cross_plot_refuses_a_latent_target_without_truth(dataset):
    """Silently falling back to the observed column would relabel the axis and
    change what the error bars mean."""
    cv = loo_by_unit(dataset, baseline_factory(dataset), "baseline").drop(columns=["latent"])
    with pytest.raises(ValueError, match="truth_points"):
        plot_prediction_vs_truth(dataset, cv, target="latent")


def test_cross_plot_handles_two_models_in_one_panel(dataset):
    cv = loo_by_unit(dataset, baseline_factory(dataset), "baseline")
    both = pd.concat([cv, cv.assign(model="baseline_copy")], ignore_index=True)
    fig = plot_prediction_vs_truth(dataset, both)
    ax = next(a for a in fig.axes if a.get_visible())
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"baseline", "baseline_copy"}
    plt.close(fig)


def test_constant_model_detection_survives_the_leave_one_out_wobble(dataset):
    """LOO moves a constant prediction by ``range(y)/(n-1)``, which at n = 22 is
    not negligible; a tolerance that ignored n would flag only the big units."""
    cv = loo_by_unit(dataset, baseline_factory(dataset), "baseline")
    for uid in dataset.unit_ids:
        assert _is_constant(cv[cv["unit_id"] == uid]), uid

    spatial = cv.assign(pred=cv["observed"])  # a model that tracks truth exactly
    assert not _is_constant(spatial[spatial["unit_id"] == "unit_1"])
