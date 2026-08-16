"""The per-unit baseline estimator."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cpt_geostat.models import UnitMeanEstimator, fit_unit_baselines, unit_baseline_table
from cpt_geostat.models.baseline import pooled_residual_var


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #


def test_fit_recovers_the_sample_mean_and_variance():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.zeros((4, 2))
    m = UnitMeanEstimator().fit(X, y)
    assert m.mean_ == pytest.approx(2.5)
    assert m.residual_var_ == pytest.approx(y.var(ddof=1))


def test_prediction_is_constant_and_ignores_location():
    m = UnitMeanEstimator().fit(np.zeros((5, 2)), np.arange(5.0))
    far_apart = np.array([[-7.0, -7.0], [0.0, 0.0], [7.0, 7.0]])
    assert np.allclose(m.predict(far_apart), m.mean_)


def test_latent_variance_is_the_standard_error_and_observation_variance_is_not():
    """The two conventions differ by exactly the residual variance.

    Getting this backwards is silent: the latent sd shrinks with n and the
    observation sd does not, so a CV scored on the latent sd looks fine at small
    n and gets worse the more data it is given.
    """
    y = np.random.default_rng(0).normal(1.5, 0.4, size=40)
    m = UnitMeanEstimator().fit(np.zeros((40, 2)), y)
    X = np.zeros((3, 2))

    _, sd_latent = m.predict(X, return_std=True)
    _, sd_obs = m.predict_observation(X)
    s = math.sqrt(m.residual_var_)

    assert sd_latent == pytest.approx(s / math.sqrt(40))
    assert sd_obs == pytest.approx(s * math.sqrt(1 + 1 / 40))
    assert np.all(sd_obs > sd_latent)


def test_params_attribute_all_variance_to_the_nugget():
    """The flat model has no sill and no identifiable range — and says so."""
    m = UnitMeanEstimator().fit(np.zeros((6, 2)), np.arange(6.0))
    p = m.params_
    assert p["sill"] == 0.0
    assert p["nugget"] == pytest.approx(m.residual_var_)
    assert p["range_km"] is None and p["aniso_angle_deg"] is None


def test_single_cpt_uses_the_fallback_variance_and_is_nan_without_one():
    X, y = np.zeros((1, 2)), np.array([2.0])
    assert math.isnan(UnitMeanEstimator().fit(X, y).residual_var_)
    assert UnitMeanEstimator(fallback_var=0.25).fit(X, y).residual_var_ == pytest.approx(0.25)


def test_fit_rejects_mismatched_coordinates():
    with pytest.raises(ValueError):
        UnitMeanEstimator().fit(np.zeros((3, 2)), np.arange(5.0))


def test_pooled_residual_var_is_within_unit_not_across_units():
    """Units differ in level by design; pooling raw values would inflate this."""
    summary = pd.DataFrame(
        {
            "unit_id": ["a", "a", "b", "b"],
            "log_Q_mean": [0.0, 2.0, 10.0, 12.0],  # same spread, levels 10 apart
        }
    )
    assert pooled_residual_var(summary) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# on a generated dataset
#
# The comparisons against what the generator actually put there live in
# tests/synthetic/test_truth.py — this file stays on the estimator's own terms.
# --------------------------------------------------------------------------- #


def test_reading_stats_satisfy_the_thickness_weighted_decomposition(dataset):
    """Reading-level scatter splits exactly into between-CPT and within-CPT
    parts — but *thickness-weighted*, which the across-CPT columns are not.

    This is why ``reading_sd`` is computed from the samples rather than by
    adding ``log_Q_sd`` and ``within_sd`` in quadrature: that sum both
    double-counts the depth-averaging error and drops the weighting, and on this
    dataset the two errors do not cancel.
    """
    table = unit_baseline_table(dataset)
    summary = dataset.unit_summary

    for uid, row in table.iterrows():
        block = summary[summary["unit_id"] == uid]
        n = block["n_samples"].to_numpy(dtype=float)
        m = block["log_Q_mean"].to_numpy(dtype=float)
        s = np.nan_to_num(block["log_Q_sd"].to_numpy(dtype=float))
        grand = float(np.sum(n * m) / n.sum())
        sst = float(np.sum(n * (m - grand) ** 2) + np.sum((n - 1) * s**2))
        assert row["reading_sd"] ** 2 == pytest.approx(sst / (n.sum() - 1), rel=1e-9)
        assert row["reading_mean"] == pytest.approx(grand, rel=1e-12)


def test_qtn_quantiles_bracket_the_median(dataset):
    table = unit_baseline_table(dataset)
    assert (table["Qtn_p10"] < table["Qtn_median"]).all()
    assert (table["Qtn_median"] < table["Qtn_p90"]).all()


def test_every_unit_gets_a_model(dataset):
    models = fit_unit_baselines(dataset)
    assert set(models) == set(dataset.unit_ids)


# --------------------------------------------------------------------------- #
# real-data path
# --------------------------------------------------------------------------- #


def test_runs_on_a_bare_unit_summary_with_no_truth_and_no_samples(dataset):
    """Real data supplies a dataframe and nothing else; the reading-level
    columns are dropped rather than faked."""
    table = unit_baseline_table(dataset.unit_summary)
    assert "log_Q_sd" in table and "reading_sd" not in table
    assert len(table) == dataset.unit_summary["unit_id"].nunique()
