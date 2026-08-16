"""The truth reference table — what the generator actually put there.

Synthetic-only by construction: every check here needs ``truth.yaml`` and
``truth_points.csv``, which is precisely why the code under test lives in
``cpt_geostat.synthetic`` rather than alongside the estimators it scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.models import unit_baseline_table
from cpt_geostat.synthetic.truth import truth_reference_table


def test_truth_sds_are_measured_over_present_cpts_not_the_whole_site(dataset):
    """A channel unit's latent spread over its own 16% of the site is much
    smaller than over the whole of it.  Comparing the baseline's sd against the
    all-CPT figure — which is what ``truth.yaml`` records — therefore credits a
    spatial model with explaining more variance than the data contains.
    """
    truth = truth_reference_table(dataset)
    assert (truth["field_sd"] < truth["field_sd_all_cpt"]).any()
    for uid in ("unit_1", "unit_2"):  # present everywhere: the two must coincide
        assert truth.loc[uid, "field_sd"] == pytest.approx(
            truth.loc[uid, "field_sd_all_cpt"], rel=1e-6
        )


def test_observation_noise_is_measured_against_the_field_not_the_true_value(dataset):
    """``obs_noise`` is nugget + depth-averaging error, in quadrature.

    Measured against ``log_Q_true`` instead it would capture only the averaging
    error and fall short by exactly the nugget — the failure mode plan 02 flags,
    and it under-reports the irreducible floor by a factor approaching two.
    """
    truth = truth_reference_table(dataset)
    lo = (truth["nugget_sd"] - truth["depth_avg_sd"]).abs()
    hi = truth["nugget_sd"] + truth["depth_avg_sd"]
    assert ((lo <= truth["obs_noise_sd"]) & (truth["obs_noise_sd"] <= hi)).all()

    # Quadrature holds only in expectation: the two components are independent
    # by construction but their empirical cross-term is order 1/sqrt(n_cpt),
    # which is 15-20% at the 20-50 CPTs a unit is held at here.
    assert np.allclose(truth["obs_noise_sd"], np.hypot(truth["nugget_sd"], truth["depth_avg_sd"]),
                       rtol=0.25)

    # every unit's noise floor exceeds the averaging error alone — differencing
    # against log_Q_true instead would make these two equal
    assert (truth["obs_noise_sd"] > truth["depth_avg_sd"]).all()
    assert truth["nugget_sd"].idxmax() == "unit_6"  # the configured noisy unit


def test_structured_fraction_stays_inside_zero_and_one(dataset):
    """The metric the CLI reports.  Defined as ``1 - noise/total`` rather than
    ``field/total``, which is a quotient of two noisy sds and exceeds 1 on the
    units where the field dominates."""
    joined = unit_baseline_table(dataset).join(truth_reference_table(dataset))
    frac = 1.0 - (joined["obs_noise_sd"] / joined["log_Q_sd"]) ** 2
    assert ((frac > 0.0) & (frac <= 1.0)).all()
    # unit 6 is the designed noise-dominated case: least for a GP to find
    assert frac.idxmin() == "unit_6"


def test_mean_is_close_to_mu_where_the_unit_is_present_everywhere(dataset):
    """Units 1 and 2 span the site, so the sample mean has no presence bias.

    Channel and patch units deliberately do not get this check: they sample a
    biased subset of a lumpy field and a gap there is data, not error.
    """
    table = unit_baseline_table(dataset)
    truth = truth_reference_table(dataset)
    for uid in ("unit_1", "unit_2"):
        assert abs(table.loc[uid, "log_Q_mean"] - truth.loc[uid, "mu"]) < 4 * table.loc[
            uid, "se_mean"
        ]


def test_table_matches_the_realised_truth_written_by_the_generator(dataset):
    """``truth.yaml`` already records the across-CPT mean and sd; agreeing with
    it proves the estimator averages what it is supposed to be averaging."""
    table = unit_baseline_table(dataset)
    realised = dataset.truth["realised"]["units"]

    for uid, row in table.iterrows():
        assert row["log_Q_mean"] == pytest.approx(realised[uid]["log_Q_mean_observed"])
        assert row["log_Q_sd"] == pytest.approx(realised[uid]["log_Q_sd_observed"])
        assert row["n_cpt"] == realised[uid]["n_cpt_present"]


def test_returns_none_on_real_data(dataset):
    """A bare dataframe carries no truth, and the answer is ``None`` rather
    than an empty table that would join as silent nans."""
    assert truth_reference_table(dataset.unit_summary) is None
