"""Kriging handed the covariance that generated the data.

The reference model: it separates *"is kriging implemented correctly"* from
*"can a variogram be fitted from 30 CPTs"*.  Both questions matter and a single
number confounds them, so the truth-covariance path exists purely to answer the
first — which is why it lives on the synthetic side and is unavailable on real
data.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.models import baseline_factory, kriging_factory
from cpt_geostat.synthetic.truth import truth_kriging_factory
from cpt_geostat.validate import loo_by_unit, score_by_unit


def test_kriging_with_the_true_covariance_beats_the_baseline(dataset):
    """The headline check.  Given the covariance that generated the data,
    kriging must extract something the constant mean cannot — except on unit 6,
    where the nugget exceeds the sill and there is nothing to extract.
    """
    base = score_by_unit(loo_by_unit(dataset, baseline_factory(dataset), "b"))
    krig = score_by_unit(loo_by_unit(dataset, truth_kriging_factory(dataset), "k"))
    gain = base.droplevel("model")["rmse"] - krig.droplevel("model")["rmse"]
    assert (gain.drop("unit_6") > 0).all(), gain
    # The size of the gain scales with how many CPTs there are to borrow from,
    # so this fixture's reduced layout is held to a weaker bar than the full run
    # (where unit 1 gains 0.054); the sign is the part that must not move.
    assert gain["unit_1"] > 0.005


def test_kriging_with_the_true_covariance_is_calibrated_except_where_there_is_a_trend(dataset):
    """Correct covariance in, correct uncertainty out.  If this drifts, suspect
    the variance convention before suspecting the estimator.

    Unit 2 is excluded, and its exclusion is the finding rather than a let-off:
    simple kriging assumes a **constant** mean, and unit 2 carries a 0.10/km
    trend.  Giving it the true covariance does not fix that — the unmodelled
    trend shows up as excess error against unchanged intervals, so it comes out
    overconfident.  Unit 2 is in the contrast set to isolate trend handling, and
    this is what "no trend handling" looks like.
    """
    krig = score_by_unit(
        loo_by_unit(dataset, truth_kriging_factory(dataset), "k")
    ).droplevel("model")
    others = krig.drop("unit_2")
    assert others["mssr"].between(0.6, 1.5).all(), others["mssr"]
    assert others["coverage95"].between(0.85, 1.0).all(), others["coverage95"]
    assert krig["mssr"].idxmax() == "unit_2"
    assert krig.loc["unit_2", "mssr"] > 1.5


def test_truth_covariance_uses_the_realised_noise_not_the_configured_nugget(dataset):
    """The model input is a depth-average, so what is uncorrelated between CPTs
    is the nugget *plus* the averaging error.  Handing kriging config.nugget
    would make the reference model overconfident by up to 2.3x in variance."""
    est = truth_kriging_factory(dataset)("unit_1")
    configured = dataset.config.units["unit_1"].property.nugget
    assert est.covariance[1] > configured


def test_truth_covariance_refuses_to_run_on_real_data(dataset):
    """Silently falling back to a fit would answer a different question."""
    bare = type("Bare", (), {"config": None, "unit_summary": dataset.unit_summary})()
    with pytest.raises(ValueError, match="config"):
        truth_kriging_factory(bare)


def test_the_estimator_layer_no_longer_offers_a_truth_covariance(dataset):
    """``kriging_factory`` runs on any data, so it cannot serve a truth model.

    It points at the synthetic one rather than accepting the argument and
    failing later, or worse, quietly fitting instead.
    """
    with pytest.raises(ValueError, match="synthetic-only"):
        kriging_factory(dataset, covariance="truth")


def test_the_reference_and_the_fit_are_different_models(dataset):
    """If these ever coincided the comparison would be vacuous."""
    ref = truth_kriging_factory(dataset)("unit_1")
    block = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"]
    X, y = block[["x", "y"]].to_numpy(), block["log_Q_mean"].to_numpy()
    fitted = kriging_factory(dataset, covariance="fit")("unit_1").fit(X, y)
    ref = ref.fit(X, y)
    assert not np.isclose(ref.cov_.len_scale, fitted.cov_.len_scale, rtol=1e-3)
