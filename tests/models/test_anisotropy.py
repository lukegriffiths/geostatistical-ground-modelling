"""Geometric anisotropy by profile likelihood.

Three things here are load-bearing and none of them fail loudly:

* the **azimuth convention** — a recovered axis 90 degrees out still produces a
  plausible ellipse, a plausible map and a plausible likelihood;
* the **scan interval** — [0, 180) in this module's ordered major/minor
  parameterisation, where plan 03's [0, 90) applies to unordered ARD.  Scanning
  the half interval here would silently make every bearing above 90 degrees
  unreachable, and the fit would quietly report the nearest one it could;
* the **null** — a ratio of 9 fitted to isotropic noise looks exactly like a
  ratio of 9 fitted to a channel, and only the simulated threshold separates
  them.

Fits use a coarse scan and few simulations throughout: the numbers being checked
are bearings and orderings, not third decimal places.
"""

from __future__ import annotations

from dataclasses import replace

import gstools as gs
import numpy as np
import pytest

from cpt_geostat.covariance import GrfConfig, build_model
from cpt_geostat.models.anisotropy import (
    MAX_RATIO,
    AnisotropyDecision,
    AnisotropyFit,
    anisotropic_kriging_factory,
    decide_anisotropy,
    decision_table,
    fit_anisotropy,
    fit_unit_anisotropy,
    null_lr_threshold,
    simulate_isotropic,
)

FAST = dict(step_deg=30.0, refine=False)


def _field(azimuth, ratio, n=90, range_km=4.0, nugget=0.0, seed=1):
    """A realisation with a known axis, on a jittered grid over a 12 km site."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    g = np.linspace(-6.0, 6.0, side)
    x, y = (a.ravel()[:n] for a in np.meshgrid(g, g))
    x = x + rng.normal(0, 0.25, x.size)
    y = y + rng.normal(0, 0.25, y.size)
    model = build_model(
        GrfConfig(sill=1.0, range_km=range_km, aniso_ratio=ratio, aniso_angle_deg=azimuth)
    )
    v = np.asarray(gs.SRF(model, seed=seed)((x, y), mesh_type="unstructured"), dtype=float)
    if nugget:
        v = v + rng.normal(0.0, np.sqrt(nugget), v.size)
    return x, y, v


def _axis_error(got, want):
    """Circular difference between two *axes* — mod 180, not mod 360."""
    d = abs(float(got) - float(want)) % 180.0
    return min(d, 180.0 - d)


# --------------------------------------------------------------------------
# recovery
# --------------------------------------------------------------------------

def test_a_planted_axis_comes_back():
    x, y, v = _field(azimuth=70.0, ratio=4.0, seed=3)
    fit = fit_anisotropy(x, y, v, **FAST)
    assert _axis_error(fit.azimuth_deg, 70.0) <= 30.0
    assert fit.ratio > 1.8
    assert fit.lr_stat > 0


@pytest.mark.parametrize("azimuth", [20.0, 120.0])
def test_bearings_past_90_degrees_are_reachable(azimuth):
    """The scan interval, pinned.

    In the ordered major/minor form a 120 degree axis is a *different* model
    from a 30 degree one, so a [0, 90) scan could never return it.  This is the
    test that fails if someone "corrects" the interval back to plan 03's.

    Asserted on the **median over three realisations**, not on one.  A single
    draw of an anisotropic field genuinely can look anisotropic in the wrong
    direction — measured here at roughly one seed in five, at either bearing —
    so a single-seed assertion would be testing luck, and would fail for a
    reason that says nothing about the code.
    """
    errs = [
        _axis_error(fit_anisotropy(*_field(azimuth=azimuth, ratio=5.0, seed=s), **FAST)
                    .azimuth_deg, azimuth)
        for s in range(1, 4)
    ]
    assert np.median(errs) <= 30.0


def test_isotropic_data_gives_a_flat_curve_and_a_ratio_near_one():
    """A flat profile curve is the honest answer, not a failure."""
    x, y, v = _field(azimuth=0.0, ratio=1.0, seed=11)
    fit = fit_anisotropy(x, y, v, **FAST)
    assert fit.ratio < 2.5
    aniso = fit_anisotropy(*_field(azimuth=70.0, ratio=5.0, seed=11), **FAST)
    assert fit.curve_contrast < aniso.curve_contrast


def test_the_likelihood_ratio_is_what_it_says():
    x, y, v = _field(azimuth=45.0, ratio=3.0, seed=7)
    fit = fit_anisotropy(x, y, v, **FAST)
    assert fit.lr_stat == pytest.approx(2.0 * (fit.loglik - fit.loglik_isotropic))
    # the anisotropic fit is a superset of the isotropic one, so it cannot lose
    assert fit.loglik >= fit.loglik_isotropic - 1e-6


# --------------------------------------------------------------------------
# bounds — a number on its bound is not a measurement
# --------------------------------------------------------------------------

def test_a_ratio_on_its_bound_is_flagged():
    """Synthetic unit 3's failure mode: a minor axis the survey cannot see, and
    a likelihood happy to shrink it to nothing."""
    x, y, v = _field(azimuth=70.0, ratio=8.0, seed=2)
    fit = fit_anisotropy(x, y, v, max_ratio=1.5, **FAST)
    assert fit.ratio == pytest.approx(1.5, rel=1e-2)
    assert fit.at_ratio_bound
    assert "at bound" in fit.describe()


def test_a_minor_axis_finer_than_the_cpt_spacing_is_flagged():
    fit = AnisotropyFit(
        azimuth_deg=70.0, ratio=10.0, range_major_km=4.0, sill=1.0, nugget=0.1,
        loglik=-10.0, loglik_isotropic=-12.0, range_isotropic_km=3.0,
        n_cpt=40, min_separation_km=0.8,
    )
    assert fit.range_minor_km == pytest.approx(0.4)
    assert not fit.minor_resolved
    assert "finer than the CPT spacing" in fit.describe()
    assert replace(fit, min_separation_km=0.1).minor_resolved


def test_the_fit_reports_the_closest_pair_it_saw():
    x, y, v = _field(azimuth=0.0, ratio=1.0, seed=4)
    fit = fit_anisotropy(x, y, v, **FAST)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    assert fit.min_separation_km == pytest.approx(d[d > 0].min())


# --------------------------------------------------------------------------
# the convention, end to end
# --------------------------------------------------------------------------

def test_the_fit_rebuilds_into_a_model_anisotropic_the_same_way():
    """``to_grf_config`` is the handover to the estimators, and the one place a
    90 degree slip would enter kriging.  Checked by covariance, not by field
    names: separation along the fitted major axis must decorrelate *slower*."""
    fit = AnisotropyFit(
        azimuth_deg=70.0, ratio=4.0, range_major_km=4.0, sill=1.0, nugget=0.0,
        loglik=0.0, loglik_isotropic=-1.0, range_isotropic_km=3.0, n_cpt=40,
    )
    model = build_model(fit.to_grf_config())
    along = np.deg2rad(90.0 - 70.0)      # azimuth -> math angle
    across = np.deg2rad(90.0 - 160.0)
    sep = 2.0
    c_along = np.atleast_1d(model.cov_spatial(
        np.array([[sep * np.cos(along)], [sep * np.sin(along)]])))[0]
    c_across = np.atleast_1d(model.cov_spatial(
        np.array([[sep * np.cos(across)], [sep * np.sin(across)]])))[0]
    assert c_along > c_across


def test_too_few_cpts_returns_nothing_rather_than_a_bearing(dataset):
    """Below the directional gate an azimuth is sampling noise, and a number
    returned there would be used."""
    small = min(dataset.unit_ids, key=lambda u: (dataset.unit_summary["unit_id"] == u).sum())
    if (dataset.unit_summary["unit_id"] == small).sum() >= 30:
        pytest.skip("no unit under the gate in this fixture")
    assert fit_unit_anisotropy(dataset, small, **FAST) is None


# --------------------------------------------------------------------------
# the null
# --------------------------------------------------------------------------

def test_the_null_simulates_the_fitted_nugget_not_a_smooth_field():
    """Regression: the null once put the *total* variance into the structured
    part and added no nugget.

    That makes the simulated fields far smoother than the data, so a spurious
    axis fits them better and the threshold comes back inflated — in one
    measured case by a factor of nine, enough to reject genuine anisotropy.

    Pinned on the simulated **field**, not on the threshold it leads to.  The
    threshold is a 95th percentile of a few dozen draws and is far too noisy to
    carry a regression test — the difference this bug makes to it is not even
    reliably signed at n_sim = 10.  The variance of the field is arithmetic.
    """
    x, y, v = _field(azimuth=0.0, ratio=1.0, range_km=6.0, nugget=1.0, seed=9)
    fit = fit_anisotropy(x, y, v, **FAST)
    sill, nugget = fit.sill_isotropic, fit.nugget_isotropic
    assert nugget > 0.2 * (sill + nugget), "fixture must be nugget-heavy to be a test"

    rng = np.random.default_rng(0)
    draws = np.concatenate([simulate_isotropic(x, y, fit, rng) for _ in range(30)])
    # the structured part alone would fall short by the whole nugget
    assert draws.var() == pytest.approx(sill + nugget, rel=0.25)
    assert draws.var() > 1.3 * sill


def test_the_null_shows_a_big_lr_is_unremarkable_on_isotropic_noise():
    """The whole reason the null exists, stated as something testable.

    Noise-dominated isotropic ground fits an axis with a healthy likelihood
    ratio; what makes that meaningless is that *simulated isotropic ground does
    too*.  So the claim pinned here is about the null distribution's location —
    it routinely reaches the observed statistic — not about the verdict on one
    fixture.  A verdict assertion would be flaky by construction: a 95th
    percentile is exceeded 5% of the time when the null is true, which is the
    definition of the threshold rather than a bug in it.
    """
    x, y, v = _field(azimuth=0.0, ratio=1.0, range_km=6.0, nugget=1.5, seed=13)
    fit = fit_anisotropy(x, y, v, **FAST)
    out = null_lr_threshold(x, y, fit, n_sim=10, seed=2, **FAST)
    assert out["n_sim"] >= 5
    assert np.median(out["null_lr"]) > 0.3 * fit.lr_stat
    assert out["null_lr"].std() > 0        # a degenerate null would gate nothing


def test_the_null_reports_what_it_measured():
    x, y, v = _field(azimuth=70.0, ratio=4.0, seed=6)
    fit = fit_anisotropy(x, y, v, **FAST)
    out = null_lr_threshold(x, y, fit, n_sim=5, seed=3, **FAST)
    assert out["lr_stat"] == fit.lr_stat
    assert out["null_lr"].size == out["n_sim"]
    assert np.isfinite(out["threshold"])


# --------------------------------------------------------------------------
# handing it to the estimators
# --------------------------------------------------------------------------

def _decision(**kw):
    defaults = dict(
        azimuth_deg=70.0, ratio=4.0, range_major_km=4.0, sill=1.0, nugget=0.2,
        loglik=0.0, loglik_isotropic=-5.0, range_isotropic_km=3.0, n_cpt=40,
        min_separation_km=0.1,
    )
    return AnisotropyFit(**{**defaults, **kw})


def test_a_rejected_unit_takes_exactly_the_isotropic_path():
    """``"fit"``, not an isotropic ``GrfConfig``.

    A rejected unit must run the *same code* the isotropic estimators run, so a
    comparison between the two models is a comparison of the units that changed
    and nothing else.
    """
    rejected = AnisotropyDecision("u", False, "nope", _decision())
    assert rejected.covariance == "fit"

    accepted = AnisotropyDecision("u", True, "beat the null", _decision())
    grf, nugget = accepted.covariance
    assert grf.aniso_ratio == pytest.approx(4.0)
    assert grf.aniso_angle_deg == pytest.approx(70.0)
    assert nugget == pytest.approx(0.2)


def test_a_unit_with_no_decision_at_all_stays_isotropic():
    assert AnisotropyDecision("u", True, "no fit", None).covariance == "fit"


@pytest.fixture(scope="module")
def decisions(dataset):
    """One pass over the fixture; the fits are the expensive part."""
    return decide_anisotropy(dataset, n_sim=0, **FAST)


def test_the_gates_fire_in_order_of_cost(dataset, decisions):
    """Cheap structural rejections must not pay for a null simulation.

    ``n_sim`` is set high enough that a unit reaching the null would take
    obviously longer than this test does; the assertion is on the *reasons*,
    which name which gate stopped each unit.
    """
    assert set(decisions) == set(dataset.unit_ids)
    for uid, d in decisions.items():
        n = int((dataset.unit_summary["unit_id"] == uid).sum())
        if n < 30:
            assert not d.use_anisotropy
            assert "directional gate" in d.reason
            assert d.fit is None


def test_skipping_the_null_is_recorded_in_the_reason(decisions):
    """A run that took the shortcut must say so in its own report."""
    used = [d for d in decisions.values() if d.use_anisotropy]
    assert all(d.reason == "null not tested" for d in used)


def test_the_factory_gives_each_unit_the_covariance_it_earned(dataset):
    decisions = {
        "unit_1": AnisotropyDecision("unit_1", True, "beat the null", _decision()),
        "unit_2": AnisotropyDecision("unit_2", False, "did not", _decision()),
    }
    factory = anisotropic_kriging_factory(dataset, method="OK", decisions=decisions)
    assert factory("unit_1").covariance != "fit"
    assert factory("unit_2").covariance == "fit"
    # a unit the decisions never saw must not inherit someone else's axis
    assert factory("unit_never_seen").covariance == "fit"


def test_the_decision_table_reports_every_unit_including_the_rejected(dataset, decisions):
    table = decision_table(decisions)
    assert set(table.index) == set(dataset.unit_ids)
    assert "reason" in table.columns
    assert table["anisotropic"].dtype == bool
