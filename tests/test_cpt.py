"""The ``Qtn`` <-> ``qt`` transform, and the stress exponent it hinges on.

The transform itself is four lines of arithmetic; what makes it worth testing is
that every way of getting it wrong is silent.  A wrong unit weight is a few
percent, invisible against a prediction band.  A wrong stress exponent is a
depth-dependent factor of two or three, which does not look like an error — it
looks like a trend, and this project's whole business is trends.  So the tests
pin the exponent convention, the ``n = 1`` identity, and the monotonicity that
lets a band be transformed edge-by-edge.

One IJmuiden reading is hardcoded as a regression anchor.  It is the only claim
here about what the *supplied* ``Qt`` column means, and it is checked against
arithmetic rather than against ``data/cpt_dataframe.csv``, which no test may
require to be present.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.cpt import (
    METADATA_KEY,
    N_MAX,
    PA_KPA,
    DEFAULT_PROFILE,
    StressProfile,
    exponent_log_sd,
    exponent_uncertainty_from_metadata,
    infer_gamma_eff,
    log_stress_ratio,
    normalisation_from_metadata,
    normalisation_metadata,
    qt_from_log_qtn,
    qt_from_qtn,
    qt_moments,
    qt_quantile,
    qt_scale,
    qtn_from_qt,
    robertson_n,
    soil_behaviour_index,
    total_log_sd,
)

#: What the IJmuiden export's own columns imply, rounded to a stated soil.
IJMUIDEN = StressProfile.from_gradients(20.1, 10.0)


# --------------------------------------------------------------------------
# the stress profile
# --------------------------------------------------------------------------

def test_default_profile_is_two_tonnes_per_cubic_metre_in_seawater():
    """The project's stated assumption, in the units a soil report quotes."""
    assert DEFAULT_PROFILE.gamma_sat == pytest.approx(19.62, abs=0.01)
    assert DEFAULT_PROFILE.gamma_w == pytest.approx(10.06, abs=0.01)
    assert DEFAULT_PROFILE.gamma_eff == pytest.approx(9.56, abs=0.01)


def test_effective_stress_is_total_minus_hydrostatic():
    z = np.array([1.0, 10.0, 50.0])
    p = StressProfile(rho_soil=2.0, rho_water=1.0)
    assert p.sigma_v0_eff(z) == pytest.approx(p.sigma_v0(z) - p.u0(z))


def test_from_gradients_round_trips_the_unit_weights():
    """Unit weights in, the same unit weights out — the forensic entry point."""
    p = StressProfile.from_gradients(20.1, 10.0)
    assert p.gamma_sat == pytest.approx(20.1)
    assert p.gamma_w == pytest.approx(10.0)
    assert p.gamma_eff == pytest.approx(10.1)


# --------------------------------------------------------------------------
# the transform
# --------------------------------------------------------------------------

def test_the_inverse_round_trips_at_any_exponent():
    z = np.array([0.5, 2.0, 10.0, 40.0, 80.0])
    qt = np.array([500.0, 2_000.0, 15_000.0, 40_000.0, 60_000.0])
    for n in (0.5, 0.75, 1.0):
        back = qt_from_qtn(qtn_from_qt(qt, z, n=n), z, n=n)
        assert back == pytest.approx(qt, rel=1e-12)


def test_n_equals_one_is_robertsons_Qt_and_pa_drops_out():
    """``Qtn(n=1) == Qt``, which is why the two names get conflated.

    Checked against the ``Qt`` expression written out longhand, and against a
    deliberately absurd ``pa``: at ``n = 1`` the reference pressure cancels
    algebraically, so a wrong ``pa`` cannot change the answer.  Away from
    ``n = 1`` it can and does.
    """
    z, qt = 12.0, 20_000.0
    p = DEFAULT_PROFILE
    longhand = (qt - p.sigma_v0(z)) / p.sigma_v0_eff(z)
    assert qtn_from_qt(qt, z, n=1.0) == pytest.approx(longhand)
    assert qtn_from_qt(qt, z, n=1.0, pa=1.0) == pytest.approx(longhand)
    assert qtn_from_qt(qt, z, n=0.5, pa=1.0) != pytest.approx(qtn_from_qt(qt, z, n=0.5))


def test_a_wrong_exponent_is_a_depth_dependent_error_not_a_constant_one():
    """Why ``n`` cannot be shrugged off: the error grows down the hole.

    Inverting an ``n = 1`` column with ``n = 0.5`` overshoots ``qt`` near the
    seabed and undershoots it deep, by ``sqrt(pa / sigma_v0_eff)`` — a factor of
    3 at 1 m and 0.35 at 80 m.  A constant bias would show up in any comparison;
    this one masquerades as stratigraphy.
    """
    z = np.array([1.0, 80.0])
    qt = np.array([1_500.0, 60_000.0])
    Q = qtn_from_qt(qt, z, n=1.0)
    ratio = (qt_from_qtn(Q, z, n=0.5) - DEFAULT_PROFILE.sigma_v0(z)) / (
        qt - DEFAULT_PROFILE.sigma_v0(z)
    )
    assert ratio == pytest.approx(np.sqrt(PA_KPA / DEFAULT_PROFILE.sigma_v0_eff(z)))
    assert ratio[0] > 3.0 > 1.0 > ratio[1]


def test_constant_Qtn_through_a_unit_gives_a_linear_qt_gradient():
    """The content of the transform: the model gives the level, depth the slope.

    With ``n = 1`` a single per-unit prediction becomes a straight line in
    ``qt``, whose gradient is ``sigma_v0`` plus ``Qtn`` buoyant gradients.  This
    is what a reconstructed profile looks like before within-unit scatter.
    """
    z = np.linspace(5.0, 15.0, 11)
    qt = qt_from_qtn(np.full_like(z, 150.0), z, n=1.0)
    grad = np.gradient(qt, z)
    assert grad == pytest.approx(
        DEFAULT_PROFILE.gamma_sat + 150.0 * DEFAULT_PROFILE.gamma_eff, rel=1e-10
    )


def test_the_transform_is_monotone_so_band_edges_map_to_band_edges():
    """What licenses transforming a log-space interval edge-by-edge.

    ``exp`` then :func:`qt_from_qtn` is strictly increasing, so quantiles are
    equivariant: the transformed edges *are* the ``qt`` quantiles, exactly, with
    no delta method.  Order preservation is the testable form of that claim.
    """
    log_q = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    qt = qt_from_log_qtn(log_q, z=8.0)
    assert np.all(np.diff(qt) > 0)
    assert qt_from_log_qtn(2.0, z=8.0) == pytest.approx(qt_from_qtn(np.exp(2.0), z=8.0))


def test_the_median_transforms_but_the_mean_does_not():
    """``exp(mean of logs)`` is a median; adding ``sigma_v0`` keeps it a median.

    A lognormal *mean* is larger by ``exp(sd**2 / 2)``, so the two answers are
    not interchangeable and the profile figures quote the median.
    """
    pred, sd, z = 5.0, 0.6, 10.0
    median = qt_from_log_qtn(pred, z)
    mean = qt_from_qtn(np.exp(pred + sd**2 / 2.0), z)
    assert mean > median
    assert (mean - DEFAULT_PROFILE.sigma_v0(z)) / (median - DEFAULT_PROFILE.sigma_v0(z)) == (
        pytest.approx(np.exp(sd**2 / 2.0))
    )


def test_normalising_at_the_seabed_is_nan_not_infinity():
    """``sigma_v0_eff`` is zero at z = 0, so ``Qtn`` is undefined there.

    Real traces start at 0.01 m and sometimes carry a leading zero; nan keeps
    that reading out of an average instead of poisoning it with inf.
    """
    out = qtn_from_qt(np.array([100.0, 100.0, 100.0]), np.array([-1.0, 0.0, 1.0]))
    assert np.isnan(out[:2]).all()
    assert np.isfinite(out[2])


def test_de_normalising_at_the_seabed_is_zero_not_nan():
    """The asymmetry: the inverse *is* defined at z = 0, and equals zero.

    A unit that outcrops at the seabed has ``z_top = 0``, and its ``qt`` there is
    zero under a constant-gradient profile whatever ``Qtn`` is — both stresses
    vanish.  Masking it to nan would put a hole at the top of every outcropping
    unit for a value the model does state.  Above the seabed there is no soil,
    so that stays nan.
    """
    out = qt_from_qtn(np.array([300.0, 300.0, 300.0]), np.array([-1.0, 0.0, 1.0]))
    assert np.isnan(out[0])
    assert out[1] == 0.0
    assert out[2] > 0.0
    assert qt_moments(5.0, 0.5, 0.0)["qt_sd"] == 0.0


# --------------------------------------------------------------------------
# carrying the uncertainty across
# --------------------------------------------------------------------------

def test_qt_is_affine_in_qtn_which_is_what_makes_the_moments_exact():
    """The structural fact the whole uncertainty transform rests on.

    ``qt = sigma_v0 + c(z) * Qtn``.  The stress exponent lives in ``c``, not on
    ``Qtn``, so this holds for every ``n`` — the moments below are not an
    ``n = 1`` special case.
    """
    z = 12.0
    for n in (0.5, 1.0):
        c = qt_scale(z, n=n, profile=IJMUIDEN)
        sv = IJMUIDEN.sigma_v0(z)
        q = np.array([50.0, 100.0, 200.0])
        assert qt_from_qtn(q, z, n=n, profile=IJMUIDEN) == pytest.approx(sv + c * q)


def test_moments_match_a_simulation():
    """Closed form against brute force — the check that catches a dropped term."""
    rng = np.random.default_rng(7)
    pred, sd, z = 5.0, 0.55, 15.0
    draws = qt_from_qtn(rng.lognormal(pred, sd, 4_000_000), z, profile=IJMUIDEN)
    m = qt_moments(pred, sd, z, profile=IJMUIDEN)

    assert m["qt_mean"] == pytest.approx(draws.mean(), rel=2e-3)
    assert m["qt_sd"] == pytest.approx(draws.std(), rel=1e-2)
    assert m["qt_median"] == pytest.approx(np.median(draws), rel=2e-3)
    assert m["qt_cv"] == pytest.approx(draws.std() / draws.mean(), rel=1e-2)


def test_the_shift_moves_the_mean_and_leaves_the_spread_alone():
    """``sigma_v0`` is a constant at fixed depth, so it cannot add scatter.

    Stated as a test because it is the step where an sd most easily acquires a
    spurious ``sigma_v0`` term.
    """
    pred, sd, z = 5.0, 0.5, 20.0
    m = qt_moments(pred, sd, z, profile=IJMUIDEN)
    net = qt_scale(z, profile=IJMUIDEN) * np.exp(pred + sd**2 / 2) * np.sqrt(np.expm1(sd**2))
    assert m["qt_sd"] == pytest.approx(net)
    assert m["qt_mean"] - IJMUIDEN.sigma_v0(z) == pytest.approx(
        qt_scale(z, profile=IJMUIDEN) * np.exp(pred + sd**2 / 2)
    )


def test_the_qt_band_is_asymmetric_about_the_median():
    """Why ``qt_median +/- qt_sd`` is not the interval, quantified.

    At an IJmuiden-scale ``sd`` the upper arm of the 95% band is several times
    the lower one, and the symmetric approximation runs below the true 2.5%
    quantile — it would understate a *low* ``qt``, which for a design is the
    wrong direction to be wrong in.
    """
    pred, sd, z = 5.0, 0.6, 15.0
    m = qt_moments(pred, sd, z, profile=IJMUIDEN)
    lo = qt_quantile(pred, sd, z, q=0.025, profile=IJMUIDEN)
    hi = qt_quantile(pred, sd, z, q=0.975, profile=IJMUIDEN)

    assert (hi - m["qt_median"]) / (m["qt_median"] - lo) > 2.0
    assert m["qt_median"] - 1.96 * m["qt_sd"] < lo


def test_the_net_resistance_cv_does_not_depend_on_depth():
    """The stable way to quote the uncertainty in ``qt`` terms.

    ``qt_sd`` grows down a unit even at constant log-space sd, because ``c(z)``
    does.  The cv of the net resistance is ``sqrt(exp(sd**2) - 1)`` at every
    depth; the cv of ``qt`` itself is slightly smaller and drifts, because
    ``sigma_v0`` dilutes it.
    """
    pred, sd = 5.0, 0.5
    z = np.array([2.0, 20.0, 60.0])
    m = qt_moments(pred, sd, z, profile=IJMUIDEN)
    net_cv = m["qt_sd"] / (m["qt_mean"] - IJMUIDEN.sigma_v0(z))
    assert net_cv == pytest.approx(np.sqrt(np.expm1(sd**2)))
    assert np.all(np.diff(m["qt_sd"]) > 0)
    assert np.all(m["qt_cv"] < net_cv)


# --------------------------------------------------------------------------
# metadata round trip
# --------------------------------------------------------------------------

def test_the_normalisation_survives_a_round_trip_through_json():
    """``qt`` is dropped at preparation time; this block is what remains."""
    import json

    meta = {METADATA_KEY: normalisation_metadata(IJMUIDEN, 1.0, basis="fitted")}
    profile, n = normalisation_from_metadata(json.loads(json.dumps(meta)))
    assert n == 1.0
    assert profile.gamma_sat == pytest.approx(IJMUIDEN.gamma_sat)
    assert profile.gamma_eff == pytest.approx(IJMUIDEN.gamma_eff)


def test_missing_metadata_raises_rather_than_defaulting():
    """A silent default here is a guessed exponent, i.e. a fabricated trend."""
    with pytest.raises(KeyError, match=METADATA_KEY):
        normalisation_from_metadata({"property": {"column": "Qtn"}})


# --------------------------------------------------------------------------
# recovering what a supplied column assumed
# --------------------------------------------------------------------------

def test_infer_gamma_eff_recovers_a_planted_gradient():
    rng = np.random.default_rng(0)
    z = rng.uniform(0.5, 60.0, 4_000)
    truth = StressProfile.from_gradients(20.1, 10.0)
    qt = qt_from_qtn(rng.lognormal(5.0, 0.6, z.size), z, n=1.0, profile=truth)

    fit = infer_gamma_eff(qt, z, qtn_from_qt(qt, z, n=1.0, profile=truth), n=1.0,
                          gamma_sat=truth.gamma_sat)
    assert fit["gamma_eff"] == pytest.approx(10.1, rel=1e-6)
    assert fit["ratio_median"] == pytest.approx(1.0, rel=1e-6)


def test_infer_gamma_eff_cannot_rescue_the_wrong_exponent():
    """The diagnostic that identified the IJmuiden column as ``n = 1``.

    Fitting a gradient under ``n = 0.5`` to an ``n = 1`` column leaves a residual
    ratio that sweeps through 1.0 with depth rather than sitting on it — the
    signature that no unit weight will fix, because the exponent is wrong.
    """
    rng = np.random.default_rng(1)
    z = rng.uniform(0.5, 60.0, 4_000)
    truth = StressProfile.from_gradients(20.1, 10.0)
    qt = qt_from_qtn(rng.lognormal(5.0, 0.3, z.size), z, n=1.0, profile=truth)
    Q = qtn_from_qt(qt, z, n=1.0, profile=truth)

    good = infer_gamma_eff(qt, z, Q, n=1.0, gamma_sat=truth.gamma_sat)
    bad = infer_gamma_eff(qt, z, Q, n=0.5, gamma_sat=truth.gamma_sat)
    assert good["ratio_p95"] - good["ratio_p05"] < 0.01
    assert bad["ratio_p95"] - bad["ratio_p05"] > 0.5


def test_infer_gamma_eff_rejects_data_it_cannot_fit():
    with pytest.raises(ValueError, match="no finite readings"):
        infer_gamma_eff([1.0, 2.0], [0.0, -1.0], [1.0, 1.0])


# --------------------------------------------------------------------------
# the supplied IJmuiden column
# --------------------------------------------------------------------------

def test_one_ijmuiden_reading_reproduces_under_n_equals_one():
    """Regression anchor for what ``cpt_samples.Qtn`` actually holds.

    IJ56-CPT-04 at 6.29 m below seabed: ``qt = 44.70735`` MPa, and the export
    gives ``Qt = 700.8464``.  Under ``n = 1`` with a buoyant gradient of
    10.1 kPa/m that reproduces to 0.1%; under ``n = 0.5`` it is out by 20%,
    which is the whole reason this file exists.

    The 10.1 kPa/m is the export's own gradient (:func:`infer_gamma_eff` over all
    532k readings gives 10.07), not this project's 2.0 t/m^3 assumption, which is
    5% lower — see the module docstring.
    """
    z, qt_kpa, supplied = 6.29, 44_707.35, 700.8464
    profile = StressProfile.from_gradients(20.1, 10.0)

    assert qtn_from_qt(qt_kpa, z, n=1.0, profile=profile) == pytest.approx(supplied, rel=2e-3)
    assert qtn_from_qt(qt_kpa, z, n=0.5, profile=profile) != pytest.approx(supplied, rel=0.1)
    assert qt_from_qtn(supplied, z, n=1.0, profile=profile) == pytest.approx(qt_kpa, rel=2e-3)


def test_the_project_density_assumption_costs_five_percent_on_qt():
    """Quantified rather than asserted, because it is the residual disagreement.

    2.0 t/m^3 in seawater gives a buoyant gradient of 9.56 against the export's
    10.07, so a ``qt`` rebuilt from a prediction lands about 5% low.  That is an
    order of magnitude inside the model's own band (``log_Q_sd`` 0.31-0.71 on
    IJmuiden, i.e. tens of percent), so it is a documented offset, not a defect —
    but it must stay a *known* 5%.
    """
    z = np.array([2.0, 10.0, 40.0])
    supplied = np.array([400.0, 150.0, 80.0])
    export = StressProfile.from_gradients(20.1, 10.0)
    ratio = qt_from_qtn(supplied, z, profile=DEFAULT_PROFILE) / qt_from_qtn(
        supplied, z, profile=export
    )
    assert np.all((ratio > 0.93) & (ratio < 0.96))


# --------------------------------------------------------------------------
# a variable exponent
# --------------------------------------------------------------------------

def test_the_exponent_acts_only_through_the_log_stress_ratio():
    """``L = log(sigma'v0 / pa)`` is the whole lever, and it is zero at pa.

    The pivot is a real depth — 9.9 m on IJmuiden's gradient — and above and
    below it two exponents disagree in opposite directions.  Everything else
    about a variable ``n`` follows from this one quantity.
    """
    z_pivot = PA_KPA / IJMUIDEN.gamma_eff
    assert log_stress_ratio(z_pivot, IJMUIDEN) == pytest.approx(0.0, abs=1e-12)
    assert qt_scale(z_pivot, n=0.5, profile=IJMUIDEN) == pytest.approx(
        qt_scale(z_pivot, n=1.0, profile=IJMUIDEN)
    )
    # ...and away from it, the exponent matters in opposite directions
    shallow, deep = 0.5, 60.0
    assert qt_scale(shallow, n=0.5, profile=IJMUIDEN) > qt_scale(shallow, n=1.0, profile=IJMUIDEN)
    assert qt_scale(deep, n=0.5, profile=IJMUIDEN) < qt_scale(deep, n=1.0, profile=IJMUIDEN)


def test_exponent_uncertainty_vanishes_at_the_pivot_and_grows_away_from_it():
    z = np.array([0.5, 2.0, PA_KPA / IJMUIDEN.gamma_eff, 20.0, 60.0])
    extra = exponent_log_sd(z, 0.1, IJMUIDEN)
    assert extra[2] == pytest.approx(0.0, abs=1e-12)
    assert extra[0] > extra[1] > extra[2] < extra[3] < extra[4]
    # Shallow ground is where an exponent argument actually costs something:
    # 0.30 at half a metre, against a between-hole log_Q_sd of 0.31-0.71.
    assert extra[0] == pytest.approx(0.30, abs=0.01)


def test_a_certain_exponent_changes_nothing():
    """``n_sd = 0`` must reproduce the earlier behaviour exactly, not nearly."""
    pred, sd, z = 5.0, 0.4, np.array([1.0, 10.0, 40.0])
    assert total_log_sd(sd, z, n_sd=0.0, profile=IJMUIDEN) == pytest.approx(sd)
    a = qt_moments(pred, sd, z, n=0.7, profile=IJMUIDEN)
    b = qt_moments(pred, sd, z, n=0.7, n_sd=0.0, profile=IJMUIDEN)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-15)


def test_uncertain_exponent_matches_a_simulation():
    """The closed form is a claim about a product of two lognormals — checked.

    ``qt - sigma_v0 = pa * exp(n L) * Qtn`` with ``n`` Gaussian and ``Qtn``
    lognormal is itself lognormal, so median, mean, sd and quantiles all stay
    exact.  Simulated at a depth well off the pivot, where the extra term is
    doing real work.
    """
    rng = np.random.default_rng(11)
    pred, sd, n, n_sd, z = 5.0, 0.45, 0.7, 0.12, 60.0
    draws = qt_from_qtn(rng.lognormal(pred, sd, 2_000_000), z,
                        n=rng.normal(n, n_sd, 2_000_000), profile=IJMUIDEN)
    m = qt_moments(pred, sd, z, n=n, n_sd=n_sd, profile=IJMUIDEN)
    assert m["qt_median"] == pytest.approx(np.median(draws), rel=3e-3)
    assert m["qt_mean"] == pytest.approx(draws.mean(), rel=3e-3)
    assert m["qt_sd"] == pytest.approx(draws.std(), rel=1e-2)
    for q in (0.025, 0.975):
        got = qt_quantile(pred, sd, z, q=q, n=n, n_sd=n_sd, profile=IJMUIDEN)
        assert got == pytest.approx(np.percentile(draws, 100 * q), rel=5e-3)


def test_exponent_uncertainty_widens_the_band_without_moving_the_median():
    """It is doubt about the transform, not a different answer.

    The exponent enters as a lognormal factor whose median is one, so the median
    ``qt`` is untouched and only the spread grows.
    """
    pred, sd, z = 5.0, 0.4, 40.0
    kw = dict(n=0.7, profile=IJMUIDEN)
    assert qt_quantile(pred, sd, z, q=0.5, n_sd=0.25, **kw) == pytest.approx(
        qt_quantile(pred, sd, z, q=0.5, n_sd=0.0, **kw)
    )
    wide = qt_quantile(pred, sd, z, q=0.975, n_sd=0.25, **kw)
    narrow = qt_quantile(pred, sd, z, q=0.975, n_sd=0.0, **kw)
    assert wide > narrow
    assert qt_moments(pred, sd, z, n_sd=0.25, **kw)["qt_sd"] > \
        qt_moments(pred, sd, z, n_sd=0.0, **kw)["qt_sd"]


def test_the_seabed_has_no_exponent_uncertainty_either():
    """qt is deterministically zero there, so there is no spread to inflate."""
    assert exponent_log_sd(0.0, 0.3, IJMUIDEN) == 0.0
    assert qt_moments(5.0, 0.5, 0.0, n_sd=0.3, profile=IJMUIDEN)["qt_sd"] == 0.0
    assert np.isnan(exponent_log_sd(-1.0, 0.3, IJMUIDEN))


# --------------------------------------------------------------------------
# Robertson's exponent
# --------------------------------------------------------------------------

def test_robertson_n_puts_sand_near_a_half_and_clay_at_the_cap():
    """The physical content: sqrt(stress) for granular, proportional for cohesive."""
    sve = 100.0  # at pa, so the stress term contributes its 0.05
    assert robertson_n(1.6, sve) == pytest.approx(0.381 * 1.6 + 0.05 - 0.15)
    assert robertson_n(1.6, sve) < 0.6          # clean sand
    assert robertson_n(3.0, sve) == N_MAX       # clay, capped
    assert np.all(robertson_n(np.array([2.8, 3.5, 4.0]), sve) <= N_MAX)


def test_the_ic_solve_reaches_a_fixed_point():
    """The three unknowns are consistent at the answer, not merely stopped at.

    Recomputing Ic and n from the returned Qtn must return the same n — the
    definition of the fixed point the iteration is chasing.
    """
    rng = np.random.default_rng(3)
    z = rng.uniform(1.0, 60.0, 3_000)
    qt = rng.lognormal(9.0, 1.0, z.size)      # kPa, spanning sand to clay
    fs = qt * rng.uniform(0.004, 0.06, z.size)

    out = soil_behaviour_index(qt, fs, z, profile=IJMUIDEN)
    assert out["converged"] and out["n_unconverged"] == 0
    ok = np.isfinite(out["n"])
    again = robertson_n(out["ic"][ok], IJMUIDEN.sigma_v0_eff(z[ok]))
    assert np.abs(again - out["n"][ok]).max() < 0.01


def test_the_ic_solve_separates_sand_from_clay():
    """A low friction ratio must come back with a lower exponent than a high one."""
    z = np.full(2, 20.0)
    qt = np.array([20_000.0, 1_500.0])         # dense sand, soft clay
    fs = np.array([80.0, 60.0])                # Fr ~0.4% vs ~4%
    out = soil_behaviour_index(qt, fs, z, profile=IJMUIDEN)
    assert out["ic"][0] < out["ic"][1]
    assert out["n"][0] < out["n"][1]
    assert out["n"][0] < 0.75 < out["n"][1]


def test_readings_with_no_soil_type_come_back_nan():
    """qt below the overburden, or a zero sleeve, is not a soil — it is a gap."""
    z = np.array([10.0, 10.0, 10.0])
    qt = np.array([100.0, 5_000.0, 5_000.0])   # first is below sigma_v0 at 10 m
    fs = np.array([20.0, 0.0, 50.0])
    out = soil_behaviour_index(qt, fs, z, profile=IJMUIDEN)
    assert np.isnan(out["n"][:2]).all()
    assert np.isfinite(out["n"][2])
    assert out["n_usable"] == 1


def test_metadata_carries_a_per_unit_exponent_and_its_spread():
    import json

    n = {"sand": 0.52, "clay": 0.98}
    meta = {METADATA_KEY: normalisation_metadata(IJMUIDEN, n, basis="Ic", n_sd={"sand": 0.08,
                                                                                "clay": 0.11})}
    meta = json.loads(json.dumps(meta))
    profile, got = normalisation_from_metadata(meta)
    assert got == n
    assert profile.gamma_eff == pytest.approx(IJMUIDEN.gamma_eff)
    assert exponent_uncertainty_from_metadata(meta) == {"sand": 0.08, "clay": 0.11}
    # absent n_sd means "treated as known", not "unknown, pick something"
    bare = {METADATA_KEY: normalisation_metadata(IJMUIDEN, 1.0)}
    assert "n_sd" not in bare[METADATA_KEY]
    assert exponent_uncertainty_from_metadata(bare) == 0.0


def test_the_exponent_decides_whether_a_unit_reads_straight_or_curved():
    """The shape of a reconstructed profile *is* the exponent, made visible.

    ``n = 1`` gives ``qt`` proportional to depth — a straight line through the
    origin.  Anything below 1 bends it: ``qt - sigma_v0`` goes as ``z**n``, so
    the median comes out concave, steep at the seabed and flattening.  This is
    why a curved profile in a suite generated with ``n = 1`` means something has
    gone wrong, and why a straight one under Robertson's sand exponent would.
    """
    z = np.linspace(0.05, 4.0, 60)
    net = lambda n: qt_from_qtn(np.full_like(z, 400.0), z, n=n, profile=IJMUIDEN) \
        - IJMUIDEN.sigma_v0(z)  # noqa: E731

    straight = np.gradient(np.gradient(net(1.0), z), z)
    curved = np.gradient(np.gradient(net(0.48), z), z)
    assert np.abs(straight).max() < 1e-6 * np.abs(net(1.0)).max()
    assert (curved < 0).all()          # concave everywhere
