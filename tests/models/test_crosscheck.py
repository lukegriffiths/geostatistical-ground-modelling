"""pykrige vs. gstools — two implementations of ordinary kriging.

The same library generating and fitting is an inverse crime; this is the one
place the kriging algebra is checked against code written by someone else.

The comparison is run **at zero nugget**, and that is the finding rather than a
convenience: the two libraries use opposite variance conventions, so with a
nugget they disagree *by construction*.  Asserting a loose tolerance there would
hide a documented convention gap behind a number that looks like agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.covariance import GrfConfig, build_model
from cpt_geostat.models.crosscheck import (
    PYKRIGE_RANGE_FACTOR,
    PyKrigeOrdinary,
    to_pykrige_params,
)
from cpt_geostat.models.kriging import OrdinaryKriging

pytest.importorskip("pykrige")


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, (40, 2))
    return X, rng.normal(3.0, 1.0, 40)


# --------------------------------------------------------------------------- #
# the fourth parameterisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", sorted(PYKRIGE_RANGE_FACTOR))
def test_the_range_bridge_reproduces_gstools_variograms_exactly(model):
    """pykrige's ``range`` is a *fourth* length-scale parameterisation.

    Checked against the library's own variogram function rather than against
    our algebra: if pykrige changes its definition, this fails instead of the
    cross-check quietly comparing two different models.
    """
    from pykrige import variogram_models as vm

    cov = build_model(GrfConfig(sill=0.8, range_km=4.0, model=model), nugget=0.15)
    psill, pk_range, nugget = to_pykrige_params(cov, model)
    fn = getattr(vm, f"{model}_variogram_model")

    d = np.linspace(0.05, 12.0, 200)
    assert np.allclose(fn([psill, pk_range, nugget], d), cov.variogram(d), atol=1e-8)


def test_the_pykrige_range_is_not_the_practical_range():
    """Why the bridge exists at all.

    For spherical the two differ by 23%.  Passing a practical range to pykrige
    as its ``range`` would produce a confident, wrong cross-check — which would
    then read as gstools being wrong.
    """
    cov = build_model(GrfConfig(sill=1.0, range_km=4.0, model="spherical"))
    _, pk_range, _ = to_pykrige_params(cov, "spherical")
    practical = 4.0
    assert abs(pk_range - practical) / practical > 0.2


def test_matern_is_rejected_rather_than_silently_substituted():
    """pykrige has no Matern.  Falling back to exponential would compare two
    different covariance models and call the difference a kriging discrepancy."""
    cov = build_model(GrfConfig(sill=1.0, range_km=3.0, model="matern25"))
    with pytest.raises(ValueError, match="matern25"):
        to_pykrige_params(cov, "matern25")


# --------------------------------------------------------------------------- #
# the cross-check itself
# --------------------------------------------------------------------------- #


def test_ordinary_kriging_agrees_with_pykrige_to_machine_precision(sample):
    """The headline cross-check: at zero nugget, two independent
    implementations of ordinary kriging agree to ~1e-12 in both mean and
    variance.  That validates the algebra in both."""
    X, y = sample
    cfg = GrfConfig(sill=0.8, range_km=4.5, model="exponential")

    gs_ok = OrdinaryKriging(covariance=(cfg, 0.0), model="exponential").fit(X, y)
    pk_ok = PyKrigeOrdinary(covariance=(cfg, 0.0), model="exponential").fit(X, y)

    grid = np.random.default_rng(9).uniform(-4, 4, (25, 2))
    m_gs, sd_gs = gs_ok.predict(grid, return_std=True)
    m_pk, sd_pk = pk_ok.predict(grid, return_std=True)

    assert np.allclose(m_gs, m_pk, atol=1e-10)
    assert np.allclose(sd_gs, sd_pk, atol=1e-10)


def test_the_two_libraries_disagree_once_there_is_a_nugget_and_why(sample):
    """The convention gap, asserted rather than tolerated.

    pykrige interpolates the conditioning data exactly and returns a *latent*
    variance — zero at the data points.  gstools with ``exact=False,
    cond_err="nugget"`` smooths them and returns an *observation* variance.
    Both are self-consistent; they are answering different questions, and a
    cross-check that averaged over the difference would be worthless.
    """
    X, y = sample
    cfg = GrfConfig(sill=0.8, range_km=4.5, model="exponential")

    gs_ok = OrdinaryKriging(covariance=(cfg, 0.2), model="exponential").fit(X, y)
    pk_ok = PyKrigeOrdinary(covariance=(cfg, 0.2), model="exponential").fit(X, y)

    _, sd_pk_at = pk_ok.predict(X[:5], return_std=True)
    _, sd_gs_at = gs_ok.predict(X[:5], return_std=True)

    assert np.allclose(sd_pk_at, 0.0, atol=1e-6)   # pykrige: data are exact
    assert np.all(sd_gs_at > 0.1)                  # gstools: data are noisy


def test_the_cross_check_refuses_anisotropy_rather_than_comparing_two_models(sample):
    X, y = sample
    cfg = GrfConfig(sill=0.8, range_km=4.5, aniso_ratio=3.0, model="exponential")
    with pytest.raises(ValueError, match="isotropic"):
        PyKrigeOrdinary(covariance=(cfg, 0.0), model="exponential").fit(X, y)
