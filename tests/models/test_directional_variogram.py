"""Directional variogram estimation — the bearing convention above all.

The azimuth bridge is this package's flagged "single most likely source of
silent bugs", and a directional variogram is where it bites hardest: a sign
slip or a from-east mix-up rotates every sector by 90 degrees and still
produces a figure that looks entirely plausible.  So the convention is checked
against a field with a *known* major axis, not against our own algebra.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.covariance import GrfConfig
from cpt_geostat.models.variogram import (
    DEFAULT_SECTORS,
    DIRECTIONAL_N_BINS,
    MIN_CPT_FOR_FIT,
    directional_variogram,
    fit_unit_variogram,
    fit_variogram,
    unit_block,
)
from cpt_geostat.synthetic.fields import Raster, grf_raster


def _anisotropic_sample(azimuth_deg, n=3000, seed=7):
    """Scattered samples from a ratio-6 field whose major axis is known."""
    raster = Raster.from_site(20.0, 0.1)
    cfg = GrfConfig(sill=1.0, range_km=6.0, aniso_ratio=6.0,
                    aniso_angle_deg=azimuth_deg, model="gaussian")
    field = grf_raster(raster, cfg, np.random.default_rng(seed))
    XX, YY = raster.meshgrid()
    idx = np.random.default_rng(1).choice(field.size, n, replace=False)
    return XX.ravel()[idx], YY.ravel()[idx], field.ravel()[idx]


# --------------------------------------------------------------------------- #
# the convention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 70.0, 135.0])
def test_the_on_axis_sector_decorrelates_slowest(azimuth):
    """A field is smoothest *along* its major axis, so that sector has the
    lowest gamma.  This is the check that catches a 90-degree inversion.

    Compared against the perpendicular sector rather than against an absolute
    number: the contrast is the signal, and it survives the field realisation
    being what it happens to be.
    """
    x, y, v = _anisotropic_sample(azimuth)
    edges = np.linspace(0.0, 6.0, 13)
    got = directional_variogram(
        x, y, v, azimuths=(azimuth, azimuth + 90.0), tol_deg=15.0,
        bandwidth_km=0.5, edges=edges,
    )
    along = got[float(azimuth)][1]
    across = got[float(azimuth + 90.0)][1]
    # Well inside the major range and well outside the minor one.
    assert along[3] < 0.6 * across[3], f"az={azimuth}: {along[3]:.3f} vs {across[3]:.3f}"


def test_a_direction_is_an_axis_not_a_bearing():
    """Pairs are unordered, so 70 and 250 degrees select the same pairs.

    If these ever disagreed, the sectors would be double-counting half the
    compass and the figure would be showing two views of the same data as
    though they were independent.
    """
    x, y, v = _anisotropic_sample(70.0, n=1200)
    edges = np.linspace(0.0, 6.0, 8)
    got = directional_variogram(x, y, v, azimuths=(70.0, 250.0), edges=edges)
    lag_a, gam_a, cnt_a = got[70.0]
    lag_b, gam_b, cnt_b = got[250.0]
    assert np.allclose(gam_a, gam_b)
    assert np.array_equal(cnt_a, cnt_b)


def test_an_isotropic_field_shows_no_preferred_sector():
    """The control.  Without it, a test suite cannot tell "found the axis" from
    "reports an axis regardless"."""
    raster = Raster.from_site(20.0, 0.1)
    cfg = GrfConfig(sill=1.0, range_km=4.0, aniso_ratio=1.0, model="gaussian")
    field = grf_raster(raster, cfg, np.random.default_rng(3))
    XX, YY = raster.meshgrid()
    idx = np.random.default_rng(2).choice(field.size, 3000, replace=False)
    x, y, v = XX.ravel()[idx], YY.ravel()[idx], field.ravel()[idx]

    edges = np.linspace(0.0, 6.0, 9)
    got = directional_variogram(x, y, v, tol_deg=22.5, edges=edges)
    at_lag = np.array([got[float(az)][1][3] for az in DEFAULT_SECTORS])
    assert at_lag.max() / at_lag.min() < 1.6, at_lag


# --------------------------------------------------------------------------- #
# binning and degenerate sectors
# --------------------------------------------------------------------------- #


def test_default_binning_is_coarser_than_the_omnidirectional_one(dataset):
    """Splitting the pairs four ways and keeping ~13 bins leaves 4-9 pairs in
    each, and the resulting curve is sampling noise that reads as dramatic
    anisotropy.  The coarser default is what makes the figure honest."""
    block = unit_block(dataset, "unit_1")
    x, y, v = block["x"].to_numpy(), block["y"].to_numpy(), block["log_Q_mean"].to_numpy()
    got = directional_variogram(x, y, v)
    iso = fit_variogram(x, y, v)
    for az in DEFAULT_SECTORS:
        assert len(got[float(az)][0]) <= DIRECTIONAL_N_BINS
    assert len(iso.lags) > DIRECTIONAL_N_BINS


def test_every_requested_sector_comes_back(dataset):
    block = unit_block(dataset, "unit_1")
    got = directional_variogram(
        block["x"].to_numpy(), block["y"].to_numpy(), block["log_Q_mean"].to_numpy()
    )
    assert set(got) == {float(a) for a in DEFAULT_SECTORS}


def test_a_sector_with_no_pairs_returns_empty_rather_than_raising():
    """Three collinear points hold no pairs perpendicular to their own line."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.zeros(4)
    v = np.array([1.0, 2.0, 1.5, 2.5])
    got = directional_variogram(x, y, v, azimuths=(90.0, 0.0), tol_deg=5.0,
                                edges=np.linspace(0, 3, 4))
    assert len(got[0.0][0]) == 0          # nothing runs north-south
    assert got[90.0][2].sum() > 0         # everything runs east-west


def test_empty_bins_are_dropped_so_lags_and_gamma_stay_aligned(dataset):
    block = unit_block(dataset, "unit_3")
    got = directional_variogram(
        block["x"].to_numpy(), block["y"].to_numpy(), block["log_Q_mean"].to_numpy()
    )
    for az, (lags, gamma, counts) in got.items():
        assert len(lags) == len(gamma) == len(counts), az
        assert (counts > 0).all(), az


# --------------------------------------------------------------------------- #
# the shared per-unit fit
# --------------------------------------------------------------------------- #


def test_fit_unit_variogram_matches_a_direct_call(dataset):
    """The CLI table and the figures must illustrate the *same* fit."""
    block = unit_block(dataset, "unit_1")
    direct = fit_variogram(
        block["x"].to_numpy(), block["y"].to_numpy(), block["log_Q_mean"].to_numpy()
    )
    shared = fit_unit_variogram(dataset, "unit_1")
    assert shared.sill == pytest.approx(direct.sill)
    assert shared.range_km == pytest.approx(direct.range_km)
    assert shared.nugget == pytest.approx(direct.nugget)


def test_too_few_cpts_returns_none_rather_than_a_meaningless_fit(dataset):
    import pandas as pd

    thin = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"].head(
        MIN_CPT_FOR_FIT - 1
    )
    bare = type("Bare", (), {"unit_summary": pd.concat([thin])})()
    assert fit_unit_variogram(bare, "unit_1") is None
