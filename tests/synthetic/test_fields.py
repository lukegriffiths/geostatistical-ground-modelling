"""Field-construction checks: range parameterisation and sill delivery."""

from __future__ import annotations

import numpy as np
import pytest

from cpt_geostat.covariance import GrfConfig, build_model, range_to_len_scale
from cpt_geostat.synthetic.fields import Raster, grf_raster


@pytest.mark.parametrize("model", ["matern25", "matern15", "exponential", "gaussian", "spherical"])
def test_practical_range_is_honoured(model):
    """`range_km` must be the distance at which correlation falls to 0.05.

    Passing it to gstools as a len_scale instead would make a Matern 2.5 field
    3.7x smoother than configured — invisible on a map, fatal to range recovery.
    """
    cfg = GrfConfig(sill=1.0, range_km=3.0, model=model)
    built = build_model(cfg)
    assert np.isclose(built.percentile_scale(0.95), 3.0, rtol=1e-6)
    assert np.isclose(built.correlation(3.0), 0.05, atol=1e-6)


def test_len_scale_kind_passes_through():
    cfg = GrfConfig(sill=1.0, range_km=3.0, model="matern25", range_kind="len_scale")
    assert range_to_len_scale(cfg) == 3.0
    assert build_model(cfg).len_scale == 3.0


def test_integral_scale_kind():
    cfg = GrfConfig(sill=1.0, range_km=3.0, model="matern25", range_kind="integral")
    assert np.isclose(build_model(cfg).integral_scale, 3.0, rtol=1e-6)


def test_unknown_range_kind_rejected():
    with pytest.raises(ValueError, match="range_kind"):
        range_to_len_scale(GrfConfig(range_kind="nonsense"))


def test_sill_recovered_on_a_domain_much_larger_than_the_range():
    """Realised variance approaches the sill only when the domain dwarfs the range.

    On the actual 15 km site it will sit well below — that gap is expected, and
    it is why truth.yaml records realised variance alongside the requested sill.
    """
    raster = Raster.from_site(120.0, 0.4)
    cfg = GrfConfig(sill=0.09, range_km=3.0, model="matern25")
    field = grf_raster(raster, cfg, np.random.default_rng(3))
    assert np.isclose(field.var(), 0.09, rtol=0.1)


def test_zero_sill_gives_a_flat_field():
    raster = Raster.from_site(5.0, 0.5)
    field = grf_raster(raster, GrfConfig(sill=0.0), np.random.default_rng(0))
    assert np.all(field == 0.0)


def test_raster_sampling_matches_cell_centres():
    raster = Raster.from_site(10.0, 0.5)
    XX, YY = raster.meshgrid()
    field = 2.0 * XX - 3.0 * YY
    x = np.array([-2.25, 0.25, 3.75])
    y = np.array([1.25, -4.25, 0.75])
    assert np.allclose(raster.sample(field, x, y), 2.0 * x - 3.0 * y, atol=1e-9)
