"""Estimators.

``base`` fixes the interface; ``baseline`` is the no-spatial-structure model
every later estimator has to beat.  Everything here runs on any data meeting
the contract; the truth comparisons live in :mod:`cpt_geostat.synthetic.truth`.
"""

from .base import SpatialEstimator
from .baseline import (
    UnitMeanEstimator,
    baseline_factory,
    fit_unit_baselines,
    unit_baseline_table,
)
from .kriging import (
    KRIGING_METHODS,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
    kriging_factory,
)
from .variogram import VariogramFit, empirical_variogram, fit_variogram

__all__ = [
    "SpatialEstimator",
    "UnitMeanEstimator",
    "SimpleKriging",
    "OrdinaryKriging",
    "UniversalKriging",
    "KRIGING_METHODS",
    "VariogramFit",
    "baseline_factory",
    "kriging_factory",
    "empirical_variogram",
    "fit_variogram",
    "fit_unit_baselines",
    "unit_baseline_table",
]
