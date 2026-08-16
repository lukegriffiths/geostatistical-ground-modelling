"""Geostatistical ground modelling toolkit for normalised cone tip resistance.

Split along the data contract: :mod:`cpt_geostat.contract` and :mod:`cpt_geostat.models`
run on any data; :mod:`cpt_geostat.synthetic` generates datasets with known truth.
Synthetic validates, real applies, and nothing on the contract side imports
from the synthetic side.
"""

from .geometry import AZIMUTH_CONVENTION

__all__ = ["Config", "AZIMUTH_CONVENTION"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy so that ``import cpt_geostat.contract`` stays free of the synthetic stack
    # (and of gstools); see tests/contract/test_architecture.py.
    if name == "Config":
        from .synthetic.config import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
