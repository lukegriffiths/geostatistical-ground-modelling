"""The data contract — what any dataset, synthetic or real, must supply.

This package is deliberately dependency-light (numpy + pandas + yaml): loading
two CSVs of real data must not drag in the field-simulation stack.  ``cpt_geostat``
splits along this line — ``cpt_geostat.contract`` and ``cpt_geostat.models`` run on any
data; ``cpt_geostat.synthetic`` needs a generator config and truth.  The invariant is
pinned by ``tests/contract/test_architecture.py``: importing this package must
not import ``cpt_geostat.synthetic`` or gstools.
"""

from .io import read_dataset, write_dataset
from .schema import (
    LAYERS_COLUMNS,
    SAMPLES_COLUMNS,
    SUMMARY_COLUMNS,
    Dataset,
    Raster,
)
from .summarise import summarise_units

__all__ = [
    "Dataset",
    "Raster",
    "LAYERS_COLUMNS",
    "SAMPLES_COLUMNS",
    "SUMMARY_COLUMNS",
    "read_dataset",
    "summarise_units",
    "write_dataset",
]
