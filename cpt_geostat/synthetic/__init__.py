"""Synthetic data generation and truth diagnostics (Part A + B1).

Everything here needs a generator config or realised truth.  The estimators and
data-driven plots live outside this package and must not import from it —
synthetic validates, real applies.
"""

from .config import Config
from .pipeline import generate

__all__ = ["Config", "generate"]
