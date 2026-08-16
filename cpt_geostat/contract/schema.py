"""Column contracts and the :class:`Dataset` every consumer works on.

``layout``/``layers``/``samples``/``unit_summary`` are what real data supplies;
``config``, ``raster``, ``rasters``, ``unit_values`` and ``truth`` are
synthetic-only and every consumer of them must tolerate their absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # synthetic-only type; never imported at runtime
    from ..synthetic.config import Config

LAYERS_COLUMNS = ["cpt_id", "unit_id", "z_top", "z_bot"]
SAMPLES_COLUMNS = ["cpt_id", "x", "y", "z", "unit_id", "Qtn"]
SUMMARY_COLUMNS = [
    "cpt_id", "x", "y", "unit_id", "log_Q_mean", "log_Q_sd", "n_samples", "thickness_m",
]


@dataclass(frozen=True)
class Raster:
    """Regular grid of cell centres, in kilometres, origin at site centre."""

    x: np.ndarray  # (nx,)
    y: np.ndarray  # (ny,)

    @classmethod
    def from_site(cls, size_km: float, res_km: float) -> "Raster":
        half = size_km / 2.0
        n = int(round(size_km / res_km))
        edges = np.linspace(-half, half, n + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return cls(x=centres, y=centres.copy())

    @classmethod
    def from_bounds(
        cls, xmin: float, xmax: float, ymin: float, ymax: float, res_km: float
    ) -> "Raster":
        """A grid covering an arbitrary rectangle, for real sites.

        :meth:`from_site` assumes a *centred square*, which a real survey is
        only by coincidence — IJmuiden happens to be centred at the origin
        because its preparation step puts it there, and 23.3 x 19.6 km is not
        square.  Building a real grid on that coincidence is the kind of thing
        that works until a project supplies coordinates in its own frame.

        Cell counts are rounded up, so the grid always *covers* the requested
        bounds rather than stopping just inside them.
        """
        if not (xmax > xmin and ymax > ymin):
            raise ValueError(f"empty bounds: x {xmin}..{xmax}, y {ymin}..{ymax}")
        if res_km <= 0:
            raise ValueError(f"res_km must be positive, got {res_km}")

        def axis(lo: float, hi: float) -> np.ndarray:
            n = max(int(np.ceil((hi - lo) / res_km)), 1)
            edges = lo + res_km * np.arange(n + 1)
            return 0.5 * (edges[:-1] + edges[1:])

        return cls(x=axis(xmin, xmax), y=axis(ymin, ymax))

    @property
    def shape(self):
        return (self.y.size, self.x.size)

    @property
    def extent(self):
        """``(xmin, xmax, ymin, ymax)`` cell *edges*, for ``imshow``."""
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        return (
            float(self.x[0] - dx / 2),
            float(self.x[-1] + dx / 2),
            float(self.y[0] - dy / 2),
            float(self.y[-1] + dy / 2),
        )

    def meshgrid(self):
        """``(XX, YY)``, each ``(ny, nx)``."""
        return np.meshgrid(self.x, self.y, indexing="xy")

    def sample(self, field: np.ndarray, x, y) -> np.ndarray:
        """Bilinear sample of a ``(ny, nx)`` field at scattered points."""
        # Imported here so the contract package stays numpy+pandas at import time.
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (self.y, self.x), np.asarray(field, dtype=float),
            method="linear", bounds_error=False, fill_value=None,
        )
        pts = np.column_stack([np.asarray(y, dtype=float), np.asarray(x, dtype=float)])
        return interp(pts)


@dataclass
class Dataset:
    """Everything a run produces.

    ``layout``/``layers``/``samples``/``unit_summary`` are what real data also
    supplies; ``raster``, ``rasters`` and ``truth`` are synthetic-only and every
    consumer of them must tolerate their absence.
    """

    layout: pd.DataFrame
    layers: pd.DataFrame
    samples: pd.DataFrame
    unit_summary: pd.DataFrame
    config: Optional["Config"] = None
    raster: Optional[Raster] = None
    rasters: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    unit_values: Optional[pd.DataFrame] = None
    truth: Optional[Dict[str, Any]] = None

    @property
    def is_synthetic(self) -> bool:
        return self.truth is not None

    @property
    def unit_ids(self):
        if self.config is not None:
            return self.config.unit_ids
        order = self.layers["unit_id"].drop_duplicates().tolist()
        return sorted(order)
