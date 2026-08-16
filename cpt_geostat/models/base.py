"""The estimator interface — what every spatial model here must provide.

Two conventions are fixed once, here, because getting either wrong is quiet:

**Coordinates.** ``X`` is ``(n, 2)`` in kilometres with the origin at the site
centre, and ``y`` is ``log(Qtn)`` — one value per CPT per unit, the depth-average
from ``unit_summary``.  Estimators are fitted per unit, never across units.

**Variance.**  There are two, they differ by the observation noise, and which one
is wanted depends entirely on the question:

============================  =========================  ===========================
Question                      Method                     Includes observation noise?
============================  =========================  ===========================
"what is the field here?"     :meth:`predict`             no  — latent field only
"what would I measure here?"  :meth:`predict_observation` yes — latent + noise
============================  =========================  ===========================

Estimator-vs-estimator comparisons and difference maps want the *latent*
variance: observation noise is not part of anyone's belief about the field.
Every cross-validation metric wants the *observation* variance, because the
held-out number is an observation — it carries the nugget and the error incurred
by depth-averaging a finite, autocorrelated CPT trace.  Scoring calibration on
latent variance against noisy held-out values guarantees under-coverage, and it
misfires worst on the units that are *supposed* to come out wide and honest.

Subclasses therefore implement :meth:`predict` (latent) and :attr:`noise_var_`;
:meth:`predict_observation` is derived from the two so the addition happens in
one place rather than at every call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


def check_coords(X) -> np.ndarray:
    """Validate and normalise an ``(n, 2)`` array of kilometre coordinates."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be (n, 2) in km, got shape {X.shape}")
    return X


class SpatialEstimator(ABC):
    """Fitted per unit; predicts ``log(Qtn)`` at map locations."""

    @abstractmethod
    def fit(self, X, y) -> "SpatialEstimator":
        """``X``: ``(n, 2)`` km.  ``y``: ``(n,)`` depth-averaged ``log(Qtn)``."""

    @abstractmethod
    def predict(self, X, return_std: bool = False):
        """Posterior over the **latent field**; observation noise excluded."""

    @property
    @abstractmethod
    def noise_var_(self) -> float:
        """Variance of an observation about the latent field.

        Nugget plus depth-averaging error — everything uncorrelated between two
        CPTs standing at the same place.
        """

    @property
    @abstractmethod
    def params_(self) -> Dict[str, Any]:
        """Fitted parameters in ``truth.yaml``'s parameterisation.

        Keys: ``mean``, ``sill``, ``range_km`` (practical), ``aniso_ratio``,
        ``aniso_angle_deg`` (CW from north), ``nugget``.  A parameter this
        estimator cannot identify is ``None`` — distinct from a fitted number,
        and distinct from ``nan``, which would read as a failed fit.
        """

    def predict_observation(self, X):
        """Predictive distribution of a **new observation**: ``(mean, sd)``."""
        mean, sd = self.predict(X, return_std=True)
        return mean, np.sqrt(np.asarray(sd, dtype=float) ** 2 + self.noise_var_)
