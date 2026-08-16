"""pykrige as an independent check on gstools' kriging — validation only.

Nothing in the production path imports this.  It exists so that "our ordinary
kriging is right" rests on two independently written implementations agreeing,
rather than on one library agreeing with itself.

Two things make the comparison harder than calling both and diffing, and both
are the reason this lives in its own module rather than inside ``kriging.py``:

**pykrige is a fourth length-scale parameterisation.**  ``practical range`` /
gstools ``len_scale`` / sklearn ``length_scale`` were already three; pykrige's
``range`` is a fourth, and it is *not* the practical range:

======================  ==========================  ==================
model                   ``pk_range / len_scale``    practical / len_scale
======================  ==========================  ==================
exponential             3                           2.9957
gaussian                7/(2*sqrt(pi)) = 1.97466    1.9530
spherical               1                           0.8114
======================  ==========================  ==================

For spherical those differ by 23%.  Feeding a practical range to pykrige as its
``range`` would therefore produce a confident, wrong cross-check — which would
be read as gstools being wrong.  The factors below are exact closed forms and
are asserted against the library's own variogram functions.

**pykrige has no Matern**, so the cross-check runs on exponential — the point is
to validate the *kriging algebra*, not the covariance model, and the algebra
does not know which correlation function it was handed.

**The two libraries use opposite variance conventions.**  pykrige interpolates
the conditioning data exactly and returns a *latent* variance (zero at the data
points, nugget excluded); gstools with ``exact=False, cond_err="nugget"``
smooths them and returns an *observation* variance.  The clean comparison is
therefore **at zero nugget**, where the conventions coincide and the two agree
to machine precision.  With a nugget they differ by construction, and asserting
a loose tolerance there would hide the convention gap instead of documenting it.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import SpatialEstimator, check_coords

#: ``pykrige_range = factor * gstools_len_scale``.  Exact, not fitted.
PYKRIGE_RANGE_FACTOR = {
    "exponential": 3.0,
    "gaussian": 7.0 / (2.0 * np.sqrt(np.pi)),
    "spherical": 1.0,
}

#: gstools model names that have a pykrige counterpart at all.
SHARED_MODELS = tuple(PYKRIGE_RANGE_FACTOR)


def to_pykrige_params(cov, model: str) -> list:
    """A gstools covariance model as pykrige ``[psill, range, nugget]``.

    The single conversion site for this bridge, for the same reason
    :mod:`cpt_geostat.covariance` is the single site for the gstools one: a forward
    conversion here and an inverse somewhere else is how the original
    ``len_scale`` bug happened.
    """
    if model not in PYKRIGE_RANGE_FACTOR:
        raise ValueError(
            f"pykrige has no {model!r} model; cross-checks must use one of "
            f"{sorted(PYKRIGE_RANGE_FACTOR)}"
        )
    return [
        float(cov.var),
        float(cov.len_scale) * PYKRIGE_RANGE_FACTOR[model],
        float(cov.nugget),
    ]


class PyKrigeOrdinary(SpatialEstimator):
    """Ordinary kriging via pykrige, conforming to this package's interface.

    A cross-check, not an estimator to use: it is restricted to the covariance
    models pykrige implements, and it treats the conditioning data as exact,
    which is the wrong assumption for depth-averaged CPT data.

    ``covariance`` is ``(GrfConfig, nugget)`` — a fitted variogram is not
    supported, because the point of the cross-check is to give both libraries
    *the same* covariance and see whether they agree.

    :attr:`noise_var_` is reported as the nugget for interface compatibility,
    but note that pykrige's returned variance already excludes it, so
    ``predict`` passes the library's number through unchanged where
    :class:`~cpt_geostat.models.kriging.OrdinaryKriging` subtracts.
    """

    def __init__(self, covariance, model: str = "exponential"):
        self.covariance = covariance
        self.model = model

    def fit(self, X, y) -> "PyKrigeOrdinary":
        from pykrige.ok import OrdinaryKriging as _PKOK

        from ..covariance import build_model

        X = check_coords(X)
        y = np.asarray(y, dtype=float).ravel()
        finite = np.isfinite(y)
        X, y = X[finite], y[finite]
        if y.size == 0:
            raise ValueError("no finite observations to fit")

        grf, nugget = self.covariance
        if grf.aniso_ratio != 1.0:
            raise ValueError(
                "the pykrige cross-check is isotropic only; an anisotropic "
                "comparison would be checking two different models"
            )
        self.cov_ = build_model(grf, dim=2, nugget=float(nugget))
        self.n_ = int(y.size)
        self.mean_ = float(np.mean(y))

        self.krige_ = _PKOK(
            X[:, 0], X[:, 1], y,
            variogram_model=self.model,
            variogram_parameters=to_pykrige_params(self.cov_, self.model),
            enable_plotting=False,
        )
        return self

    def predict(self, X, return_std: bool = False):
        """Latent field.  pykrige already excludes the nugget, so no subtraction."""
        X = check_coords(X)
        mean, var = self.krige_.execute("points", X[:, 0], X[:, 1])
        mean = np.asarray(mean, dtype=float)
        if not return_std:
            return mean
        return mean, np.sqrt(np.maximum(np.asarray(var, dtype=float), 0.0))

    @property
    def noise_var_(self) -> float:
        return float(self.cov_.nugget)

    @property
    def params_(self) -> Dict[str, Any]:
        from ..covariance import model_params

        p = model_params(self.cov_, self.model)
        p["mean"] = self.mean_
        p["mean_model"] = "unknown constant"
        p["n"] = self.n_
        return p

    def __repr__(self) -> str:
        if not hasattr(self, "cov_"):
            return "PyKrigeOrdinary(unfitted)"
        return f"PyKrigeOrdinary(n={self.n_}, model={self.model!r})"
