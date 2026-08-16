"""Kriging — simple, ordinary and universal, over one covariance convention.

The three differ only in what they assume about the mean, and that difference is
the whole point of having all three:

``SimpleKriging``
    The mean is **known**.  The direct upgrade from
    :class:`~cpt_geostat.models.baseline.UnitMeanEstimator`: same constant mean, but
    the residual is now spatially correlated instead of being called noise.
``OrdinaryKriging``
    The mean is an **unknown constant**, estimated from the data.  Its extra
    variance is the price of not knowing it, and it is not optional — it is the
    honest cost of the assumption simple kriging quietly gets for free by being
    handed the answer.
``UniversalKriging``
    The mean is an unknown **linear function of position**.  This is the one
    that can handle unit 2's 0.10/km trend, which simple and ordinary kriging
    structurally cannot: both assume a constant, and an unmodelled trend leaves
    the variogram unbounded and the intervals too narrow.

Where the covariance has no structure to offer, kriging collapses back onto the
baseline — which is the correct behaviour, and is why the baseline is the thing
to compare against rather than a straw man.

The variance convention, verified empirically in
``tests/models/test_kriging.py`` rather than assumed, and checked for **all
three** methods rather than extrapolated from one:

    gstools' kriging variance is the variance of a **new observation**.  Far
    from any data it tends to ``sill + nugget`` (plus, for OK and UK, the
    variance of estimating the mean or the drift).

So :meth:`predict` — which owes the caller the *latent* field, per
:mod:`cpt_geostat.models.base` — must subtract the nugget, and
:meth:`predict_observation` gets it back by adding ``noise_var_``.  Taking
gstools' number as latent would overstate the field uncertainty by the whole
nugget, which on unit 6 is most of the variance; taking it as latent *and* then
adding the nugget for CV would double-count it.

Conditioning data are treated as **noisy**, not exact (``exact=False``,
``cond_err="nugget"``).  They are: a depth-average over a finite autocorrelated
trace carries the nugget plus the averaging error, which is precisely what the
fitted nugget estimates.  Interpolating them exactly would force the surface
through values it should be smoothing, and would report zero uncertainty at the
CPTs, where the truth is that we know the *measurement* there, not the field.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gstools as gs
import numpy as np

from ..covariance import build_model, model_params
from ..trend import fit_linear_trend
from .base import SpatialEstimator, check_coords
from .variogram import fit_variogram


class _GstoolsKriging(SpatialEstimator):
    """Shared machinery: covariance selection, the variance convention, params.

    Subclasses supply only :meth:`_make_krige` — what gstools object to build
    and what it assumes about the mean.  Everything that is easy to get subtly
    wrong (which variance is returned, how a range is reported, when a
    parameter is not identifiable) is therefore written once.
    """

    #: What this estimator assumes about the mean, for reports and reprs.
    mean_model = "known constant"
    #: Degrees of freedom spent on the mean — 0 for SK, 1 for OK, 3 for UK.
    _mean_dof = 0

    def __init__(self, covariance="fit", model: str = "matern25"):
        self.covariance = covariance
        self.model = model

    # -- subclass hooks -----------------------------------------------------
    def _make_krige(self, cov, X, y):
        raise NotImplementedError

    def _variogram_values(self, X, y):
        """What the variogram is fitted to.

        A variogram works on increments, so it is already blind to an unknown
        *constant* mean — SK and OK can fit it to the raw values.  A **drift**
        is different, and :class:`UniversalKriging` overrides this.
        """
        return y

    # -- fitting ------------------------------------------------------------
    def fit(self, X, y):
        X = check_coords(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != y.size:
            raise ValueError(f"X has {len(X)} rows but y has {y.size}")
        finite = np.isfinite(y)
        X, y = X[finite], y[finite]
        if y.size == 0:
            raise ValueError("no finite observations to fit")
        if y.size <= self._mean_dof:
            raise ValueError(
                f"{type(self).__name__} spends {self._mean_dof} degrees of freedom on the "
                f"mean and cannot be fitted to {y.size} point(s)"
            )

        self.X_, self.y_ = X, y
        self.n_ = int(y.size)

        if self.covariance == "fit":
            v = self._variogram_values(X, y)
            self.variogram_ = fit_variogram(X[:, 0], X[:, 1], v, model=self.model)
            cov = self.variogram_.model
        else:
            grf, nugget = self.covariance
            self.variogram_ = None
            cov = build_model(grf, dim=2, nugget=float(nugget))
        self.cov_ = cov

        self.krige_ = self._make_krige(cov, X, y)
        return self

    # -- prediction ---------------------------------------------------------
    def predict(self, X, return_std: bool = False):
        """Latent field: gstools' variance with the nugget taken back out."""
        X = check_coords(X)
        mean, var = self.krige_((X[:, 0], X[:, 1]), return_var=True)
        mean = np.asarray(mean, dtype=float)
        if not return_std:
            return mean
        # Clipped at zero: with a pseudo-inverse solve the subtraction can land
        # a hair below it at a conditioning point, and a negative variance would
        # propagate as nan through every metric downstream.
        latent = np.maximum(np.asarray(var, dtype=float) - self.noise_var_, 0.0)
        return mean, np.sqrt(latent)

    @property
    def noise_var_(self) -> float:
        return float(self.cov_.nugget)

    @property
    def params_(self) -> Dict[str, Any]:
        p = model_params(self.cov_, self.model)
        p["mean"] = getattr(self, "mean_", None)
        p["mean_model"] = self.mean_model
        p["n"] = self.n_
        if self.variogram_ is not None and not self.variogram_.resolved:
            # An unresolvable range is not a fitted number, and reporting it as
            # one is how a confident, meaningless range reaches a results table.
            p["range_km"] = None
            p["not_identifiable"] = self.variogram_.why_not_resolved()
        return p

    def __repr__(self) -> str:
        name = type(self).__name__
        if not hasattr(self, "cov_"):
            return f"{name}(unfitted)"
        return (
            f"{name}(n={self.n_}, sill={self.cov_.var:.3f}, "
            f"nugget={self.cov_.nugget:.4f}, mean={self.mean_model})"
        )


class SimpleKriging(_GstoolsKriging):
    """Simple kriging with a constant, *known* mean.

    ``mean`` is the known mean simple kriging assumes.  ``None`` uses the sample
    mean, which is what makes this comparable with the baseline; it is a mild
    cheat — simple kriging is supposed to be *given* the mean — and it is the
    same cheat the baseline makes, so the comparison stays fair.  Ordinary
    kriging is the estimator that pays for the mean honestly.

    ``covariance`` selects where the model comes from:

    ``"fit"``
        Fit an isotropic variogram to the data.  The only option available on
        real data.
    ``(GrfConfig, nugget)``
        Use a covariance given outright.  On synthetic runs this can be the
        truth, which separates *"is kriging implemented correctly"* from *"can a
        variogram be fitted from 30 CPTs"* — two questions that a single number
        confounds, and the second is much harder than the first.
    """

    mean_model = "known constant"
    _mean_dof = 0

    def __init__(self, mean: Optional[float] = None, covariance="fit",
                 model: str = "matern25"):
        super().__init__(covariance=covariance, model=model)
        self.mean = mean

    def _make_krige(self, cov, X, y):
        self.mean_ = float(np.mean(y)) if self.mean is None else float(self.mean)
        return gs.krige.Simple(
            cov, (X[:, 0], X[:, 1]), y, mean=self.mean_,
            exact=False, cond_err="nugget",
        )


class OrdinaryKriging(_GstoolsKriging):
    """Ordinary kriging: the mean is an unknown constant, estimated from the data.

    The difference from simple kriging is one degree of freedom, and it shows up
    as *wider* intervals — everywhere, but conspicuously far from data, where
    simple kriging returns to a mean it was handed and ordinary kriging admits
    it does not know the level.  That gap is a real modelling difference to be
    quantified, not an error to be tuned away, which is why both estimators are
    kept rather than one being declared the right answer.

    On a unit whose mean is genuinely unknown — every real unit — this is the
    more defensible of the two.  Simple kriging's advantage in cross-validation
    comes partly from being handed the sample mean of the data it is scored on.
    """

    mean_model = "unknown constant"
    _mean_dof = 1

    def _make_krige(self, cov, X, y):
        # Reported for comparability with SK; OK solves for it internally per
        # prediction point, so this is the unweighted sample mean, not the
        # kriging estimate of the drift.
        self.mean_ = float(np.mean(y))
        return gs.krige.Ordinary(
            cov, (X[:, 0], X[:, 1]), y, exact=False, cond_err="nugget",
        )


class UniversalKriging(_GstoolsKriging):
    """Universal kriging: the mean is an unknown linear function of position.

    The estimator for a unit carrying a trend.  Simple and ordinary kriging both
    assume a constant mean, so on unit 2 (0.10/km at 115°) they leave the
    variogram unbounded — the fitted range runs to the identifiability bound —
    and produce intervals that are too narrow because the trend's contribution
    to the spread is being called neither trend nor noise.

    ``drift`` is ``"linear"`` (the default: ``1, x, y``) or a list of callables
    gstools will evaluate at each position.  Three degrees of freedom go on the
    drift, so a unit held at a handful of CPTs cannot support this — the
    estimability gate, not this class, is what should be deciding that.

    Unlike a GP fitted on OLS residuals, the predictive variance here **does**
    include the uncertainty in the fitted drift, which is why extrapolating a
    universal kriging surface well outside the data gives an honestly enormous
    interval rather than a confident straight line.

    **The variogram is fitted to the OLS residuals, not to the raw values**, and
    that is not a refinement — without it universal kriging does not work at
    all.  A variogram of trended data is unbounded: the trend keeps adding
    variance as the lag grows, so the fit runs to the range bound and puts
    nearly all the variance into a long-range structured component with a tiny
    nugget.  Kriging then predicts with intervals several times too narrow.
    Measured on unit 2 (0.10/km) this is the difference between MSSR 31 and
    MSSR 1.4 — the estimator looks broken until the variogram is detrended.

    The residual sill is what universal kriging wants anyway: the drift is
    modelled explicitly, so it must not also be counted as spatial structure.
    """

    mean_model = "unknown linear drift"
    _mean_dof = 3

    def __init__(self, covariance="fit", model: str = "matern25", drift="linear"):
        super().__init__(covariance=covariance, model=model)
        self.drift = drift

    def _variogram_values(self, X, y):
        """Detrended residuals — see the class docstring; this is load-bearing."""
        trend = fit_linear_trend(X[:, 0], X[:, 1], y)
        if not np.isfinite(trend.gradient):
            return y
        return y - trend.predict(X[:, 0], X[:, 1])

    def _make_krige(self, cov, X, y):
        # The fitted plane is reported for readability and for comparison with
        # the truth trend; gstools solves the drift internally, so this is a
        # description of the data, not the coefficients it uses.
        self.trend_ = fit_linear_trend(X[:, 0], X[:, 1], y)
        self.mean_ = float(np.mean(y))
        return gs.krige.Universal(
            cov, (X[:, 0], X[:, 1]), y, drift_functions=self.drift,
            exact=False, cond_err="nugget",
        )

    @property
    def params_(self) -> Dict[str, Any]:
        p = super().params_
        trend = getattr(self, "trend_", None)
        if trend is not None and trend.gradient_is_identifiable:
            p["trend_grad"] = trend.gradient
            p["trend_azimuth_deg"] = trend.azimuth_deg
        else:
            # A bearing fitted to a surface that is statistically flat is a
            # number with no content; the angle rule anisotropy follows at
            # ratio 1 applies here too.
            p["trend_grad"] = trend.gradient if trend is not None else None
            p["trend_azimuth_deg"] = None
        return p


#: The estimators the CLI compares, in increasing order of what they assume.
KRIGING_METHODS = {
    "SK": SimpleKriging,
    "OK": OrdinaryKriging,
    "UK": UniversalKriging,
}


def kriging_factory(ds, covariance="fit", model: str = "matern25", method: str = "SK"):
    """``unit_id -> fresh unfitted kriging estimator``, for cross-validation.

    Runs on any data.  ``method`` is ``"SK"``, ``"OK"`` or ``"UK"``.  The
    kriging-with-the-right-covariance reference is synthetic-only and lives with
    the rest of the truth machinery in
    :func:`cpt_geostat.synthetic.truth.truth_kriging_factory`; asking for it here
    raises rather than silently falling back to a fit — the two answer
    different questions and the difference between them is the result worth
    having.
    """
    if covariance == "truth":
        raise ValueError(
            "covariance='truth' is synthetic-only; use "
            "cpt_geostat.synthetic.truth.truth_kriging_factory(ds)"
        )
    try:
        cls = KRIGING_METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown kriging method {method!r}; have {sorted(KRIGING_METHODS)}"
        ) from None
    return lambda unit_id: cls(covariance=covariance, model=model)
