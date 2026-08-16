"""Linear trend surfaces — fitted once, here, and shared.

``log Q = intercept + bx*x + by*y`` is wanted in three places that must not each
grow their own copy: the trend-check diagnostic, universal kriging's drift, and
a GP fitted on detrended residuals.  The diagnostic previously fitted it inline
and returned ``(gradient, azimuth)``, discarding the intercept — enough to draw
a line, not enough to *add the trend back* at a prediction point, which is what
the estimators need.

The module sits at the package root, peer to :mod:`cpt_geostat.geometry`, because
``models/`` must not import from ``plots/``: that inverts the layering, and the
estimators would then depend on the plotting stack.

Azimuth follows :mod:`cpt_geostat.geometry` — degrees clockwise from north — and the
coefficients are exactly the ones that convention implies::

    trend = gradient * project_on_azimuth(x, y, azimuth_deg)
          = gradient * (x*sin(az) + y*cos(az))

so ``bx = gradient*sin(az)`` and ``by = gradient*cos(az)``.  Unlike an
anisotropy axis, a trend azimuth is a *direction of increase* and wraps mod 360,
not mod 180: a gradient rising to the north is not the same as one falling to
the north.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Significance level for calling a gradient distinguishable from zero.
_ALPHA = 0.05


@dataclass(frozen=True)
class LinearTrend:
    """An OLS plane through ``(x, y, v)``, in this package's conventions.

    ``gradient``/``azimuth_deg`` are the same plane in polar form — the pair a
    report quotes — while ``intercept``/``bx``/``by`` are what :meth:`predict`
    needs to put the trend back at a new location.
    """

    intercept: float
    bx: float  # d(value)/dx, per km
    by: float  # d(value)/dy, per km
    n: int = 0
    residual_var: float = float("nan")  # about the fitted plane, ddof = 3
    se_bx: float = float("nan")
    se_by: float = float("nan")

    @property
    def gradient(self) -> float:
        """Steepest slope, per km — always non-negative."""
        return float(np.hypot(self.bx, self.by))

    @property
    def azimuth_deg(self) -> float:
        """Bearing of steepest *increase*, degrees CW from north, in [0, 360)."""
        return float(np.rad2deg(np.arctan2(self.bx, self.by)) % 360.0)

    @property
    def gradient_is_identifiable(self) -> bool:
        """Is the gradient distinguishable from zero at the 5% level?

        Where it is not, :attr:`azimuth_deg` is the bearing of what is
        statistically a flat surface — a number with no content.  Recovery
        reports must render that as *not identifiable* rather than scoring it
        against a true azimuth, which is the same rule anisotropy angles follow
        at ratio 1.
        """
        if self.n < 4 or not np.isfinite(self.se_bx) or not np.isfinite(self.se_by):
            return False
        if self.se_bx <= 0 or self.se_by <= 0:
            return False
        # Two independent-enough one-sided checks; a joint F-test is stricter
        # than needed for a flag whose only job is to suppress a meaningless
        # bearing, and this stays readable.
        z = np.hypot(self.bx / self.se_bx, self.by / self.se_by)
        return bool(z > 2.45)  # ~ chi2(2) at 5%

    def predict(self, x, y):
        """The trend surface at ``(x, y)`` — broadcasting, shape-preserving."""
        return self.intercept + self.bx * np.asarray(x, dtype=float) + self.by * np.asarray(
            y, dtype=float
        )

    def __repr__(self) -> str:
        if not np.isfinite(self.gradient):
            return "LinearTrend(undetermined)"
        return (
            f"LinearTrend(gradient={self.gradient:.4g}/km at {self.azimuth_deg:.1f}°, "
            f"intercept={self.intercept:.4g}, n={self.n})"
        )


#: What an unfittable trend returns — nan rather than a silent zero plane.
_UNDETERMINED = LinearTrend(
    intercept=float("nan"), bx=float("nan"), by=float("nan"), n=0
)


def fit_linear_trend(x, y, v) -> LinearTrend:
    """OLS ``v ~ 1 + x + y``.

    Fewer than three points cannot determine a plane, and a degenerate layout
    (every CPT on one line) cannot determine it uniquely; both return a trend of
    ``nan`` rather than a least-norm answer that would read as a fitted result.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if not (x.size == y.size == v.size):
        raise ValueError(f"x, y, v must be the same length; got {x.size}, {y.size}, {v.size}")

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    n = x.size
    if n < 3:
        return _UNDETERMINED

    A = np.column_stack([np.ones(n), x, y])
    # Rank-deficient layouts (collinear CPTs) have no unique plane; lstsq would
    # return the minimum-norm one, which is a choice the data does not support.
    if np.linalg.matrix_rank(A) < 3:
        return _UNDETERMINED

    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    intercept, bx, by = (float(c) for c in coef)

    dof = n - 3
    if dof > 0:
        resid = v - A @ coef
        residual_var = float(resid @ resid / dof)
        xtx_inv = np.linalg.inv(A.T @ A)
        se_bx = float(np.sqrt(residual_var * xtx_inv[1, 1]))
        se_by = float(np.sqrt(residual_var * xtx_inv[2, 2]))
    else:
        # Exactly determined: the plane passes through every point, so there is
        # no residual to estimate a standard error from.
        residual_var = se_bx = se_by = float("nan")

    return LinearTrend(
        intercept=intercept, bx=bx, by=by, n=n,
        residual_var=residual_var, se_bx=se_bx, se_by=se_by,
    )


def detrend(x, y, v, trend: Optional[LinearTrend] = None):
    """``(residuals, trend)`` — fit a plane and subtract it.

    The companion to ``trend.predict`` at the far end: an estimator fitted on
    the residuals must add the *same* plane back, so both halves come from one
    object rather than from two fits of the same data.
    """
    if trend is None:
        trend = fit_linear_trend(x, y, v)
    v = np.asarray(v, dtype=float)
    if not np.isfinite(trend.gradient):
        return v, trend
    return v - trend.predict(x, y), trend
