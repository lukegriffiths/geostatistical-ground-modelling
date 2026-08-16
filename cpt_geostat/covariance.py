"""Covariance models and the one place ranges are converted.

gstools parameterises a covariance by ``len_scale``.  That is **not** the range
a variogram fit reports: for Matern 2.5 the practical range — the separation at
which correlation has fallen to 0.05 — is 3.74x the ``len_scale``.  Feeding a
practical range straight in as a ``len_scale`` makes every field several times
smoother than intended, which is invisible on a map and fatal to any claim about
recovered hyperparameters.

The conversion therefore lives here and nowhere else, together with its inverse,
and both the generator (:mod:`cpt_geostat.synthetic.fields`) and the estimators
(:mod:`cpt_geostat.models`) import from this module.  Splitting the forward conversion
across the generator and the inverse across the estimators would recreate
exactly the two-homes situation that let the original bug through.

:class:`GrfConfig` — the covariance specification both sides speak — lives here
for the same reason: it is not generator config, it is the shared vocabulary in
which a covariance is stated and a fit is reported back.

Angles follow :mod:`cpt_geostat.geometry`: azimuths are degrees clockwise from north
and ``aniso_angle_deg`` is the azimuth of the **major** axis.  gstools wants
``anis = minor/major`` and radians counter-clockwise from +x.
"""

from __future__ import annotations

from dataclasses import dataclass

import gstools as gs

from .geometry import azimuth_to_math_angle, math_angle_to_azimuth


@dataclass
class GrfConfig:
    """A Gaussian random field component — the shared covariance specification.

    Both sides of the package speak it: the generator realises fields from it
    and the estimators (:mod:`cpt_geostat.models`) report fits back in it, which is
    why it lives here rather than in the synthetic-only config.

    ``range_km`` is the **practical range** by default — the separation at which
    the correlation has fallen to 0.05, i.e. what a fitted variogram reports.
    This is *not* gstools' ``len_scale`` (for Matern 2.5 the practical range is
    3.74x the len_scale), so ``range_kind`` records which is meant and the
    conversion happens in one place, :func:`range_to_len_scale` below.
    """

    sill: float = 0.09
    range_km: float = 3.5
    aniso_ratio: float = 1.0  # major / minor range; 1.0 = isotropic
    aniso_angle_deg: float = 0.0  # azimuth of the MAJOR axis
    model: str = "matern25"
    range_kind: str = "practical"  # practical | integral | len_scale


MODELS = {
    "matern25": lambda **kw: gs.Matern(nu=2.5, **kw),
    "matern15": lambda **kw: gs.Matern(nu=1.5, **kw),
    "exponential": gs.Exponential,
    "gaussian": gs.Gaussian,
    "spherical": gs.Spherical,
}

#: Correlation level defining the practical range.
PRACTICAL_LEVEL = 0.95


def _probe(model: str, dim: int = 2):
    if model not in MODELS:
        raise ValueError(f"unknown covariance model {model!r}; have {sorted(MODELS)}")
    return MODELS[model](dim=dim, var=1.0, len_scale=1.0)


def range_factor(model: str = "matern25", kind: str = "practical", dim: int = 2) -> float:
    """``range = factor * len_scale`` for this model and range definition.

    3.74 for Matern 2.5 practical, 1.95 for Gaussian, 0.81 for Spherical.
    """
    if kind == "len_scale":
        return 1.0
    probe = _probe(model, dim)
    if kind == "practical":
        return float(probe.percentile_scale(PRACTICAL_LEVEL))
    if kind == "integral":
        return float(probe.integral_scale)
    raise ValueError(f"unknown range_kind {kind!r}; use practical, integral or len_scale")


def range_to_len_scale(cfg: GrfConfig, dim: int = 2) -> float:
    """Configured range -> gstools ``len_scale``."""
    return float(cfg.range_km) / range_factor(cfg.model, cfg.range_kind, dim)


def len_scale_to_range(
    len_scale: float, model: str = "matern25", kind: str = "practical", dim: int = 2
) -> float:
    """gstools ``len_scale`` -> range, the inverse of :func:`range_to_len_scale`.

    This is what turns a fitted model back into a number comparable with
    ``truth.yaml``, which quotes practical ranges.
    """
    return float(len_scale) * range_factor(model, kind, dim)


def build_model(cfg: GrfConfig, dim: int = 2, nugget: float = 0.0):
    """A gstools covariance model carrying this package's conventions.

    ``nugget`` is separate from ``cfg`` because the two live in different places:
    the generator's nugget is a property of the *point support*, not of the GRF,
    and an estimator's nugget is fitted rather than configured.
    """
    if cfg.model not in MODELS:
        raise ValueError(f"unknown covariance model {cfg.model!r}; have {sorted(MODELS)}")
    return MODELS[cfg.model](
        dim=dim,
        var=cfg.sill,
        len_scale=range_to_len_scale(cfg, dim),
        anis=1.0 / cfg.aniso_ratio,
        angles=azimuth_to_math_angle(cfg.aniso_angle_deg),
        nugget=nugget,
    )


def model_params(model, name: str = "matern25") -> dict:
    """A fitted gstools model, expressed in ``truth.yaml``'s parameterisation.

    Anisotropy is reported as major/minor with the **major** axis azimuth, which
    is the opposite of gstools' ``anis`` (minor/major) and its mathematical
    angle.  Doing the inversion here rather than at each call site is what keeps
    a recovered axis from coming back 90 degrees out.
    """
    anis = float(model.anis[0]) if len(model.anis) else 1.0
    ratio = 1.0 / anis if anis > 0 else 1.0
    angle = float(model.angles[0]) if len(model.angles) else 0.0
    return {
        "sill": float(model.var),
        "range_km": len_scale_to_range(model.len_scale, name, "practical", model.dim),
        "aniso_ratio": ratio,
        "aniso_angle_deg": math_angle_to_azimuth(angle) if ratio != 1.0 else None,
        "nugget": float(model.nugget),
    }
