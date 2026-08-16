"""Evaluating a fitted estimator over a grid — the model's answer as a map.

The estimators are otherwise only ever seen as per-CPT numbers: a cross plot, a
CV score, a depth profile.  This is what turns one into the thing a ground model
is actually for — a surface — and the companion to
:func:`cpt_geostat.synthetic.plots.plot_unit_truth_panel`, which maps the truth the
same way.

Fitted **in-sample**, on every CPT holding the unit.  That is deliberate and it
is not the same job as cross-validation: this is the model's best statement
about the ground given everything known, whereas
:mod:`cpt_geostat.models.profile` and :mod:`cpt_geostat.validate.cv` are where it is held to
account on data it was not shown.  Using leave-one-out predictions here would
answer a question nobody asked of a map.

**Nothing is masked.**  A unit held at 4 of 194 holes still gets a surface across
the whole site, because presence is not modelled yet — that needs the classifier
plan 02 defers.  The honest counterweight is the sd field: it grows away from the
data, so the places where the mean map is inventing are exactly the places the sd
map is brightest.  Read the pair, never the mean alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..contract.schema import Raster

#: Default grid spacing for a real site, in km.
DEFAULT_RES_KM = 0.15

#: Margin added around the CPT extent so the surface does not stop at the
#: outermost hole — the edge is where extrapolation is most worth seeing.
DEFAULT_PAD_KM = 0.5


@dataclass
class FieldPrediction:
    """A fitted estimator evaluated over a raster, for one unit."""

    unit_id: str
    mean: np.ndarray  # (ny, nx), nan where the unit could not be fitted
    sd: np.ndarray  # (ny, nx), the *latent* sd — see cpt_geostat.models.base
    n_cpt: int
    n_cpt_total: int
    note: Optional[str] = None  # why it is empty, when it is

    @property
    def fitted(self) -> bool:
        return self.note is None

    @property
    def cpt_fraction(self) -> float:
        return self.n_cpt / self.n_cpt_total if self.n_cpt_total else float("nan")


def field_raster(
    ds, res_km: float = DEFAULT_RES_KM, pad_km: float = DEFAULT_PAD_KM
) -> Raster:
    """The grid to predict on.

    **Returns the dataset's own raster unchanged when it has one.**  On a
    synthetic run that makes the prediction map pixel-aligned with the truth
    map, so the two can be compared by flipping between figures rather than by
    eye across different grids — which is the entire point of drawing them in
    the same style.

    Real data has no raster, so one is built from the CPT extent plus a margin.
    """
    existing = getattr(ds, "raster", None)
    if existing is not None:
        return existing

    x = ds.layout["x"].to_numpy(dtype=float)
    y = ds.layout["y"].to_numpy(dtype=float)
    return Raster.from_bounds(
        float(x.min()) - pad_km, float(x.max()) + pad_km,
        float(y.min()) - pad_km, float(y.max()) + pad_km,
        res_km,
    )


def predict_field(
    ds,
    unit_id: str,
    factory: Callable[[str], object],
    raster: Optional[Raster] = None,
) -> FieldPrediction:
    """Fit ``factory(unit_id)`` on this unit's CPTs and evaluate it on ``raster``.

    ``factory`` is the same ``unit_id -> unfitted estimator`` callable the
    cross-validation harness takes, so
    :func:`~cpt_geostat.models.baseline.baseline_factory` and
    :func:`~cpt_geostat.models.kriging.kriging_factory` both work unchanged and a new
    estimator needs no adapter here.

    ``sd`` is the **latent** sd — uncertainty about the field itself, per
    :mod:`cpt_geostat.models.base`.  That is the right one for a map: observation
    noise is a property of taking a measurement, not of the ground, and adding
    it would make every model's map look uniformly and misleadingly wide.

    A unit too thin to fit comes back as an all-``nan`` field carrying the
    reason, rather than raising — one unfittable unit out of 23 must not take
    the whole figure down.
    """
    raster = raster if raster is not None else field_raster(ds)
    block = ds.unit_summary[ds.unit_summary["unit_id"] == unit_id]
    n_total = int(ds.unit_summary["cpt_id"].nunique())
    empty = np.full(raster.shape, np.nan)

    if not len(block):
        return FieldPrediction(unit_id, empty, empty.copy(), 0, n_total,
                               note="unit absent from unit_summary")

    X = block[["x", "y"]].to_numpy(dtype=float)
    y = block["log_Q_mean"].to_numpy(dtype=float)
    try:
        est = factory(unit_id).fit(X, y)
    except ValueError as exc:
        # Too few CPTs for this estimator's mean model, or no finite values.
        return FieldPrediction(unit_id, empty, empty.copy(), len(block), n_total,
                               note=f"not fitted — {exc}")

    XX, YY = raster.meshgrid()
    points = np.column_stack([XX.ravel(), YY.ravel()])
    mean, sd = est.predict(points, return_std=True)
    return FieldPrediction(
        unit_id=unit_id,
        mean=np.asarray(mean, dtype=float).reshape(raster.shape),
        sd=np.asarray(sd, dtype=float).reshape(raster.shape),
        n_cpt=len(block),
        n_cpt_total=n_total,
    )


def predict_fields(ds, factory, raster=None, unit_ids=None) -> dict:
    """``{unit_id: FieldPrediction}`` for every unit, in the dataset's order."""
    raster = raster if raster is not None else field_raster(ds)
    unit_ids = list(unit_ids or ds.unit_ids)
    return {uid: predict_field(ds, uid, factory, raster) for uid in unit_ids}
