"""Prediction field maps.

In ``cpt_geostat.plots``, so the binding constraint is real data — no truth, no
config, no palette.  Beyond that the thing worth guarding is that an unfittable
unit degrades to an annotated blank panel rather than taking the figure down,
because on IJmuiden that will happen for some unit on most runs.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from cpt_geostat.contract import read_dataset
from cpt_geostat.models import baseline_factory, kriging_factory
from cpt_geostat.plots import plot_prediction_map

#: Columns of the combined figure: predicted, latent sd, lower, upper.
N_COLUMNS = 4


@pytest.fixture(scope="module")
def real(real_data_dir):
    return read_dataset(real_data_dir)


@pytest.fixture(scope="module")
def coarse(real):
    """A cheap grid — these tests are about rendering, not resolution."""
    from cpt_geostat.models.field import field_raster

    return field_raster(real, res_km=1.5)


# --------------------------------------------------------------------------- #
# the Part B contract
# --------------------------------------------------------------------------- #


def test_renders_on_real_data_without_truth_or_config(real, coarse):
    fig = plot_prediction_map(real, kriging_factory(real, method="OK"), "OK",
                              raster=coarse)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_renders_for_the_baseline_whose_field_is_flat(real, coarse):
    """A flat field must not break the colour scaling — vmin == vmax otherwise."""
    fig = plot_prediction_map(real, baseline_factory(real), "baseline", raster=coarse)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_each_unit_gets_four_columns(dataset):
    """predicted / latent sd / lower / upper, on one row per unit."""
    units = ["unit_1", "unit_2"]
    fig = plot_prediction_map(dataset, baseline_factory(dataset), "baseline",
                              unit_ids=units)
    maps = [ax for ax in fig.axes if ax.images]
    assert len(maps) == N_COLUMNS * len(units)
    headers = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert any("predicted" in h for h in headers)
    assert any("lower" in h for h in headers) and any("upper" in h for h in headers)
    matplotlib.pyplot.close(fig)


def test_the_bounds_bracket_the_prediction_on_a_shared_scale(dataset):
    """The three value columns must share limits, or a wide interval would look
    identical to a narrow one — which is the whole reason to draw them side by
    side.  The sd column is a different quantity and keeps its own."""
    from cpt_geostat.models.field import field_raster, predict_fields
    from cpt_geostat.plots.fields import shared_colour_limits

    r = field_raster(dataset)
    by_model = {"OK": predict_fields(dataset, kriging_factory(dataset, method="OK"), r,
                                     unit_ids=["unit_1"])}
    lim = shared_colour_limits(dataset, by_model, level=0.95)
    pred = by_model["OK"]["unit_1"]
    z = 1.959963984540054
    assert lim["value"]["unit_1"][0] <= np.nanmin(pred.mean - z * pred.sd) + 1e-9
    assert lim["value"]["unit_1"][1] >= np.nanmax(pred.mean + z * pred.sd) - 1e-9
    assert lim["sd"]["unit_1"] != lim["value"]["unit_1"]


def test_renders_on_synthetic_using_the_truth_grid(dataset):
    """On synthetic the map shares the truth raster, so the two figures can be
    compared by flipping between them."""
    fig = plot_prediction_map(dataset, baseline_factory(dataset), "baseline",
                              unit_ids=["unit_1", "unit_2"])
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# degradation and guards
# --------------------------------------------------------------------------- #


def test_an_unfittable_unit_becomes_an_annotated_panel_not_an_exception(dataset, coarse):
    import pandas as pd

    one = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"].head(1)
    bare = type("Bare", (), {"unit_summary": pd.concat([one]), "unit_ids": ["unit_1"],
                             "raster": None, "config": None, "layout": dataset.layout})()
    fig = plot_prediction_map(bare, kriging_factory(bare, method="OK"), "OK", raster=coarse)
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "not fitted" in texts
    matplotlib.pyplot.close(fig)


def test_a_constant_sd_field_never_shows_a_negative_standard_deviation(real, coarse):
    """A unit whose variogram fits pure nugget gets a flat sd field.

    Padding it symmetrically to give matplotlib a range put the colourbar below
    zero — not a value an sd can take, and it read as a broken figure rather
    than as "kriging found no structure here".  Five of IJmuiden's 23 units hit
    this, so it is the common case, not an edge one.
    """
    from cpt_geostat.plots.fields import _colour_limits

    lo, hi, uniform = _colour_limits(0.084, 0.084, "sd")
    assert uniform and lo >= 0.0 and hi > lo
    # a mean may legitimately be negative, so it is padded symmetrically
    lo_m, hi_m, uniform_m = _colour_limits(-1.5, -1.5, "mean")
    assert uniform_m and lo_m < -1.5 < hi_m


def test_a_uniform_panel_says_why_it_is_uniform(real, coarse):
    fig = plot_prediction_map(real, baseline_factory(real), "baseline",
                              raster=coarse, unit_ids=list(real.unit_ids)[:1])
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "uniform" in texts
    matplotlib.pyplot.close(fig)


def test_shared_limits_span_every_model_so_colours_are_comparable(dataset):
    """The point of drawing the models alike is flipping between their figures.

    Auto-scaling each to its own surface makes the same colour mean a different
    value in each, which quietly defeats that.
    """
    from cpt_geostat.models.field import field_raster, predict_fields
    from cpt_geostat.plots.fields import shared_colour_limits

    r = field_raster(dataset)
    by_model = {
        "baseline": predict_fields(dataset, baseline_factory(dataset), r, unit_ids=["unit_1"]),
        "OK": predict_fields(dataset, kriging_factory(dataset, method="OK"), r,
                             unit_ids=["unit_1"]),
    }
    lim = shared_colour_limits(dataset, by_model)["sd"]["unit_1"]
    for per_unit in by_model.values():
        sd = per_unit["unit_1"].sd
        assert lim[0] <= np.nanmin(sd) and np.nanmax(sd) <= lim[1]

    # and the union is strictly wider than the flat baseline's own range,
    # which is what makes its claimed precision visible next to kriging's
    flat = by_model["baseline"]["unit_1"].sd
    assert lim[1] - lim[0] > np.ptp(flat) + 1e-9


def test_shared_limits_are_per_unit_not_across_units(dataset):
    """Units differ in level by design — unit 2 sits near 1.0 and unit 6 near
    2.4 — so one scale for all of them would flatten every panel."""
    from cpt_geostat.models.field import field_raster, predict_fields
    from cpt_geostat.plots.fields import shared_colour_limits

    r = field_raster(dataset)
    by_model = {"baseline": predict_fields(dataset, baseline_factory(dataset), r)}
    lim = shared_colour_limits(dataset, by_model)["value"]
    assert len(lim) > 1
    assert lim["unit_2"] != lim["unit_6"]


def test_precomputed_fields_avoid_refitting(dataset):
    """A caller drawing four models must not evaluate each one twice."""
    from cpt_geostat.models.field import field_raster, predict_fields

    r = field_raster(dataset)
    fields = predict_fields(dataset, baseline_factory(dataset), r, unit_ids=["unit_1"])
    fig = plot_prediction_map(dataset, model_name="baseline", raster=r,
                              unit_ids=["unit_1"], fields=fields)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_neither_a_factory_nor_fields_is_an_error(dataset):
    with pytest.raises(ValueError, match="factory|fields"):
        plot_prediction_map(dataset, model_name="x", unit_ids=["unit_1"])


def test_only_the_requested_units_are_drawn(real, coarse):
    units = list(real.unit_ids)[:2]
    fig = plot_prediction_map(real, baseline_factory(real), "baseline",
                              raster=coarse, unit_ids=units)
    # Units label their row; the titles are the four column headers.
    labels = [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()]
    for u in units:
        assert any(u in la for la in labels), u
    assert len([ax for ax in fig.axes if ax.images]) == N_COLUMNS * len(units)
    matplotlib.pyplot.close(fig)


def test_the_panel_states_how_many_cpts_hold_the_unit(dataset):
    """An unmasked map over-claims for a sparse unit, so the count has to be on
    the figure rather than in a csv somewhere."""
    fig = plot_prediction_map(dataset, baseline_factory(dataset), "baseline",
                              unit_ids=["unit_3"])
    labels = " ".join(ax.get_ylabel() for ax in fig.axes if ax.get_ylabel())
    assert "CPTs" in labels and "%" in labels
    matplotlib.pyplot.close(fig)
