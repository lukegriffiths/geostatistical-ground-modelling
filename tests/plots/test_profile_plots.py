"""Depth-profile figures.

They live in ``cpt_geostat.plots``, so the binding constraint is that they render on
real data.  The case worth guarding beyond that is a run the model could not
predict: the trace must still be drawn and the band must **not** be, because a
band bridging an unpredicted interval is an invention.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from cpt_geostat.contract import read_dataset
from cpt_geostat.models import kriging_factory
from cpt_geostat.cpt import StressProfile
from cpt_geostat.models.profile import qt_readings, reading_predictions
from cpt_geostat.plots import plot_depth_profile, plot_profile_calibration
from cpt_geostat.validate import loo_by_unit


@pytest.fixture(scope="module")
def real(real_data_dir):
    return read_dataset(real_data_dir)


@pytest.fixture(scope="module")
def real_readings(real):
    cv = loo_by_unit(real, kriging_factory(real, method="OK"), "OK")
    return reading_predictions(real, cv)


@pytest.fixture(scope="module")
def readings(dataset):
    cv = loo_by_unit(dataset, kriging_factory(dataset, method="OK"), "OK")
    return reading_predictions(dataset, cv)


# --------------------------------------------------------------------------- #
# the Part B contract
# --------------------------------------------------------------------------- #


def test_profile_renders_on_real_data_without_truth_or_config(real, real_readings):
    cid = real_readings["cpt_id"].iloc[0]
    fig = plot_depth_profile(real, real_readings, cid)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_calibration_renders_on_real_data(real, real_readings):
    fig = plot_profile_calibration(real, real_readings)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_profile_renders_on_synthetic(dataset, readings):
    fig = plot_depth_profile(dataset, readings, readings["cpt_id"].iloc[0])
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# gaps
# --------------------------------------------------------------------------- #


def test_an_unpredicted_run_draws_no_band(dataset):
    """The trace survives, the band does not.

    Counted rather than eyeballed: one fewer band than there are units in the
    hole, for the unit whose predictions were removed.
    """
    cv = loo_by_unit(dataset, kriging_factory(dataset, method="OK"), "OK")
    full = reading_predictions(dataset, cv)
    holed = reading_predictions(dataset, cv[cv["unit_id"] != "unit_1"])

    cid = next(
        c for c, g in full.groupby("cpt_id")
        if "unit_1" in set(g["unit_id"]) and g["unit_id"].nunique() > 1
    )
    fig_full = plot_depth_profile(dataset, full, cid)
    fig_hole = plot_depth_profile(dataset, holed, cid)

    bands = lambda f: len(f.axes[0].collections)  # noqa: E731 — fill_betweenx
    assert bands(fig_hole) < bands(fig_full)
    # the measured trace is still there in both
    assert any(len(ln.get_xdata()) > 10 for ln in fig_hole.axes[0].get_lines())
    matplotlib.pyplot.close("all")


def test_a_hole_with_no_predictions_at_all_still_renders(dataset):
    cv = loo_by_unit(dataset, kriging_factory(dataset, method="OK"), "OK")
    none = reading_predictions(dataset, cv.iloc[0:0])
    fig = plot_depth_profile(dataset, none, none["cpt_id"].iloc[0])
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# framing and refusal
# --------------------------------------------------------------------------- #


def test_the_depth_axis_is_framed_on_the_sampled_interval(real, real_readings):
    """Many IJmuiden boreholes only log below 30 m; anchoring at the seabed
    would spend half the figure on ground the hole never saw."""
    deep = (
        real_readings.groupby("cpt_id")["z"].min().idxmax()
    )
    g = real_readings[real_readings["cpt_id"] == deep]
    fig = plot_depth_profile(real, real_readings, deep)
    bottom, top = fig.axes[0].get_ylim()  # inverted axis
    if g["z"].min() > 2.0:
        assert top > 0.0
    assert bottom >= g["z"].max()
    matplotlib.pyplot.close(fig)


def test_an_unknown_cpt_id_raises(dataset, readings):
    with pytest.raises(ValueError, match="no readings"):
        plot_depth_profile(dataset, readings, "not-a-hole")


def test_the_bands_are_nested_on_the_figure(dataset, readings):
    """The reading band must be drawn wider than the unit-mean band; drawn the
    other way round the figure would say the opposite of the maths."""
    cid = readings["cpt_id"].iloc[0]
    g = readings[(readings["cpt_id"] == cid) & readings["pred"].notna()]
    row = g.iloc[0]
    assert row["Qtn_lo_read"] < row["Qtn_lo_mean"]
    assert row["Qtn_hi_read"] > row["Qtn_hi_mean"]
    fig = plot_depth_profile(dataset, readings, cid)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# the qt suite
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_qt(real_readings):
    """The same predictions in qt, under IJmuiden's fitted stress profile."""
    return qt_readings(real_readings, profile=StressProfile.from_gradients(20.1, 10.0), n=1.0)


def test_the_qt_suite_renders_on_real_data(real, real_qt):
    fig = plot_depth_profile(real, real_qt, real_qt["cpt_id"].iloc[0], units="qt")
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_each_unit_system_gets_the_axis_it_is_legible_on(real, real_readings, real_qt):
    """Log for Qtn, linear MPa for qt — the reason these are two suites.

    A linear Qtn axis would crush a quantity spanning two decades; a log qt axis
    would not be a CPT log.  The qt axis also starts at zero, as a CPT log does.
    """
    cid = real_qt["cpt_id"].iloc[0]
    fig_q = plot_depth_profile(real, real_readings, cid, units="Qtn")
    fig_t = plot_depth_profile(real, real_qt, cid, units="qt")

    assert fig_q.axes[0].get_xscale() == "log"
    assert fig_t.axes[0].get_xscale() == "linear"
    assert fig_t.axes[0].get_xlim()[0] == 0.0
    matplotlib.pyplot.close("all")


def test_the_qt_trace_is_drawn_in_megapascals(real, real_qt):
    """kPa in the tables, MPa on the figure — the unit a CPT log is read in."""
    cid = real_qt["cpt_id"].iloc[0]
    g = real_qt[real_qt["cpt_id"] == cid]
    fig = plot_depth_profile(real, real_qt, cid, units="qt")
    # The trace is the one line carrying every reading; the median lines are
    # drawn per run and carry the run plus its two boundaries.
    traces = [ln for ln in fig.axes[0].get_lines() if len(ln.get_xdata()) == len(g)]
    assert len(traces) == 1
    drawn = float(np.nanmax(traces[0].get_xdata()))
    assert drawn == pytest.approx(g["qt"].max() / 1000.0, rel=1e-9)
    matplotlib.pyplot.close(fig)


def test_qt_is_framed_on_the_data_not_on_the_band(real, real_qt):
    """The band is several times the width of the data in qt.

    Framing on it squeezes the trace — the thing being compared against — into
    the left quarter of the axis, so the frame follows the trace and the median
    and the band is allowed to overflow.
    """
    cid = real_qt["cpt_id"].iloc[0]
    g = real_qt[real_qt["cpt_id"] == cid]
    fig = plot_depth_profile(real, real_qt, cid, units="qt")
    upper = fig.axes[0].get_xlim()[1]
    assert upper < g["qt_hi_read"].max() / 1000.0
    assert upper >= g["qt"].max() / 1000.0
    matplotlib.pyplot.close(fig)


def test_asking_for_qt_without_converting_first_says_so(real, real_readings):
    """The failure a user will actually hit, with the fix in the message."""
    with pytest.raises(ValueError, match="qt_readings"):
        plot_depth_profile(real, real_readings, real_readings["cpt_id"].iloc[0], units="qt")


def test_an_unknown_unit_system_raises(real, real_readings):
    with pytest.raises(ValueError, match="unknown units"):
        plot_depth_profile(real, real_readings, real_readings["cpt_id"].iloc[0], units="MPa")
