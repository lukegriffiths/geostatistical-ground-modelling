"""Variogram figures.

These live in ``cpt_geostat.plots``, so the binding constraint is that they render on
real data — no truth, no config, no palette.  The rest of the file covers the
cases that are *supposed* to look wrong: a unit whose range ran to the bound and
one that is nearly pure nugget still have to produce a readable figure, because
those are exactly the units a reader most needs to look at.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from cpt_geostat.contract import read_dataset
from cpt_geostat.plots import plot_directional_variogram, plot_variogram

FIGURES = (plot_variogram, plot_directional_variogram)


@pytest.fixture(scope="module")
def real(real_data_dir):
    return read_dataset(real_data_dir)


# --------------------------------------------------------------------------- #
# the Part B contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", FIGURES)
def test_renders_on_real_data_without_truth_or_config(real, fn):
    fig = fn(real, "unit_1")
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("fn", FIGURES)
def test_renders_on_synthetic_data_too(dataset, fn):
    fig = fn(dataset, "unit_1")
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_the_truth_overlay_appears_only_when_a_config_is_present(dataset, real):
    """The optional-truth pattern: same function, more information when the
    generator's parameters happen to be available, and no crash when not."""
    labels = lambda fig: {  # noqa: E731
        t.get_label() for t in fig.axes[0].get_lines() + fig.axes[0].collections
    }
    with_truth = plot_variogram(dataset, "unit_1")
    without = plot_variogram(real, "unit_1")
    assert any("true" in str(la) for la in labels(with_truth))
    assert not any("true" in str(la) for la in labels(without))
    matplotlib.pyplot.close("all")


# --------------------------------------------------------------------------- #
# the units that are supposed to look wrong
# --------------------------------------------------------------------------- #


def test_a_unit_whose_range_ran_to_the_bound_still_renders(dataset):
    """unit_2's trend makes its variogram unbounded.  The figure exists to show
    that, so it must not be the case that it fails to draw."""
    from cpt_geostat.models.variogram import fit_unit_variogram

    assert fit_unit_variogram(dataset, "unit_2").at_range_bound  # the precondition
    fig = plot_variogram(dataset, "unit_2")
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_a_pure_nugget_unit_renders_despite_a_zero_sill(dataset):
    """Sill 0 makes the model curve flat and the practical range meaningless;
    neither may produce a division by zero or an empty axes."""
    from cpt_geostat.models.variogram import fit_unit_variogram

    worst = min(
        (u for u in dataset.unit_ids if fit_unit_variogram(dataset, u) is not None),
        key=lambda u: fit_unit_variogram(dataset, u).structured_fraction,
    )
    fig = plot_variogram(dataset, worst)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_an_anisotropic_unit_draws_both_truth_axes_as_a_band(dataset):
    """An anisotropic model has no single variogram, so the omnidirectional
    empirical estimate is a mixture and must be shown against both axis curves
    rather than against one that it is not expected to match."""
    fig = plot_variogram(dataset, "unit_3")  # ratio 4 at 70 deg
    labels = [ln.get_label() for ln in fig.axes[0].get_lines()]
    assert any("major axis" in la for la in labels)
    assert any("minor axis" in la for la in labels)
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# refusal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", FIGURES)
def test_a_unit_too_thin_to_fit_raises_rather_than_drawing_a_blank(dataset, fn):
    """A figure of nothing invites the reader to conclude something.  The
    baseline is the estimator for these units and the error says so."""
    import pandas as pd

    thin = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_1"].head(2)
    bare = type("Bare", (), {"unit_summary": pd.concat([thin]), "config": None,
                             "unit_ids": ["unit_1"]})()
    with pytest.raises(ValueError, match="too few"):
        fn(bare, "unit_1")


def test_the_directional_figure_labels_sectors_with_their_pair_counts(dataset):
    """A sector the layout cannot support is a fact about the survey; the count
    on the legend is what stops it being read as a measurement."""
    fig = plot_directional_variogram(dataset, "unit_1")
    labels = [ln.get_label() for ln in fig.axes[0].get_lines()]
    assert sum("pairs" in la for la in labels) >= 4
    matplotlib.pyplot.close(fig)
