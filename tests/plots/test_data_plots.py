"""The B2 plots — every one of them must run on real data.

This is the test that makes the ``cpt_geostat.plots`` / ``cpt_geostat.synthetic.plots``
split more than a naming convention: anything that needs truth belongs in the
other package, and the parametrised sweep below is what says so.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from cpt_geostat import plots
from cpt_geostat.contract import read_dataset


@pytest.mark.parametrize(
    "fn",
    [
        plots.plot_layout,
        plots.plot_presence_map,
        plots.plot_value_map,
        plots.plot_thickness_map,
        plots.plot_trend_check,
        plots.plot_lag_coverage,
        plots.plot_depth_traces,
        plots.plot_within_unit_scatter,
    ],
)
def test_data_diagnostics_run_unchanged_on_real_data(real_data_dir, fn):
    """The B2 plots must not need truth, a config, or a colour palette."""
    ds = read_dataset(real_data_dir)
    fig = fn(ds)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_every_exported_plot_runs_on_real_data(real_data_dir):
    """Nothing may be added to this package that needs the generator.

    The sweep above names its functions; this one catches a new export that
    forgot to add itself to the list.
    """
    ds = read_dataset(real_data_dir)
    # Need extra arguments; each is covered against this same fixture by
    # tests/plots/test_variogram_plots.py and test_profile_plots.py.
    skip = {"plot_prediction_vs_truth", "plot_section", "project_to_section",
            "plot_configured_sections", "fit_trend_azimuth",
            "plot_variogram", "plot_directional_variogram",
            "plot_depth_profile", "plot_profile_calibration",
            "plot_prediction_map"}
    for name in plots.__all__:
        if name in skip:
            continue
        fig = getattr(plots, name)(ds)
        assert isinstance(fig, Figure), name
        matplotlib.pyplot.close(fig)


def test_configured_sections_render(written):
    ds = read_dataset(written)
    figs = plots.plot_configured_sections(ds)
    assert len(figs) == len(ds.config.sections)
    for fig in figs:
        matplotlib.pyplot.close(fig)


def test_sections_are_skipped_rather_than_guessed_without_a_config(real_data_dir):
    """Real data has no configured section lines, and inventing some would be a
    claim about where the interesting geology is."""
    ds = read_dataset(real_data_dir)
    assert plots.plot_configured_sections(ds) == []


def test_section_projection_is_a_rotation(dataset):
    """Chainage/offset must preserve distances — a scaling bug here would shift
    every section without changing its shape."""
    import numpy as np

    sel = plots.project_to_section(dataset.layout, (-7.0, 0.0), (7.0, 0.0), corridor_km=99.0)
    assert np.allclose(sel["chainage_km"], sel["x"] + 7.0)
    assert np.allclose(sel["offset_km"], sel["y"])


def test_section_with_no_nearby_cpts_raises(dataset):
    with pytest.raises(ValueError, match="no CPTs"):
        plots.plot_section(dataset, (-7.0, -7.4), (-6.9, -7.4), corridor_km=0.001)


def test_fitted_trend_recovers_a_strong_one(dataset):
    """unit_2 carries a strong oblique trend; the fit must find its bearing."""
    sub = dataset.unit_summary[dataset.unit_summary["unit_id"] == "unit_2"]
    grad, azimuth = plots.fit_trend_azimuth(sub)
    true = dataset.config.units["unit_2"].property.trend
    assert abs(grad - true.grad) < 0.02
    assert min(abs(azimuth - true.azimuth_deg), 360 - abs(azimuth - true.azimuth_deg)) < 15
