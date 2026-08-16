"""Map plotting and diagnostics (Part B).

Every function here takes a dataframe or a ``Dataset`` and returns a matplotlib
Figure.  No file I/O happens inside a plotting function, and nothing here needs
truth or a generator config — every plot in this package runs on real data.
The truth panels (B1) live in :mod:`cpt_geostat.synthetic.plots`.
"""

from .diagnostics import (
    fit_trend_azimuth,
    plot_depth_traces,
    plot_lag_coverage,
    plot_trend_check,
    plot_within_unit_scatter,
)
from .maps import plot_layout, plot_presence_map, plot_thickness_map, plot_value_map
from .predictions import plot_prediction_vs_truth
from .fields import plot_prediction_map
from .profiles import plot_depth_profile, plot_profile_calibration
from .sections import plot_configured_sections, plot_section, project_to_section
from .variograms import plot_directional_variogram, plot_variogram

__all__ = [
    "fit_trend_azimuth",
    "plot_configured_sections",
    "plot_depth_profile",
    "plot_depth_traces",
    "plot_directional_variogram",
    "plot_profile_calibration",
    "plot_lag_coverage",
    "plot_layout",
    "plot_prediction_map",
    "plot_prediction_vs_truth",
    "plot_presence_map",
    "plot_section",
    "plot_thickness_map",
    "plot_trend_check",
    "plot_value_map",
    "plot_variogram",
    "plot_within_unit_scatter",
    "project_to_section",
]
