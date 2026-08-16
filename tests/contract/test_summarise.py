"""Depth-averaging — the one contract function projects also call.

``projects/IJmuiden/prepare_data.py`` used to reimplement this rather than
import it, because importing it meant importing the generator (and gstools)
through ``cpt_geostat.generate.series``.  It now imports the real thing; these tests
cover what the duplicate never had — agreement on the convention itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpt_geostat.contract import SUMMARY_COLUMNS, summarise_units


@pytest.fixture
def tiny():
    """Two holes, one unit each, with hand-checkable values."""
    samples = pd.DataFrame(
        {
            "cpt_id": ["A", "A", "A", "B", "B"],
            "x": [0.0, 0.0, 0.0, 1.0, 1.0],
            "y": [0.0, 0.0, 0.0, 2.0, 2.0],
            "z": [1.0, 2.0, 3.0, 1.0, 2.0],
            "unit_id": ["u1", "u1", "u1", "u1", "u1"],
            "Qtn": [np.e, np.e**2, np.e**3, np.e, np.e**3],
        }
    )
    layers = pd.DataFrame(
        {
            "cpt_id": ["A", "B"],
            "unit_id": ["u1", "u1"],
            "z_top": [0.5, 0.5],
            "z_bot": [3.5, 2.5],
            "thickness_m": [3.0, 2.0],
        }
    )
    layout = pd.DataFrame({"cpt_id": ["A", "B"], "x": [0.0, 1.0], "y": [0.0, 2.0]})
    return samples, layers, layout


def test_averages_in_log_space_not_linear_space(tiny):
    """The convention: natural log first, then average.

    Averaging Qtn and taking the log afterwards would give 1.55 rather than 2.0
    for hole A — a bias that grows with the scatter and always upwards.
    """
    out = summarise_units(*tiny).set_index("cpt_id")
    assert out.loc["A", "log_Q_mean"] == pytest.approx(2.0)
    assert out.loc["B", "log_Q_mean"] == pytest.approx(2.0)


def test_sd_is_over_readings_with_the_pandas_default_ddof(tiny):
    out = summarise_units(*tiny).set_index("cpt_id")
    assert out.loc["A", "log_Q_sd"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
    assert out.loc["A", "n_samples"] == 3


def test_thickness_comes_from_layers_not_from_the_reading_span(tiny):
    """The layer is thicker than the readings it holds, and the contract says
    so: thickness is logged geometry, not a range of samples."""
    out = summarise_units(*tiny).set_index("cpt_id")
    assert out.loc["A", "thickness_m"] == pytest.approx(3.0)
    assert out.loc["B", "thickness_m"] == pytest.approx(2.0)


def test_columns_are_exactly_the_contract(tiny):
    assert list(summarise_units(*tiny).columns) == SUMMARY_COLUMNS


def test_agrees_with_the_generator_on_a_full_dataset(dataset):
    """The real-data path recomputes this; it must agree with the generated one."""
    again = summarise_units(dataset.samples, dataset.layers, dataset.layout)
    pd.testing.assert_frame_equal(
        dataset.unit_summary.sort_values(["cpt_id", "unit_id"]).reset_index(drop=True),
        again.sort_values(["cpt_id", "unit_id"]).reset_index(drop=True),
    )
