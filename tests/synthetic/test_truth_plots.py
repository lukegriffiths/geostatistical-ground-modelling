"""B1 — the truth panels.  Synthetic only, and they say so on real data."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from cpt_geostat.contract import read_dataset
from cpt_geostat.synthetic.plots import (
    TruthUnavailable,
    plot_anisotropy_check,
    plot_unit_truth_panel,
)


def test_truth_panels_render_for_every_unit(written):
    ds = read_dataset(written)
    for uid in ds.unit_ids:
        fig = plot_unit_truth_panel(ds, uid)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)


def test_anisotropy_check_renders(written):
    ds = read_dataset(written)
    fig = plot_anisotropy_check(ds)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_truth_diagnostics_refuse_real_data_clearly(real_data_dir):
    """Raised, not skipped: asking for a truth panel on real data is a mistake
    about what the data is, and a blank figure would hide it."""
    ds = read_dataset(real_data_dir)
    with pytest.raises(TruthUnavailable):
        plot_anisotropy_check(ds)
    with pytest.raises(TruthUnavailable):
        plot_unit_truth_panel(ds, "unit_1")
