"""The dependency direction, pinned.

Synthetic validates, real applies.  The contract and the estimators run on any
data; the generator is a *consumer* of them.  That direction is easy to reverse
by accident — one convenient import of ``Dataset`` from the generator and
loading two CSVs of real data starts pulling in the field-simulation stack —
and nothing else in the suite would notice.  So it is checked here rather than
documented and hoped for.

Each check runs in a fresh interpreter: import side effects are global and
per-process, so asserting on ``sys.modules`` inside the already-loaded test
session would prove nothing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _imports_after(statement: str) -> set:
    """The ``cpt_geostat.*`` and third-party modules loaded by ``statement`` alone."""
    code = textwrap.dedent(
        f"""
        import sys
        {statement}
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in
                        {{"cpt_geostat", "gstools", "matplotlib", "scipy"}})
        print("\\n".join(loaded))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def test_the_contract_does_not_import_the_generator():
    """The real-data entry point must not route through the synthetic side.

    This is the inversion the restructure removed: ``read_dataset`` used to
    import ``Dataset`` from the generator package, so reading a real dataset
    imported the field simulator.
    """
    loaded = _imports_after("import cpt_geostat.contract")
    assert not [m for m in loaded if m.startswith("cpt_geostat.synthetic")]


def test_reading_real_data_does_not_import_gstools():
    """The visible cost of the old inversion, and the cheapest thing to check.

    ``gstools`` is only needed to *simulate* or to *krige*; loading two tables
    of real data needs neither.
    """
    assert "gstools" not in _imports_after("import cpt_geostat.contract")


def test_the_estimators_do_not_import_the_generator():
    """``cpt_geostat.models`` needs gstools, but never truth.

    The truth-covariance reference used to live in ``kriging_factory``, which
    made the estimator layer depend on the truth machinery it is supposed to be
    scored against.
    """
    loaded = _imports_after("import cpt_geostat.models")
    assert not [m for m in loaded if m.startswith("cpt_geostat.synthetic")]


def test_the_data_plots_do_not_import_the_generator():
    """Everything left in ``cpt_geostat.plots`` provably runs on real data."""
    loaded = _imports_after("import cpt_geostat.plots")
    assert not [m for m in loaded if m.startswith("cpt_geostat.synthetic")]


def test_the_generator_may_import_the_contract():
    """The permitted direction — stated so the guard cannot be read as symmetric."""
    loaded = _imports_after("import cpt_geostat.synthetic")
    assert "cpt_geostat.contract.schema" in loaded


@pytest.mark.parametrize(
    "name", ["cpt_geostat.contract", "cpt_geostat.models", "cpt_geostat.plots", "cpt_geostat.synthetic"]
)
def test_every_subpackage_imports_on_its_own(name):
    """No subpackage relies on another having been imported first."""
    subprocess.run([sys.executable, "-c", f"import {name}"], check=True)


def test_preparation_does_not_import_the_generator():
    """``cpt_geostat.prepare`` is a real-data path and must stay one.

    It sits outside ``contract`` precisely so it may import ``cpt_geostat.cpt``
    and its scipy dependency; that licence does not extend to the synthetic
    stack, which would make wrangling a client's export pull in the field
    simulator it is supposed to be independent of.
    """
    loaded = _imports_after("import cpt_geostat.prepare")
    assert not [m for m in loaded if m.startswith("cpt_geostat.synthetic")]
    assert "gstools" not in loaded
