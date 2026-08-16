"""Command-line selection logic.

The CLI is mostly wiring and is exercised end to end by running it, but the
selection helpers decide *what gets written* from a free-text argument, and
their failure mode is silence: a typo'd model name that produces an empty
directory rather than an error looks identical to a run that was never asked
for those figures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cpt_geostat.cli import _profile_models, _safe_name


@pytest.fixture
def cv():
    """A cv table in the order the estimators are meant to be compared in."""
    return pd.DataFrame({
        "model": ["baseline", "SK (fitted)", "OK (fitted)", "UK (fitted)"],
        "cpt_id": ["C1"] * 4,
        "unit_id": ["u1"] * 4,
    })


def test_all_gives_every_estimator_in_the_tables_own_order(cv):
    """Order is the cv table's, not the argument's — increasing in what each
    model assumes about the mean, which is how they are meant to be read."""
    assert _profile_models(cv, "all") == [
        "baseline", "SK (fitted)", "OK (fitted)", "UK (fitted)"
    ]


def test_a_single_name_selects_just_that_one(cv):
    assert _profile_models(cv, "OK (fitted)") == ["OK (fitted)"]


def test_a_comma_list_is_reordered_to_the_tables_order(cv):
    """So `all` and an explicit list produce directories in the same sequence."""
    assert _profile_models(cv, "UK (fitted),baseline") == ["baseline", "UK (fitted)"]


def test_whitespace_around_names_is_tolerated(cv):
    assert _profile_models(cv, " baseline , OK (fitted) ") == ["baseline", "OK (fitted)"]


def test_an_unknown_name_is_reported_and_dropped(cv, capsys):
    """Named but absent must say so.  Silently writing nothing is the failure
    mode this whole file exists for."""
    assert _profile_models(cv, "nope") == []
    assert "nope" in capsys.readouterr().err

    assert _profile_models(cv, "nope,baseline") == ["baseline"]
    assert "nope" in capsys.readouterr().err


def test_model_names_become_safe_directory_names():
    """``SK (fitted)`` is a directory, and the brackets and space cannot survive."""
    assert _safe_name("SK (fitted)") == "SK_fitted"
    assert _safe_name("baseline") == "baseline"
    assert _safe_name("") == "model"
    # distinct estimators must not collide onto one directory
    names = ["baseline", "SK (fitted)", "OK (fitted)", "UK (fitted)"]
    assert len({_safe_name(n) for n in names}) == len(names)
