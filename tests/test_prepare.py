"""The shared preparation pipeline — the rules that are wrong by default.

This code used to live in ``projects/IJmuiden/prepare_data.py`` with no tests at
all, which was tolerable while there was one project and intolerable the moment
there were two: the layer-geometry rules are subtle in a way that is *invisible
when wrong*.  A contact placed at a reading rather than between two makes every
layer short by one sample interval and looks entirely plausible in the output; a
label dropped before run detection silently fuses the units either side of it.

So each test below pins a rule against the plausible-looking wrong answer, not
merely against a regression.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cpt_geostat.cpt import StressProfile
from cpt_geostat.prepare import (
    PrepareConfig,
    build_intervals,
    collapse_intervals,
    normalise_raw,
    prepare,
    report,
    write,
)

DZ = 0.5  # exactly representable, so midpoints are exact and assertions can be


def _raw(holes, value=100.0, easting=1000.0, northing=2000.0) -> pd.DataFrame:
    """A raw frame from ``{cpt_id: [(unit_id, n_readings), ...]}``, top down.

    Readings are ``DZ`` apart starting at ``DZ``, so the first contact is never
    at zero and the seabed clamp is exercised rather than assumed.
    """
    rows = []
    for i, (cpt_id, runs) in enumerate(holes.items()):
        depth = DZ
        for unit_id, n in runs:
            for _ in range(n):
                rows.append({
                    "cpt_id": cpt_id,
                    "easting": easting + 500.0 * i,
                    "northing": northing + 500.0 * i,
                    "z": depth,
                    "unit_id": unit_id,
                    "value": value,
                    "survey_year": 2020 + i,
                })
                depth += DZ
    return pd.DataFrame(rows)


def _config(**kw) -> PrepareConfig:
    """Filters off unless a test is about filters, so geometry stays unconfounded."""
    base = dict(crs="TEST / metres", min_thickness_m=0.0, min_samples=0)
    return PrepareConfig(**{**base, **kw})


# --------------------------------------------------------------------------
# the raw frame contract
# --------------------------------------------------------------------------

def test_a_missing_column_names_itself_and_the_target():
    """The error has to be actionable: a project hits this while writing a
    rename map, so it must say which name was expected."""
    raw = _raw({"H1": [("A", 4)]}).drop(columns=["unit_id"])
    with pytest.raises(ValueError, match="unit_id"):
        normalise_raw(raw)


def test_normalise_raw_sorts_so_a_project_cannot_forget_to():
    """Run detection uses ``shift()``.  An unsorted frame does not raise — it
    produces a *different and wrong* layer geometry, which is why the sort is
    done here rather than documented as a caller's duty."""
    raw = _raw({"H1": [("A", 4), ("B", 4)]})
    shuffled = raw.sample(frac=1.0, random_state=0).reset_index(drop=True)

    from_sorted = build_intervals(normalise_raw(raw))
    from_shuffled = build_intervals(normalise_raw(shuffled))
    pd.testing.assert_frame_equal(from_sorted, from_shuffled)

    # and the shuffle really was disruptive, so the test is not vacuous
    assert not shuffled["z"].is_monotonic_increasing


def test_the_rename_map_is_applied_before_the_column_check():
    raw = _raw({"H1": [("A", 4)]}).rename(columns={"unit_id": "soil_unit"})
    out = normalise_raw(raw, rename={"soil_unit": "unit_id"})
    assert "unit_id" in out.columns


# --------------------------------------------------------------------------
# contacts
# --------------------------------------------------------------------------

def test_a_contact_sits_between_two_readings_not_on_one():
    """The rule that costs a sample interval per layer when it is wrong.

    Readings at 2.0 (last A) and 2.5 (first B) put the contact at 2.25.  Taking
    either reading instead loses DZ from one layer and gives it to the other —
    a 0.5 m error per contact, invisible in any plot.
    """
    intervals = build_intervals(normalise_raw(_raw({"H1": [("A", 4), ("B", 4)]})))
    a, b = intervals.iloc[0], intervals.iloc[1]

    assert a["z_bot"] == pytest.approx(2.25)
    assert b["z_top"] == pytest.approx(2.25)
    assert a["z_bot"] == b["z_top"], "tops and bases must meet exactly, leaving no void"


def test_the_first_layer_starts_at_its_first_reading_not_at_the_seabed():
    """Unlogged ground above the first reading stays unassigned, deliberately.

    Many holes only start logging well below the datum — IJmuiden has some
    beginning at 30 m — and stretching the shallowest unit up to zero would
    invent material nobody measured.  The clamp below is a guard against a
    reading *above* the datum, not an anchor to it.
    """
    intervals = build_intervals(normalise_raw(_raw({"H1": [("A", 4)]})))
    assert intervals.iloc[0]["z_top"] == pytest.approx(DZ)


def test_a_reading_above_the_datum_is_clamped_rather_than_given_a_negative_top():
    raw = _raw({"H1": [("A", 4)]})
    raw["z"] = raw["z"] - 1.0  # first reading now at -0.5 m
    intervals = build_intervals(normalise_raw(raw))
    assert intervals.iloc[0]["z_top"] == 0.0


def test_a_dropped_label_leaves_a_gap_rather_than_a_false_contact():
    """Why run detection must precede filtering.

    ``A D B`` with ``D`` unclassified must leave a void where ``D`` was.  If the
    label were dropped first, the runs would read ``A B`` and the two real units
    would be fused across a contact that does not exist.
    """
    raw = _raw({"H1": [("A", 4), ("D", 4), ("B", 4)]})
    tables, _ = prepare(raw, _config(drop_units=("D",)))
    layers = tables["layers"].set_index("unit_id")

    assert set(layers.index) == {"A", "B"}
    assert layers.loc["A", "z_bot"] < layers.loc["B", "z_top"], "the void must survive"
    # the fused answer, stated so the assertion above cannot be read as loose
    assert layers.loc["A", "z_bot"] != layers.loc["B", "z_top"]


# --------------------------------------------------------------------------
# repeated units
# --------------------------------------------------------------------------

def test_a_repeated_unit_collapses_to_one_row_summing_its_runs():
    """``thickness_m`` is occupancy, not span.

    ``A B A`` spans the whole hole but occupies only its two ``A`` runs.
    Crediting it with the span would hand it the intervening ``B`` — and
    thickness feeds the depth-average weighting, so the error propagates.
    """
    raw = _raw({"H1": [("A", 4), ("B", 4), ("A", 4)]})
    layers = collapse_intervals(
        build_intervals(normalise_raw(raw))
    ).set_index("unit_id")

    a = layers.loc["A"]
    span = a["z_bot"] - a["z_top"]
    assert a["n_intervals"] == 2
    assert a["thickness_m"] < span
    assert a["thickness_m"] == pytest.approx(layers.loc["B", "thickness_m"] * 2, rel=0.3)


def test_the_per_run_detail_survives_in_intervals():
    """``layers`` is collapsed for the contract's merge key; anything needing
    the individual occurrences must still be able to get them."""
    raw = _raw({"H1": [("A", 4), ("B", 4), ("A", 4)]})
    tables, _ = prepare(raw, _config())
    assert (tables["intervals"]["unit_id"] == "A").sum() == 2
    assert (tables["layers"]["unit_id"] == "A").sum() == 1


# --------------------------------------------------------------------------
# values
# --------------------------------------------------------------------------

def test_unusable_readings_leave_the_samples_but_still_count_as_ground():
    """A unit is logged whether or not its resistance reduced successfully.

    Dropping the readings from the layer geometry as well would shorten the
    layer — the material is there, the measurement is not.
    """
    raw = _raw({"H1": [("A", 4), ("B", 4)]})
    raw.loc[raw["unit_id"] == "A", "value"] = [100.0, np.nan, 0.0, 100.0]

    tables, _ = prepare(raw, _config())
    a_layer = tables["layers"].set_index("unit_id").loc["A"]
    assert a_layer["z_bot"] == pytest.approx(2.25), "geometry unaffected by bad values"
    assert (tables["cpt_samples"]["unit_id"] == "A").sum() == 2, "NaN and zero dropped"


def test_a_non_positive_value_is_dropped_not_passed_to_the_log():
    """``log(0)`` is ``-inf`` and would poison the depth-average silently."""
    raw = _raw({"H1": [("A", 4)]})
    raw.loc[0, "value"] = 0.0
    tables, _ = prepare(raw, _config())
    assert np.isfinite(tables["unit_summary"]["log_Q_mean"]).all()


# --------------------------------------------------------------------------
# filters and consistency
# --------------------------------------------------------------------------

def test_filters_drop_slivers_and_say_how_many_of_each():
    raw = _raw({"H1": [("A", 1), ("B", 10)]})
    tables, meta = prepare(raw, _config(min_samples=5))

    assert set(tables["unit_summary"]["unit_id"]) == {"B"}
    assert meta["filters"]["dropped_too_few_samples"] == 1
    assert meta["filters"]["dropped_unit_occurrences"] == 1


def test_a_filtered_occurrence_leaves_every_table_at_once():
    """"Present at this CPT" must mean the same in all five files.

    Disagreement here does not raise; it surfaces much later as a merge that
    quietly loses rows.
    """
    raw = _raw({"H1": [("A", 1), ("B", 10)]})
    tables, _ = prepare(raw, _config(min_samples=5))
    for name in ["cpt_samples", "layers", "unit_summary", "intervals"]:
        assert "A" not in set(tables[name]["unit_id"]), name


def test_a_hole_that_retains_no_unit_stops_being_a_location():
    raw = _raw({"H1": [("A", 1)], "H2": [("B", 10)]})
    tables, meta = prepare(raw, _config(min_samples=5))
    assert set(tables["locations"]["cpt_id"]) == {"H2"}
    assert meta["counts"]["n_cpt"] == 1


# --------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------

def test_the_origin_defaults_to_the_extent_centre():
    raw = _raw({"H1": [("A", 4)], "H2": [("A", 4)], "H3": [("A", 4)]})
    tables, meta = prepare(raw, _config())
    x = tables["locations"]["x"]
    assert x.min() == pytest.approx(-x.max())
    assert meta["coordinates"]["origin_easting"] == pytest.approx(1500.0)


def test_a_pinned_origin_survives_subsetting():
    """The reason to pin one.  A centre-derived origin *moves* when the input is
    subset, so two runs over overlapping data land in different frames and their
    coordinates are quietly incomparable.
    """
    raw = _raw({"H1": [("A", 4)], "H2": [("A", 4)], "H3": [("A", 4)]})
    subset = raw[raw["cpt_id"] != "H3"]
    pinned = _config(origin=(1000.0, 2000.0))

    full_x = prepare(raw, pinned)[0]["locations"].set_index("cpt_id")["x"]
    part_x = prepare(subset, pinned)[0]["locations"].set_index("cpt_id")["x"]
    assert full_x["H1"] == part_x["H1"]

    drifting = _config()
    a = prepare(raw, drifting)[0]["locations"].set_index("cpt_id")["x"]["H1"]
    b = prepare(subset, drifting)[0]["locations"].set_index("cpt_id")["x"]["H1"]
    assert a != b, "the unpinned origin really does drift — that is the hazard"


def test_coordinates_already_in_kilometres_are_not_divided_again():
    """Not every export is in metres, and a silent factor of 1000 in the
    coordinates would make every fitted range meaningless."""
    raw = _raw({"H1": [("A", 4)], "H2": [("A", 4)]})
    tables, _ = prepare(raw, _config(coordinate_units_per_km=1.0, origin=(1000.0, 2000.0)))
    assert tables["locations"].set_index("cpt_id").loc["H2", "x"] == pytest.approx(500.0)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

def test_a_dataset_must_state_its_coordinate_system():
    """Predictions that cannot be mapped back to the real world are not a
    deliverable, so this fails at config time rather than at handover."""
    with pytest.raises(ValueError, match="crs"):
        PrepareConfig(crs="  ")


def test_the_normalisation_block_is_written_when_a_stress_profile_is_given():
    """``qt`` is dropped from the prepared tables, so this is recorded here or
    it is lost — ``cli model`` reads it back to convert predictions to ``qt``."""
    raw = _raw({"H1": [("A", 4)]})
    _, meta = prepare(raw, _config(
        stress_profile=StressProfile.from_gradients(gamma_sat=20.1, gamma_w=10.0),
        stress_exponent=1.0, stress_basis="fitted to the export",
    ))
    block = meta["normalisation"]
    assert block["gamma_sat_kn_m3"] == pytest.approx(20.1)
    assert block["n"] == pytest.approx(1.0)
    assert block["basis"] == "fitted to the export"


def test_a_dataset_with_no_stress_profile_omits_the_block_rather_than_guessing():
    """Absent is honest; a default would be a fabricated unit weight that
    silently converts every prediction wrongly."""
    _, meta = prepare(_raw({"H1": [("A", 4)]}), _config())
    assert "normalisation" not in meta


def test_the_metadata_records_what_was_thrown_away():
    _, meta = prepare(_raw({"H1": [("A", 1), ("B", 10)]}),
                      _config(drop_units=("D",), min_samples=5))
    assert meta["dropped_units"] == ["D"]
    assert meta["filters"]["min_samples"] == 5
    assert meta["counts"]["n_unit_occurrences"] == 1


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def test_the_written_directory_round_trips_through_read_dataset(tmp_path):
    """The end the whole module exists to reach: what it writes is what
    ``cpt_geostat.contract`` reads, with no adapter in between."""
    from cpt_geostat.contract import read_dataset

    raw = _raw({"H1": [("A", 4), ("B", 4)], "H2": [("A", 4), ("B", 4)]})
    tables, meta = prepare(raw, _config())
    write(tables, meta, tmp_path)

    ds = read_dataset(tmp_path)
    assert set(ds.unit_ids) == {"A", "B"}
    assert len(ds.layout) == 2
    assert json.loads((tmp_path / "prepare_metadata.json").read_text())["crs"] == "TEST / metres"


def test_survey_metadata_is_carried_onto_locations_when_asked_for(tmp_path):
    raw = _raw({"H1": [("A", 4)], "H2": [("A", 4)]})
    tables, _ = prepare(raw, _config(location_columns=("survey_year",)))
    assert tables["locations"].set_index("cpt_id").loc["H2", "survey_year"] == 2021


def test_report_runs_on_a_prepared_dataset(capsys):
    """It is the operator's only view of a run, so a crash in it loses the run."""
    tables, meta = prepare(_raw({"H1": [("A", 4), ("B", 4)]}), _config())
    report(tables, meta)
    out = capsys.readouterr().out
    assert "1 CPTs, 2 units" in out
    assert "coverage" in out
