"""Raw CPT export to the dataset contract — the part that is the same everywhere.

Every project has to map some supplier's export onto ``cpt_samples`` /
``layers`` / ``unit_summary``, and almost all of that work is identical across
sites: finding contiguous runs of a unit down a hole, placing contacts between
them, collapsing repeated occurrences onto the contract's merge key,
depth-averaging, dropping slivers, and recording what was done.  Only the
*declarations* differ — which columns the export uses, what its coordinates
mean, which labels are not units, and how its normalised column was produced.

So the split is: **a project supplies a normalised raw frame and a
:class:`PrepareConfig`; this module does the rest.**  That keeps one
implementation of the geometry rules rather than one per project, which matters
because the rules are subtle in ways that are invisible when wrong — a contact
placed at a reading rather than between two makes every layer short by one
sample interval, and it looks entirely plausible in the output.

The project's remaining job — reading its own file, renaming its own columns —
is genuinely per-site and stays in ``projects/<name>/prepare_data.py``.

Raw frame contract
------------------
:func:`prepare` takes a dataframe with :data:`RAW_COLUMNS`:

    cpt_id      hole identifier
    easting     projected coordinate, in ``config.coordinate_units_per_km``
    northing    likewise
    z           depth, metres, positive down
    unit_id     stratigraphic label, including any to be dropped
    value       the normalised resistance to model, linear (becomes ``Qtn``)

**Rows must be sorted by hole then depth**, and must still contain the unit
labels destined for ``drop_units`` — run detection has to see them (see
:func:`build_intervals`).  :func:`normalise_raw` does the sorting and checks the
columns; a project that renames its columns to these names can call it and be
sure of both.

Any further columns are carried through to ``locations`` as per-hole survey
metadata if named in ``config.location_columns``.

This module sits outside ``cpt_geostat.contract`` on purpose.  The contract is
dependency-light because reading real data must stay cheap; preparation runs
once, may import ``cpt_geostat.cpt``, and nothing downstream imports it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contract import LAYERS_COLUMNS, SAMPLES_COLUMNS, SUMMARY_COLUMNS, summarise_units
from .cpt import METADATA_KEY, StressProfile, normalisation_metadata

#: What a project must hand :func:`prepare`.  Everything else is optional metadata.
RAW_COLUMNS = ["cpt_id", "easting", "northing", "z", "unit_id", "value"]

#: ``layers`` as written here — the contract's four columns plus provenance.
PREPARED_LAYERS_COLUMNS = LAYERS_COLUMNS + ["thickness_m", "n_intervals"]


@dataclass(frozen=True)
class PrepareConfig:
    """The site-specific declarations, and nothing that is derivable from them.

    Defaults are the ones IJmuiden validated, not universal truths; a project
    that wants different filters should say so rather than inherit them
    silently.  ``crs`` has no default at all, because a prepared dataset whose
    coordinate system is unrecorded cannot be mapped back to the real world and
    is therefore not a deliverable.
    """

    crs: str
    #: Labels present in the export that are not stratigraphic units.  Dropped
    #: *after* run detection, so an unclassified interval leaves a real gap
    #: rather than being bridged into a false contact.
    drop_units: Tuple[str, ...] = ()
    min_thickness_m: float = 0.5
    min_samples: int = 20
    #: Raw coordinate units per model kilometre — 1000 for metres, 1 for km.
    coordinate_units_per_km: float = 1000.0
    #: Pin the origin in raw coordinate units.  ``None`` puts it at the extent
    #: centre, which *moves if the input is subset* — pin it for anything whose
    #: coordinates must stay comparable between runs.
    origin: Optional[Tuple[float, float]] = None
    #: Extra per-hole columns from the raw frame to carry into ``locations``.
    location_columns: Tuple[str, ...] = ()
    #: How the supplied normalised column was produced.  Unrecoverable from the
    #: prepared tables (``qt`` is dropped), so it is recorded or it is lost.
    stress_profile: Optional[StressProfile] = None
    stress_exponent: float = 1.0
    stress_exponent_sd: float = 0.0
    stress_basis: str = ""
    #: Provenance for the modelled column, for the metadata file.
    raw_value_column: str = ""
    value_note: str = "linear; the model takes the natural log downstream"
    depth_datum: str = "below seabed, positive down"
    extra_metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.crs).strip():
            raise ValueError("crs must be stated: a dataset that cannot be mapped "
                             "back to real coordinates is not a deliverable")
        if self.coordinate_units_per_km <= 0:
            raise ValueError(
                f"coordinate_units_per_km must be positive, got {self.coordinate_units_per_km}"
            )


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

def normalise_raw(df: pd.DataFrame, rename: Optional[Dict[str, str]] = None,
                  keep: Sequence[str] = ()) -> pd.DataFrame:
    """Rename onto :data:`RAW_COLUMNS`, check them, and sort by hole then depth.

    The sort is not cosmetic: :func:`build_intervals` detects runs with a
    ``shift()``, so an unsorted frame silently produces a different — and
    wrong — layer geometry rather than an error.  Doing it here means no project
    has to remember.
    """
    out = df.rename(columns=dict(rename or {}))
    missing = [c for c in RAW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"raw frame is missing required column(s): {missing}. "
            f"Pass a `rename` mapping onto {RAW_COLUMNS}"
        )
    cols = list(RAW_COLUMNS) + [c for c in keep if c in out.columns and c not in RAW_COLUMNS]
    return out[cols].sort_values(["cpt_id", "z"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def resolve_origin(raw: pd.DataFrame, origin=None) -> Tuple[float, float]:
    """Origin in raw coordinate units — the midpoint of the location extent."""
    if origin is not None:
        return float(origin[0]), float(origin[1])
    loc = raw.drop_duplicates("cpt_id")
    return (
        float((loc["easting"].min() + loc["easting"].max()) / 2.0),
        float((loc["northing"].min() + loc["northing"].max()) / 2.0),
    )


def build_locations(raw: pd.DataFrame, origin: Tuple[float, float],
                    config: PrepareConfig) -> pd.DataFrame:
    """One row per CPT: model coordinates in km, plus any survey metadata kept."""
    extra = [c for c in config.location_columns if c in raw.columns]
    loc = raw.drop_duplicates("cpt_id")[["cpt_id", "easting", "northing", *extra]].copy()
    scale = config.coordinate_units_per_km
    loc["x"] = (loc["easting"] - origin[0]) / scale
    loc["y"] = (loc["northing"] - origin[1]) / scale
    loc["kind"] = "cpt"  # the layout carries a `kind`; real holes are not grid nodes
    return loc.sort_values("cpt_id").reset_index(drop=True)


def build_intervals(raw: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of one unit down a hole, with contacts at midpoints.

    Two rules that are wrong by default and right here:

    * Run detection must see the **unfiltered** record.  Dropping an
      unclassified label first would close the gap it leaves, joining the units
      either side of it into one false contact.
    * A contact goes at the **midpoint between the last reading of one run and
      the first of the next**, not at either reading, so tops and bases meet
      exactly and no layer is short by a sample interval.
    """
    new_run = (raw["unit_id"] != raw["unit_id"].shift()) | (raw["cpt_id"] != raw["cpt_id"].shift())
    runs = (
        raw.assign(run_id=new_run.cumsum())
        .groupby("run_id", sort=True)
        .agg(
            cpt_id=("cpt_id", "first"),
            unit_id=("unit_id", "first"),
            z_first=("z", "min"),
            z_last=("z", "max"),
            n_readings=("z", "size"),
        )
        .reset_index(drop=True)
    )

    same_hole = runs["cpt_id"].eq(runs["cpt_id"].shift(-1))
    mid = (runs["z_last"] + runs["z_first"].shift(-1)) / 2.0
    runs["z_bot"] = np.where(same_hole, mid, runs["z_last"])
    runs["z_top"] = np.where(
        runs["cpt_id"].eq(runs["cpt_id"].shift()), runs["z_bot"].shift(), runs["z_first"]
    )
    # The first run in each hole starts at its first reading, clamped to the datum.
    runs["z_top"] = runs["z_top"].astype(float).clip(lower=0.0)
    runs["thickness_m"] = runs["z_bot"] - runs["z_top"]
    return runs[["cpt_id", "unit_id", "z_top", "z_bot", "thickness_m", "n_readings"]]


def collapse_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(cpt_id, unit_id)`` — the contract's merge key.

    ``thickness_m`` **sums** the occupied runs, so a unit that re-enters lower
    down is not credited with the intervening material it does not occupy.  The
    span is still recoverable from ``z_top``/``z_bot``, and the per-run detail
    survives in ``intervals``.
    """
    layers = (
        intervals.groupby(["cpt_id", "unit_id"], sort=False)
        .agg(
            z_top=("z_top", "min"),
            z_bot=("z_bot", "max"),
            thickness_m=("thickness_m", "sum"),
            n_intervals=("z_top", "size"),
        )
        .reset_index()
    )
    return layers.sort_values(["cpt_id", "z_top"])[PREPARED_LAYERS_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------
# samples and summary
# --------------------------------------------------------------------------

def build_samples(raw: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """``cpt_samples``: readings with a usable value, in model coordinates.

    Non-positive values go too, not just nulls — the model works in ``log(Qtn)``
    and a zero would propagate as ``-inf`` through the depth-average into every
    downstream fit.
    """
    keep = raw["value"].notna() & (raw["value"] > 0)
    samples = raw.loc[keep, ["cpt_id", "z", "unit_id", "value"]].rename(columns={"value": "Qtn"})
    samples = samples.merge(locations[["cpt_id", "x", "y"]], on="cpt_id", how="left")
    return samples.sort_values(["cpt_id", "z"])[SAMPLES_COLUMNS].reset_index(drop=True)


def build_unit_summary(samples: pd.DataFrame, layers: pd.DataFrame,
                       locations: pd.DataFrame) -> pd.DataFrame:
    """Depth-averaged ``log Qtn`` per CPT per unit — the model input.

    Delegates to :func:`cpt_geostat.contract.summarise_units` rather than
    reimplementing it: the natural-log-then-average convention is part of the
    contract the estimators are calibrated against, and a second copy could
    drift from it without anything failing.
    """
    out = summarise_units(samples, layers, locations)
    return out.sort_values(["cpt_id", "unit_id"])[SUMMARY_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------

def apply_filters(layers: pd.DataFrame, summary: pd.DataFrame,
                  min_thickness_m: float, min_samples: int):
    """Drop unit occurrences too thin or too sparsely sampled to average.

    A depth-average over a handful of readings of a 0.1 m sliver is noise
    dressed as an observation, and the estimator has no way to down-weight it.
    """
    thin = summary["thickness_m"] < min_thickness_m
    sparse = summary["n_samples"] < min_samples
    drop = thin | sparse

    keep_key = set(map(tuple, summary.loc[~drop, ["cpt_id", "unit_id"]].to_numpy()))
    layer_key = list(map(tuple, layers[["cpt_id", "unit_id"]].to_numpy()))
    layers = layers.loc[[k in keep_key for k in layer_key]].reset_index(drop=True)
    summary = summary.loc[~drop].reset_index(drop=True)

    report = {
        "min_thickness_m": float(min_thickness_m),
        "min_samples": int(min_samples),
        "dropped_unit_occurrences": int(drop.sum()),
        "dropped_too_thin": int(thin.sum()),
        "dropped_too_few_samples": int(sparse.sum()),
    }
    return layers, summary, report


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def prepare(raw: pd.DataFrame, config: PrepareConfig):
    """Raw frame to the prepared tables plus the metadata that documents them.

    Returns ``(tables, meta)``, where ``tables`` holds ``cpt_samples``,
    ``layers``, ``unit_summary`` — the contract — alongside ``intervals`` and
    ``locations``, which are extra detail the contract does not read.
    """
    raw = normalise_raw(raw, keep=config.location_columns)
    origin = resolve_origin(raw, config.origin)
    locations = build_locations(raw, origin, config)

    # Runs first, so a dropped label leaves a gap rather than a false contact.
    intervals = build_intervals(raw)
    intervals = intervals[~intervals["unit_id"].isin(config.drop_units)].reset_index(drop=True)

    kept = raw[~raw["unit_id"].isin(config.drop_units)]
    samples = build_samples(kept, locations)

    layers = collapse_intervals(intervals)
    summary = build_unit_summary(samples, layers, locations)
    layers, summary, filter_report = apply_filters(
        layers, summary, config.min_thickness_m, config.min_samples
    )

    tables = _make_consistent(samples, layers, summary, intervals, locations)
    return tables, _metadata(config, origin, filter_report, tables)


def _make_consistent(samples, layers, summary, intervals, locations) -> Dict[str, pd.DataFrame]:
    """Trim every table to the same ``(cpt_id, unit_id)`` set, then the same holes.

    A layer whose readings were all unusable leaves no summary row, and a hole
    that retains no unit is not a location any more.  Without this pass
    "present at this CPT" would mean something different in each file, and the
    disagreement would surface much later as a merge that silently loses rows.
    """
    key = summary[["cpt_id", "unit_id"]]
    samples = samples.merge(key, on=["cpt_id", "unit_id"], how="inner")
    intervals = intervals.merge(key, on=["cpt_id", "unit_id"], how="inner")

    holes = summary["cpt_id"].unique()
    keep = lambda df: df[df["cpt_id"].isin(holes)].reset_index(drop=True)  # noqa: E731
    return {
        "cpt_samples": keep(samples),
        "layers": keep(layers),
        "unit_summary": keep(summary),
        "intervals": keep(intervals),
        "locations": keep(locations),
    }


def _metadata(config: PrepareConfig, origin, filter_report, tables) -> dict:
    summary, samples = tables["unit_summary"], tables["cpt_samples"]
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crs": config.crs,
        "coordinates": {
            "units": "km",
            "origin_easting": origin[0],
            "origin_northing": origin[1],
            "units_per_km": config.coordinate_units_per_km,
            "note": (f"x = (easting - origin_easting) / {config.coordinate_units_per_km:g}; "
                     "y likewise for northing."),
        },
        "depth": {"units": "m", "datum": config.depth_datum},
        "property": {
            "column": "Qtn",
            "raw_column": config.raw_value_column,
            "note": config.value_note,
        },
        "dropped_units": list(config.drop_units),
        "filters": filter_report,
        "counts": {
            "n_cpt": int(summary["cpt_id"].nunique()),
            "n_units": int(summary["unit_id"].nunique()),
            "n_unit_occurrences": int(len(summary)),
            "n_samples": int(len(samples)),
        },
        "units": sorted(summary["unit_id"].unique().tolist()),
        "extent_km": {
            "x": [float(summary["x"].min()), float(summary["x"].max())],
            "y": [float(summary["y"].min()), float(summary["y"].max())],
            "z": [float(samples["z"].min()), float(samples["z"].max())],
        },
    }
    if config.stress_profile is not None:
        meta[METADATA_KEY] = normalisation_metadata(
            config.stress_profile, config.stress_exponent,
            basis=config.stress_basis, n_sd=config.stress_exponent_sd,
        )
    meta.update(config.extra_metadata)
    return meta


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write(tables: Dict[str, pd.DataFrame], meta: dict, out_dir, also_csv: bool = False) -> Path:
    """Parquet by default — real exports are large and parquet round-trips dtypes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False, compression="zstd")
        if also_csv:
            df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    (out / "prepare_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return out


def report(tables: Dict[str, pd.DataFrame], meta: dict, out_dir=None) -> None:
    """What was produced and what was thrown away, on one screen.

    The per-unit coverage table is the point: it is the cheapest check that unit
    labels and values line up (lithology should separate in the expected
    direction), and it shows immediately which units have too few holes to fit a
    variogram against.
    """
    c = meta["counts"]
    if out_dir is not None:
        print(f"wrote {out_dir}")
    print(f"  {c['n_cpt']} CPTs, {c['n_units']} units, "
          f"{c['n_unit_occurrences']} unit occurrences, {c['n_samples']:,} readings")
    f = meta["filters"]
    print(f"  dropped {f['dropped_unit_occurrences']} occurrences "
          f"({f['dropped_too_thin']} thinner than {f['min_thickness_m']} m, "
          f"{f['dropped_too_few_samples']} under {f['min_samples']} readings)")
    ext = meta["extent_km"]
    print(f"  extent {ext['x'][1] - ext['x'][0]:.1f} x {ext['y'][1] - ext['y'][0]:.1f} km, "
          f"{ext['z'][1]:.1f} m deep")

    per_unit = (
        tables["unit_summary"]
        .groupby("unit_id")
        .agg(n_cpt=("cpt_id", "size"), mean_log_Q=("log_Q_mean", "mean"),
             mean_thickness_m=("thickness_m", "mean"))
        .sort_values("n_cpt", ascending=False)
    )
    per_unit["coverage"] = per_unit["n_cpt"] / c["n_cpt"]
    print(per_unit.to_string(float_format=lambda v: f"{v:.2f}"))


# --------------------------------------------------------------------------
# the argument parser every project's script wants
# --------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser, config: PrepareConfig,
                  default_source=None, default_out=None) -> argparse.ArgumentParser:
    """The flags that are the same for every project, defaulted from ``config``.

    A project adds its own on top; :func:`config_from_args` folds whatever the
    user overrode back in.
    """
    parser.add_argument("--source", type=Path, default=default_source,
                        help="raw CPT export")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="output directory")
    parser.add_argument("--drop-units", nargs="*", default=list(config.drop_units),
                        help=f"unit labels to exclude (default: {list(config.drop_units)})")
    parser.add_argument("--min-thickness-m", type=float, default=config.min_thickness_m,
                        help="drop unit occurrences thinner than this")
    parser.add_argument("--min-samples", type=int, default=config.min_samples,
                        help="drop unit occurrences with fewer usable readings than this")
    parser.add_argument("--origin", type=float, nargs=2, metavar=("EASTING", "NORTHING"),
                        default=config.origin,
                        help="pin the coordinate origin in raw units (default: extent centre)")
    parser.add_argument("--also-csv", action="store_true",
                        help="write csv alongside the parquet")
    return parser


def config_from_args(config: PrepareConfig, args) -> PrepareConfig:
    """``config`` with whatever :func:`add_arguments` collected folded in."""
    from dataclasses import replace

    return replace(
        config,
        drop_units=tuple(getattr(args, "drop_units", config.drop_units) or ()),
        min_thickness_m=getattr(args, "min_thickness_m", config.min_thickness_m),
        min_samples=getattr(args, "min_samples", config.min_samples),
        origin=tuple(args.origin) if getattr(args, "origin", None) else None,
    )
