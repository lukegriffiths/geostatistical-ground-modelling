"""IJmuiden (IJ56) CPT data — raw export to the dataset contract.

Everything here is a *declaration about this export*.  The pipeline that acts on
them — run detection, contacts at midpoints, collapsing repeated units,
depth-averaging, sliver filtering, metadata — lives in ``cpt_geostat.prepare``
and is shared with every other project, so the geometry rules have one
implementation rather than one per site.

    python projects/IJmuiden/prepare_data.py
    cpt-geostat plot --data projects/IJmuiden/data --out projects/IJmuiden/figures

The dependency runs one way only: this script imports ``cpt_geostat``, and
``cpt_geostat`` never imports from ``projects/``.

What this export needs said about it
------------------------------------
* **Coordinates** are ETRS89 / UTM 31N metres; the model works in kilometres
  with the origin at the site centre.
* **``Default``** is an unclassified interval, not a stratigraphic unit.  It is
  dropped *after* run detection, so it leaves a real gap rather than bridging
  the units either side of it into a false contact.
* **Normalisation.**  ``qt`` is dropped from the prepared tables, so the stress
  exponent and unit weights that produced ``Qt`` would be unrecoverable.  They
  are fitted from the export (see ``STRESS_PROFILE``) and written to
  ``prepare_metadata.json``, which is what anyone converting a prediction back
  to ``qt`` should read.
* **Missing ``Qt``** — 30,170 readings, 5.4% — needs no declaration: the shared
  pipeline drops them from ``cpt_samples`` but still counts them towards layer
  geometry, because a soil unit is logged whether or not the resistance reduced
  successfully.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cpt_geostat.cpt import StressProfile
from cpt_geostat.prepare import (
    PrepareConfig,
    add_arguments,
    config_from_args,
    prepare,
    report,
    write,
)

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]

DEFAULT_SOURCE = REPO_ROOT / "data" / "cpt_dataframe.csv"
DEFAULT_OUT = PROJECT_DIR / "data"

# Raw columns actually needed.  The rest of the export is survey admin.
SOURCE_COLUMNS = [
    "Location_ID_ind", "Location_ID_group", "Loc_type", "Easting", "Northing",
    "WD", "Area_code", "Year", "z_bsf", "soil_unit", "qc", "qt", "Qt",
]

#: This export's column names, onto the shared pipeline's.
RENAME = {
    "Location_ID_ind": "cpt_id",
    "Easting": "easting",
    "Northing": "northing",
    "z_bsf": "z",
    "soil_unit": "unit_id",
    "Qt": "value",
}

# How the export's `Qt` column was normalised, recovered from the export itself:
# `cpt_geostat.cpt.infer_gamma_eff` over all 532,321 readings carrying both `qt`
# and `Qt` returns gamma_eff = 10.07 kN/m3 at n = 1, reproducing the column to a
# median 1.002 (p5-p95 0.993-1.016).  Rounding that to a stated soil — saturated
# 20.1, seawater 10.0 — holds to +/-1%.
#
# n = 1 is not an assumption.  Fitting n = 0.5 leaves a residual that sweeps
# 0.37 to 1.61 with depth and no unit weight repairs it, so the column is
# Robertson's `Qt`, whatever the `Qtn` rename downstream suggests.
STRESS_PROFILE = StressProfile.from_gradients(gamma_sat=20.1, gamma_w=10.0)
STRESS_BASIS = (
    "fitted to the export's own qt and Qt columns with cpt_geostat.cpt.infer_gamma_eff "
    "(gamma_eff 10.07 kN/m3 at n=1, reproduced to a median ratio of 1.002), then "
    "rounded to a stated soil; holds to +/-1%"
)

CONFIG = PrepareConfig(
    crs="ETRS89 / UTM zone 31N (EPSG:25831)",
    drop_units=("Default",),
    min_thickness_m=0.5,
    min_samples=20,
    coordinate_units_per_km=1000.0,
    location_columns=("Location_ID_group", "Loc_type", "Area_code", "Year", "WD"),
    stress_profile=STRESS_PROFILE,
    stress_exponent=1.0,
    stress_basis=STRESS_BASIS,
    raw_value_column="Qt",
)


def load_raw(source: Path) -> pd.DataFrame:
    """Read the raw export, keeping only the columns the pipeline needs.

    ``low_memory=False`` because the 129 MB export has mixed types in its admin
    columns that chunked dtype inference gets wrong.
    """
    df = pd.read_csv(source, usecols=SOURCE_COLUMNS, low_memory=False)
    return df.rename(columns=RENAME)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(p, CONFIG, default_source=DEFAULT_SOURCE, default_out=DEFAULT_OUT)
    args = p.parse_args(argv)

    if not args.source.exists():
        p.error(f"source not found: {args.source}")

    config = config_from_args(CONFIG, args)
    tables, meta = prepare(load_raw(args.source), config)
    meta["source"] = str(args.source)
    write(tables, meta, args.out, args.also_csv)
    report(tables, meta, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
