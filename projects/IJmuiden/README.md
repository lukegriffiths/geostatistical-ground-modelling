# IJmuiden (IJ56)

Project-specific wrangling for the IJmuiden Ver CPT dataset. Nothing here is
imported by `cpt_geostat`; the dependency runs the other way.

`prepare_data.py` is 122 lines and almost all of it is *declaration* — which
columns this export uses, what its coordinates mean, which labels are not units,
how its `Qt` column was normalised. The pipeline acting on those declarations
lives in `cpt_geostat.prepare` and is shared with every other project, so the
layer-geometry rules have one implementation and one set of tests
(`tests/test_prepare.py`) rather than a copy per site.

## Prepare

```bash
python projects/IJmuiden/prepare_data.py
cpt-geostat plot --data projects/IJmuiden/data --out projects/IJmuiden/figures
```

Reads `data/cpt_dataframe.csv` (129 MB, 562,491 readings, 205 locations) and
writes `projects/IJmuiden/data/` (4.1 MB of zstd parquet). Both directories are
gitignored.

| Output | Contents |
|---|---|
| `cpt_samples.parquet` | `cpt_id, x, y, z, unit_id, Qtn` — one row per depth reading |
| `layers.parquet` | `cpt_id, unit_id, z_top, z_bot, thickness_m, n_intervals` |
| `unit_summary.parquet` | `cpt_id, x, y, unit_id, log_Q_mean, log_Q_sd, n_samples, thickness_m` — the model input |
| `intervals.parquet` | per-run layer detail before collapsing to one row per unit |
| `locations.parquet` | survey metadata per CPT (`Loc_type`, `Area_code`, `Year`, `WD`, UTM coords). Written for reference — `read_dataset` rebuilds the layout from `cpt_samples` and does not read this file |
| `prepare_metadata.json` | coordinate origin, filters applied, realised counts |

`--also-csv` writes csv alongside; `--min-thickness-m`, `--min-samples`,
`--drop-units` and `--origin` control the filters and the frame.

## Mapping onto the contract

| Raw | Model | Note |
|---|---|---|
| `Location_ID_ind` | `cpt_id` | 205 individual holes across 148 group locations |
| `Easting`, `Northing` | `x`, `y` | m → km, origin at the extent centre; see `prepare_metadata.json` |
| `z_bsf` | `z` | already m below seabed, positive down |
| `soil_unit` | `unit_id` | `Default` dropped — unclassified, not a unit |
| `Qt` | `Qtn` | linear; the model takes the natural log downstream |

Coordinates are ETRS89 / UTM 31N (EPSG:25831). To map a prediction back:
`Easting = x * 1000 + origin_easting`, reading `origin_easting` and
`units_per_km` from the `coordinates` block of `prepare_metadata.json`.

## Things the raw data does that the contract does not

* **Repeated units.** 19 holes re-enter a unit deeper down — usually `GGM_31_S`
  below an intervening clay. `layers` keeps one row per `(cpt_id, unit_id)`
  because that is the contract's merge key, spanning first top to last base,
  with `thickness_m` the **sum** of the occupied runs rather than the span.
  `intervals.parquet` keeps the per-run detail for anything that needs it.
* **Layer boundaries.** Raw rows carry a unit label, not contacts. Boundaries go
  at the midpoint between the last reading of one run and the first of the next,
  so tops and bases meet exactly. Runs are detected *before* `Default` is
  dropped, so an unclassified interval leaves a real gap instead of being
  bridged into a false contact.
* **Missing `Qt`.** 30,170 readings (5.4%) have none. They are dropped from
  `cpt_samples` but still count towards layer geometry — the soil unit is logged
  whether or not the resistance reduced successfully.
* **Slivers.** Unit occurrences under 0.5 m or 20 usable readings are dropped
  (20 of 960). A depth-average over a handful of readings is noise dressed as an
  observation, and the estimator has no way to down-weight it.
* **Empty holes.** 11 of 205 locations retain no unit after filtering — mostly
  the very shallow holes, one of which bottoms out at 0.31 m.

## Realised dataset

194 CPTs, 23 units, 940 unit occurrences, 483,349 readings, over a 23.3 × 19.6 km
footprint to 62.5 m below seabed.

Coverage is steeply unequal, which matters for what can be estimated: `GGM_01_S`
appears at 76% of holes, `GGM_02_S_C` at 62%, `GGM_31_S` at 47%, and then it
falls away — nine units sit under 10% and four are at four to seven holes, too
few to fit a variogram against.

Mean depth-averaged `log Qtn` separates on lithology in the expected direction —
sands 4.7–6.1, clays 2.8–3.9 — which is the cheapest available check that the
unit labels and the `Qt` column line up.
