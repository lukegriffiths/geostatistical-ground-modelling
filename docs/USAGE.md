# Usage and reference

Commands, conventions, the data contract, every file the CLI writes, and the
package layout.  *Why* any of it is this way is in [DECISIONS.md](DECISIONS.md);
the mathematics is in [METHODOLOGY.md](METHODOLOGY.md).

---

## 1. Install and run

```bash
python -m venv .venv
.venv/bin/pip install -e .            # Windows: .venv/Scripts/pip
```

`pip install -e .` is not optional convenience — it is what puts the
`cpt-geostat` command on the path and makes `import cpt_geostat` work from a
project directory rather than only from the repository root.

```bash
cpt-geostat synth run --out run              # generate + plot + model, end to end

cpt-geostat synth generate --config config.yaml --out data/
cpt-geostat plot  --data data/ --out figures/
cpt-geostat model --data data/ --out models/
```

`synth` needs a generator config and produces truth.  `plot` and `model` need
only a directory holding `cpt_samples` and `layers`, and run identically on real
data; both print `mode = synthetic (truth found)` or `mode = real data (no
truth)`.

### 1.1 Flags

`--data`, `--out` and `--dpi` are per-command; `synth generate` and `synth run`
also take `--config` and `--seed`.  The output options below are shared by
`model` and `synth run`.

| Flag | Default | Effect |
|---|---|---|
| `--fields default\|all\|none\|A,B` | `default` | which models get field maps; `default` is baseline, SK and OK, `all` adds UK |
| `--field-res-km KM` | synthetic raster, else 0.15 | grid spacing for the field maps |
| `--anisotropy N_SIM` | `0` (off) | fit a major axis per unit, gated on `N_SIM` simulated isotropic nulls; adds the `OK (aniso)` estimator.  30–60 is a usable gate and takes minutes |
| `--anisotropy-seed S` | — | seed for the null simulations, for a reproducible gate |
| `--profiles all\|none\|N` | `all` | how many per-hole depth profiles to write — `all` is 194 figures on IJmuiden |
| `--profile-model NAME\|all\|A,B` | one estimator | which cross-validated estimator(s) the profiles show, each into its own directory |
| `--profile-units both\|Qtn\|qt` | `both` | which profile suites to write |
| `--stress-profile GAMMA_SAT GAMMA_W` | from `prepare_metadata.json` | unit weights in kN/m³ for the qt conversion |
| `--stress-exponent N` | `1.0` | stress exponent for the qt conversion; only read alongside `--stress-profile` |
| `--stress-exponent-sd SD` | whatever the dataset records, else 0 | uncertainty on the exponent, widening the qt bands |

`--profile-model all` on IJmuiden is four estimators × 194 holes × 2 suites, so
it is not the default.  The profiles are the one output where the estimators
differ *hole by hole* rather than in a summary statistic.

---

## 2. Conventions

| | |
|---|---|
| Modelling variable | `log(Qtn)` — stress exponent `n = 1` on IJmuiden, so its `Qtn` *is* Robertson's `Qt`; see `cpt_geostat/cpt.py` |
| Coordinates | kilometres, origin at site centre |
| Depth | metres below seabed, positive down |
| **Azimuth** | **degrees clockwise from north** (0 = +y = N, 90 = +x = E) |
| `range_km` | **practical range** — separation at which correlation reaches 0.05 |

Azimuth and range are load-bearing and enforced by tests rather than by comment
— see [decisions §1](DECISIONS.md#1-conventions).  Both live in
`cpt_geostat/geometry.py` and `cpt_geostat/covariance.py` respectively, each the
single conversion site.

---

## 3. The data contract

| File | Contents |
|---|---|
| `cpt_samples.csv` | `cpt_id, x, y, z, unit_id, Qtn` — one row per depth reading |
| `layers.csv` | `cpt_id, unit_id, z_top, z_bot` — absent units are missing rows |
| `unit_summary.csv` | `cpt_id, x, y, unit_id, log_Q_mean, log_Q_sd, n_samples, thickness_m` — the model input |
| `truth.yaml` | resolved generative parameters + realised statistics (synthetic only) |
| `truth_fields.npz`, `truth_points.csv` | truth rasters and true CPT node values, for the B1 panels (synthetic only) |

Each table may be `.csv` or `.parquet`; parquet wins where both exist.

### 3.1 Running on real data

Supply `cpt_samples` and `layers` only.  `unit_summary` is recomputed from the
samples if absent, truth-dependent diagnostics are skipped, and the unit palette
falls back to a default cycle:

```bash
python -m cpt_geostat.cli plot --data /path/to/real --out figures/
```

`tests/plots/test_data_plots.py` runs every B2 plot against a directory holding
nothing but those two tables.

### 3.2 Getting a real export into that shape

`cpt_geostat.prepare` does the part that is the same at every site.  A project
supplies a raw frame renamed onto `cpt_id, easting, northing, z, unit_id, value`
and a `PrepareConfig` of declarations about its own export; the module handles
run detection, layer contacts, repeated units, depth-averaging, sliver filtering
and the metadata file:

```python
from cpt_geostat.prepare import PrepareConfig, prepare, report, write

CONFIG = PrepareConfig(
    crs="ETRS89 / UTM zone 31N (EPSG:25831)",
    drop_units=("Default",),          # labels that are not stratigraphic units
    min_thickness_m=0.5, min_samples=20,
    coordinate_units_per_km=1000.0,   # 1.0 if the export is already in km
    raw_value_column="Qt",
)

raw = pd.read_csv(source).rename(columns=RENAME)
tables, meta = prepare(raw, CONFIG)
write(tables, meta, out_dir)
report(tables, meta, out_dir)
```

That is the whole of a new project's preparation step bar its own reader and
rename map — [projects/IJmuiden/prepare_data.py](../projects/IJmuiden/prepare_data.py)
is 122 lines, of which the pipeline is six.
[projects/IJmuiden/](../projects/IJmuiden/README.md) is the worked example
(194 CPTs, 23 units, 483k readings).  The four wrangling rules the module pins
are in [decisions §3.1](DECISIONS.md#31-the-shared-rules-live-in-one-tested-module).

---

## 4. What `cli model` writes

```
models/
  unit_baseline.csv  cv_scores_*.csv  cross_plot_*.png
  anisotropy.csv                 # written whether or not any unit passes the gate
  variograms/                    # the fitted covariance SK and OK use —
    variogram_<unit>.png         #   UK refits on detrended residuals, so its
    variogram_directional_<unit>.png   #   covariance is NOT the one drawn here
  fields/<model>/
    prediction.png               # one row per unit x 4 columns
  profiles/<model>/          # one directory per estimator asked for
    profile_calibration.png  profile_coverage.csv   # one calibration, both suites
    Qtn/profile_<cpt>.png    # log axis — the space the model is Gaussian in
    qt/profile_<cpt>.png     # linear MPa — needs a stress profile; see §4.5
```

`cli synth run --out run` puts the four contract files in `run/data/`, 18 figures
in `run/figures/`, and the above in `run/models/`.

### 4.1 `unit_baseline.csv`

`cpt_geostat/models/baseline.py` fits one mean and one sd per unit and no spatial
term — the model a GI report already contains implicitly, written down so that
everything spatial has something to beat.  On synthetic data
`unit_baseline_vs_truth.csv` is written alongside.

| Column | Population | Reducible by more CPTs? |
|---|---|---|
| `log_Q_mean` ± `log_Q_sd` | per-CPT depth-averages, **one weight per CPT** | no — this is the spread of a new location |
| `se_mean` | uncertainty on the level itself | yes, as `1/√n` |
| `within_sd` | depth-to-depth scatter inside one hole | no — it is the texture of the trace |
| `reading_mean` ± `reading_sd` | every reading, **thickness-weighted** | no — quote this for "what will the cone see" |

The truth comparison reports `structured_frac`, the share of the baseline's
variance a perfect spatial model could remove, and `obs_noise_sd`, the noise
floor (nugget plus depth-averaging error).

### 4.2 Cross plots

`cross_plot_observed.png` and — synthetic only — `cross_plot_latent.png`:
predicted against true, one panel per unit, with `cv_predictions.csv` and
`cv_scores.csv` behind them.  Predictions are **leave-one-out**.  Both scoring
targets are carried on every row of `cv_predictions.csv` and each figure labels
which it used.

### 4.3 Variogram figures

`variograms/variogram_<unit>.png` for every unit that can be fitted.  Each
identifiability guard is drawn rather than asserted, and a refused fit gets
`why_not_resolved()` printed on its face.  On synthetic runs the generating
covariance is overlaid, which turns the figure into a direct recovery check; for
an anisotropic unit both axis curves are drawn as a band, because the
omnidirectional estimate is a mixture and is not expected to match either.

`variograms/variogram_directional_<unit>.png` splits the lags into four azimuth
sectors, and is the only thing that distinguishes *"no structure"* from
*"structure I was averaging away"*.  A compass panel carries the sectors and, on
synthetic data, the true anisotropy ellipse — if the lowest sector lies *across*
that ellipse rather than along it, the bearing convention is inverted.  Written
only where **n ≥ 30 CPTs**.

### 4.4 Field maps

`fields/<model>/prediction.png` for the baseline, SK and OK (`--fields all` adds
UK).  One **row per unit, four columns**:

| column | shows |
|---|---|
| `predicted` | the median surface |
| `latent sd` | how well the field itself is known — what more drilling reduces |
| `lower 95%` / `upper 95%` | the two ends of the interval, as maps in their own right |

These are the companion to the B1 truth maps: same style, and on a synthetic run
**the same raster**, so `fields/OK/prediction.png` and `figures/truth_unit_3.png`
can be compared by flipping between them rather than by eye across different
grids.  Fitted in-sample, and not masked by presence — read the mean and sd maps
as a pair.  Colour scales are shared within a row and across models, never across
units.

Units where kriging has collapsed onto the baseline are labelled *"uniform — no
spatial structure resolved"*.

### 4.5 Depth profiles

One figure per hole in `profiles/<model>/`, reconstructing `Qtn` down the hole
from the per-unit predictions and laying it over that hole's own readings.  Every
hole is predicted **leave-one-out**.  Three nested uncertainty bands are drawn:
latent, unit mean, and single reading — only the third is comparable with a raw
trace.  `profile_coverage.csv` and `profile_calibration.png` report realised
coverage per unit.

### 4.6 The `qt` suite

`cpt_geostat/cpt.py` inverts the normalisation, and `models.profile.qt_readings`
/ `qt_by_unit` apply it to a prediction table:

```python
profile, n = normalisation_from_metadata(json.load(open("data/prepare_metadata.json")))
in_qt = qt_readings(reading_predictions(ds, cv, model="OK (fitted)"), profile, n)
```

The stress profile and exponent come from the dataset's `prepare_metadata.json`;
a dataset that records none is skipped with a message rather than defaulted.
`--stress-profile GAMMA_SAT GAMMA_W [--stress-exponent N]` is the override —
including for the synthetic datasets, which have no underlying `qt` at all.
`--stress-exponent-sd` widens the bands for exponent doubt and sets
`attrs['coverage_transfers'] = False`.

`n` may be a single number or a `unit_id -> n` mapping, everywhere an exponent is
taken.  To *get* one, `cpt.soil_behaviour_index` runs Robertson's iteration over
raw `qt` and `fs`; it needs sleeve friction, so it belongs at preparation time.
The mathematics of the transform, including the uncertain-exponent case, is in
[METHODOLOGY §5](METHODOLOGY.md#5-the-q_tn-leftrightarrow-q_t-transform).

---

## 5. Package layout

The package is split by *what a module needs*, not by what it does: everything
above the line runs on any dataset meeting the contract; everything below it
needs a generator config or realised truth.

```
cpt_geostat/
  geometry.py        # azimuth conventions — the single source of truth
  covariance.py      # GrfConfig + range <-> len_scale — the single conversion site
  trend.py           # the linear trend surface, shared by plots and estimators
  cpt.py             # Qtn <-> qt: stress profile, stress exponent (fixed,
                     #   per-unit or uncertain), Robertson's Ic iteration
  contract/          # ---- the data contract: numpy + pandas only ----
    schema.py        # column contracts, Dataset, Raster
    summarise.py     # depth-averaging; projects import this
    io.py            # readers/writers; the real-data entry point
  models/
    base.py          # the estimator interface — and the two variance conventions
    baseline.py      # per-unit mean and sd; no spatial term
    variogram.py     # empirical + fitted + directional, identifiability guards
    anisotropy.py    # profile likelihood over the axis + a simulated null
    kriging.py       # simple / ordinary / universal kriging
    profile.py       # per-unit predictions -> Qtn with depth, per reading
    field.py         # per-unit predictions -> a gridded surface + its sd
    crosscheck.py    # pykrige, as an independent check — validation only
  validate/
    metrics.py       # rmse, bias, r2, coverage, mssr
    cv.py            # leave-one-CPT-out; carries both variances on every row
  plots/             # every plot here runs on real data
    style.py         # shared conventions (B3)
    maps.py          # B2 — presence / value / thickness maps
    sections.py      # B2 — fence sections
    diagnostics.py   # B2 — depth traces, trend checks, lag coverage
    variograms.py    # B2 — variogram fits and directional sectors
    profiles.py      # B2 — predicted resistance with depth vs the measured
                     #      trace, as two suites: Qtn (log) and qt (linear MPa)
    fields.py        # B2 — maps of the predicted field and its uncertainty
    predictions.py   # predicted-vs-true cross plots
  synthetic/         # ---- needs a config or truth ----
    config.py        # schema, validation, named RNG streams
    layout.py        # site + CPT positions (jittered grid, cluster, thinned corner)
    strat.py         # presence probability, thickness, stratigraphic assembly
    fields.py        # rasters, GRFs, trend surfaces, the gstools bridge
    series.py        # property sampling and depth-series synthesis
    pipeline.py      # A1–A7 end to end
    truth.py         # realised truth, the truth reference table, truth-covariance kriging
    plots.py         # B1 — truth diagnostics
  cli.py
```

Tests mirror the same split: `tests/contract/`, `tests/models/`, `tests/plots/`
and `tests/synthetic/`.  `validate/` holds only leave-one-out and the metrics a
cross plot needs; the GP, hyperparameter recovery and the remaining scores belong
to [plan 02](../plans/02_estimators_and_validation.md).

```bash
.venv/bin/python -m pytest -q
```

---

## 6. Regenerating the source dump

`docs/CODE.md` is the whole project as one self-contained markdown file, for
handing to a reader with no other context.  It is a build artefact, not a source
of truth, and a stale dump is worse than none:

```bash
python tools/dump_code.py            # -> docs/CODE.md
python tools/dump_code.py --no-docs  # code only
```
