# Geostatistical ground modelling — CPT data

Spatial modelling of normalised cone tip resistance (`Qtn`) across an offshore
wind farm site: a synthetic CPT dataset with planted truth, a per-unit baseline
estimator, simple/ordinary/universal kriging on top of it, and the validation
harness and figures that say which of them is actually working.

The same code runs on real ground investigation data.  Anything meeting the
two-table data contract — `cpt_samples` and `layers` — goes through the plotting
and modelling paths unchanged;
[projects/IJmuiden/](projects/IJmuiden/README.md) is the worked example
(194 CPTs, 23 units, 483k readings).

## What it can do

* **Generate a synthetic site** whose truth is known — six soil units, each
  isolating one effect (trend, anisotropy, sparse presence, noise domination) so
  a failure points at a cause.
* **Estimate a per-unit level and spread** with a baseline model that has no
  spatial term — the model a GI report already contains implicitly, written down
  so everything spatial has something to beat.
* **Krige** — simple, ordinary and universal — with variogram fitting under
  explicit identifiability guards, and a fitted-anisotropy axis behind a
  simulated null.
* **Validate** by leave-one-CPT-out, scoring against both the held-out
  observation and (on synthetic data) the latent field, with RMSE, bias, r²,
  coverage and MSSR.
* **Plot** presence/value/thickness maps, fence sections, variograms and
  directional sectors, per-model field maps with their uncertainty, and per-hole
  depth profiles in both `Qtn` and `qt`.
* **Prepare real exports** onto the contract, with the wrangling rules that are
  wrong by default and invisible when wrong pinned in one tested module.

Not built here: the Gaussian process and hyperparameter recovery, specified
separately in [plans/02](plans/02_estimators_and_validation.md).

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e .            # Windows: .venv/Scripts/pip

.venv/bin/cpt-geostat synth run --out run   # generate + plot + model
.venv/bin/python -m pytest -q
```

`run/data/` gets the four contract files, `run/figures/` 18 figures, and
`run/models/` the baseline and CV tables, cross plots, variograms, field maps
and depth profiles.

## How it works

The CLI is three commands, and the split is the architecture made visible:

```bash
cpt-geostat synth generate --config config.yaml --out data/   # needs a config, makes truth
cpt-geostat plot  --data data/ --out figures/                 # needs only the contract
cpt-geostat model --data data/ --out models/                  # needs only the contract
```

`plot` and `model` require nothing but a directory holding `cpt_samples` and
`layers`, so they run identically on real data; both announce which mode they are
in, so a real-data run never looks like a synthetic one whose truth diagnostics
silently went missing.

Inside the package the same line is drawn again.  `contract/`, `models/`,
`validate/` and `plots/` run on any conforming dataset; `synthetic/` needs a
generator config or realised truth, and is a *consumer* of the others.  The
dependency direction is pinned by a test, not by convention — importing
`cpt_geostat.contract` loads six modules and neither gstools nor scipy.

The modelling variable is `log(Qtn)` throughout; estimators work on per-unit
depth-averages, and the depth profiles reconstruct a trace from them.  Every
estimator distinguishes the *latent* field from an *observation* — the second
carries the nugget and the depth-averaging error — because conflating the two is
how a calibration number goes quietly wrong.

## Conventions

| | |
|---|---|
| Modelling variable | `log(Qtn)` |
| Coordinates | kilometres, origin at site centre |
| Depth | metres below seabed, positive down |
| **Azimuth** | **degrees clockwise from north** (0 = +y = N, 90 = +x = E) |
| `range_km` | **practical range** — separation at which correlation reaches 0.05 |

The last two are load-bearing and enforced by tests rather than by comment; see
[decisions §1](docs/DECISIONS.md#1-conventions).

## The data contract

| File | Contents |
|---|---|
| `cpt_samples.csv` | `cpt_id, x, y, z, unit_id, Qtn` — one row per depth reading |
| `layers.csv` | `cpt_id, unit_id, z_top, z_bot` — absent units are missing rows |
| `unit_summary.csv` | `cpt_id, x, y, unit_id, log_Q_mean, log_Q_sd, n_samples, thickness_m` — the model input |
| `truth.yaml`, `truth_fields.npz`, `truth_points.csv` | generative parameters, realised statistics and truth rasters (synthetic only) |

Real data supplies the first two; `unit_summary` is recomputed if absent,
truth-dependent diagnostics are skipped.  Either csv or parquet.

## Repository layout

```
cpt_geostat/     the package — see docs/USAGE.md §5 for the module map
config.yaml      the synthetic generator's single source of truth
projects/        per-project wrangling of real exports; the package never imports from here
plans/           specifications for work not yet built
docs/            documentation (below)
tests/           mirrors the package split
tools/           dump_code.py, which regenerates docs/CODE.md
```

## Documentation

| | |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Commands and flags, conventions, the data contract, every file the CLI writes, the module map |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Why it is built this way — each decision, the alternative rejected, and the evidence |
| [docs/RESULTS.md](docs/RESULTS.md) | What the estimators measurably do, on the contrast set and on IJmuiden |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | The mathematics: generative model, estimators, validation, the `qt` transform |
| [docs/CODE.md](docs/CODE.md) | The whole project as one file, for a reader with no other context (generated) |
| [cpt_gp_plan.md](cpt_gp_plan.md), [plans/](plans/) | The original plan and the specifications for what is not built |

## Status

Built: the synthetic generator (Parts A and B of `cpt_gp_plan.md`), the baseline,
simple/ordinary/universal kriging, variogram and anisotropy fitting,
leave-one-out validation, and the full plotting suite — all of it running on
IJmuiden as well as on synthetic data.

Not built: the Gaussian process, hyperparameter recovery and a presence model.
`validate/cv.py` is the harness they slot into — pass another factory and the
cross plots draw every model in the same panel.
