# Geostatistical ground modelling — synthetic CPT dataset & diagnostics

Implements Parts A and B of [cpt_gp_plan.md](cpt_gp_plan.md): a synthetic CPT
dataset for an offshore wind farm site, plus the map and diagnostic plots, plus
the per-unit baseline estimator everything spatial gets measured against.
**Simple, ordinary and universal kriging** are built on top of it, with
leave-one-CPT-out validation, variogram and directional-variogram figures,
per-hole depth profiles and per-model field maps.  The Gaussian process and
hyperparameter recovery are specified separately and are **not** built here —
see [plans/02](plans/02_estimators_and_validation.md).

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e .            # Windows: .venv/Scripts/pip

.venv/bin/cpt-geostat synth run --out run   # generate + plot + model
.venv/bin/python -m pytest -q
```

`pip install -e .` is not optional convenience — it is what puts the
`cpt-geostat` command on the path and makes `import cpt_geostat` work from a
project directory rather than only from the repository root.

`run/data/` gets the four output files; `run/figures/` gets 18 figures;
`run/models/` gets the baseline and CV tables, the cross plots, and the
variogram, field-map and depth-profile figures (see
[what `cli model` writes](#what-cli-model-writes)).

```bash
cpt-geostat synth generate --config config.yaml --out data/
cpt-geostat plot  --data data/ --out figures/
cpt-geostat model --data data/ --out models/
```

The split is deliberate and is the package's architecture made visible:
`synth` needs a generator config and produces truth; `plot` and `model` need
only a directory holding `cpt_samples` and `layers`, and run identically on
real data.  Both print the mode they are in (`mode = synthetic (truth found)`
or `mode = real data (no truth)`), so a real-data run never looks like a
synthetic one whose truth diagnostics silently went missing.

## Conventions (fixed once — see `cpt_geostat/geometry.py`)

| | |
|---|---|
| Modelling variable | `log(Qtn)` — stress exponent `n = 1` on IJmuiden, so its `Qtn` *is* Robertson's `Qt`; see `cpt_geostat/cpt.py` |
| Coordinates | kilometres, origin at site centre |
| Depth | metres below seabed, positive down |
| **Azimuth** | **degrees clockwise from north** (0 = +y = N, 90 = +x = E) |
| `range_km` | **practical range** — separation at which correlation reaches 0.05 |

Two of these are load-bearing and are enforced by tests rather than by comment:

* **Azimuth.** The plan flagged this as the most likely source of silent bugs.
  It is applied identically to trend directions, anisotropy angles and channel
  orientations, converted for gstools in exactly one place
  (`azimuth_to_math_angle`), and checked numerically in
  `tests/synthetic/test_geometry.py::test_anisotropy_major_axis_follows_its_azimuth`
  — a field that is anisotropic *the wrong way* looks perfectly plausible on a map.
* **Range.** gstools parameterises by `len_scale`, which is not the range a
  variogram reports: for Matérn 2.5 the practical range is 3.74× the `len_scale`.
  Passing a practical range straight through makes every field several times
  smoother than configured — invisible by eye, fatal to hyperparameter recovery.
  `GrfConfig.range_kind` records which is meant (`practical` by default) and
  `truth.yaml` writes out all three resolved scales.  The conversion and its
  inverse now live together in `cpt_geostat/covariance.py`, so the generator and the
  estimators share one site rather than one each.

## Outputs

| File | Contents |
|---|---|
| `cpt_samples.csv` | `cpt_id, x, y, z, unit_id, Qtn` — one row per depth reading |
| `layers.csv` | `cpt_id, unit_id, z_top, z_bot` — absent units are missing rows |
| `unit_summary.csv` | `cpt_id, x, y, unit_id, log_Q_mean, log_Q_sd, n_samples, thickness_m` — the model input |
| `truth.yaml` | resolved generative parameters + realised statistics (synthetic only) |
| `truth_fields.npz`, `truth_points.csv` | truth rasters and true CPT node values, for the B1 panels (synthetic only) |

`truth_fields.npz` / `truth_points.csv` are an addition to the plan's Part A7:
the B1 panels need the generating rasters, and keeping them out of the four
contract files leaves the real-data path clean.

### Running on real data

Supply `cpt_samples` and `layers` only.  `unit_summary` is recomputed from the
samples if absent, truth-dependent diagnostics are skipped, and the unit palette
falls back to a default cycle:

```bash
python -m cpt_geostat.cli plot --data /path/to/real --out figures/
```

Each table may be `.csv` or `.parquet`; parquet wins where both exist, and is
what real exports should use — the IJmuiden site is 129 MB of csv against 4 MB
of parquet.  This is covered by `tests/plots/test_data_plots.py`, which runs
every B2 plot against a directory holding nothing but those two tables.

### Getting a real export into that shape

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
rename map — [projects/IJmuiden/prepare_data.py](projects/IJmuiden/prepare_data.py)
is 122 lines, of which the pipeline is six.

The rules it applies are the ones that are **wrong by default and invisible when
wrong**, which is why they live in one tested place rather than in a copy per
project:

* Contacts go at the **midpoint between the last reading of one run and the
  first of the next**.  Taking either reading instead makes every layer short by
  a sample interval and changes nothing you could see on a plot.
* Runs are detected **before** `drop_units` is applied, so an unclassified
  interval leaves a real gap instead of fusing the units either side of it.
* A unit that re-enters lower down collapses to one row on the contract's merge
  key, with `thickness_m` the **sum** of its occupied runs rather than the span,
  so it is not credited with the material between them.
* Readings with no usable value leave `cpt_samples` but still count towards
  layer geometry — the unit is logged whether or not the resistance reduced.

`tests/test_prepare.py` pins each of these against the plausible-looking wrong
answer.  [projects/IJmuiden/](projects/IJmuiden/README.md) is the worked example
(194 CPTs, 23 units, 483k readings).

## Contrast set

Each unit isolates one effect, so a failure points at a cause:

| Unit | Presence | Trend | Anisotropy | Purpose |
|---|---|---|---|---|
| 1 | everywhere | none | isotropic | GP ≡ simple kriging identity test |
| 2 | everywhere | 0.10/km at 115° | isotropic | trend handling only |
| 3 | channel (~28%) | none | ratio 4 at 70° | anisotropy only |
| 4 | patch (~45%) | 0.06/km at 25° | ratio 3 at 135° | both, on different bearings |
| 5 | narrow channel (~16%) | weak | isotropic, 1.5 km range | near data-limited |
| 6 | patch (~30%) | none | isotropic, nugget > sill | noise-dominated failure case |

Realised coverage is written to `truth.yaml`; it differs from the target because
presence is drawn, and it is what should be quoted when interpreting which units
the estimators struggle on.

## The baseline model

`cpt_geostat/models/baseline.py` fits one mean and one sd per unit and no spatial term
— the model a GI report already contains implicitly, written down so that
everything spatial has something to beat.  `cpt_geostat.cli model` writes
`unit_baseline.csv` and, on synthetic data, `unit_baseline_vs_truth.csv`.

Three different "average and std" appear in that table and they answer three
different questions.  Conflating them is the easy mistake, so each has its own
column:

| Column | Population | Reducible by more CPTs? |
|---|---|---|
| `log_Q_mean` ± `log_Q_sd` | per-CPT depth-averages, **one weight per CPT** | no — this is the spread of a new location |
| `se_mean` | uncertainty on the level itself | yes, as `1/√n` |
| `within_sd` | depth-to-depth scatter inside one hole | no — it is the texture of the trace |
| `reading_mean` ± `reading_sd` | every reading, **thickness-weighted** | no — quote this for "what will the cone see" |

`reading_sd` is computed from `cpt_samples` rather than by adding the other two
in quadrature, which both double-counts the depth-averaging error and drops the
thickness weighting; on this dataset the two errors do not cancel.

The truth comparison reports `structured_frac` — the share of the baseline's
variance a perfect spatial model could remove.  It runs 0.83–0.97 on units 1–5
and **0.44 on unit 6**, which is the designed noise-dominated case: the baseline
is close to the best available answer there, and an estimator that appears to
beat it is overfitting.  The remainder is `obs_noise_sd`, measured as
`Var(log_Q_mean − log_Q_field)` — nugget *plus* depth-averaging error.  Measured
against `log_Q_true` instead it would capture only the averaging error and
understate the noise floor by nearly a factor of two, since `log_Q_true` already
carries the nugget.

### Cross plots

`cli model` also writes `cross_plot_observed.png` and — synthetic only —
`cross_plot_latent.png`: predicted against true, one panel per unit, with
`cv_predictions.csv` and `cv_scores.csv` behind them.  Predictions are
**leave-one-out**; the baseline's fitted value *is* the mean of the points it
would otherwise be plotted against, so an in-sample cross plot would flatter it
for the one reason that cannot generalise.

The two figures are the same folds scored against different things, and the
contrast is the point:

| | vs. the held-out **observation** | vs. the **latent** field |
|---|---|---|
| error bar | `sd_obs` (latent + nugget + averaging error) | `sd_latent` |
| coverage 95 | 0.91–0.97 | 0.13–0.50 |
| MSSR | 1.01–1.09 | 12–116 |

Both are correct. Scored on what it actually predicts — one noisy
depth-average — the constant model is **well calibrated**: it attributes all
spatial variation to noise and gets the interval width right. Scored against
the field, the same intervals are out by up to 11× in sd, because "the ground is
flat, and I know its level to ±0.03" is a false claim. Reporting only one of
these numbers would mislead whichever one you picked, which is why both targets
are carried on every row of `cv_predictions.csv` and the figures label which
they used.

The leave-one-out `r²` is `1 − (n/(n−1))²` exactly — always slightly negative,
never zero. That is the line a spatial estimator has to clear.

## Kriging — simple, ordinary and universal

`cpt_geostat/models/kriging.py` holds all three.  They differ **only** in what they
assume about the mean, and that is the whole reason to keep all three:

| | mean | dof | when it is the right answer |
|---|---|---|---|
| `SimpleKriging` | known constant | 0 | the mean is genuinely known |
| `OrdinaryKriging` | unknown constant | 1 | the honest default on real data |
| `UniversalKriging` | unknown linear drift | 3 | the unit carries a trend |

`SimpleKriging` is the direct upgrade from the baseline: same constant mean, but
the residual is now spatially correlated instead of being called noise.
`OrdinaryKriging` pays for the mean instead of being handed it — one degree of
freedom, which shows up as strictly wider intervals everywhere and conspicuously
so far from data.  On the 117-CPT synthetic run that gap is negligible (the mean
is well determined by then); it is the units with 20-odd CPTs where it matters.

**Universal kriging needs a detrended variogram, and that is not a refinement —
without it it does not work at all.**  A variogram of trended data is unbounded:
the trend keeps adding variance as the lag grows, the fit runs to the range
bound, and nearly all the variance lands in a long-range structured component
with a near-zero nugget.  Kriging then predicts with intervals several times too
narrow.  Fitting the variogram to the OLS residuals instead is what makes the
drift model actually pay off, and the residual sill is what UK wants anyway —
the drift is modelled explicitly, so it must not also be counted as structure.

On unit 2 (0.10/km at 115°), leave-one-out against the held-out observation:

| | RMSE | coverage 95 | MSSR |
|---|---|---|---|
| baseline | 0.547 | 0.966 | 1.01 |
| SK (fitted) | 0.242 | 0.812 | 2.29 |
| OK (fitted) | 0.242 | 0.812 | 2.29 |
| **UK (fitted)** | **0.224** | **0.974** | **0.90** |

Scored against the *latent* field the contrast is starker still — SK's MSSR is
12.3 against UK's 1.07 — because there the unmodelled trend has nowhere to hide.

UK is not free, and the suite asserts that it is not: on the four units with no
trend it *loses* to SK, because three drift parameters fitted to noise cost
variance and buy nothing.  An implementation that won everywhere would mean the
drift was being fitted to noise and rewarded for it.  On unit 5 — 22 CPTs in a
narrow channel — UK is badly overconfident, which is the argument for the
IJmuiden estimability gate rather than for tuning UK.

Leave-one-out RMSE against the held-out observation, and MSSR alongside:

| unit | baseline | SK (fitted) | SK (truth cov) | | MSSR fitted | MSSR truth |
|---|---|---|---|---|---|---|
| 1 | 0.300 | 0.247 | **0.246** | | 1.05 | 1.00 |
| 2 | 0.547 | **0.242** | 0.272 | | 2.29 | 1.38 |
| 3 | 0.324 | 0.337 | **0.291** | | 1.32 | 0.98 |
| 4 | 0.422 | 0.367 | **0.345** | | 1.25 | 1.35 |
| 5 | 0.291 | **0.253** | 0.264 | | 1.31 | 0.87 |
| 6 | **0.239** | 0.247 | 0.256 | | 1.21 | 0.96 |

**The truth covariance is not a cheat, it is the control.**
(`cpt_geostat.synthetic.truth.truth_kriging_factory` — synthetic-only, which is why it
sits there and not in `models/`.)  It hands kriging the
generating covariance, which separates *"is kriging implemented correctly"* from
*"can a variogram be fitted from 30 CPTs"* — two questions a single number
confounds, and the second is much the harder. With the true covariance, kriging
beats the baseline on every unit except 6 and is calibrated (MSSR 0.87–1.38).
That validates the implementation. With a *fitted* covariance it predicts about
as well but is consistently overconfident, which localises the remaining error
to variogram estimation rather than to the kriging.

Three things that are working as designed rather than failing:

* **Unit 6 is the one unit the baseline wins.** Nugget above sill, `structured_frac`
  0.44 — there is nothing for kriging to exploit, and it correctly declines to
  invent any. An estimator that appeared to beat the baseline here would be
  overfitting.
* **Unit 2 is flagged `resolved=False`.** Its 0.10/km trend makes the variogram
  unbounded, so the fitted range runs to the identifiability bound. Simple
  kriging assumes a *constant* mean; the honest output is a flag, not a number,
  so `params_["range_km"]` comes back `None` with a `not_identifiable` note.
  Fixing it needs universal kriging, which is the estimator document's job.
* **Units 3 and 5 fit a nugget of exactly zero.** Unit 3's minor axis is 1 km
  against a ~1.3 km CPT spacing, so an isotropic variogram sees noise. Nearly
  all nugget is the *correct* fit, not a failure — and `VariogramFit.resolved`
  says which.

### Fitting the variogram is where this gets fragile

An unconstrained three-parameter fit on 20–120 scattered points does not fail
loudly: the first attempt returned a length scale of **1700 km** on unit 2 and
22 km on unit 4. Three constraints, in `models/variogram.py`:

1. **Bins resolve short lags** (first edge at 200 m, not 600 m). The nugget is
   only identifiable from pairs closer than the correlation range, which on a
   turbine grid means the four clustered CPTs `config.yaml` puts 60 m apart.
   With uniform bins those six pairs get averaged away and the nugget fits to
   zero; resolving them recovers 0.0076 against a true 0.0083.
2. **Lags capped at half the maximum separation**, and `len_scale` bounded to
   match — a range beyond the longest fitted lag is not identifiable, so it is
   not offered, and `at_range_bound` records when the fit ends up there.
3. **Total sill pinned to the sample variance.** It is the one quantity these
   data estimate well, and pinning it stops sill, nugget and range trading off
   freely.

A fourth guard floors the nugget at γ(h_min), but **only when h_min is inside
10% of the fitted range** — γ(h) is nugget *plus* structure, so past that it is a
range estimate in disguise. It fixed unit 1 (MSSR 2.27 → 1.05) and correctly
declined to act on units 3, 5 and 6.

### Seeing the fit, not just its verdict

`cli model` writes `variograms/variogram_<unit>.png` for every unit it can fit.
Each guard above is drawn rather than asserted: the sample variance the sill is
pinned to, the nugget and the short-lag bins that are the only thing identifying
it, the practical range, and a shaded band beyond half the maximum separation
where a range is not identifiable at all. A refused fit gets
`why_not_resolved()` printed on its face — so unit 2's range pressed against the
9.8 km bound is *visible* as a curve climbing past a wall, rather than a
`resolved=False` in a csv. On synthetic runs the generating covariance is
overlaid, which turns the figure into a direct recovery check; for an
anisotropic unit both axis curves are drawn as a band, because the
omnidirectional estimate is a mixture and is not expected to match either.

`variograms/variogram_directional_<unit>.png` splits the lags into four azimuth
sectors, and is the only thing that distinguishes *"no structure"* from
*"structure I was averaging away"*. Unit 3 is the case in point: its minor axis
is finer than the CPT spacing, so the isotropic fit correctly sees noise, while
the sector nearest its 70° major axis stays visibly lowest. A compass panel
carries the sectors and, on synthetic data, the true anisotropy ellipse — if the
lowest sector lies *across* that ellipse rather than along it, the bearing
convention is inverted.

Directional figures are only written where **n ≥ 30 CPTs**, the estimability
tier from plan 02: four sectors over ten lag bins needs roughly that many, and
below it the sector curves are sampling noise that reads as dramatic anisotropy.
The sectors also use deliberately coarser bins than the omnidirectional fit, for
the same reason — splitting the pairs four ways and keeping the same bins leaves
4–9 pairs in each.

### Fitting an axis, not just drawing one

The sector figures are a *diagnostic*; `models/anisotropy.py` fits the axis, by
plan 03's chosen method — profile the likelihood over the major-axis azimuth on
a grid, fit the rest at each angle, and read the likelihood-vs-angle curve as
the identifiability check. A flat curve is a result, not a failure.

Two departures from the plan as written, both deliberate:

* **No sklearn.** It is not a dependency here, and adding a GP estimator stack
  alongside the gstools one is far more than "one anisotropy method". The
  likelihood is computed directly — a Cholesky of the covariance matrix — with
  the covariance built through `covariance.build_model`, so the azimuth
  convention and the gstools `anis`/`angles` inversion stay in the one place
  already tested for them. The method is unchanged; only the objective's
  arithmetic is.
* **The scan runs 0–180°, not 0–90°.** Plan 03's half interval is correct for
  the *unordered ARD* parameterisation it assumed, where `(θ, l₁, l₂)` and
  `(θ+90, l₂, l₁)` are the same model. This module uses the package's ordered
  form — `aniso_ratio = major/minor ≥ 1`, azimuth naming the **major** axis — in
  which no such degeneracy exists and a 120° axis is genuinely different from a
  30° one. Scanning [0, 90) here would silently make half the bearings
  unreachable; a parametrised test pins it.

**A fitted ratio is not evidence of anisotropy, and this is the trap.** Synthetic
unit 6 is isotropic by construction, noise-dominated, and comes back `ratio 9.35
at 80°` — indistinguishable at a glance from a real channel. Three things guard
against reading that as a finding:

| guard | what it catches |
|---|---|
| `at_ratio_bound` | the ratio sitting on `MAX_RATIO` — the likelihood shrinking a minor axis it cannot see to nothing |
| `minor_resolved` | a fitted minor range shorter than the closest pair of CPTs, i.e. structure finer than the survey sampled |
| `null_lr_threshold` | the likelihood ratio an *isotropic* field of the same covariance produces at the same locations |

The null is simulated rather than looked up, because under isotropy the azimuth
is not a parameter at all — it is unidentified, which is Davies' problem, and a
`chi2(2)` p-value would manufacture significance. Simulating at the survey's own
locations also inherits its geometry, which is what generates spurious axes in
the first place.

Recovery against the contrast set, at 60 simulations per unit:

| unit | truth | fitted | LR | gate | called |
|---|---|---|---|---|---|
| 1 | isotropic | 1.07 @ 160° | 2.0 | null p95 13.1 | isotropic ✓ |
| 2 | isotropic + trend | 1.27 @ 40° | 4.3 | null p95 9.5 | isotropic ✓ |
| 3 | **ratio 4 @ 70°** | 7.87 @ 100° | 5.1 | minor axis unresolved | isotropic ✗ |
| 4 | **ratio 3 @ 135°** | 3.16 @ 100° | 4.5 | null p95 5.5 | isotropic ✗ |
| 5 | isotropic | — | — | n = 22, under the gate | isotropic ✓ |
| 6 | isotropic, noise-dominated | 9.37 @ 80° | 2.4 | null p95 5.6 | isotropic ✓ |

**Every isotropic unit is correctly rejected, and both anisotropic units are
missed.** That is the honest read: the method has no useful power at these
sample sizes and these ratios. Unit 3 has 31 CPTs and a minor axis finer than
the CPT spacing — the structural gates catch it before the null is even reached.
Unit 4 has 58 CPTs, recovers the ratio well (3.16 against a true 3) and the
bearing poorly (100° against 135°), and lands just under its threshold.

That conclusion is *weaker* than an earlier version of this section claimed, and
the reason is worth recording. Two numerical traps, both fixed and both pinned
by tests:

* **The null simulated the total variance as structured, with no nugget.**
  Fields far smoother than the data, so spurious axes fit them better, so the
  threshold came back inflated — ninefold on unit 2.
* **`(range, nugget)` is a ridge, not a basin.** A short range with no nugget
  and a long range with a large one explain the same data to within a fraction
  of a log-likelihood, so a single optimiser start seeded from the variogram fit
  never left its basin. The isotropic reference is now multi-started from twelve
  points, and the winning azimuth is refined from three, so the two sides of the
  likelihood ratio are searched alike.

The second fix *lowered* every LR statistic, because the isotropic reference it
is measured against had been under-fitted. Unit 4's LR fell from 18.0 to 4.5 and
crossed from a detection to a miss. **An inflated likelihood ratio against an
under-fitted null is how a method like this manufactures findings**, and it took
a validation harness with planted truth to see it — which is what the contrast
set is for.

### Kriging with a fitted axis

`--anisotropy N_SIM` adds an `OK (aniso)` estimator that uses each unit's fitted
axis wherever it survives the gates, and the isotropic covariance everywhere
else. Nothing in `models/kriging.py` changed: the estimators already accept a
covariance given outright — the path the truth-covariance reference uses — and
`AnisotropyDecision.covariance` returns either that or the literal string
`"fit"`, so a rejected unit runs the *same code* an isotropic run would. A
comparison between the two estimators is then a comparison of the units that
changed and nothing else.

The gate is not conservatism for its own sake. An axis fitted to noise rotates
the prediction **and** narrows its variance along a direction the noise chose,
which is worse than isotropy rather than merely no better.

`anisotropy.csv` is written whether or not any unit passes, because "every unit
stayed isotropic, and here is why each one did" is the result on a site with no
fabric. On the synthetic contrast set no unit passes, and the run says so
outright rather than leaving a reader to hunt for a difference between two
identical figures.

## Field maps — what each model says across the site

`cli model` writes `fields/<model>/prediction.png` for the baseline, SK and OK
(`--fields all` adds UK).  **Everything from one estimator lives under that
estimator's name** — a folder of similar-looking surfaces distinguished only by
a filename suffix is exactly how a model comparison gets mixed up.

One **row per unit, four columns**:

| column | shows |
|---|---|
| `predicted` | the median surface |
| `latent sd` | how well the field itself is known — what more drilling reduces |
| `lower 95%` / `upper 95%` | the two ends of the interval, as maps in their own right |

The bounds are the reason to draw all four together.  A mean map alone invites
being read as *the* answer; putting the interval either side of it turns "how
uncertain is this" from a second figure you hold in your head into something
visible in one glance across a row.  On the synthetic run units 3 and 5 make the
point immediately: their **lower map is almost uniformly dark and their upper
almost uniformly bright**, because away from the holes the interval spans nearly
the unit's whole range.  Units 1 and 2, at 117 CPTs, stay tight.

These are the companion to the B1 truth maps: same style, and on a synthetic run
**the same raster**, so `fields/OK/prediction.png` and
`figures/truth_unit_3.png` can be compared by flipping between them rather than
by eye across different grids.

Fitted **in-sample** — this is the model's statement about the ground given
everything known, not a held-out test.  The depth profiles and cross plots are
where it gets scored on data it was not shown.

**Nothing is masked by presence**, because presence is not modelled yet.  So a
unit held at 4 of 194 holes still gets a surface across the whole site, and the
sd map is what keeps that honest: it grows away from the data, so wherever the
mean map is inventing, the sd map is brightest.  **Read the pair, never the mean
alone.**  Two things the pair makes visible that no number does:

* On the synthetic run, unit 1's sd map is dark along the CPT grid and bright in
  the **thinned NE corner** — the sensitivity-sweep hole in the layout, showing
  up as uncertainty exactly where it should.
* Unit 3's truth is fine ENE–WSW striping (ratio 4 at 70°, 1 km minor axis
  against ~1.3 km CPT spacing).  Its prediction map is bullseyes on a flat
  background, because an isotropic variogram genuinely cannot resolve that —
  which is the documented "correct answer to the wrong question", now visible
  side by side with the truth rather than inferred from a `resolved=False`.

**Colour scales are shared two ways.**  Within a row, predicted / lower / upper
share one scale — scaled separately, a wide interval would look identical to a
narrow one, which is the very thing the row exists to show.  The sd column is a
different quantity and keeps its own.  Across models, the scale for a given unit
spans every estimator, so the same colour means the same value in
`fields/OK/prediction.png` as in `fields/UK/prediction.png` and flipping between
them is a real comparison.

They are *not* shared across units: units differ in level by design (unit 2 sits
near 1.0, unit 6 near 2.4), so one scale for all of them would flatten every
panel to a single hue.

That shared scale earns its keep immediately.  The baseline's map is a flat
colour and its sd map is flat too — but on the common scale that flat sd sits
right at the bottom, because the baseline's latent sd is the standard error of
its one constant, `s/√n`.  On unit 1 it claims to know the field to **±0.028
everywhere**, against ordinary kriging's **±0.28** away from the holes.  A
ten-fold difference in claimed precision, and per-figure auto-scaling hid it
completely: the baseline's map looked unremarkable next to everything else.
That is the point `models/baseline.py` makes in prose — the shrinking latent sd
"is not a claim that the field is flat to that precision" — made visible.

Five of IJmuiden's 23 units come out flat under OK as well, labelled *"uniform —
no spatial structure resolved"*: the pure-nugget fits, where kriging has
correctly collapsed onto the baseline.

## Qtn with depth — the prediction against the measured trace

`cli model` writes one figure per hole to `profiles/`, reconstructing `Qtn` down
the hole from the per-unit predictions and laying it over that hole's own
readings.  Every hole is predicted **leave-one-out**, so each figure answers
*"if this hole had never been drilled, what would we have said?"* — an in-sample
profile would be showing the model its own answer.

Three nested uncertainties, and they are not interchangeable:

| band | sd | question |
|---|---|---|
| latent | `sd_latent` | where is this unit's *true* value here |
| unit mean | `sd_obs` | what would a new *depth-average* here be |
| single reading | `√(sd_obs² + within_sd²)` | where will one 2 cm reading fall |

Only the third is comparable with a raw trace, and on IJmuiden it is much the
widest: `within_sd` runs 0.45–0.56 against a between-hole `log_Q_sd` of
0.31–0.71, so **depth texture inside one hole is as large as the variation
between holes.** Quoting the narrow band against a trace would claim a precision
the model does not have; quoting only the wide one would hide how much of it a
spatial model can actually reduce. Both are drawn, nested.

`exp(pred)` is a **median**, not a mean — it is the exponential of a mean of
logs. The bands are built in log space and exponentiated, which is exact rather
than approximate, because `exp` is monotone and quantiles are equivariant under
a monotone map. Qtn is plotted on a log axis so the intervals read symmetrically.

**The band's width is measured, not asserted.** It is built from a decomposition
that knowingly double-counts the depth-averaging error (`sd_obs` already contains
it, and it is itself produced by the within-unit scatter), so
`profiles/<model>/profile_coverage.csv` and `profile_calibration.png` report the
*realised* coverage per unit. On IJmuiden, ordinary kriging gives:

| | realised coverage | MSSR |
|---|---|---|
| 17 of 23 units | 0.94–0.97 | 0.89–1.13 |
| `GGM_23_C`, `GGM_31_Si`, `GGM_22_Si` | 0.82–0.91 | 1.42–1.54 (narrow) |
| `GGM_27_S`, `GGM_24_C_S` | 1.00 | 0.64–0.73 (over-wide, few CPTs) |

Against a nominal 0.95 that is close calibration on real ground, and the
double-counting turns out to be negligible next to everything else. On synthetic
data the residuals come out essentially N(0,1) with bias −0.01; the two units
that miss are unit 2 and unit 4, the two carrying trends that ordinary kriging
structurally cannot model — the same failure that shows up as MSSR 2.29 in the
cross-validation table, and the reason universal kriging exists.

Profiles are written for **every hole** by default (194 on IJmuiden, a couple of
minutes) into `profiles/<model>/`; `--profiles none` or `--profiles N` limits
it, and `--profile-model` selects which cross-validated estimator to show.

### Back to `qt` — the units a design is specified in

`cpt_geostat/cpt.py` inverts the normalisation, and `models.profile.qt_readings` /
`qt_by_unit` apply it to a prediction table:

```python
profile, n = normalisation_from_metadata(json.load(open("data/prepare_metadata.json")))
in_qt = qt_readings(reading_predictions(ds, cv, model="OK (fitted)"), profile, n)
```

The stress exponent and unit weights come from the *data*, not from this
package's defaults: `prepare_metadata.json` records what the supplied column was
built with, because `cpt_samples` keeps `Qtn` and drops `qt`, so nothing else
remembers. On IJmuiden that is `n = 1` with a buoyant gradient of 10.1 kN/m³,
both recovered from the export by `cpt.infer_gamma_eff`. **`n` is worth checking
before trusting a column's name** — inverting an `n = 1` column as if it were
`n = 0.5` is wrong by a factor of 2–3, and wrong in a way that varies with depth,
so it reads as stratigraphy rather than as an error.

Two properties make the uncertainty survive the trip:

* At fixed depth `qt = σv0 + c(z)·Qtn` is **affine** in `Qtn`, so a lognormal
  `Qtn` gives a shifted, scaled lognormal `qt` — its median, mean and sd are
  closed-form, and `sd` scales with `c(z)` while the shift by `σv0` moves only
  the mean.
* The map is **monotone**, so band edges transform exactly. A 95% band in
  `log(Qtn)` is a 95% band in `qt`, and the coverage `profile_coverage` measures
  carries over unchanged — 0.956 on IJmuiden either side of the conversion.

`qt` is right-skewed at these sds, so `qt_median ± qt_sd` is *not* the interval;
the upper arm of the 95% band is 2–3× the lower. Quote quantiles, and treat
`qt_sd` as a moment for propagating into a downstream calculation. And a
per-unit prediction is a *line*, not a number: constant `Qtn` through a unit
de-normalises to a `qt` rising linearly with depth, which is why `qt_by_unit`
returns top, mid and base.

`cli model` writes both suites — `profiles/<model>/Qtn/` and
`profiles/<model>/qt/` — on the axis each is legible on: **log** for `Qtn`, the
space the model is Gaussian in, and **linear MPa** for `qt`, the way a CPT log
is read. The qt figure frames on the trace and the median and lets the bands
overflow, captioning which ones did: in `qt` a `log_Q_sd` of 0.4 is a factor of
2.2 each way, so a band-framed axis would squeeze the trace into the left
quarter of the page. `profile_coverage.csv` and `profile_calibration.png` are
written once, at the model level — the transform is monotone, so the calibration
is one result about both suites. `--profile-units Qtn|qt|both` selects.

`--profile-model` takes one estimator, a comma-separated set, or `all`, each
into its own directory. The profiles are the one output where the estimators
differ *hole by hole* rather than in a summary statistic — the baseline draws
the same band at every location, and how much a kriged one narrows it at this
particular hole is only visible side by side. On IJmuiden `all` is four
estimators × 194 holes × 2 suites, so it is not the default.

### A variable stress exponent, and the distribution of `qt`

`n` may be a single number or a `unit_id -> n` mapping, everywhere an exponent
is taken. To *get* one, `cpt.soil_behaviour_index` runs Robertson's iteration —
`Ic` from `Qtn` and `Fr`, `n = 0.381·Ic + 0.05·σ'v0/pa − 0.15` capped at 1.0,
repeat — over raw `qt` and `fs`. It needs sleeve friction, so it belongs at
preparation time: the contract keeps `Qtn` and drops both `qt` and `fs`, which
is exactly why the transform takes `n` as an input. On IJmuiden's export it
converges in 26 iterations and sorts the units by soil type unprompted:

| | `Ic` | fitted `n` |
|---|---|---|
| sands (`GGM_01_S` … `GGM_27_S`) | 1.56–1.92 | 0.48–0.72 |
| silts (`GGM_22_Si`, `GGM_31_Si`) | 1.90–2.24 | 0.62–0.91 |
| clays (`GGM_03_C` … `GGM_27_C`) | 2.41–2.89 | 0.80–0.98 |

**That table is not licence to use those exponents on this dataset.** IJmuiden's
supplied column was built with `n = 1`, and a column must be inverted with the
exponent that made it. Using per-unit `n` here would mean re-normalising from
raw `qt` and re-running the model — a different dataset, not a different plot.

The distribution of `qt` survives all of this, because `qt = σv0 + pa·e^(nL)·Qtn`
is affine in `Qtn` at any fixed `n`. Writing `L = log(σ'v0/pa)`, a Gaussian `n`
contributes a *lognormal factor*, and a product of lognormals is lognormal:

```
log(qt − σv0) ~ N( log(pa) + n·L + μ,  sd² + (n_sd·L)² )
```

So `qt` stays a shifted lognormal, every closed form still holds, and the only
change is a wider sd — `total_log_sd` combines the two in quadrature. Verified
against 2M-draw simulation to 3–4 significant figures at five depths.

`L` is the whole lever, and it is **zero at σ'v0 = pa**, which on IJmuiden's
gradient is 9.9 m below seabed. At that depth the exponent is irrelevant; away
from it the doubt grows logarithmically in both directions. `n_sd = 0.1` adds
0.30 to the log-sd at 0.5 m and 0.18 at 60 m, against a between-hole `log_Q_sd`
of 0.31–0.71 — so **at the seabed the exponent is as large a source of doubt as
the spatial model, and at 10 m it is nothing.** That is the opposite of most
people's intuition and is the reason to carry `n_sd` at all.

Two caveats, both stated in the code. The closed form assumes `n ⟂ Qtn`, and
Robertson's `n` is a function of `Ic` which is a function of `Qtn` — sample
instead if the dependence matters. And a Gaussian `n` ignores the `n ≤ 1` cap,
so for a unit sitting at the cap the upper half of the assumed spread is
unphysical. Setting `n_sd > 0` also sets `attrs['coverage_transfers'] = False`
and captions the figure: the measured `qt` on the same row is itself conditional
on the central `n`, so a band widened for exponent doubt is no longer being
scored fairly against it.

### The stress profile

The qt suite needs a stress profile, and takes it from the dataset's
`prepare_metadata.json`; a dataset that records none is **skipped with a
message** rather than defaulted, since a guessed exponent is a depth-dependent
error that reads as stratigraphy. `--stress-profile GAMMA_SAT GAMMA_W
[--stress-exponent N]` is the override for datasets whose preparation step
predates the metadata block — including the synthetic ones, which have no
underlying `qt` at all. `--stress-exponent-sd` widens the qt bands for exponent
doubt; `n` itself is a property of the data, but how much doubt to carry is the
caller's judgement, so the flag overrides whatever the metadata records.

### What `cli model` writes

```
models/
  unit_baseline.csv  cv_scores_*.csv  cross_plot_*.png
  variograms/                    # the fitted covariance SK and OK use —
    variogram_<unit>.png         #   UK refits on detrended residuals, so its
    variogram_directional_<unit>.png   #   covariance is NOT the one drawn here
  fields/<model>/
    prediction.png               # one row per unit x 4 columns
  profiles/<model>/          # one directory per estimator asked for
    profile_calibration.png  profile_coverage.csv   # one calibration, both suites
    Qtn/profile_<cpt>.png    # log axis — the space the model is Gaussian in
    qt/profile_<cpt>.png     # linear MPa — needs a stress profile; see above
```

Two conventions worth knowing before reading any number out of `models/`:

* **The two variances are not interchangeable.** `predict` returns the *latent*
  field and `predict_observation` adds `noise_var_`.  Estimator comparisons want
  the first; every cross-validation metric wants the second, because a held-out
  value is an observation.  `models/base.py` states which is which.
  For kriging this is load-bearing and was **verified, not assumed**: gstools'
  kriging variance tends to `sill + nugget` far from data, so it is an
  *observation* variance and `predict` subtracts the nugget back off. Taking it
  as latent would overstate field uncertainty by the whole nugget — most of the
  variance on unit 6. `test_gstools_kriging_variance_includes_the_nugget` pins
  it, so a library upgrade fails a test rather than quietly shifting every
  calibration number.
* **Truth sds are measured over the CPTs a unit is actually present at**, not
  over the whole site as `truth.yaml` records them.  A narrow channel's latent
  spread over its own 16% of the site is 1.5× smaller than over all of it, and
  comparing across that difference credits a spatial model with explaining more
  variance than the data contains.

## Layout

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

**The dependency direction only ever points one way**: `contract` and `models`
never import from `synthetic`; the generator is a *consumer* of them.  That is
easy to reverse by accident — one convenient import of `Dataset` from the
generator and loading two CSVs of real data starts pulling in the field
simulator, which is exactly what the previous layout did — so it is pinned by
`tests/contract/test_architecture.py` rather than documented and hoped for.
Importing `cpt_geostat.contract` loads six modules and neither gstools nor scipy.

`validate/` holds only leave-one-out and the metrics a cross plot needs; the GP,
ordinary/universal kriging, hyperparameter recovery and the remaining scores
belong to the estimator document.

Tests mirror the same split: `tests/contract/`, `tests/models/`, `tests/plots/`
and `tests/synthetic/`.

Every plotting function takes a dataframe or `Dataset` and returns a
`matplotlib.figure.Figure`.  No plotting function touches the filesystem — `cli.py`
owns all saving — so the harness can call them on real data unchanged.

## Notes on the generator

* **Presence** is a *probability* raster; the Bernoulli draw happens once, at the
  CPT locations.  Drawing per raster cell would give salt-and-pepper edges that
  no downstream sampling could undo.  Channel edges are made ragged by perturbing
  the bank position with a short-range GRF rather than by blurring.
* **Thickness** maps a unit-variance GRF through a logistic into `[min_m, max_m]`,
  with the logistic's two parameters solved by Gauss–Hermite quadrature so the
  realised mean and sd match the config.  An unattainable sd warns rather than
  silently delivering something else.
* **Missingness has two correlated sources**: lateral absence, and pinch-out
  where shallow units are thick enough to push a deep unit past the 50 m cut-off.
  A unit retaining less than `min_thickness_m` is marked absent, not kept as a sliver.
* **Within-unit scatter** is AR(1) with a configurable correlation length, so the
  effective sample size behind a depth-average is much smaller than `n_samples`.
  This is why the realised within-unit sd sits below the configured one, and it is
  what the depth-averaging step has to survive being tested against.
* **RNG streams are keyed by name**, not draw order, so editing one unit does not
  re-roll the rest of the site (`test_streams_are_named_not_ordered`).

## Relationship to the plan document

`cpt_gp_plan.md` originally wrote the A5 trend as `grad · [cos(az)·x + sin(az)·y]`
— the mathematical convention (counter-clockwise from east) — which contradicted
its own Section 0 declaration of azimuths as clockwise from north.  The code
resolved it in favour of Section 0, and the plan document has since been
corrected to `grad · [sin(az)·x + cos(az)·y]` so the two agree.

The GP, ordinary/universal kriging and hyperparameter recovery are not built
yet; see [plans/02_estimators_and_validation.md](plans/02_estimators_and_validation.md).
`validate/cv.py` is the harness they slot into: pass another factory and the
cross plots draw every model in the same panel.
