# Design decisions

Why this repository is built the way it is.  Each entry records the decision, the
alternative that was rejected, and — where there is one — the evidence that
settled it.  Measured numbers quoted here in support of a decision are given in
full in [RESULTS.md](RESULTS.md); the mathematics is in
[METHODOLOGY.md](METHODOLOGY.md).

---

## 1. Conventions

### 1.1 Azimuths are degrees clockwise from north

The plan flagged this as the most likely source of silent bugs.  It is applied
identically to trend directions, anisotropy angles and channel orientations,
converted for gstools in exactly one place (`azimuth_to_math_angle`), and
checked numerically in
`tests/synthetic/test_geometry.py::test_anisotropy_major_axis_follows_its_azimuth`
— a field that is anisotropic *the wrong way* looks perfectly plausible on a map.

### 1.2 `range_km` means the practical range, not gstools' `len_scale`

gstools parameterises by `len_scale`, which is not the range a variogram
reports: for Matérn 2.5 the practical range is 3.74× the `len_scale`.  Passing a
practical range straight through makes every field several times smoother than
configured — invisible by eye, fatal to hyperparameter recovery.
`GrfConfig.range_kind` records which is meant (`practical` by default) and
`truth.yaml` writes out all three resolved scales.  The conversion and its
inverse live together in `cpt_geostat/covariance.py`, so the generator and the
estimators share one site rather than one each.

---

## 2. Architecture

### 2.1 The CLI is split three ways, and the split is the architecture

`synth` needs a generator config and produces truth; `plot` and `model` need
only a directory holding `cpt_samples` and `layers`, and run identically on real
data.  Both print the mode they are in (`mode = synthetic (truth found)` or
`mode = real data (no truth)`), so a real-data run never looks like a synthetic
one whose truth diagnostics silently went missing.

### 2.2 The dependency direction only ever points one way

`contract` and `models` never import from `synthetic`; the generator is a
*consumer* of them.  That is easy to reverse by accident — one convenient import
of `Dataset` from the generator and loading two CSVs of real data starts pulling
in the field simulator, which is exactly what the previous layout did — so it is
pinned by `tests/contract/test_architecture.py` rather than documented and hoped
for.  Importing `cpt_geostat.contract` loads six modules and neither gstools nor
scipy.

### 2.3 No plotting function touches the filesystem

Every plotting function takes a dataframe or `Dataset` and returns a
`matplotlib.figure.Figure`; `cli.py` owns all saving.  The harness can therefore
call them on real data unchanged.

### 2.4 `truth_fields.npz` / `truth_points.csv` sit outside the four contract files

An addition to the plan's Part A7: the B1 panels need the generating rasters, and
keeping them out of the four contract files leaves the real-data path clean.

### 2.5 Truth-covariance kriging lives in `synthetic/`, not `models/`

`cpt_geostat.synthetic.truth.truth_kriging_factory` hands kriging the generating
covariance.  It is a synthetic-only control, so it sits on the synthetic side of
the line rather than in the estimator package.

---

## 3. Preparing real exports

### 3.1 The shared rules live in one tested module

`cpt_geostat.prepare` does the part that is the same at every site, because the
rules it applies are the ones that are **wrong by default and invisible when
wrong**:

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
answer.

### 3.2 Parquet is the intended real-data format

Each table may be `.csv` or `.parquet`; parquet wins where both exist.  The
IJmuiden site is 129 MB of csv against 4 MB of parquet.

---

## 4. The baseline

### 4.1 Three different "average and standard deviation" get three columns

Conflating them is the easy mistake, so each answers its own question — see the
[column reference](USAGE.md#41-unit_baselinecsv).  `reading_sd` is computed from
`cpt_samples` rather than by adding the other two in quadrature, which both
double-counts the depth-averaging error and drops the thickness weighting; on
this dataset the two errors do not cancel.

### 4.2 `obs_noise_sd` is measured against the field, not against `log_Q_true`

`Var(log_Q_mean − log_Q_field)` is nugget *plus* depth-averaging error.  Measured
against `log_Q_true` instead it would capture only the averaging error and
understate the noise floor by nearly a factor of two, since `log_Q_true` already
carries the nugget.

### 4.3 Cross plots are leave-one-out, never in-sample

The baseline's fitted value *is* the mean of the points it would otherwise be
plotted against, so an in-sample cross plot would flatter it for the one reason
that cannot generalise.  The leave-one-out `r²` is `1 − (n/(n−1))²` exactly —
always slightly negative, never zero.  That is the line a spatial estimator has
to clear.

### 4.4 Both scoring targets are carried on every row

`cross_plot_observed.png` and `cross_plot_latent.png` are the same folds scored
against different things, and both are correct.  Scored on what it actually
predicts — one noisy depth-average — the constant model is **well calibrated**.
Scored against the latent field, the same intervals are out by up to 11× in sd,
because "the ground is flat, and I know its level to ±0.03" is a false claim.
Reporting only one of these numbers would mislead whichever one you picked, which
is why both targets are on every row of `cv_predictions.csv` and the figures
label which they used.

---

## 5. Kriging

### 5.1 All three of SK, OK and UK are kept

They differ **only** in what they assume about the mean, and that is the whole
reason to keep all three.  `SimpleKriging` is the direct upgrade from the
baseline: same constant mean, but the residual is now spatially correlated
instead of being called noise.  `OrdinaryKriging` pays for the mean instead of
being handed it — one degree of freedom, which shows up as strictly wider
intervals everywhere and conspicuously so far from data.  On the 117-CPT
synthetic run that gap is negligible; it is the units with 20-odd CPTs where it
matters.

### 5.2 Universal kriging refits the variogram on detrended residuals

Not a refinement — without it it does not work at all.  A variogram of trended
data is unbounded: the trend keeps adding variance as the lag grows, the fit runs
to the range bound, and nearly all the variance lands in a long-range structured
component with a near-zero nugget.  Kriging then predicts with intervals several
times too narrow.  Fitting to the OLS residuals is what makes the drift model pay
off, and the residual sill is what UK wants anyway — the drift is modelled
explicitly, so it must not also be counted as structure.

**Consequence for reading the outputs:** the variogram figures under
`models/variograms/` are the covariance SK and OK use.  UK's is not drawn there.

### 5.3 UK is expected to lose where there is no trend

The suite asserts that it does.  Three drift parameters fitted to noise cost
variance and buy nothing; an implementation that won everywhere would mean the
drift was being fitted to noise and rewarded for it.

### 5.4 The truth covariance is a control, not a cheat

It separates *"is kriging implemented correctly"* from *"can a variogram be
fitted from 30 CPTs"* — two questions a single number confounds, and the second
is much the harder.

### 5.5 `predict` returns the latent field; `predict_observation` adds the nugget

Estimator comparisons want the first; every cross-validation metric wants the
second, because a held-out value is an observation.  For kriging this is
load-bearing and was **verified, not assumed**: gstools' kriging variance tends
to `sill + nugget` far from data, so it is an *observation* variance and
`predict` subtracts the nugget back off.  Taking it as latent would overstate
field uncertainty by the whole nugget — most of the variance on unit 6.
`test_gstools_kriging_variance_includes_the_nugget` pins it, so a library upgrade
fails a test rather than quietly shifting every calibration number.

### 5.6 Truth sds are measured over the CPTs a unit is present at

Not over the whole site as `truth.yaml` records them.  A narrow channel's latent
spread over its own 16% of the site is 1.5× smaller than over all of it, and
comparing across that difference credits a spatial model with explaining more
variance than the data contains.

---

## 6. Variogram fitting

An unconstrained three-parameter fit on 20–120 scattered points does not fail
loudly: the first attempt returned a length scale of **1700 km** on unit 2 and
22 km on unit 4.  Three constraints, in `models/variogram.py`:

1. **Bins resolve short lags** (first edge at 200 m, not 600 m).  The nugget is
   only identifiable from pairs closer than the correlation range, which on a
   turbine grid means the four clustered CPTs `config.yaml` puts 60 m apart.
   With uniform bins those six pairs get averaged away and the nugget fits to
   zero; resolving them recovers 0.0076 against a true 0.0083.
2. **Lags capped at half the maximum separation**, and `len_scale` bounded to
   match — a range beyond the longest fitted lag is not identifiable, so it is
   not offered, and `at_range_bound` records when the fit ends up there.
3. **Total sill pinned to the sample variance.**  It is the one quantity these
   data estimate well, and pinning it stops sill, nugget and range trading off
   freely.

A fourth guard floors the nugget at γ(h_min), but **only when h_min is inside
10% of the fitted range** — γ(h) is nugget *plus* structure, so past that it is a
range estimate in disguise.  It fixed unit 1 (MSSR 2.27 → 1.05) and correctly
declined to act on units 3, 5 and 6.

### 6.1 A refused fit is a flag, not a number

Unit 2's 0.10/km trend makes the variogram unbounded, so the fitted range runs to
the identifiability bound.  Simple kriging assumes a *constant* mean; the honest
output is `params_["range_km"] = None` with a `not_identifiable` note, and
`VariogramFit.resolved` says so.

### 6.2 Every guard is drawn on the figure, not just asserted

The sample variance the sill is pinned to, the nugget and the short-lag bins that
are the only thing identifying it, the practical range, and a shaded band beyond
half the maximum separation where a range is not identifiable at all.  A refused
fit gets `why_not_resolved()` printed on its face — so unit 2's range pressed
against the 9.8 km bound is *visible* as a curve climbing past a wall, rather
than a `resolved=False` in a csv.

### 6.3 Directional figures need n ≥ 30 CPTs, and coarser bins

The estimability tier from plan 02: four sectors over ten lag bins needs roughly
that many, and below it the sector curves are sampling noise that reads as
dramatic anisotropy.  The sectors also use deliberately coarser bins than the
omnidirectional fit, for the same reason — splitting the pairs four ways and
keeping the same bins leaves 4–9 pairs in each.

---

## 7. Anisotropy

The sector figures are a *diagnostic*; `models/anisotropy.py` fits the axis, by
plan 03's chosen method — profile the likelihood over the major-axis azimuth on a
grid, fit the rest at each angle, and read the likelihood-vs-angle curve as the
identifiability check.  A flat curve is a result, not a failure.

### 7.1 Two departures from plan 03 as written

* **No sklearn.**  It is not a dependency here, and adding a GP estimator stack
  alongside the gstools one is far more than "one anisotropy method".  The
  likelihood is computed directly — a Cholesky of the covariance matrix — with
  the covariance built through `covariance.build_model`, so the azimuth
  convention and the gstools `anis`/`angles` inversion stay in the one place
  already tested for them.  The method is unchanged; only the objective's
  arithmetic is.
* **The scan runs 0–180°, not 0–90°.**  Plan 03's half interval is correct for
  the *unordered ARD* parameterisation it assumed, where `(θ, l₁, l₂)` and
  `(θ+90, l₂, l₁)` are the same model.  This module uses the package's ordered
  form — `aniso_ratio = major/minor ≥ 1`, azimuth naming the **major** axis — in
  which no such degeneracy exists and a 120° axis is genuinely different from a
  30° one.  Scanning [0, 90) here would silently make half the bearings
  unreachable; a parametrised test pins it.

### 7.2 A fitted ratio is not evidence of anisotropy

This is the trap.  Synthetic unit 6 is isotropic by construction,
noise-dominated, and comes back `ratio 9.35 at 80°` — indistinguishable at a
glance from a real channel.  Three guards:

| guard | what it catches |
|---|---|
| `at_ratio_bound` | the ratio sitting on `MAX_RATIO` — the likelihood shrinking a minor axis it cannot see to nothing |
| `minor_resolved` | a fitted minor range shorter than the closest pair of CPTs, i.e. structure finer than the survey sampled |
| `null_lr_threshold` | the likelihood ratio an *isotropic* field of the same covariance produces at the same locations |

### 7.3 The null is simulated, not looked up

Under isotropy the azimuth is not a parameter at all — it is unidentified, which
is Davies' problem, and a `chi2(2)` p-value would manufacture significance.
Simulating at the survey's own locations also inherits its geometry, which is
what generates spurious axes in the first place.

### 7.4 The gate is applied before kriging uses an axis

`--anisotropy N_SIM` adds an `OK (aniso)` estimator that uses each unit's fitted
axis wherever it survives the gates, and the isotropic covariance everywhere
else.  Nothing in `models/kriging.py` changed: the estimators already accept a
covariance given outright — the path the truth-covariance reference uses — and
`AnisotropyDecision.covariance` returns either that or the literal string
`"fit"`, so a rejected unit runs the *same code* an isotropic run would.  A
comparison between the two estimators is then a comparison of the units that
changed and nothing else.

The gate is not conservatism for its own sake.  An axis fitted to noise rotates
the prediction **and** narrows its variance along a direction the noise chose,
which is worse than isotropy rather than merely no better.

`anisotropy.csv` is written whether or not any unit passes, because "every unit
stayed isotropic, and here is why each one did" is the result on a site with no
fabric.

---

## 8. Field maps

### 8.1 Everything from one estimator lives under that estimator's name

A folder of similar-looking surfaces distinguished only by a filename suffix is
exactly how a model comparison gets mixed up.

### 8.2 The interval is drawn as maps, beside the mean

A mean map alone invites being read as *the* answer; putting the interval either
side of it turns "how uncertain is this" from a second figure you hold in your
head into something visible in one glance across a row.

### 8.3 Colour scales are shared two ways, and not a third

* **Within a row**, predicted / lower / upper share one scale — scaled
  separately, a wide interval would look identical to a narrow one, which is the
  very thing the row exists to show.  The sd column is a different quantity and
  keeps its own.
* **Across models**, the scale for a given unit spans every estimator, so the
  same colour means the same value in `fields/OK/prediction.png` as in
  `fields/UK/prediction.png`.
* **Not across units**: units differ in level by design (unit 2 sits near 1.0,
  unit 6 near 2.4), so one scale for all of them would flatten every panel to a
  single hue.

That shared scale earns its keep immediately — see
[the baseline's flat sd map](RESULTS.md#5-field-maps).  Per-figure auto-scaling hid
a ten-fold difference in claimed precision completely.

### 8.4 Field maps are fitted in-sample, and nothing is masked by presence

In-sample is the model's statement about the ground given everything known, not a
held-out test; the depth profiles and cross plots are where it gets scored on
data it was not shown.  Presence is not modelled yet, so a unit held at 4 of 194
holes still gets a surface across the whole site — the sd map is what keeps that
honest.  It grows away from the data, so wherever the mean map is inventing, the
sd map is brightest.  **Read the pair, never the mean alone.**

---

## 9. Depth profiles

### 9.1 Every hole is predicted leave-one-out

Each figure answers *"if this hole had never been drilled, what would we have
said?"*  An in-sample profile would be showing the model its own answer.

### 9.2 Three nested bands, because only one is comparable with a trace

| band | sd | question |
|---|---|---|
| latent | `sd_latent` | where is this unit's *true* value here |
| unit mean | `sd_obs` | what would a new *depth-average* here be |
| single reading | `√(sd_obs² + within_sd²)` | where will one 2 cm reading fall |

Only the third is comparable with a raw trace, and on IJmuiden it is much the
widest: `within_sd` runs 0.45–0.56 against a between-hole `log_Q_sd` of
0.31–0.71, so **depth texture inside one hole is as large as the variation
between holes.**  Quoting the narrow band against a trace would claim a precision
the model does not have; quoting only the wide one would hide how much of it a
spatial model can actually reduce.  Both are drawn, nested.

### 9.3 `exp(pred)` is a median, not a mean

It is the exponential of a mean of logs.  The bands are built in log space and
exponentiated, which is exact rather than approximate, because `exp` is monotone
and quantiles are equivariant under a monotone map.  `Qtn` is plotted on a log
axis so the intervals read symmetrically.

### 9.4 The band's width is measured, not asserted

The decomposition knowingly double-counts the depth-averaging error (`sd_obs`
already contains it, and it is itself produced by the within-unit scatter), so
`profile_coverage.csv` and `profile_calibration.png` report the *realised*
coverage per unit.  On IJmuiden the double-counting turns out to be negligible
next to everything else.

---

## 10. Back to `qt`

### 10.1 The stress exponent comes from the data, never from a default

`prepare_metadata.json` records what the supplied column was built with, because
`cpt_samples` keeps `Qtn` and drops `qt`, so nothing else remembers.  A dataset
that records no stress profile is **skipped with a message** rather than
defaulted, since a guessed exponent is a depth-dependent error that reads as
stratigraphy.  `--stress-profile` is the override for datasets whose preparation
predates the metadata block — including the synthetic ones, which have no
underlying `qt` at all.

**`n` is worth checking before trusting a column's name.**  Inverting an `n = 1`
column as if it were `n = 0.5` is wrong by a factor of 2–3, and wrong in a way
that varies with depth.  On IJmuiden, `cpt.infer_gamma_eff` recovers `n = 1` with
a buoyant gradient of 10.1 kN/m³ from the export itself.

### 10.2 Robertson's per-unit `n` is a diagnostic, not a licence

`cpt.soil_behaviour_index` sorts IJmuiden's units by soil type unprompted, but
IJmuiden's supplied column was built with `n = 1`, and a column must be inverted
with the exponent that made it.  Using per-unit `n` there would mean
re-normalising from raw `qt` and re-running the model — a different dataset, not
a different plot.

### 10.3 Quote quantiles for `qt`, not `median ± sd`

`qt` is right-skewed at these sds; the upper arm of the 95% band is 2–3× the
lower.  Treat `qt_sd` as a moment for propagating into a downstream calculation.
And a per-unit prediction is a *line*, not a number: constant `Qtn` through a
unit de-normalises to a `qt` rising linearly with depth, which is why
`qt_by_unit` returns top, mid and base.

### 10.4 Each suite is plotted on the axis it is legible on

**Log** for `Qtn`, the space the model is Gaussian in; **linear MPa** for `qt`,
the way a CPT log is read.  The qt figure frames on the trace and the median and
lets the bands overflow, captioning which ones did: in `qt` a `log_Q_sd` of 0.4
is a factor of 2.2 each way, so a band-framed axis would squeeze the trace into
the left quarter of the page.  `profile_coverage.csv` and
`profile_calibration.png` are written once, at the model level — the transform is
monotone, so the calibration is one result about both suites.

### 10.5 `n_sd > 0` disables the coverage claim

Setting it sets `attrs['coverage_transfers'] = False` and captions the figure:
the measured `qt` on the same row is itself conditional on the central `n`, so a
band widened for exponent doubt is no longer being scored fairly against it.
Two further caveats, both stated in the code: the closed form assumes
`n ⟂ Qtn`, and Robertson's `n` is a function of `Ic` which is a function of
`Qtn` — sample instead if the dependence matters; and a Gaussian `n` ignores the
`n ≤ 1` cap, so for a unit sitting at the cap the upper half of the assumed
spread is unphysical.

---

## 11. The generator

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
  A unit retaining less than `min_thickness_m` is marked absent, not kept as a
  sliver.
* **Within-unit scatter** is AR(1) with a configurable correlation length, so the
  effective sample size behind a depth-average is much smaller than `n_samples`.
  This is why the realised within-unit sd sits below the configured one, and it
  is what the depth-averaging step has to survive being tested against.
* **RNG streams are keyed by name**, not draw order, so editing one unit does not
  re-roll the rest of the site (`test_streams_are_named_not_ordered`).

---

## 12. Relationship to the plan documents

`cpt_gp_plan.md` originally wrote the A5 trend as `grad · [cos(az)·x + sin(az)·y]`
— the mathematical convention (counter-clockwise from east) — which contradicted
its own Section 0 declaration of azimuths as clockwise from north.  The code
resolved it in favour of Section 0, and the plan document has since been
corrected to `grad · [sin(az)·x + cos(az)·y]` so the two agree.

The Gaussian process and hyperparameter recovery are not built; see
[plans/02_estimators_and_validation.md](../plans/02_estimators_and_validation.md).
`validate/cv.py` is the harness they slot into: pass another factory and the
cross plots draw every model in the same panel.
