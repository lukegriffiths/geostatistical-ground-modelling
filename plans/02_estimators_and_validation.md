# Plan 02 — Estimator layer and validation harness

**Status:** started. Built: `covariance.py` (the conversion site), `trend.py` (the shared
linear trend), `models/base.py` (the interface and the two variance conventions),
`models/baseline.py`, `models/variogram.py`, `models/kriging.py` (**simple, ordinary and
universal kriging**), `models/crosscheck.py` (the pykrige cross-check), **directional variogram estimation**,
`validate/metrics.py`, `validate/cv.py` (leave-one-CPT-out), `plots/predictions.py`
(predicted-vs-true cross plots), `plots/variograms.py` (variogram fit and directional panels),
`models/profile.py` + `plots/profiles.py` (**Qtn with depth against the measured trace**, with
measured coverage) and `models/field.py` + `plots/fields.py` (**prediction and sd maps per
model**); 260 tests passing. Outstanding: `params.py` (the sklearn bridge), `gp.py`,
`hyper.py`, fitting an *anisotropic model* to the directional estimates, the GP−kriging
difference map, presence masking and profiles at **unvisited** locations (both blocked on the
presence classifier — see below) and the rest of `plots/validation.py` (PIT, calibration).
**Depends on:** Parts A and B (built).
**Companion:** [`03_gp_anisotropy_alternatives.md`](03_gp_anisotropy_alternatives.md).

> **Note on paths.** This document predates the contract/synthetic package split. Where it
> says `cpt_geostat/generate/…` read `cpt_geostat/synthetic/…`; `models/baseline.py::truth_reference_table`
> is now `synthetic/truth.py::truth_reference_table`; `tests/test_*.py` are now under
> `tests/{contract,models,plots,synthetic}/`. Two placement calls follow from that split and
> supersede what is written below: **`hyper.py` belongs in `cpt_geostat/synthetic/`**, not
> `validate/` — this document already says it is synthetic-only, and `validate/` is
> contract-driven — and the sklearn bridge belongs beside the gstools one in
> `cpt_geostat/covariance.py` rather than in `models/params.py`, for the same one-conversion-site
> reason that put `range_to_len_scale` there.

### What building OK/UK turned up that this plan did not anticipate

1. **Universal kriging needs a *residual* variogram, and it is load-bearing.** The plan treats
   OK/UK as thin wrappers. They are, except for this: fitting the variogram to trended values
   leaves it unbounded, the range runs to the identifiability bound, and the nugget collapses.
   Wired that way UK is no better than SK on unit 2 (MSSR 31 on the reduced fixture). Fitting
   it to the OLS residuals instead gives MSSR 1.2 and restores 95% coverage from 0.28 to 0.90.
   This is why `trend.py` had to land before UK, not merely before `gp.py`.
2. **pykrige is a *fourth* length-scale parameterisation**, and it is not the practical range:
   `pk_range / len_scale` is 3 for exponential, 7/(2√π) for gaussian and **1 for spherical**,
   against practical-range factors of 2.996, 1.953 and 0.811. For spherical that is a 23%
   error. pykrige also has no Matérn, so the cross-check runs on exponential. Both facts are
   encoded once in `models/crosscheck.py` and asserted against pykrige's own variogram
   functions.
3. **The pykrige cross-check must be run at zero nugget.** The two libraries use opposite
   conventions — pykrige interpolates the data exactly and returns a *latent* variance (zero
   at the data points); gstools smooths and returns an *observation* variance. At zero nugget
   they agree to ~1e-14 in both mean and variance, which validates the algebra in both. With a
   nugget they differ by construction, and a loose tolerance there would hide the gap rather
   than document it. This is the pykrige-at-the-data-points behaviour risk C predicted.
4. **`fit_variogram` could raise, and one bad fold killed the whole CV run.** A near-flat
   residual variogram can exhaust the optimiser. It now falls back to pure nugget and records
   `fit_failed`, which feeds `resolved` / `why_not_resolved()` like every other identifiability
   verdict.
5. **A directional variogram needs far coarser bins than an omnidirectional one.** Splitting
   the pairs across four sectors leaves each with a quarter of them; keeping the ~13
   hand-spaced bins puts 4-9 pairs in each and produces a curve that is pure sampling noise —
   and noise in four colours reads as dramatic anisotropy. `DIRECTIONAL_N_BINS = 6` uniform
   bins, plus the n >= 30 gate before a directional figure is drawn at all. The short-lag
   hand-spaced edges are also dropped for directional estimates: they exist to pin the nugget,
   which no single sector holds enough close pairs to do.
6. **The single-reading interval needs three nested variances, and the widest one dominates.**
   Reconstructing `Qtn` with depth needs `sd_latent` < `sd_obs` < `sqrt(sd_obs^2 + within_sd^2)`,
   and only the last is comparable with a raw trace. On IJmuiden `within_sd` (0.45-0.56)
   *exceeds* the between-hole `log_Q_sd` (0.31-0.71), so most of a reading's predictive
   variance is depth texture that no amount of drilling removes — a sobering but honest
   result, and the argument for reporting both bands rather than one. The decomposition
   knowingly double-counts the depth-averaging error; measured realised coverage is 0.94-0.97
   on 17 of 23 real units against a nominal 0.95, so the double-count is negligible in
   practice. The units that miss are the ones with trends, which is the known OK failure.
7. **An unmasked prediction map is defensible only when read with its sd map.** The field maps
   are drawn without a presence mask, so a unit held at 4 of 194 holes still gets a surface
   across 23 x 20 km. What stops that being a lie is the sd field, which grows away from the
   data — the plan already says masking "stops being optional", and this is the concrete form
   of that: the mask is deferred, the sd map is the interim safeguard, and the figures say so
   on their face. Note also that a **pure-nugget fit gives a flat sd field** (5 of 23 IJmuiden
   units), which needed its own colour-scale handling: padding a constant field symmetrically
   produced a colourbar running below zero, which is not a value a standard deviation can take.
8. **Profiles at unvisited locations are blocked, and it is worth being explicit why.** A
   profile needs the layer stack — which units, at what depths. At a drilled hole that is
   logged; anywhere else it needs the presence classifier *and* a thickness model. This is the
   concrete deliverable that the deferred classifier gates, and it is what turns per-unit
   predictions into a block model.
9. **Leave-one-out is pathological on co-located holes.** IJmuiden has 63 CPT/borehole pairs
   under 10 m apart. The nugget is only identifiable *from* those pairs, so a fold that removes
   one of a pair fits a near-zero nugget and then predicts the held-out hole from its twin
   5 m away with a predictive sd of ~0.003 — against a genuine difference of ~0.7 in log Qtn.
   MSSR reaches 1.7e4 on `GGM_22_Si` and `GGM_31_Si`. This affects SK, OK and UK identically
   and is not caused by any of them; it is a property of LOO on clustered data and needs
   either leave-one-*cluster*-out or a nugget floor that survives the fold.

---

## Context

The synthetic generator and the diagnostic plots are done. `models/` and `validate/` were
left out on purpose — `cpt_gp_plan.md` defers them to a separate document, and this is it.

The point of this chunk is not to produce predictions. It is to establish **whether a GP and
kriging agree when they must, differ where they should, and recover the parameters that
generated the data**. That last question is only answerable because `truth.yaml` records the
generative parameters — which is the reason the whole synthetic apparatus exists.

The six-unit contrast set was designed so a failure points at a cause: unit 1 isolates the
GP/simple-kriging identity, unit 2 isolates trend handling, unit 3 isolates anisotropy, unit 4
confounds trend and anisotropy on different bearings, unit 5 is deliberately data-limited, and
unit 6 is deliberately noise-dominated. The harness below is built to exploit that structure
rather than to report a single average score across all six.

---

## Decisions taken, and why they depart from `cpt_gp_plan.md`

The plan document specifies `gstools` + `skgstat` for variograms and `pykrige` for OK/UK.
Three changes, all verified against the installed environment (Python 3.9.2, numpy 2.0.2):

| Decision | Choice | Reason |
|---|---|---|
| Simple kriging | **gstools** | **pykrige has no simple kriging.** The plan's headline identity test requires SK. gstools ships `krige.Simple / Ordinary / Universal / Detrended / ExtDrift`. |
| Variogram estimation | **gstools** | `gstools.vario_estimate` already does directional variograms (`direction`, `angles_tol`, `bandwidth`) and `model.fit_variogram` fits them. |
| `scikit-gstat` | **dropped** | It resolves (1.0.23) but pulls numba + llvmlite, and adds a *third* length-scale parameterisation to bridge, for a capability gstools already has. |
| OK/UK cross-check | **pykrige 1.7.3** | Kept, as an independent implementation to check gstools' OK/UK against. It already depends on gstools, so it costs one line in `requirements.txt`. |
| GP | **scikit-learn 1.6.1** | Genuinely independent of gstools, which makes GP↔kriging agreement a meaningful cross-validation of both. |

**Deferred out of this chunk:** the sensitivity sweep (CPT count, presence fraction, trend
strength) and the two-stage probit-GP presence classifier.

---

## The three things most likely to go silently wrong

Ranked by how quietly they would corrupt results. Each repeats the shape of the bug already
found in Part A — `len_scale` vs. practical range, invisible on a map, fatal to recovery.

### A. Three incompatible length-scale parameterisations

`sklearn.gaussian_process.kernels.Matern`, `gstools.Matern` and "practical range" are three
different parameterisations of the same kernel. Comparing recovered numbers across them
without an explicit bridge produces a confident, wrong recovery table.

**Resolution — one conversion site, not two.** Move `range_to_len_scale` out of
`cpt_geostat/generate/fields.py` into a new `cpt_geostat/covariance.py` (package root, peer to
`geometry.py`), together with its inverse `len_scale_to_range`, `build_model`, and the
sklearn↔gstools bridge. `generate/fields.py` and `models/params.py` both import from there.

Splitting the forward conversion across `fields.py` and the inverse across `models/params.py`
would recreate exactly the two-homes situation that let the original `len_scale` bug through.
The bridge is verified **empirically, not algebraically**: `test_params.py`
evaluates both library kernels at a set of distances and asserts the correlation values agree
to 1e-10. Recovery is always reported in `truth.yaml`'s parameterisation — sill, practical
range (km), aniso ratio, aniso angle (degrees CW from north), nugget — reusing the existing
`cpt_geostat.geometry.math_angle_to_azimuth`.

### A2. The nugget recovery target is not the configured nugget

The model input is `log_Q_mean`, a depth-average. A variogram or `WhiteKernel` fitted to it
absorbs *everything* uncorrelated at CPT separation: the configured nugget **plus the
depth-averaging error**. Because the within-unit scatter is AR(1), that averaging error is far
larger than `sd²/n` — it is `(sd²/n)·(1+ρ)/(1−ρ)` with `ρ = exp(−dz/corr_len)`, an inflation
factor of 30–50 at the configured correlation lengths.

Scoring a recovered nugget against `config.nugget` therefore reports a ~2x failure on
correctly fitted estimators, worst on the best-behaved units. Measured on the current dataset:

| Unit | configured nugget | averaging error | **true target** | ratio |
|---|---|---|---|---|
| 1 | 0.005 | 0.0049 | **0.0083** | 1.66x |
| 2 | 0.005 | 0.0021 | **0.0076** | 1.53x |
| 3 | 0.006 | 0.0050 | **0.0129** | 2.15x |
| 4 | 0.008 | 0.0031 | **0.0083** | 1.04x |
| 5 | 0.006 | 0.0112 | **0.0138** | 2.30x |
| 6 | 0.060 | 0.0033 | **0.0294** | 0.49x |

**Resolution.** One line in the generator: `realised_truth` in
`cpt_geostat/generate/pipeline.py` gains a per-unit `obs_noise_var`, and `hyper.py` scores the
recovered nugget against that.

**Built instead in `models/baseline.py::truth_reference_table`**, as `obs_noise_sd` with its
`nugget_sd` / `depth_avg_sd` split, computed from `truth_points.csv` rather than written into
`truth.yaml`. Two reasons to prefer that placement, and `hyper.py` should read it from there:
it keeps the noise floor **restricted to the CPTs each unit is present at** — see below — and
it keeps a derived quantity out of the generator's output contract, where it would have to be
regenerated to change.

**One thing this plan gets wrong:** it treats `truth.yaml`'s realised sds as the comparison
target. They are computed over *every* CPT, present or not, while every fitted number is
computed over the CPTs holding the unit. For unit 5 the two populations differ by a factor of
1.5 — a compact 16% of a lumpy field is far less variable than the whole of it — so scoring
across that difference credits an estimator with explaining more variance than the data
contains. It produces a "structured fraction" above 1 on units 2 and 4. Every truth-side
statistic in `hyper.py` must be recomputed over present CPTs, not read from `truth.yaml`.

It must be measured as **`Var(log_Q_mean − log_Q_field)`, not `Var(log_Q_mean − log_Q_true)`**.
`log_Q_true` already contains the nugget, so differencing against it captures only the
averaging error and would leave the target short by exactly the nugget — the same
factor-of-two error, in the other direction. Both columns already exist in `truth_points.csv`;
record `depth_avg_var` and `nugget_var` alongside the total so the split stays visible.

The unit 6 row (0.49x) is a low realisation of the nugget draw, not a generator fault:
`Var(log_Q_true − log_Q_field)` over all 117 CPTs sits within 1.5 sd of configured for every
unit. It is also the argument for recording the **realised** value rather than the configured
one — which is the principle `truth.yaml` already follows for sill and coverage.

The SK identity test is unaffected: both sides use the same fixed values, whatever they are.

### B. A `ConstantKernel` is not a constant mean

This one breaks the headline test. Simple kriging assumes a **known** mean. Adding
`ConstantKernel` to a sklearn GP does not give a constant mean — it gives a zero-mean GP with
a random constant offset, which is nearer to *ordinary* kriging. Wired that way, the SK
identity test fails for a reason that looks numerical and will be debugged in the wrong place.

**Resolution.** The SK-equivalent GP is: subtract the known mean, fit a **zero-mean** GP with
`Matern(nu=2.5) + WhiteKernel`, no `ConstantKernel`, `normalize_y=False`, `alpha=0`, and
hyperparameters **fixed** to truth (`optimizer=None`) for the identity test. The OK gap is
then measured separately against `krige.Ordinary` — it is a real modelling difference to be
quantified, not an error to be tuned away.

### C. Two different variances, needed in two different places

sklearn's `predict(return_std=True)` uses the fitted kernel, which *includes* the
`WhiteKernel`, so its variance carries observation noise at the target (`alpha` does not appear
in it). gstools' kriging variance is the variance of the latent field, and its nugget handling
depends on `exact` / `cond_err`. Comparing them directly compares two different things.

**Resolution — and the convention splits by purpose.** Getting this wrong in either direction
breaks something:

| Purpose | Variance | Why |
|---|---|---|
| SK identity test, GP↔kriging comparisons, difference maps | **latent** (`gp_var_total - nugget`) | Comparing two estimators' beliefs about the *field*; observation noise is not part of that. |
| Every CV metric — PIT, coverage, NLPD, CRPS, MSSR | **latent + observation noise** | The held-out value is an *observation*. It carries the nugget and the depth-averaging error. |

Scoring calibration on latent variance against noisy held-out observations guarantees
under-coverage, and it fails worst exactly where this harness is supposed to succeed: unit 6
(nugget ≫ sill) is meant to come out *well calibrated and uninformative*, and latent-only
variance would render it catastrophically miscalibrated instead. The function signature names
which convention a number is in; it is never left to the reader.

A dedicated test pins down each library's convention empirically *before* anything is built on
it — including **pykrige's behaviour at the data points**, where it treats the nugget as exact
by default. That is precisely where an OK cross-check would silently disagree with gstools
while both are "correct".

---

## Design

### `cpt_geostat/models/base.py` — **built**

```python
class SpatialEstimator(ABC):
    def fit(self, X, y) -> "SpatialEstimator"   # X: (n, 2) km; y: (n,) log Qtn
    def predict(self, X, return_std=False)      # latent field; nugget excluded
    @property
    def noise_var_(self) -> float               # nugget + depth-averaging error
    @property
    def params_(self) -> dict                   # in truth.yaml's parameterisation
    def predict_observation(self, X)            # concrete: latent + noise_var_
```

`noise_var_` and `predict_observation` are additions to what this plan specified, and they
are how risk C is discharged: the two variance conventions become two named methods, the
addition happens once rather than at every call site, and a metric can no longer be scored on
the wrong one by omission.

`params_` returning truth-comparable numbers from *every* estimator is what makes
`validate/hyper.py` a table lookup instead of per-estimator special-casing. A parameter an
estimator cannot identify is `None` — distinct from a fitted number and from `nan`, which
would read as a failed fit. This is what makes "not identifiable" reportable as its own
outcome on IJmuiden's thin units.

### `cpt_geostat/models/baseline.py` — **built**

`UnitMeanEstimator`: constant mean, all residual scatter attributed to the nugget, `sill = 0`
and `range_km = None` asserted rather than fitted. It is the estimator to beat, and on unit 6
(`structured_frac` 0.44) it is close to the best available answer — anything that appears to
beat it there is overfitting. It is also the pooled fallback the IJmuiden estimability gate
requires, via `fallback_var` for units held at a single CPT.

### `cpt_geostat/models/gp.py` — `GPEstimator`

**Rotation by profile likelihood over angle.** For each candidate azimuth on a grid, rotate
coordinates using the existing `cpt_geostat.geometry` helpers, fit a standard sklearn ARD
`Matern(nu=2.5) + White`, and keep the highest log-marginal-likelihood. No custom gradients,
and immune to the angle-wrapping multimodality that traps a joint optimiser.

**Scan 0–90°, not 0–180°.** An axis-aligned ARD kernel at rotation θ is the *same model* as
θ+90° with the two length scales swapped. Scanning 0–180° therefore produces two exactly equal
maxima, and "refine around the best" is ill-defined between them — recovery can land 90° from
truth for reasons that have nothing to do with the fit. Every distinct unordered axis pair
appears exactly once in [0°, 90°), so that interval is complete *and* non-degenerate.

The recovered major azimuth is then defined as **the axis carrying the longer length scale**,
wrapped mod 180°, resolved inside `gp.py` before it ever reaches the circular-difference
comparison in `hyper.py`.

The profile-likelihood curve is retained on the fitted object and plotted. It shows whether
the angle is identifiable **at all** — which for unit 5 it will not be. A flat curve is a
result, not a failure, and the harness must be able to say so.

`mean=` handling, always explicit and never via `normalize_y`:

- `"zero"` — for the SK identity test, on pre-centred data.
- `"constant"` — sample mean subtracted, added back at predict.
- `"ols_detrend"` — OLS on `[1, x, y]`, GP on residuals, trend added back at predict.
  Must flag that the predictive variance excludes trend-estimation uncertainty; universal
  kriging includes it.

**New shared module `cpt_geostat/trend.py`.** `models/` must not import from `plots/` — that inverts
the layering. And the existing `fit_trend_azimuth` is insufficient anyway: it returns
`(grad, azimuth)` and discards the intercept, so it cannot add a trend back at predict time.
Move the fit into `cpt_geostat/trend.py` (package root, peer to `geometry.py`, no circular imports)
returning full coefficients:

```python
@dataclass
class LinearTrend:
    intercept: float; bx: float; by: float
    @property gradient      # hypot(bx, by), per km
    @property azimuth_deg   # CW from north
    def predict(self, x, y)

def fit_linear_trend(x, y, v) -> LinearTrend
```

`plots/diagnostics.py::fit_trend_azimuth` becomes a thin wrapper over it, so both callers share
one implementation and one convention.

### `cpt_geostat/models/kriging.py` and `variogram.py`

Thin wrappers over `gstools.krige.Simple / Ordinary / Universal` conforming to the interface,
plus `PyKrigeOrdinary` / `PyKrigeUniversal` used only as cross-checks. Variogram work lives in
`variogram.py`: empirical (isotropic and directional) via `gstools.vario_estimate`, model
fitting via `model.fit_variogram`, returning parameters already converted through `params.py`.

**`SimpleKriging` is built**, with the isotropic variogram. Three findings the plan did not
anticipate, each of which cost a debugging cycle:

1. **The variance convention resolved the other way from the plan's guess.** gstools' kriging
   variance tends to `sill + nugget` far from data — it is an *observation* variance. So
   `predict` **subtracts** the nugget to return the latent field, rather than the plan's
   assumed "gstools gives you the latent field". Pinned in
   `test_gstools_kriging_variance_includes_the_nugget`, which is a statement about the
   library, not about our code, and so must survive a gstools upgrade or fail loudly.
2. **Unconstrained variogram fitting fails silently and badly** — 1700 km on unit 2, 22 km on
   unit 4, and a nugget of exactly zero on three units. The plan treats `fit_variogram` as a
   one-liner. It needs three constraints (short-lag bins, a lag cap with a matching
   `len_scale` bound, and the sill pinned to the sample variance) before it returns anything
   defensible. `VariogramFit.resolved` and `why_not_resolved()` carry the identifiability
   verdict, which is the mechanism the IJmuiden estimability gate needs and can reuse.
3. **`covariance="truth"` earns its place as a control, not a cheat.** Kriging given the
   generating covariance is calibrated (MSSR 0.87–1.38) and beats the baseline everywhere but
   unit 6; kriging given a *fitted* covariance predicts as well but is overconfident. That
   pair localises the error to variogram estimation rather than to the kriging, which no
   single number can. `hyper.py` should keep both columns for the same reason.

The truth covariance uses the **realised observation noise**, not `config.nugget` — see A2.
Handing kriging the configured value makes the reference model overconfident, which then
reads as a flaw in kriging rather than in what it was told.

Anisotropic and directional variograms are outstanding, and unit 3 is why they matter: its
minor axis is 1 km against a ~1.3 km CPT spacing, so an isotropic fit correctly sees noise and
fits nearly all nugget. That is the *right* answer to the wrong question.

### `cpt_geostat/validate/`

- **`metrics.py`** — `rmse`, `bias`, `r2`, `coverage(level=0.95)`, `mssr` are **built** and
  tested against analytic cases. `mae`, `nlpd`, `crps_gaussian`
  (closed form: `sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))`) and `pit` are outstanding.
- **`cv.py`** — **built**, refitting every fold. `loo_predict` takes an estimator *factory*
  and returns `pred`, `sd_latent` and `sd_obs` — **both variances, on every row**. That is how
  risk C is discharged in practice: `target_columns` pairs each target with its own sd, so the
  pairing is chosen at the call site and is visible in the figure, and a metric cannot be
  scored on the wrong variance by omission. The fit-once/`refit=False` variant is deferred
  until there is an estimator expensive enough to need it — the baseline refits in
  microseconds, and a fit-once default would have been optimistic for no gain.

  The baseline's leave-one-out `r2` is `1 - (n/(n-1))**2` exactly, for any data. That closed
  form is asserted in `test_validate.py` and is the reference line for every spatial estimator.
- **`hyper.py`** — recovery table vs. `truth.yaml`. Three rules:
  - Angles compared with a **circular difference mod 180°** — an anisotropy axis at 179° and
    one at 1° are 2° apart, not 178°.
  - **Angle recovery is conditional on the ratio.** Where the fitted ratio is ≈ 1 (unit 1, and
    likely unit 5) the axis is meaningless and must be reported as *not identifiable*, not as a
    pass/fail number. The same applies to the trend azimuth where the fitted gradient is
    indistinguishable from zero.
  - Nugget scored against `obs_noise_var`, never `config.nugget` (see A2).

### Plots

Both follow the Part B contract in `cpt_geostat/plots/style.py`: take a dataframe or `Dataset`,
return a Figure, no file I/O.

- **`plots/predictions.py`** — prediction map, sd map, and GP−kriging difference map per unit,
  masked where the unit is absent. On synthetic the mask is the truth presence raster; on real
  data no mask exists until the presence classifier lands, so `mask=None` is supported and the
  panel says so on its face.
- **`plots/predictions.py`** — **built**, so far the cross plot only:
  `plot_prediction_vs_truth(ds, cv, target=...)`, a panel per unit, a series per model, square
  axes so the 1:1 line is at 45°, and per-panel `rmse / bias / r2 / coverage / mssr`. It flags
  a constant model as flat by construction and flags intervals that are the wrong size, so a
  correct-but-alarming result — the baseline scores `mssr` 12–116 against the latent field —
  reads as a finding rather than a bug. The prediction and sd maps go in the same module.
- **`plots/validation.py`** — PIT histogram, coverage/calibration curve, variogram fit panels,
  and the profile-likelihood-vs-angle curve. Observed-vs-predicted is done, above.

### CLI

`python -m cpt_geostat.cli model --data run/data --out run/models` and
`... validate --data run/data --out run/validation`, matching the existing
`generate` / `plot` / `run` structure in `cpt_geostat/cli.py`.

---

## Applying to IJmuiden — the estimability gate

The synthetic contrast set is where the estimators get **validated**; IJmuiden is where they
get **applied**. That division is clean and worth keeping: only the synthetic side has truth,
so `hyper.py` is synthetic-only, and the real site gets CV, calibration and maps.

The real site is not a bigger version of the synthetic one. It has 194 CPTs and **23 units**
over 23.3 x 19.6 km, with steeply unequal coverage:

| CPTs holding the unit | Units | What is estimable |
|---|---|---|
| ≥ 30 (`GGM_01_S` 148 … `GGM_26_S` 43) | 10 | Full: anisotropy, trend, range, nugget |
| 12–29 (`GGM_11_S` 25 … `GGM_25_S` 12) | 3 | Isotropic range + nugget only |
| < 12 (down to `GGM_03_C` and `GGM_21_S` at 4) | 10 | Nothing per-unit |

Fitting a directional variogram to four holes does not fail loudly — it returns numbers. So
the gate is explicit, in config, and its tiers are justified by pair counts rather than taste:
anisotropy needs pairs spread across ~4 azimuth sectors and ~10 lag bins, which needs roughly
n ≥ 30 (435 pairs, ~11 per bin); an isotropic sill/range/nugget needs ~12 (66 pairs, ~7 per
bin). Below that, fall back to the unit mean with a variance from the pooled residual.

Three consequences for the design:

1. **`params_` must be able to say "not identifiable"** — for a whole unit, and for individual
   parameters within a fitted unit (the angle when the ratio is ≈ 1). The recovery and CV
   reports must render that as a distinct outcome from a bad number. `GGM_24_C_S` at 7 CPTs
   with an sd of 1.27 is the case that will otherwise produce a confident, meaningless range.
2. **Prediction masking stops being optional.** With 23 units mostly absent at most holes, an
   unmasked prediction map is misleading over most of its area. The presence classifier moves
   from "natural extension" to the thing that gates real prediction maps — still out of this
   chunk, but the reason to schedule it next.
3. **Scale check.** The largest unit has 148 CPTs, so exact GP and the profile-likelihood scan
   stay cheap (dense Cholesky at n ≈ 150, times ~18 angles, times 23 units). No approximation
   is needed, and none should be introduced speculatively.

Real `log Qtn` runs 2.8–6.1 against the synthetic 1.0–2.5, so any hard-coded axis limit or
tolerance calibrated on synthetic data must be relative, not absolute.

---

## Build order

Geometry bugs surfaced before the property field buried them in Part A; the same logic applies
here — pin the parameterisations before anything depends on them.

0. **Prerequisites in existing code**, all small and all removing a trap:
   - ~~move `range_to_len_scale` into new `cpt_geostat/covariance.py`~~ — **done**, with
     `len_scale_to_range`, `build_model` and `model_params` (risk A);
   - ~~new `cpt_geostat/trend.py` with `fit_linear_trend`, rewiring `plots/diagnostics.py`~~ —
     **done**; `LinearTrend` carries the intercept and standard errors, so it can add a trend
     back at predict time and can say when a bearing is not identifiable;
   - ~~add `obs_noise_var` / `depth_avg_var` / `nugget_var` to `realised_truth`~~ — **done**
     in `synthetic/truth.py::truth_reference_table` instead, over present CPTs (risk A2).
0b. **Done:** `covariance.py`, `models/base.py`, `models/baseline.py`, `models/variogram.py`,
   `models/kriging.py`, `validate/metrics.py`, `validate/cv.py`, `plots/predictions.py`,
   `cli model`, `tests/test_models.py`, `tests/test_validate.py`, `tests/test_kriging.py`.
1. `params.py` + `test_params.py` — the **sklearn** side of the bridge; the gstools side and
   the range round-trip are done in `covariance.py`.
2. ~~`base.py`, then `kriging.py` (SK first) and `variogram.py`~~ — **done**, including
   ~~ordinary and universal kriging~~ over a shared `_GstoolsKriging` base and
   ~~directional variograms~~ (`directional_variogram`, rendered by `plots/variograms.py`).
   Outstanding: *fitting* an anisotropic model to those directional estimates — this
   renders the evidence, it does not yet recover a ratio and an axis. That belongs with
   `gp.py`'s 0-90 degree profile-likelihood scan, which solves the same problem.
3. ~~**The variance-convention pinning test** (risk C)~~ — **done for gstools**, separately
   for SK, OK and UK rather than extrapolated from one; ~~pykrige at data points~~ **done**
   (it treats them as exact — see finding 3 above).
4. `gp.py` with `mean="zero"` only.
5. **The SK identity test.** Nothing else proceeds until this passes.
6. `gp.py` rotation (0–90° scan) and mean handling; ~~OK/UK; pykrige cross-check~~ — **done**.
7. `validate/metrics.py`, then `cv.py`, then `hyper.py`.
8. `plots/predictions.py`, `plots/validation.py`, CLI wiring.

---

## Verification

The identity test is load-bearing; the rest is scoring.

1. **Parameterisation bridge** — `pytest tests/test_params.py`: sklearn and gstools kernels
   evaluated at the same distances agree to 1e-10 after conversion; round-trips through
   practical range are lossless.
2. **SK identity (headline)** — on unit 1 (no trend, isotropic, by design), a zero-mean GP with
   hyperparameters fixed to truth reproduces `krige.Simple` to ~1e-8 in both mean and latent
   variance at a grid of test points. sklearn and gstools are independent implementations, so
   agreement validates both.
3. **OK gap quantified, not hidden** — the same comparison against `krige.Ordinary` must
   *fail* to match, with the variance inflation reported as a number.
4. **pykrige cross-check** — pykrige OK agrees with gstools OK to tolerance on unit 1.
5. **Rotation recovery** — on unit 3 (ratio 4 at 70°) and unit 4 (ratio 3 at 135°), the
   recovered azimuth lands within ~15° of truth and the recovered ratio exceeds 2. On unit 1
   the profile likelihood is flat, the fitted ratio is near 1, and the angle is reported as
   *not identifiable* rather than as a number. A dedicated test asserts the 0–90° scan does not
   return an axis 90° from truth on units 3 and 4.
6. **Calibration, scored on observation variance** (latent + `obs_noise_var`, per risk C) —
   leave-one-CPT-out on units 1 and 2 gives 95% coverage in roughly 0.90–0.98 and MSSR near 1.
   Unit 6 (nugget > sill) should be **well calibrated and uninformative** — high RMSE with
   honest variance. That combination is the correct outcome, not a bad one, and the report must
   not penalise it as failure. If unit 6 comes out badly *miscalibrated*, suspect the variance
   convention before suspecting the estimator.
6b. **Nugget recovery** scored against `obs_noise_var` lands near 1.0x; scored against
   `config.nugget` it would land near 1.5–2.3x. Worth asserting both in one test, so the
   distinction cannot quietly regress.
7. **End to end** — `python -m cpt_geostat.cli run --out run`, then `... model` and `... validate`,
   then read the prediction, difference and PIT figures. The GP−kriging difference map on
   unit 1 should be visually flat at the colour-scale limits.
8. **Full suite** — `pytest -q` stays green (58 existing + new).

---

## Risks

- **The same library generates and fits.** gstools on both sides is an inverse crime, and the
  recovery numbers are optimistic because of it. Partly mitigated by the sklearn GP being a
  genuinely independent implementation and by the pykrige OK/UK cross-check. This belongs in
  the results as a stated caveat rather than something to engineer around.
- **Units 5 and 6 will not recover well — by design.** The harness must report "not
  identifiable" distinctly from "fitted badly". A flat profile likelihood and a wide interval
  are the intended output, and the plots need to make that legible at a glance.
- **Prediction masking on real data is unavailable** until the presence classifier lands.
  Synthetic runs use the truth presence raster; real-data runs pass `mask=None`. On IJmuiden,
  where most units are absent at most holes, this makes real prediction maps provisional —
  the strongest argument for scheduling the classifier immediately after this chunk.
- **Ten of IJmuiden's 23 units are not estimable per-unit at all.** The deliverable there is a
  defensible refusal plus a pooled fallback, not 23 fitted models. A harness that silently
  returns 23 looks more successful and is worth less.
- **Python 3.9 pins.** scikit-learn 1.6.1 is the last release supporting 3.9. If the project
  moves to a newer interpreter, re-resolve rather than assuming the pins still hold.
