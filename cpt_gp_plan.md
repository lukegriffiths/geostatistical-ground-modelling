# Synthetic CPT Dataset & Spatial Model — Plan

**Scope of this document:** Part A covers synthetic data generation; Part B covers map plotting and
diagnostics. The estimator layer (GP vs. kriging) and validation harness are summarised at the end
as context, but are specified in a separate document.

**Project goal:** build a Gaussian-process / kriging toolkit for normalised cone tip resistance
(`Qtn`) across an offshore wind farm site, modelling each soil unit as a laterally varying field,
where units are not present at every CPT location.

---

## 0. Core modelling decisions (assumptions — challenge these)

| Decision | Choice | Rationale |
|---|---|---|
| Modelling variable | `log(Qtn)` | Q is positive and right-skewed; log makes Gaussian assumptions defensible |
| Spatial dimension | 2D per unit (6 independent models) | Matches "constant within a unit"; within-unit scatter becomes a measurable nugget contribution |
| Aggregation | Depth-average `log Qtn` over each unit interval → one value per CPT per unit | Makes the model input inspectable rather than hidden inside a fit |
| Coordinates | Kilometres, origin at site centre | Well-conditioned hyperparameter optimisation |
| Azimuth convention | Degrees **clockwise from north**, used for both trend direction and anisotropy angle | Defined once; the single most likely source of silent bugs |
| 3D option | Kept behind the same interface (x, y, z with short vertical range) | Not the first target, but the interface shouldn't preclude it |

---

## Part A — Data generation

### A1. Config (single source of truth)

One `config.yaml` plus a `seed`. Everything downstream is a pure function of these, so the generator
is reproducible and `truth.yaml` is simply the resolved config written back out alongside realised
summary statistics.

Per-unit block:

```yaml
unit_3:
  presence:   {mode: channel, width_km: 2.2, sinuosity: 0.35, edge_softness: 0.3}
  thickness:  {mean_m: 6, sd_m: 2, range_km: 4, min_m: 1, max_m: 10}
  property:
    mu: 1.6                                  # mean of log Qtn
    trend: {grad: 0.08, azimuth_deg: 115}    # per km, oblique
    grf:  {sill: 0.09, range_km: 3.5, aniso_ratio: 1.0, aniso_angle_deg: 0, model: matern25}
    nugget: 0.01
    within_unit_sd: 0.18                     # depth-to-depth scatter about the unit value
```

### A2. Site and CPT layout

- Site: 15 × 15 km square.
- ~120 CPTs on a jittered grid (turbine-like layout).
- **Plus** a small cluster of 3–4 tightly spaced CPTs. Without a few short lag pairs, the variogram
  near the origin is unconstrained and nugget/range trade off against each other freely.
- Optionally drop CPTs from one corner to create a data-sparse region for the sensitivity sweep.

### A3. Presence fields

Generate on a fine raster (~100 m), then sample at CPT locations.

- **Units 1–2:** present everywhere.
- **Channel units (3, 5):** centreline as `y = y0 + A·sin(2πx/λ + φ)` in a rotated frame, so channels
  run oblique rather than axis-aligned. (A correlated random walk works too if you want the geometry
  less regular.) Presence probability falls off smoothly from the centreline over `width_km`, with
  `edge_softness` controlling raggedness; draw Bernoulli per location.
- **Patch units (4, 6):** threshold a smooth GRF at a quantile chosen to hit a target coverage
  fraction (e.g. 0.45, 0.30). Produces blobby, spatially coherent presence.

Record **realised** coverage per unit in `truth.yaml` — needed when interpreting which units the
estimators struggle on.

### A4. Thickness and stratigraphic assembly

- Per unit, a smooth GRF (range ~4 km) mapped through a logistic to `[min_m, max_m]`, so thickness
  varies coherently rather than i.i.d.
- Per CPT: walk units in order, accumulate `z_top` / `z_bot`, skip absent units, truncate at 50 m.
- Deep units pinch out naturally where shallow ones are thick — realistic, and gives a second,
  correlated source of missingness.
- **Guard:** enforce a minimum 1 m retained thickness; otherwise mark the unit absent at that CPT.

### A5. Property field

Per unit, on the same raster:

```
log Q(x, y) = mu
            + grad · [sin(az)·x + cos(az)·y]
            + GRF(sill, range_km, aniso_ratio, aniso_angle)
```

Anisotropic GRF via `gstools`: rotate coordinates, scale the minor axis by `1/ratio`, then apply the
isotropic model. Sample at CPT locations and add nugget.

The projection above is `sin(az)·x + cos(az)·y` because azimuths are clockwise from north
(Section 0): the unit vector along azimuth `az` is `(sin az, cos az)`. Note also that
`range_km` is the **practical range** (correlation 0.05), which is *not* gstools' `len_scale` —
for Matérn 2.5 the practical range is 3.74x the `len_scale`.

**Contrast set** — assigned so each comparison isolates one effect:

| Unit | Trend | Anisotropy | Purpose |
|---|---|---|---|
| 1 | none | isotropic | GP ≡ simple kriging identity test |
| 2 | oblique, strong | isotropic | trend handling only (OK vs UK vs GP + mean function) |
| 3 | none | ratio 4, oblique angle | anisotropy only |
| 4 | oblique | ratio 3, different angle | both, plus trend/anisotropy confounding |
| 5 | weak | isotropic, short range | sparse channel unit, near data-limited |
| 6 | none | isotropic, high nugget | noise-dominated failure case |

### A6. Depth series

For each CPT × present unit, sample depths from `z_top` to `z_bot` at 2 cm (real CPT spacing;
subsample to 20 cm if file size becomes awkward):

```
Qtn = exp(unit_log_value + N(0, within_unit_sd²))
```

Optional refinements: a slow within-unit drift with depth, and a short-range AR(1) component, so the
depth series doesn't look like pure white noise. These matter if you later want to test the
depth-averaging step honestly (effective sample size per unit is much smaller than the raw sample
count when residuals are correlated).

### A7. Outputs

| File | Contents |
|---|---|
| `cpt_samples.csv` | `cpt_id, x, y, z, unit_id, Qtn` — long format, one row per depth reading |
| `layers.csv` | `cpt_id, unit_id, z_top, z_bot` — absent units are simply missing rows |
| `unit_summary.csv` | `cpt_id, x, y, unit_id, log_Q_mean, log_Q_sd, n_samples, thickness_m` — the actual model input |
| `truth.yaml` | Resolved per-unit generative parameters + realised coverage. Synthetic only; validation compares recovered vs. these |

The same pipeline runs on real data by supplying `cpt_samples.csv` and `layers.csv` only —
`truth.yaml` is absent and truth-dependent diagnostics are skipped.

---

## Part B — Map plotting

Two tiers: **truth diagnostics** (synthetic only, confirms the generator works) and **data
diagnostics** (runs unchanged on real data).

### B1. Truth diagnostics

- **Per-unit 2×2 panel:** presence raster / thickness raster / trend-only surface / full property
  surface. Overlay CPT locations coloured by sampled value. If the points don't visually match the
  background field, the sampling step has a bug.
- **Anisotropy check:** per unit, the GRF field with an ellipse glyph inset at the stated
  `aniso_angle` and `ratio`. Cheap, and catches angle-convention errors — rotation sign, and whether
  the angle is measured from north or east.

### B2. Data diagnostics

- **Presence map** — all CPTs, filled where the unit is present, hollow where absent; 2×3 grid, one
  panel per unit. Shows channel geometry as the estimator sees it.
- **Value map** — CPTs coloured by depth-averaged `log Q`, one panel per unit, shared colourbar per
  unit. The plot you'll stare at most.
- **Fence / section plots** — 2–3 transects (one along a channel, one across); stacked unit intervals
  vs. chainage with unit colours. This is how a geotechnical reviewer will want to check the
  stratigraphy.
- **Thickness map** per unit (scatter, sized or coloured).
- **Depth trace panel** — `Qtn` vs. depth for a handful of CPTs, units shaded. Confirms within-unit
  scatter looks plausible.
- **Trend check** — per unit, `log Q` projected onto the true azimuth, scatter + OLS line. On real
  data, project onto the *fitted* azimuth instead.
- **Lag coverage** — histogram of pairwise distances and a rose diagram of pair azimuths. Tells you
  whether an anisotropic variogram is identifiable at all before you try to fit one.

### B3. Conventions to fix now

- Equal aspect ratio on all maps.
- Unit colour palette defined once in the config.
- Azimuth convention stated in a module docstring and applied consistently to trend and anisotropy.
- Kilometre coordinates on axes.
- Every plotting function takes a dataframe and returns a figure — no file I/O inside, so the harness
  can call them on real data unchanged.

---

## Build order

1. A1–A2 — config schema, site, CPT layout
2. B1 — presence/thickness truth panels (stubs against dummy fields)
3. A3–A4 — presence fields, thickness, stratigraphic assembly
4. B2 — presence maps and section plots
5. A5–A6 — property fields, depth series
6. Remaining plots

Geometry bugs surface before they get buried under the property field.

---

## Downstream context (specified separately)

- **Estimators** share a `fit(X, y)` / `predict(X, return_std=True)` interface.
  GP: `Matern(ν=2.5, ARD) + White + Constant`, wrapped in a **learned rotation** since ARD length
  scales are axis-aligned only and the anisotropy here is oblique. sklearn has no mean function, so
  either detrend with OLS first (≈ residual kriging) or add a linear basis — explicitly, not via
  `normalize_y`.
  Kriging: `gstools` + `skgstat` for variogram estimation, `pykrige` for OK/UK.
- **Equivalence test:** on unit 1, GP with constant mean and isotropic Matern should reproduce
  **simple** kriging almost exactly. It will *not* equal ordinary kriging, which re-estimates the mean
  locally and inflates variance. Compare to SK for the identity test; quantify the OK gap separately.
- **Validation:** hyperparameter recovery vs. `truth.yaml`; leave-one-CPT-out CV (RMSE/MAE plus
  calibration — 95% coverage, NLPD/CRPS, PIT histogram, mean standardised squared residual ≈ 1);
  prediction and variance maps with GP−kriging difference maps; sensitivity sweep over CPT count,
  presence fraction, trend strength.
- Mask predictions where the unit is absent. Natural extension: a probit-GP presence classifier
  giving P(present) × E[Q | present] as a two-stage model.

---

## Proposed layout

```
cpt_geostat/
  config.py
  generate/
    layout.py      # site + CPT positions
    strat.py       # presence, thickness, assembly
    fields.py      # trend + GRF property fields
    series.py      # depth sampling
  io/              # readers/writers, truth.yaml handling
  models/
    base.py        # fit/predict interface
    gp.py
    kriging.py
  validate/
    cv.py
    metrics.py
    hyper.py       # recovery vs truth
  plots/
    truth.py
    maps.py
    sections.py
    diagnostics.py
  cli.py
```
