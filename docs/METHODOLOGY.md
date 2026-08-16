# cpt_geostat — Methodology

Geostatistical ground modelling of normalised cone tip resistance ($Q_{tn}$) across an
offshore wind farm site. This document defines the notation, the generative model behind the
synthetic dataset, the estimators, the validation harness, and the mapping of real data onto
the same contract. Every equation here is the one the code implements; where a library
convention was verified empirically rather than assumed, that is noted.

---

## 1. Conventions and definitions

| Symbol / term | Definition |
|---|---|
| $Q_{tn}$ | Normalised cone tip resistance (dimensionless, positive, right-skewed) |
| $y = \log Q_{tn}$ | The modelling variable throughout — natural log |
| $(x, y)$ coordinates | Kilometres, origin at the site centre. East = $+x$, north = $+y$ |
| $z$ | Depth in metres below seabed, positive down |
| Azimuth $a$ | **Degrees clockwise from north**: $0°=+y$ (N), $90°=+x$ (E) |
| Sill $\sigma^2$ | Variance of the spatially structured (GRF) component |
| Nugget $\tau^2$ | Variance of the spatially uncorrelated component |
| Range $r_p$ | **Practical range** — separation at which correlation falls to 0.05 |
| Anisotropy ratio $\kappa$ | Major range / minor range, $\kappa \ge 1$ |
| Anisotropy azimuth | Compass azimuth of the **major** (slowest-decorrelating) axis |
| Latent field | The estimator's belief about the ground itself, excluding observation noise |
| Observation | A measured depth-average; carries nugget **and** depth-averaging error |

### 1.1 Azimuth geometry

The unit vector along azimuth $a$ is

$$\mathbf{u}(a) = (\sin a, \cos a),$$

**not** $(\cos a, \sin a)$. The signed distance of a point along the azimuth (the coordinate a
linear trend runs on) is

$$s(x, y; a) = x \sin a + y \cos a,$$

and the across-azimuth coordinate (positive to the left of the heading) is
$s(x, y; a - 90°)$. gstools uses the mathematical convention (radians counter-clockwise from
$+x$); the bridge, applied in exactly one place, is

$$\theta_{\text{math}} = \frac{\pi}{180}\,(90° - a), \qquad
  a = \left(90° - \frac{180°}{\pi}\theta\right) \bmod 180°.$$

### 1.2 Range parameterisation

gstools parameterises covariance models by a length scale $\ell$ (`len_scale`), which is
**not** the range a variogram fit reports. The practical range is defined by
$\rho(r_p) = 0.05$, giving $r_p = \kappa_m\, \ell$ with a model-dependent factor
(verified against gstools 1.7):

| Model | $\kappa_m = r_p/\ell$ |
|---|---|
| Matérn $\nu = 2.5$ | 3.743 |
| Matérn $\nu = 1.5$ | 3.873 |
| Exponential | 2.996 ($= \ell \ln 20$) |
| Gaussian | 1.953 |
| Spherical | 0.811 |

Feeding a practical range into gstools as a `len_scale` makes a Matérn-2.5 field ~3.7×
smoother than configured — invisible on a map, fatal to hyperparameter recovery. The
conversion and its inverse live in a single module (`cpt_geostat/covariance.py`) used by both the
generator and the estimators.

---

## 2. Generative model (synthetic data)

Each soil unit $u$ is generated independently as a laterally varying 2D field on a fine
raster (0.1 km), then sampled at CPT locations. Randomness is drawn from named substreams of
one master seed, so editing one unit does not re-roll the others.

### 2.1 Presence

Presence is a **probability raster** $p_u(x,y) \in [0,1]$; the Bernoulli draw happens once,
at each CPT location, from the bilinearly interpolated probability.

**Channel units.** In the frame of the channel azimuth $a$, with along/across coordinates
$(s, t)$, the centreline is a sine

$$c(s) = t_0 + A \sin\!\left(\frac{2\pi s}{\lambda} + \varphi\right), \qquad
A = \frac{\text{sinuosity} \cdot \lambda}{4},$$

and the probability falls off smoothly across the banks,

$$p(s,t) = \operatorname{expit}\!\left(\frac{w/2 - |t - c(s)| + \varepsilon(x,y)}{\delta}\right),
\qquad \delta = \max(\text{softness}, 10^{-3}) \cdot \tfrac{w}{2},$$

where $w$ is the channel width and $\varepsilon = \delta\, G_{\text{ragged}}(x,y)$ perturbs
the bank position with a short-range unit-variance GRF, producing ragged rather than merely
blurred edges. (Note: the `sinuosity` parameter is an amplitude fraction, not a true
sinuosity — path length over chord.)

**Patch units.** A smooth unit-variance GRF $g$ is soft-thresholded at the quantile hitting
the target coverage $c$:

$$p = \operatorname{expit}\!\left(\frac{g - q_{1-c}(g)}{\text{softness} \cdot \hat\sigma_g}\right).$$

*Known small bias:* the quantile is exact for a hard threshold; the sigmoid softening
inflates expected coverage slightly (≈ +1.5–2 pp at softness 0.25), because the normal
density is higher on the sub-threshold side. Realised coverage is recorded in `truth.yaml`,
which is the number to quote.

### 2.2 Thickness

A unit-variance GRF $g$ (standardised per realisation) is mapped through a logistic into
$[T_{\min}, T_{\max}]$:

$$T(x,y) = T_{\min} + (T_{\max} - T_{\min})\,\operatorname{expit}\!\big(\alpha + \beta\, g(x,y)\big).$$

$(\alpha, \beta)$ are solved deterministically by Gauss–Hermite quadrature (64 nodes) so that
$\mathbb{E}[T]$ and $\operatorname{sd}[T]$ under $g \sim \mathcal N(0,1)$ match the configured
mean and sd. A bounded variable cannot have arbitrary sd; an unattainable target warns and
delivers the closest achievable fit.

### 2.3 Stratigraphic assembly

Units are walked in configured order at each CPT, accumulating depth. Absent units are
skipped (no row). Everything is truncated at the maximum depth (50 m), so deep units pinch
out where shallow ones are thick — a second, *correlated* source of missingness on top of
lateral absence. A unit retaining less than the minimum thickness (1 m) is marked absent
rather than kept as a sliver (its depth budget is released to the units below).

### 2.4 Property field

Per unit, on the raster:

$$\log Q(x,y) \;=\; \mu \;+\; \beta_t \, s(x, y; a_t) \;+\; Z(x,y),$$

where $\beta_t$ is the trend gradient (per km) along trend azimuth $a_t$, and $Z$ is a
zero-mean anisotropic GRF with sill $\sigma^2$, practical range $r_p$, anisotropy ratio
$\kappa$ and major-axis azimuth $a_\kappa$ (Matérn $\nu=2.5$ unless configured otherwise).

The **true unit value at a CPT** adds the nugget as a point-support term:

$$y^{\text{true}}_i = \underbrace{\mu + \beta_t s_i + Z(x_i)}_{y^{\text{field}}_i}
\;+\; \epsilon_i, \qquad \epsilon_i \sim \mathcal N(0, \tau^2) \ \text{i.i.d.}$$

True values are generated at *every* CPT, present or not, so sensitivity sweeps over
presence hold the field fixed.

### 2.5 Depth series

For each present (CPT, unit) interval $[z_{\text{top}}, z_{\text{bot}}]$, readings are
placed at $n = \lfloor (z_{\text{bot}} - z_{\text{top}})/\Delta z \rfloor$ mid-cell depths
($\Delta z = 0.02$ m) and

$$\log Q(z_k) \;=\; y^{\text{true}} \;+\; d\,(z_k - \bar z) \;+\; e_k,$$

with a slow, mean-centred within-unit drift $d$ (per m) and an AR(1) residual

$$e_k = \rho\, e_{k-1} + w_k, \qquad
\rho = e^{-\Delta z / \ell_z}, \qquad
w_k \sim \mathcal N\!\big(0,\, s_w^2 (1 - \rho^2)\big),$$

initialised from the stationary distribution so $\operatorname{Var}(e_k) = s_w^2$ exactly.
$Q_{tn} = \exp(\log Q)$.

### 2.6 Depth averaging and the observation-noise decomposition

The model input is the per-(CPT, unit) depth average $\bar y_i = \frac1n \sum_k \log Q(z_k)$.
Because the residual is AR(1), the variance of the average about the true unit value is
inflated far above $s_w^2/n$:

$$\operatorname{Var}(\bar y_i - y^{\text{true}}_i)
\;\approx\; \frac{s_w^2}{n} \cdot \frac{1+\rho}{1-\rho}
\qquad (\text{large } n),$$

a factor of 30–50 at the configured correlation lengths. The total **observation noise** —
everything uncorrelated between two CPTs standing at the same place — is therefore

$$\sigma^2_{\text{obs}} \;=\; \underbrace{\tau^2}_{\text{nugget}}
\;+\; \underbrace{\operatorname{Var}(\bar y - y^{\text{true}})}_{\text{depth-averaging error}},$$

and it, **not the configured nugget**, is the correct recovery target for any fitted nugget
and the correct noise to hand a truth-covariance reference model. On the synthetic site it
runs 1.0–2.3× the configured nugget depending on the unit.

Truth-side statistics are measured as root-mean-squares of errors with known zero mean
(RMS, not sd — centring would discard a real bias), and are **restricted to the CPTs where
the unit is present**. A narrow channel's latent spread over its own 16 % of the site is
~1.5× smaller than over the whole site; comparing against the all-CPT figure credits a
spatial model with explaining more variance than the data contains.

---

## 3. Estimators

All estimators share one interface: fitted per unit on $X$ ($(n,2)$ km) and
$y$ (depth-averaged $\log Q_{tn}$ from `unit_summary`), with **two named predictive
variances**:

| Question | Method | Includes observation noise? |
|---|---|---|
| "what is the field here?" | `predict` | no — latent only |
| "what would I measure here?" | `predict_observation` | yes — latent $+\ \sigma^2_{\text{obs}}$ |

Estimator-vs-estimator comparisons and difference maps use the latent variance; **every
cross-validation metric uses the observation variance**, because a held-out value is an
observation. Scoring calibration on latent variance against noisy held-out values guarantees
under-coverage.

### 3.1 Baseline: per-unit constant mean

Model: $y_i = m + e_i$, $e_i$ i.i.d. Fitted: $\hat m = \bar y$,
$\hat s^2 = \frac{1}{n-1}\sum (y_i - \bar y)^2$ (pooled fallback variance when $n = 1$).

$$\text{predict:}\quad \big(\hat m,\ \hat s^2/n\big) \qquad
\text{predict\_observation:}\quad \big(\hat m,\ \hat s^2 (1 + 1/n)\big).$$

The latent sd is the standard error of the level — it shrinks with $n$ and is *not* a claim
that the ground is flat to that precision. All residual scatter is attributed to the nugget;
sill is asserted 0 and range reported as *not identifiable* (`None`), distinct from a fitted
number and from `nan`.

Three "average and sd" pairs are reported and must not be conflated:

- `log_Q_mean` ± `log_Q_sd`: over per-CPT depth-averages, one weight per CPT — the spread of
  a new *location*; the number a spatial model has to reduce.
- `within_sd`: depth-to-depth scatter inside one hole, pooled with weights $n_i - 1$ — the
  texture of the trace, not reducible by more CPTs.
- `reading_mean` ± `reading_sd`: over every reading, implicitly thickness-weighted —
  computed directly from the samples (adding the other two in quadrature would double-count
  the averaging error and drop the weighting). Quantiles quoted in $Q_{tn}$ use
  $\exp(\text{mean} + z_p \cdot \text{sd})$; $\exp$ of a mean of logs is a **median**, not a
  mean, and is labelled as such. P10–P90 rather than 95 % bounds, since 20–120 CPTs do not
  resolve the tails.

### 3.2 Empirical variogram and model fit

Semivariance is estimated with the Matheron estimator (via gstools),

$$\hat\gamma(h) = \frac{1}{2 N(h)} \sum_{(i,j)\,\in\, h\text{-bin}} \big(y_i - y_j\big)^2,$$

on bins built to be identifiable rather than uniform: hand-placed short-lag edges
(0, 0.2, 0.5, 1.0 km) so the tight CPT cluster's ~60 m pairs are resolved, then uniform bins
to half the maximum pairwise separation. The fitted model is

$$\gamma(h) = \tau^2 + \sigma^2\big(1 - \rho_m(h)\big)$$

with three constraints, each removing a documented silent-failure mode:

1. **Short lags resolved.** The nugget is only identifiable from pairs well inside the
   range; averaging the cluster into a 600 m bin fits it to zero.
2. **Range bounded by the longest fitted lag.** $\ell$ is bounded so
   $r_p \le h_{\max}$, and the fit records `at_range_bound` when it sits on the bound
   (an unconstrained fit returned a 1700 km range on a trended unit without raising).
3. **Total sill pinned to the sample variance:** $\sigma^2 + \tau^2 = s_y^2$ — the one
   quantity these data estimate well; pinning it stops sill, nugget and range trading off
   freely.

A **nugget floor** takes $\tau^2 \ge \hat\gamma(h_1)$, but only when the shortest resolved
lag satisfies $h_1 \le 0.1\, r_p$ — past that, $\hat\gamma(h_1)$ is mostly structure and
would be a range estimate in disguise. When floored, the excess is taken out of the sill so
the pinned total is preserved.

The fit carries an identifiability verdict:

$$\text{resolved} \iff \frac{\sigma^2}{\sigma^2 + \tau^2} > 0.1
\ \wedge\ \text{not at the range bound} \ \wedge\ N > 0,$$

with a stated reason otherwise (pure nugget → structure finer than the CPT spacing; range at
bound → trend or too few lags). A near-pure-nugget fit on a unit whose minor axis is finer
than the grid spacing is the **correct answer**, not a fitting failure.

### 3.3 Simple kriging

Known constant mean $m$ (the sample mean in practice — the same mild cheat the baseline
makes, so the comparison is fair) plus a covariance model $C(\cdot)$ with nugget $\tau^2$.
With $\mathbf{K} = C(X, X) + \tau^2 I$ (data treated as noisy: `exact=False`,
`cond_err="nugget"`) and $\mathbf{c}(x_*) = C(X, x_*)$:

$$\hat y(x_*) = m + \mathbf{c}^\top \mathbf{K}^{-1} (\mathbf{y} - m),$$

$$\sigma^2_{\text{latent}}(x_*) = \sigma^2 - \mathbf{c}^\top \mathbf{K}^{-1} \mathbf{c},
\qquad
\sigma^2_{\text{obs}}(x_*) = \sigma^2_{\text{latent}}(x_*) + \tau^2.$$

**Library convention, verified empirically rather than assumed:** gstools' kriging variance
is the *observation* variance — far from data it tends to $\sigma^2 + \tau^2$, not
$\sigma^2$ (pinned by a test against the library, so an upgrade fails loudly instead of
drifting calibration numbers). `predict` therefore subtracts the nugget to return the latent
field (clipped at zero against pseudo-inverse round-off), and `predict_observation` adds it
back. The subtraction is an exact identity of the SK equations above, not an asymptotic
approximation.

Conditioning data are noisy on purpose: a depth-average over a finite autocorrelated trace
carries $\sigma^2_{\text{obs}}$, which is precisely what the fitted nugget estimates.
Interpolating them exactly would force the surface through values it should be smoothing and
report zero uncertainty at the CPTs.

Two covariance sources:

- `"fit"` — isotropic variogram fitted per fold (the only option on real data);
- the generating covariance (**the control, not a cheat**) — separates "is kriging
  implemented correctly" from "can a variogram be fitted from 30 CPTs". Its nugget is the
  **realised** $\sigma^2_{\text{obs}}$, not the configured $\tau^2$ (§2.6); handing it the
  configured value would make the reference overconfident and misattribute the error.

---

## 4. Validation

### 4.1 Leave-one-CPT-out cross-validation

Folds are per unit over that unit's present CPTs; the estimator is **refitted on every
fold** (a prediction plotted against a value it was fitted on is not a test). Every fold row
carries both variances, and the pairing of target and error bar is fixed in one place:

| Compared against | Column | Error bar |
|---|---|---|
| the held-out observation | `observed` ($\bar y_i$) | `sd_obs` |
| the field that made it | `latent` ($y^{\text{field}}_i$) | `sd_latent` |

$y^{\text{true}}$ is deliberately not offered as a target: it contains that CPT's own
independent nugget draw, which no estimator can predict, and scoring against it penalises a
perfect model by exactly $\tau^2$.

### 4.2 Metrics

For predictions $\hat y_i$ with stated sd $\hat\sigma_i$ against targets $y_i$
(non-finite pairs dropped):

$$\text{RMSE} = \sqrt{\tfrac1n \sum (\hat y_i - y_i)^2}, \qquad
\text{bias} = \tfrac1n \sum (\hat y_i - y_i),$$

$$R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}, \qquad
\text{cov}_{95} = \tfrac1n \sum \mathbf{1}\!\left[\,|y_i - \hat y_i| \le 1.96\, \hat\sigma_i\right],$$

$$\text{MSSR} = \frac1n \sum \left(\frac{y_i - \hat y_i}{\hat\sigma_i}\right)^{\!2}
\quad (= 1 \text{ if calibrated; } > 1 \text{ overconfident}).$$

MSSR resolves what coverage cannot at small $n$ (coverage moves in steps of $1/n$).
Accuracy and calibration are read as a pair: inaccurate-but-calibrated is the *correct*
outcome on a unit with no exploitable structure; accurate-but-overconfident is the dangerous
direction.

**The reference line.** Under leave-one-out, a constant-mean model predicts
$\hat m_{-i} = (n\bar y - y_i)/(n-1)$, so $y_i - \hat m_{-i} = \frac{n}{n-1}(y_i - \bar y)$
and its LOO $R^2$ is exactly

$$R^2_{\text{LOO, const}} = 1 - \left(\frac{n}{n-1}\right)^{\!2},$$

always slightly negative, never zero. That is the line a spatial estimator has to clear.

### 4.3 Truth accounting (synthetic only)

The share of the baseline's variance a perfect spatial model could remove is

$$\text{structured\_frac} = 1 - \frac{\sigma^2_{\text{obs}}}{s_y^2},$$

defined by *subtracting* the measured noise rather than as the quotient
$(\text{field sd}/s_y)^2$ — the ratio of two independently estimated sds exceeds 1 on the
units where the field dominates. The remainder ($\sigma^2_{\text{obs}}$) is irreducible by
any estimator. The observation noise is measured as
$\operatorname{RMS}(\bar y - y^{\text{field}})$, **not**
$\operatorname{RMS}(\bar y - y^{\text{true}})$; the latter would omit the nugget and
understate the floor by nearly a factor of two.

---

## 5. Real data (IJmuiden Ver)

Real exports are wrangled onto the same contract per project, outside the package.
`cpt_geostat` consumes only `cpt_samples` (`cpt_id, x, y, z, unit_id, Qtn`) and `layers`
(`cpt_id, unit_id, z_top, z_bot`); `unit_summary` is recomputed from the samples if absent,
truth-dependent diagnostics are skipped, and every estimator and metric above runs unchanged.

Mapping decisions for IJmuiden (IJ56):

- **Coordinates.** UTM (ETRS89 / zone 31N) metres → site-centred kilometres:
  $x = (\text{Easting} - E_0)/1000$, likewise $y$. The origin is recorded in metadata and
  can be pinned so a re-run on a subset does not shift the frame.
- **Layer boundaries.** Raw rows carry a unit label, not contacts. Boundaries are placed at
  the midpoint between the last reading of one run and the first of the next, so tops and
  bases meet exactly. Runs are detected *before* dropping the unclassified `Default`
  interval, so a dropped interval leaves a real gap rather than a false contact.
- **Repeated units.** Holes that re-enter a unit deeper down are collapsed to one row per
  `(cpt_id, unit_id)` — the contract's merge key — spanning first top to last base, with
  `thickness_m` the **sum** of the occupied runs (the intervening material is not credited).
  Per-run detail is preserved separately. Note this makes
  $z_{\text{bot}} - z_{\text{top}} \ne$ `thickness_m` for those rows, and the collapsed row
  overlaps the intervening units' rows.
- **Missing $Q_t$.** Readings without a usable value are dropped from `cpt_samples` but
  still count toward layer geometry.
- **Filters.** Unit occurrences thinner than 0.5 m or with fewer than 20 usable readings
  are dropped: a depth-average over a handful of readings of a sliver is noise dressed as an
  observation, and the estimator has no way to down-weight it.
- **Estimability gate.** With 23 units of steeply unequal coverage, what can be fitted is
  tiered by pair counts, not taste: full directional fitting needs roughly $n \ge 30$ CPTs,
  an isotropic sill/range/nugget roughly $n \ge 12$; below that the deliverable is the
  baseline with a pooled fallback variance and an explicit *not identifiable* verdict — a
  defensible refusal, not 23 fitted models.

---

## 6. Stated limitations

- **Inverse crime.** gstools both generates and fits the synthetic fields, so recovery
  numbers are optimistic; the planned sklearn GP and pykrige cross-checks are the mitigation.
- **Simple kriging assumes a constant mean.** On a trended unit the variogram is unbounded,
  the fitted range runs to the identifiability bound, and the honest output is a flag — the
  fix is universal kriging / a GP mean function, specified separately.
- **Isotropic variograms only, so far.** A unit whose minor axis is finer than the CPT
  spacing correctly fits as almost pure nugget under an isotropic model — the right answer
  to the wrong question until directional variograms land.
- **No presence model.** Predictions are meaningful only where a unit exists; on real data
  no mask is available until a presence classifier is built, so real prediction maps are
  provisional.
- **Soft-threshold coverage bias** in patch presence (§2.1) — small, recorded via realised
  coverage.
