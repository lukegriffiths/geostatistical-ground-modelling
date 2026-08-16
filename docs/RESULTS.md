# Measured results

What the estimators actually do, on the synthetic contrast set and on IJmuiden.
The reasoning behind the design choices these numbers test is in
[DECISIONS.md](DECISIONS.md).

---

## 1. The contrast set

The synthetic dataset is built so that each unit isolates one effect, and a
failure therefore points at a cause:

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

---

## 2. The baseline

`structured_frac` — the share of the baseline's variance a perfect spatial model
could remove — runs **0.83–0.97 on units 1–5** and **0.44 on unit 6**, which is
the designed noise-dominated case: the baseline is close to the best available
answer there, and an estimator that appears to beat it is overfitting.

### 2.1 Cross plots: the same folds, two targets

| | vs. the held-out **observation** | vs. the **latent** field |
|---|---|---|
| error bar | `sd_obs` (latent + nugget + averaging error) | `sd_latent` |
| coverage 95 | 0.91–0.97 | 0.13–0.50 |
| MSSR | 1.01–1.09 | 12–116 |

Both are correct.  Scored on what it actually predicts — one noisy depth-average
— the constant model is **well calibrated**: it attributes all spatial variation
to noise and gets the interval width right.  Scored against the field, the same
intervals are out by up to 11× in sd.

---

## 3. Kriging

### 3.1 The mean assumption is the only difference

| | mean | dof | when it is the right answer |
|---|---|---|---|
| `SimpleKriging` | known constant | 0 | the mean is genuinely known |
| `OrdinaryKriging` | unknown constant | 1 | the honest default on real data |
| `UniversalKriging` | unknown linear drift | 3 | the unit carries a trend |

### 3.2 Universal kriging on the trended unit

Unit 2 (0.10/km at 115°), leave-one-out against the held-out observation:

| | RMSE | coverage 95 | MSSR |
|---|---|---|---|
| baseline | 0.547 | 0.966 | 1.01 |
| SK (fitted) | 0.242 | 0.812 | 2.29 |
| OK (fitted) | 0.242 | 0.812 | 2.29 |
| **UK (fitted)** | **0.224** | **0.974** | **0.90** |

Scored against the *latent* field the contrast is starker still — SK's MSSR is
12.3 against UK's 1.07 — because there the unmodelled trend has nowhere to hide.

UK is not free, and the suite asserts that it is not: on the four units with no
trend it *loses* to SK.  On unit 5 — 22 CPTs in a narrow channel — UK is badly
overconfident, which is the argument for the IJmuiden estimability gate rather
than for tuning UK.

### 3.3 Fitted covariance against the truth covariance

Leave-one-out RMSE against the held-out observation, and MSSR alongside:

| unit | baseline | SK (fitted) | SK (truth cov) | | MSSR fitted | MSSR truth |
|---|---|---|---|---|---|---|
| 1 | 0.300 | 0.247 | **0.246** | | 1.05 | 1.00 |
| 2 | 0.547 | **0.242** | 0.272 | | 2.29 | 1.38 |
| 3 | 0.324 | 0.337 | **0.291** | | 1.32 | 0.98 |
| 4 | 0.422 | 0.367 | **0.345** | | 1.25 | 1.35 |
| 5 | 0.291 | **0.253** | 0.264 | | 1.31 | 0.87 |
| 6 | **0.239** | 0.247 | 0.256 | | 1.21 | 0.96 |

With the true covariance, kriging beats the baseline on every unit except 6 and
is calibrated (MSSR 0.87–1.38).  That validates the implementation.  With a
*fitted* covariance it predicts about as well but is consistently overconfident,
which localises the remaining error to variogram estimation rather than to the
kriging.

### 3.4 Three things working as designed rather than failing

* **Unit 6 is the one unit the baseline wins.**  Nugget above sill,
  `structured_frac` 0.44 — there is nothing for kriging to exploit, and it
  correctly declines to invent any.
* **Unit 2 is flagged `resolved=False`.**  Its trend makes the variogram
  unbounded, so the fitted range runs to the identifiability bound and
  `params_["range_km"]` comes back `None` with a `not_identifiable` note.  Fixing
  it needs universal kriging.
* **Units 3 and 5 fit a nugget of exactly zero.**  Unit 3's minor axis is 1 km
  against a ~1.3 km CPT spacing, so an isotropic variogram sees noise.  Nearly
  all nugget is the *correct* fit, not a failure — and `VariogramFit.resolved`
  says which.

---

## 4. Anisotropy recovery

At 60 simulations per unit:

| unit | truth | fitted | LR | gate | called |
|---|---|---|---|---|---|
| 1 | isotropic | 1.07 @ 160° | 2.0 | null p95 13.1 | isotropic ✓ |
| 2 | isotropic + trend | 1.27 @ 40° | 4.3 | null p95 9.5 | isotropic ✓ |
| 3 | **ratio 4 @ 70°** | 7.87 @ 100° | 5.1 | minor axis unresolved | isotropic ✗ |
| 4 | **ratio 3 @ 135°** | 3.16 @ 100° | 4.5 | null p95 5.5 | isotropic ✗ |
| 5 | isotropic | — | — | n = 22, under the gate | isotropic ✓ |
| 6 | isotropic, noise-dominated | 9.37 @ 80° | 2.4 | null p95 5.6 | isotropic ✓ |

**Every isotropic unit is correctly rejected, and both anisotropic units are
missed.**  That is the honest read: the method has no useful power at these
sample sizes and these ratios.  Unit 3 has 31 CPTs and a minor axis finer than
the CPT spacing — the structural gates catch it before the null is even reached.
Unit 4 has 58 CPTs, recovers the ratio well (3.16 against a true 3) and the
bearing poorly (100° against 135°), and lands just under its threshold.

On the synthetic contrast set no unit passes the gate, and the run says so
outright rather than leaving a reader to hunt for a difference between two
identical figures.

### 4.1 Two numerical traps found on the way, both now pinned by tests

That conclusion is *weaker* than an earlier version of this analysis claimed, and
the reason is worth recording.

* **The null simulated the total variance as structured, with no nugget.**
  Fields far smoother than the data, so spurious axes fit them better, so the
  threshold came back inflated — ninefold on unit 2.
* **`(range, nugget)` is a ridge, not a basin.**  A short range with no nugget
  and a long range with a large one explain the same data to within a fraction
  of a log-likelihood, so a single optimiser start seeded from the variogram fit
  never left its basin.  The isotropic reference is now multi-started from twelve
  points, and the winning azimuth is refined from three, so the two sides of the
  likelihood ratio are searched alike.

The second fix *lowered* every LR statistic, because the isotropic reference it
is measured against had been under-fitted.  Unit 4's LR fell from 18.0 to 4.5 and
crossed from a detection to a miss.  **An inflated likelihood ratio against an
under-fitted null is how a method like this manufactures findings**, and it took
a validation harness with planted truth to see it — which is what the contrast
set is for.

---

## 5. Field maps

On the synthetic run:

* Units 3 and 5 make the case for drawing the interval immediately: their
  **lower map is almost uniformly dark and their upper almost uniformly
  bright**, because away from the holes the interval spans nearly the unit's
  whole range.  Units 1 and 2, at 117 CPTs, stay tight.
* Unit 1's sd map is dark along the CPT grid and bright in the **thinned NE
  corner** — the sensitivity-sweep hole in the layout, showing up as uncertainty
  exactly where it should.
* Unit 3's truth is fine ENE–WSW striping (ratio 4 at 70°, 1 km minor axis
  against ~1.3 km CPT spacing).  Its prediction map is bullseyes on a flat
  background, because an isotropic variogram genuinely cannot resolve that —
  the documented "correct answer to the wrong question", now visible side by
  side with the truth rather than inferred from a `resolved=False`.
* The baseline's map is a flat colour and its sd map is flat too — but on the
  scale shared across models that flat sd sits right at the bottom, because the
  baseline's latent sd is the standard error of its one constant, `s/√n`.  On
  unit 1 it claims to know the field to **±0.028 everywhere**, against ordinary
  kriging's **±0.28** away from the holes.  A ten-fold difference in claimed
  precision, which per-figure auto-scaling hid completely.

On IJmuiden, five of the 23 units come out flat under OK as well, labelled
*"uniform — no spatial structure resolved"*: the pure-nugget fits, where kriging
has correctly collapsed onto the baseline.

---

## 6. Depth-profile calibration

On IJmuiden, ordinary kriging:

| | realised coverage | MSSR |
|---|---|---|
| 17 of 23 units | 0.94–0.97 | 0.89–1.13 |
| `GGM_23_C`, `GGM_31_Si`, `GGM_22_Si` | 0.82–0.91 | 1.42–1.54 (narrow) |
| `GGM_27_S`, `GGM_24_C_S` | 1.00 | 0.64–0.73 (over-wide, few CPTs) |

Against a nominal 0.95 that is close calibration on real ground, and the
knowing double-count of the depth-averaging error turns out to be negligible next
to everything else.  Coverage is 0.956 either side of the `qt` conversion, since
the transform is monotone.

On synthetic data the residuals come out essentially N(0,1) with bias −0.01; the
two units that miss are unit 2 and unit 4, the two carrying trends that ordinary
kriging structurally cannot model — the same failure that shows up as MSSR 2.29
in the cross-validation table, and the reason universal kriging exists.

`within_sd` on IJmuiden runs 0.45–0.56 against a between-hole `log_Q_sd` of
0.31–0.71: **depth texture inside one hole is as large as the variation between
holes.**

---

## 7. Stress exponent on IJmuiden

`cpt.infer_gamma_eff` over all 532,321 readings carrying both `qt` and `Qt`
returns `gamma_eff = 10.07 kN/m³` at `n = 1`, reproducing the export's column to
a median 1.002 (p5–p95 0.993–1.016).  Fitting `n = 0.5` instead leaves a residual
that sweeps 0.37 to 1.61 with depth and no unit weight repairs it, so the column
is Robertson's `Qt` whatever the `Qtn` rename downstream suggests.

`cpt.soil_behaviour_index` runs Robertson's iteration over raw `qt` and `fs`; on
IJmuiden's export it converges in 26 iterations and sorts the units by soil type
unprompted:

| | `Ic` | fitted `n` |
|---|---|---|
| sands (`GGM_01_S` … `GGM_27_S`) | 1.56–1.92 | 0.48–0.72 |
| silts (`GGM_22_Si`, `GGM_31_Si`) | 1.90–2.24 | 0.62–0.91 |
| clays (`GGM_03_C` … `GGM_27_C`) | 2.41–2.89 | 0.80–0.98 |

**That table is not licence to use those exponents on this dataset** — see
[decisions §10.2](DECISIONS.md#102-robertsons-per-unit-n-is-a-diagnostic-not-a-licence).

### 7.1 How much the exponent matters, and where

`L = log(σ'v0/pa)` is the whole lever, and it is **zero at σ'v0 = pa**, which on
IJmuiden's gradient is 9.9 m below seabed.  At that depth the exponent is
irrelevant; away from it the doubt grows logarithmically in both directions.
`n_sd = 0.1` adds 0.30 to the log-sd at 0.5 m and 0.18 at 60 m, against a
between-hole `log_Q_sd` of 0.31–0.71 — so **at the seabed the exponent is as
large a source of doubt as the spatial model, and at 10 m it is nothing.**  That
is the opposite of most people's intuition and is the reason to carry `n_sd` at
all.

The closed form for an uncertain exponent was verified against a 2M-draw
simulation to 3–4 significant figures at five depths.
