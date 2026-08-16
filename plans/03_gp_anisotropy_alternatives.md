# Plan 03 — GP anisotropy: the approaches not taken

**Status:** reference. Nothing here is scheduled.
**Companion:** [`02_estimators_and_validation.md`](02_estimators_and_validation.md).

---

## The problem

The anisotropy in this dataset is **oblique** — unit 3 has a ratio-4 major axis at 70°, unit 4
a ratio-3 axis at 135°. `scikit-learn`'s ARD length scales are **axis-aligned only**: one
length scale per input dimension, no cross terms. An ARD Matérn fitted to unit 3 can stretch
along x and along y independently but cannot tilt, so it will report a compromise that matches
neither the true ratio nor the true bearing.

Something has to supply the rotation. There are three ways to do it, and plan 02 takes the
first. This document keeps the other two costed, so switching later is a decision rather than
a rediscovery.

---

## Chosen: profile likelihood over the rotation angle

For each candidate azimuth on a grid, rotate the coordinates, fit a standard ARD Matérn, and
keep the highest log-marginal-likelihood.

```python
for az in np.arange(0, 90, 5):           # 0-90 ONLY; see the degeneracy below
    Xr = rotate_to_azimuth_frame(X[:, 0], X[:, 1], az)
    gp = GaussianProcessRegressor(Matern(length_scale=[1.0, 1.0], nu=2.5) + WhiteKernel())
    gp.fit(np.column_stack(Xr), y)
    lml[az] = gp.log_marginal_likelihood_value_
# recovered major axis = the one with the longer fitted length scale, mod 180
```

### The 90° degeneracy — scan 0–90°, not 0–180°

An axis-aligned ARD kernel at rotation θ is the **same model** as θ+90° with the two length
scales swapped. A 0–180° scan therefore has two *exactly* equal maxima, and "refine around the
best" is ill-defined between them — the recovered azimuth can land 90° from truth for reasons
that have nothing to do with the data.

Every distinct unordered axis pair appears exactly once in [0°, 90°), so that interval is both
complete and non-degenerate. The recovered major azimuth is then the axis carrying the longer
length scale, wrapped mod 180°, resolved before it reaches any circular-difference comparison.

This is a property of the *rotated-ARD parameterisation*, not of the search strategy: it
applies equally to Alternative A below, where it is harder to handle because a continuous
optimiser cannot simply be given a half-open interval.

**Why it wins here.** No custom gradient code. The degeneracy is disposed of by construction
rather than by restarts. Roughly 18 cheap fits is seconds. And the likelihood-vs-angle curve is
itself the diagnostic that tells you whether the angle is identifiable at all — for unit 5
(1.5 km range, 22 CPTs), and for the ten IJmuiden units under 12 holes, it will be flat, and a
flat curve is a *result*, not a failure. Neither alternative below produces that curve for free.

**What it costs.** The angle is resolved only to the grid plus refinement, and the reported
uncertainty on the angle is the curve's shape rather than a proper posterior. Fitting is
`n_angles` times slower than a single joint optimisation — irrelevant at the sizes in play
(largest real unit: 148 CPTs), and the binding constraint only if this ever moves to thousands
of points.

---

## Alternative A — custom `sklearn` `Kernel` subclass

Subclass `sklearn.gaussian_process.kernels.Kernel` with `length_scale_major`,
`length_scale_minor` and `angle` as hyperparameters, applying the rotation and scaling inside
`__call__` before delegating to a Matérn.

**Buys.** A single joint optimisation over all hyperparameters at once, so the angle is
continuous and drops out of the same L-BFGS run as everything else. Composes properly with the
rest of sklearn — `WhiteKernel`, `ConstantKernel`, `kernel.theta`, restart handling.

**Costs.** `__call__(eval_gradient=True)` must return `dK/dtheta` for every hyperparameter, and
the derivative with respect to the rotation angle is the fiddly one. Finite-differencing it
inside the kernel works and sidesteps the algebra, at the price of accuracy and speed in the
optimiser's inner loop.

More seriously, it inherits the **exact 90° degeneracy** described above — two identical optima,
not merely two near-equal ones — and a continuous optimiser cannot be restricted to [0°, 90°)
as cleanly as a grid can. L-BFGS from a single start picks one arbitrarily and reports it with
unwarranted confidence. Mitigating that means `n_restarts_optimizer` with varied starting
angles plus a post-hoc canonicalisation by which length scale came out longer — at which point
the cost advantage over the grid has largely evaporated and the canonicalisation step is needed
anyway.

**Trigger to switch.** The 5° grid proving too coarse to separate the recovered angle from
truth on units 3 and 4, *and* n growing enough that `n_angles` sequential fits stop being free.

---

## Alternative B — hand-rolled exact Cholesky GP

Roughly 60 lines of numpy: build `K` from a rotated-anisotropic Matérn, `cho_factor`, solve,
and return the posterior mean and variance directly.

**Buys.** Total control of the mean function. This is the real draw — sklearn has no mean
function at all, which is why plan 02 has to route around it with OLS detrending or an explicit
basis. A hand-rolled GP can integrate out a linear basis under a flat prior, which *is*
universal kriging, giving a like-for-like comparison against `krige.Universal` that sklearn
cannot express. It would also make the simple-kriging identity exact by construction rather
than a numerical near-miss to be measured, and it removes the "`ConstantKernel` is not a
constant mean" trap (risk B in plan 02) entirely, since the mean would be explicit.

**Costs.** Drops sklearn's tested optimiser, restart logic and numerical safeguards, and
replaces them with code that has to earn the same trust. It also **weakens the headline
result**: the SK identity test is persuasive precisely *because* sklearn and gstools are
independent implementations that agree. A hand-rolled GP sharing conventions with our own
kriging wrapper would be checking our arithmetic against itself.

**Trigger to switch.** Needing a proper universal-kriging-equivalent GP with the linear basis
integrated out, or the OLS-detrend variance understatement (it ignores trend-estimation
uncertainty) becoming a material problem in the calibration numbers on units 2 and 4.

**Middle path worth remembering.** Keep sklearn as the production GP and write the exact
Cholesky GP as a **test oracle only** — a few dozen lines used to prove the sklearn path does
what we think on a handful of small cases. That captures most of the assurance at a fraction of
the risk, and it is a cheap addition to plan 02 if the identity test proves awkward to pin down.

---

## Measured: is there anisotropy on IJmuiden to model? (2026-07-30)

Asked before scheduling any of the above, because the cost is only worth paying against a
signal. Directional variograms already exist as a **diagnostic** — `directional_variogram`,
`plot_directional_variogram`, the 30-CPT gate, 10 figures written on IJmuiden. The estimators
remain isotropic, so nothing consumes them.

Test: per unit, the ratio of the largest to smallest sector half-sill lag, against a parametric
null — isotropic fields simulated with that unit's *own* fitted covariance at its *own* CPT
locations, measured identically, 60 realisations.

| | |
|---|---|
| eligible units (≥30 CPTs, resolved) | 9 |
| observed contrast above the null's 95th pct | **1** |
| expected by chance at 9 tests | 0.45 |

Median observed ratio 3.0 against null medians of 1.8 — but null 95th percentiles run 3–9, so
the observed values sit inside what an isotropic field produces at these sample sizes. **No
signal. Not scheduled, and now for a measured reason rather than an unexamined one.**

Two limits on that conclusion, so it is not over-read. The 6 directional lag bins quantise the
ratio, giving power only against strong anisotropy (ratio ≳3); a ratio-1.5 fabric would not
show. And the recovered azimuths are uninformative — ties resolve to the first sector — so the
test says "no strong contrast", not "the axes are isotropic".

**Trigger to revisit.** A site whose units clear the null, or a decision to validate the method
generally rather than answer this site: the synthetic contrast set already carries unit 3
(ratio 4 at 70°) and unit 4 (ratio 3 at 135°) for exactly that, and they are currently unused
by any estimator.

---

## Summary

| | Rotation | Mean function | Angle diagnostic | Independence of gstools | Code risk |
|---|---|---|---|---|---|
| **Profile likelihood** (chosen) | grid + refine | via detrend | **free, and informative** | full | low |
| Custom kernel | continuous, joint | via detrend | needs extra work | full | medium |
| Hand-rolled GP | continuous, joint | **exact, incl. UK** | needs extra work | weakened | high |
