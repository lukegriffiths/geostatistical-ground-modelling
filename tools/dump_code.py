"""Emit the whole project as one self-contained markdown file.

    python tools/dump_code.py                 # -> docs/CODE.md
    python tools/dump_code.py --out X.md      # somewhere else
    python tools/dump_code.py --no-docs       # code only, skip README/plan

The output is meant to be handed to a reader with no other context — another
model, or a colleague opening the project cold — so it leads with the
conventions and the run instructions before any code.

Regenerate it after changing the package; it is a build artefact, not a source
of truth, and a stale dump is worse than none.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]

# Ordered: the reader should meet the conventions before the code that applies them,
# and the contract before either side of the split that depends on it.
CODE_SECTIONS = [
    ("Package — core", [
        "cpt_geostat/__init__.py",
        "cpt_geostat/geometry.py",
        "cpt_geostat/covariance.py",
        "cpt_geostat/trend.py",
    ]),
    ("Package — the data contract (any dataset, synthetic or real)", [
        "cpt_geostat/contract/__init__.py",
        "cpt_geostat/contract/schema.py",
        "cpt_geostat/contract/summarise.py",
        "cpt_geostat/contract/io.py",
    ]),
    ("Package — raw export to the contract (shared by every project)", [
        "cpt_geostat/prepare.py",
    ]),
    ("Package — estimators", [
        "cpt_geostat/models/__init__.py",
        "cpt_geostat/models/base.py",
        "cpt_geostat/models/baseline.py",
        "cpt_geostat/models/variogram.py",
        "cpt_geostat/models/kriging.py",
        "cpt_geostat/models/crosscheck.py",
        "cpt_geostat/models/profile.py",
        "cpt_geostat/models/field.py",
    ]),
    ("Package — validation", [
        "cpt_geostat/validate/__init__.py",
        "cpt_geostat/validate/metrics.py",
        "cpt_geostat/validate/cv.py",
    ]),
    ("Package — plotting (Part B, runs on real data)", [
        "cpt_geostat/plots/__init__.py",
        "cpt_geostat/plots/style.py",
        "cpt_geostat/plots/maps.py",
        "cpt_geostat/plots/sections.py",
        "cpt_geostat/plots/diagnostics.py",
        "cpt_geostat/plots/variograms.py",
        "cpt_geostat/plots/profiles.py",
        "cpt_geostat/plots/fields.py",
        "cpt_geostat/plots/predictions.py",
    ]),
    ("Package — synthetic generation and truth (Part A + B1)", [
        "cpt_geostat/synthetic/__init__.py",
        "cpt_geostat/synthetic/config.py",
        "cpt_geostat/synthetic/fields.py",
        "cpt_geostat/synthetic/layout.py",
        "cpt_geostat/synthetic/strat.py",
        "cpt_geostat/synthetic/series.py",
        "cpt_geostat/synthetic/pipeline.py",
        "cpt_geostat/synthetic/truth.py",
        "cpt_geostat/synthetic/plots.py",
    ]),
    ("Package — CLI", ["cpt_geostat/cli.py"]),
    ("Tests", [
        "tests/conftest.py",
        "tests/test_trend.py",
        "tests/contract/test_architecture.py",
        "tests/contract/test_io.py",
        "tests/contract/test_summarise.py",
        "tests/test_prepare.py",
        "tests/models/test_baseline.py",
        "tests/models/test_kriging.py",
        "tests/models/test_ok_uk.py",
        "tests/models/test_crosscheck.py",
        "tests/models/test_directional_variogram.py",
        "tests/models/test_profile.py",
        "tests/models/test_field.py",
        "tests/models/test_validate.py",
        "tests/plots/test_data_plots.py",
        "tests/plots/test_variogram_plots.py",
        "tests/plots/test_profile_plots.py",
        "tests/plots/test_field_plots.py",
        "tests/synthetic/test_geometry.py",
        "tests/synthetic/test_fields.py",
        "tests/synthetic/test_generate.py",
        "tests/synthetic/test_truth.py",
        "tests/synthetic/test_truth_kriging.py",
        "tests/synthetic/test_universal_kriging.py",
        "tests/synthetic/test_truth_plots.py",
    ]),
    ("Configuration and packaging", [
        "config.yaml",
        "pyproject.toml",
        "requirements.txt",
    ]),
    ("Tooling", ["tools/dump_code.py"]),
    # Out of the package on purpose: per-project wrangling that maps a real
    # export onto the dataset contract. `cpt_geostat` never imports from here.
    ("Real-data project — IJmuiden", ["projects/IJmuiden/prepare_data.py"]),
]

DOC_SECTIONS = [
    ("Reference documents", [
        "README.md",
        "docs/USAGE.md",
        "docs/DECISIONS.md",
        "docs/RESULTS.md",
        "docs/METHODOLOGY.md",
        "projects/IJmuiden/README.md",
        "cpt_gp_plan.md",
        "plans/02_estimators_and_validation.md",
        "plans/03_gp_anisotropy_alternatives.md",
    ]),
]

LANGUAGES = {
    ".py": "python",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".txt": "text",
    ".cfg": "ini",
}

PREAMBLE = """\
This is the complete source of a geostatistical ground-modelling toolkit for normalised
cone tip resistance (`Qtn`) across an offshore wind farm site. It generates a synthetic CPT
dataset in which each soil unit is a laterally varying 2D field, and units are **not** present
at every CPT location, then plots maps and diagnostics against it.

Parts A (data generation) and B (map plotting) of the design document are implemented, as are
the per-unit baseline estimator and **simple, ordinary and universal kriging** (`models/`),
directional variograms, leave-one-CPT-out cross-validation (`validate/`), the predicted-vs-true
cross plots, per-unit variogram figures and per-hole **depth profiles** reconstructing `Qtn`
with depth against the measured trace. The Gaussian process and hyperparameter recovery are
**not** — they are specified separately. The plan for them is included below (`plans/02`,
`plans/03`), annotated with what is already built and with what it got wrong.

The synthetic generator exists to **validate** estimators against known truth. A real dataset
(IJmuiden Ver, 194 CPTs, 23 units, 483k readings) is wrangled onto the same contract under
`projects/`, which is where they get **applied**. `cpt_geostat` never imports from `projects/`.

The wrangling itself is split the same way. `cpt_geostat.prepare` holds the part that is
identical at every site — run detection, contacts at midpoints, collapsing repeated units,
depth-averaging, sliver filtering, metadata — and a project supplies only a normalised raw
frame and a `PrepareConfig` of declarations about its own export. IJmuiden's script is 122
lines of declaration against 413 before the split.

That principle is the package layout, not just a slogan. `cpt_geostat.contract` (the data contract),
`cpt_geostat.models` (estimators), `cpt_geostat.validate` and `cpt_geostat.plots` run on **any** dataset meeting
the contract; `cpt_geostat.synthetic` needs a generator config or realised truth. The dependency
only ever points one way — the generator is a consumer of the contract, never the reverse —
and `tests/contract/test_architecture.py` pins it: importing `cpt_geostat.contract` loads six
modules and neither gstools nor scipy.

## Running it

```bash
python -m venv .venv
.venv/bin/pip install -e .                  # Windows: .venv/Scripts/pip

.venv/bin/cpt-geostat synth run --out run   # generate + plot + model
.venv/bin/python -m pytest -q
```

`run/data/` receives the four output files; `run/figures/` receives 18 figures;
`run/models/` receives the per-unit baseline table, the cross plots, per-unit variogram
figures under `variograms/`, per-model field maps under `fields/` and one depth
profile per hole under `profiles/`.

The CLI splits the same way: `synth generate` / `synth run` need a generator config, while
`plot` and `model` need only a directory holding `cpt_samples` and `layers` and run
identically on real data. Each prints the mode it is in, so a real-data run never looks like
a synthetic one whose truth diagnostics silently went missing.

## Conventions

These are fixed once and applied everywhere. Two of them are load-bearing enough that they
are enforced by tests rather than by comment.

| | |
|---|---|
| Modelling variable | `log(Qtn)` |
| Coordinates | kilometres, origin at site centre |
| Depth | metres below seabed, positive down |
| **Azimuth** | **degrees clockwise from north** (0 = +y = N, 90 = +x = E) |
| `range_km` | **practical range** — the separation at which correlation reaches 0.05 |

**Azimuth.** The unit vector along azimuth `a` is `(sin a, cos a)`, *not* `(cos a, sin a)`.
The same convention governs trend directions, anisotropy angles and channel orientations.
gstools uses the mathematical convention (radians counter-clockwise from +x); the bridge is
`azimuth_to_math_angle` in `cpt_geostat/geometry.py` and exists in exactly one place. A field that
is anisotropic *the wrong way* looks entirely plausible on a map, so the convention is checked
numerically in `test_anisotropy_major_axis_follows_its_azimuth`, not only by eye.

**Range.** gstools parameterises covariance by `len_scale`, which is **not** the range a
variogram fit reports: for Matérn 2.5 the practical range is 3.74x the `len_scale`. Passing a
practical range straight through as a `len_scale` makes every field several times smoother
than configured — invisible by eye, fatal to hyperparameter recovery. `GrfConfig.range_kind`
records which is meant (`practical` by default) and the conversion lives only in
`cpt_geostat/covariance.py`, together with its inverse — the generator and the estimators share one
site rather than one each, which is what let the original bug through.

**Variance.** There are two and they are not interchangeable. `predict` returns the **latent**
field — what an estimator believes about the ground — and is what estimator-vs-estimator
comparisons and difference maps use. `predict_observation` adds `noise_var_` (nugget *plus*
depth-averaging error) and is what every cross-validation metric uses, because a held-out
value is an observation. Scoring calibration on latent variance against noisy held-out values
guarantees under-coverage, and it misfires worst on the units meant to come out wide and
honest. `cpt_geostat/models/base.py` states which method is which; the convention is never left to
the reader. gstools' kriging variance turns out to be the **observation** one — it tends to
`sill + nugget` far from data — so `SimpleKriging.predict` subtracts the nugget back off.
That is pinned by a test against the library, not inferred from its documentation.

## Output contract

| File | Contents |
|---|---|
| `cpt_samples` | `cpt_id, x, y, z, unit_id, Qtn` — one row per depth reading |
| `layers` | `cpt_id, unit_id, z_top, z_bot` — absent units are missing rows |
| `unit_summary` | `cpt_id, x, y, unit_id, log_Q_mean, log_Q_sd, n_samples, thickness_m` — the model input |
| `truth.yaml` | resolved generative parameters + realised statistics (synthetic only) |
| `truth_fields.npz`, `truth_points.csv` | truth rasters and true CPT node values, for the B1 panels (synthetic only) |
| `unit_baseline.csv` | per-unit mean and sd, written by `cli model` — not an input to anything |
| `cv_predictions.csv` | leave-one-out folds: `pred`, `sd_latent`, `sd_obs`, `observed`, `latent` |

Each table may be `.csv` or `.parquet`; parquet wins where both exist and is what real exports
should use (IJmuiden is 129 MB of csv against 4 MB of parquet).

**Real data** supplies `cpt_samples` and `layers` only. `unit_summary` is then recomputed from
the samples, truth-dependent diagnostics are skipped, and the unit palette falls back to a
default cycle. Every Part B plotting function takes a dataframe or `Dataset` and returns a
`matplotlib.figure.Figure`; none touch the filesystem, so the same calls work on real data
unchanged.

## The six-unit contrast set

Each unit isolates one effect, so a downstream failure points at a cause:

| Unit | Presence | Trend | Anisotropy | Purpose |
|---|---|---|---|---|
| 1 | everywhere | none | isotropic | GP ≡ simple kriging identity test |
| 2 | everywhere | 0.10/km at 115° | isotropic | trend handling only |
| 3 | channel (~28%) | none | ratio 4 at 70° | anisotropy only |
| 4 | patch (~45%) | 0.06/km at 25° | ratio 3 at 135° | both, on different bearings |
| 5 | narrow channel (~16%) | weak | isotropic, 1.5 km range | near data-limited |
| 6 | patch (~30%) | none | isotropic, nugget > sill | noise-dominated failure case |

## Generator notes

* **Presence** is a *probability* raster; the Bernoulli draw happens once, at the CPT
  locations. Drawing per raster cell would give salt-and-pepper edges that no downstream
  sampling could undo. Channel edges are made ragged by perturbing the bank position with a
  short-range GRF rather than by blurring.
* **Thickness** maps a unit-variance GRF through a logistic into `[min_m, max_m]`, with the
  logistic's two parameters solved by Gauss–Hermite quadrature so the realised mean and sd
  match the config. An unattainable sd warns rather than silently delivering something else.
* **Missingness has two correlated sources**: lateral absence, and pinch-out where shallow
  units are thick enough to push a deep unit past the 50 m cut-off. A unit retaining less than
  `min_thickness_m` is marked absent, not kept as a sliver.
* **Within-unit scatter** is AR(1) with a configurable correlation length, so the effective
  sample size behind a depth-average is much smaller than `n_samples`. This is why the
  realised within-unit sd sits below the configured one.
* **RNG streams are keyed by name**, not draw order, so editing one unit does not re-roll the
  rest of the site.
"""


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")


def fence_for(content: str) -> str:
    """A fence longer than the longest backtick run inside, so nesting markdown is safe."""
    longest = max((len(m) for m in re.findall(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def anchor(rel: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", rel.lower()).strip("-")


def render(sections, out_lines: List[str], missing: List[str]) -> None:
    for title, rel_paths in sections:
        out_lines.append(f"\n## {title}\n")
        for rel in rel_paths:
            path = ROOT / rel
            if not path.exists():
                missing.append(rel)
                continue
            content = read(path)
            fence = fence_for(content)
            lang = LANGUAGES.get(path.suffix, "")
            out_lines.append(f"\n### `{rel}`\n")
            out_lines.append(f"{fence}{lang}\n{content}\n{fence}\n")


def build(include_docs: bool = True) -> str:
    sections = list(CODE_SECTIONS) + (list(DOC_SECTIONS) if include_docs else [])

    counted = [
        ROOT / rel
        for _, rels in CODE_SECTIONS
        for rel in rels
        if (ROOT / rel).exists() and (ROOT / rel).suffix == ".py"
    ]
    n_lines = sum(len(read(p).splitlines()) for p in counted)
    stamp = _dt.date.today().isoformat()

    lines: List[str] = [
        "# cpt_geostat — full source",
        "",
        f"*Generated by `tools/dump_code.py` on {stamp} — "
        f"{len(counted)} Python files, {n_lines} lines. Regenerate rather than edit.*",
        "",
        PREAMBLE,
        "",
        "## Contents",
        "",
    ]
    for title, rels in sections:
        lines.append(f"- **{title}**")
        for rel in rels:
            if (ROOT / rel).exists():
                lines.append(f"  - [`{rel}`](#{anchor(rel)})")
    lines.append("")
    lines.append("---")

    missing: List[str] = []
    render(sections, lines, missing)

    if missing:
        lines.append("\n## Not found at generation time\n")
        lines.extend(f"- `{rel}`" for rel in missing)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "docs" / "CODE.md"))
    parser.add_argument("--no-docs", action="store_true",
                        help="omit README.md and cpt_gp_plan.md")
    args = parser.parse_args(argv)

    text = build(include_docs=not args.no_docs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines, {len(text) / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
