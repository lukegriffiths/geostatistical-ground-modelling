"""Command line entry points.

    python -m cpt_geostat.cli synth generate --config config.yaml --out data/
    python -m cpt_geostat.cli synth run      --config config.yaml --out run/
    python -m cpt_geostat.cli plot  --data data/ --out figures/
    python -m cpt_geostat.cli model --data data/ --out models/

The split is the package's architecture made visible.  ``synth`` needs a
generator config and produces truth; ``plot`` and ``model`` need only a
directory holding ``cpt_samples`` and ``layers``, and run identically on real
data.  Each command prints the mode it is in, so a run against real data never
looks like a run whose truth diagnostics silently went missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # figures are written, never shown

from . import plots
from .contract import read_dataset, write_dataset
from .cpt import (
    StressProfile,
    exponent_uncertainty_from_metadata,
    normalisation_from_metadata,
)
from .models import baseline_factory, kriging_factory, unit_baseline_table
from .models.field import field_raster, predict_fields
from .plots.fields import shared_colour_limits
from .models.profile import profile_coverage, qt_readings, reading_predictions
from .models.variogram import MIN_CPT_FOR_DIRECTIONAL, fit_unit_variogram, unit_block
from .validate import loo_by_unit, score_by_unit


def _save(fig, out_dir: Path, name: str, dpi: int) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _banner(ds, action: str) -> None:
    """State the mode up front — synthetic and real runs differ in what they can say."""
    mode = "synthetic (truth found)" if ds.is_synthetic else "real data (no truth)"
    print(f"{action}: mode = {mode}")


def cmd_generate(args) -> int:
    from .synthetic import Config, generate

    cfg = Config.load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    ds = generate(cfg)
    out = write_dataset(ds, args.out)

    realised = ds.truth["realised"]
    print(f"wrote {out}")
    print(f"  {realised['n_cpt']} CPTs, {len(ds.layers)} unit intervals, "
          f"{realised['n_depth_readings']} depth readings")
    for uid, stats in realised["units"].items():
        print(f"  {uid}: present at {stats['n_cpt_present']:>3} CPTs "
              f"({stats['coverage_cpt']:.0%}), mean thickness "
              f"{stats['thickness_mean_m']:.1f} m")
    return 0


def cmd_plot(args) -> int:
    ds = read_dataset(args.data)
    _banner(ds, "plot")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    written.append(_save(plots.plot_layout(ds), out, "layout", args.dpi))
    written.append(_save(plots.plot_presence_map(ds), out, "presence_map", args.dpi))
    written.append(_save(plots.plot_value_map(ds), out, "value_map", args.dpi))
    written.append(_save(plots.plot_thickness_map(ds), out, "thickness_map", args.dpi))
    written.append(_save(plots.plot_trend_check(ds), out, "trend_check", args.dpi))
    written.append(_save(plots.plot_lag_coverage(ds), out, "lag_coverage", args.dpi))
    written.append(_save(plots.plot_depth_traces(ds), out, "depth_traces", args.dpi))
    written.append(
        _save(plots.plot_within_unit_scatter(ds), out, "within_unit_scatter", args.dpi)
    )

    for fig, section in zip(
        plots.plot_configured_sections(ds), getattr(ds.config, "sections", []) or []
    ):
        written.append(_save(fig, out, f"section_{section.name}", args.dpi))

    if ds.is_synthetic:
        # The B1 panels are synthetic-only, so the import is too.
        from .synthetic.plots import (
            TruthUnavailable,
            plot_anisotropy_check,
            plot_unit_truth_panel,
        )

        try:
            written.append(
                _save(plot_anisotropy_check(ds), out, "anisotropy_check", args.dpi)
            )
            for uid in ds.unit_ids:
                written.append(
                    _save(plot_unit_truth_panel(ds, uid), out, f"truth_{uid}", args.dpi)
                )
        except TruthUnavailable as exc:
            print(f"skipping truth diagnostics: {exc}", file=sys.stderr)
    else:
        print("truth diagnostics skipped — nothing to compare against")

    print(f"wrote {len(written)} figures to {out}")
    return 0


def cmd_model(args) -> int:
    """The per-unit baseline: one mean and one sd per unit, no spatial term."""
    ds = read_dataset(args.data)
    _banner(ds, "model")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table = unit_baseline_table(ds)
    table.to_csv(out / "unit_baseline.csv", float_format="%.6g")

    cols = ["n_cpt", "cpt_fraction", "log_Q_mean", "log_Q_sd", "se_mean", "within_sd"]
    print("per-unit baseline (log Qtn, across CPTs):")
    print(table[cols].to_string(float_format=lambda v: f"{v:.3f}"))

    if not ds.is_synthetic:
        print("\ntruth comparison skipped — nothing to compare against")
    else:
        from .synthetic.truth import truth_reference_table

        # structured_frac is defined by subtracting the *measured* noise rather
        # than by dividing by the measured field sd.  The two agree in
        # expectation, but the ratio form is a quotient of two independently
        # estimated sds and comes out above 1 on the units where the field
        # dominates — reporting that a spatial model could explain 104% of the
        # variance.
        joined = table.join(truth_reference_table(ds), how="left")
        joined["mean_error"] = joined["log_Q_mean"] - joined["mu"]
        joined["structured_frac"] = 1.0 - (joined["obs_noise_sd"] / joined["log_Q_sd"]) ** 2
        joined.to_csv(out / "unit_baseline_vs_truth.csv", float_format="%.6g")
        print("\nvs. truth:")
        print(
            joined[
                ["log_Q_mean", "mu", "mean_error", "log_Q_sd", "field_sd",
                 "obs_noise_sd", "structured_frac"]
            ].to_string(float_format=lambda v: f"{v:.3f}")
        )
        print("\nstructured_frac = 1 - (obs_noise_sd / log_Q_sd)^2 — the share of the")
        print("baseline's variance a perfect spatial model could explain.  The rest is")
        print("nugget + depth-averaging error and no estimator can remove it.")

    return _cross_plots(ds, out, args.dpi, args)


def _fitted_variograms(ds) -> pd.DataFrame:
    """Per-unit variogram fits, with what it is safe to say about each."""
    rows = []
    for uid in ds.unit_ids:
        vf = fit_unit_variogram(ds, uid)
        if vf is None:
            continue
        rows.append({
            "unit_id": uid, "n_cpt": len(unit_block(ds, uid)), "sill": vf.sill,
            "range_km": vf.range_km, "nugget": vf.nugget,
            "structured": vf.structured_fraction, "short_pairs": vf.n_short_pairs,
            "resolved": vf.resolved, "note": vf.why_not_resolved() or "",
        })
    return pd.DataFrame(rows).set_index("unit_id")


def _variogram_figures(ds, out: Path, dpi: int) -> int:
    """A variogram figure per unit, and a directional one where it is estimable.

    The directional gate is plan 02's estimability tier rather than a display
    preference: anisotropy needs pairs spread over four azimuth sectors and ten
    lag bins, which is roughly 30 CPTs.  Below that the sector curves are noise,
    and inviting a reader to compare them is worse than not drawing them.
    """
    vdir = out / "variograms"
    vdir.mkdir(parents=True, exist_ok=True)

    written = 0
    for uid in ds.unit_ids:
        if fit_unit_variogram(ds, uid) is None:
            continue
        _save(plots.plot_variogram(ds, uid), vdir, f"variogram_{uid}", dpi)
        written += 1
        if len(unit_block(ds, uid)) >= MIN_CPT_FOR_DIRECTIONAL:
            _save(plots.plot_directional_variogram(ds, uid), vdir,
                  f"variogram_directional_{uid}", dpi)
            written += 1

    print(f"wrote {written} variogram figures to {vdir}")
    print("  these are the fitted covariance SK and OK use; UK refits on detrended")
    print("  residuals, so its covariance is not the one drawn here")
    return written


def _safe_name(name: str) -> str:
    """A model label as a directory name — ``SK (fitted)`` -> ``SK_fitted``."""
    keep = [c if (c.isalnum() or c in "-_") else " " for c in str(name)]
    return "_".join("".join(keep).split()) or "model"


def _field_factories(ds):
    """``name -> (unit_id -> unfitted estimator)`` for the mapped models.

    In increasing order of what they assume, so the figures read as a
    progression: the flat baseline is what "no spatial model" claims, and every
    kriged surface is read against it.
    """
    return {
        "baseline": baseline_factory(ds),
        "SK": kriging_factory(ds, covariance="fit", method="SK"),
        "OK": kriging_factory(ds, covariance="fit", method="OK"),
        "UK": kriging_factory(ds, covariance="fit", method="UK"),
    }


#: Mapped by default.  UK is available via --fields but is off by default: its
#: three drift parameters make it the most likely to extrapolate wildly beyond
#: the data, which is exactly where an unmasked map is least trustworthy.
_DEFAULT_FIELD_MODELS = ("baseline", "SK", "OK")


def _field_maps(ds, out: Path, dpi: int, which="default", res_km=None) -> int:
    """Per-model maps of the predicted field and its uncertainty.

    Fitted in-sample: this is the model's statement about the ground given
    everything known, not a held-out test — the depth profiles and the cross
    plots are where it is scored on data it was not shown.
    """
    if which in (None, "none", "None"):
        return 0
    available = _field_factories(ds)
    if which in ("default", None):
        names = [n for n in _DEFAULT_FIELD_MODELS if n in available]
    elif which == "all":
        names = list(available)
    else:
        names = [n.strip() for n in str(which).split(",") if n.strip()]
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown field model(s) {unknown}; have {sorted(available)}")

    raster = field_raster(ds, **({"res_km": res_km} if res_km else {}))

    # Evaluated once per model, then shared: the colour limits have to span
    # every model before any figure is drawn, and re-predicting to work them
    # out would double the most expensive step in the command.
    fields_by_model = {n: predict_fields(ds, available[n], raster) for n in names}
    limits = shared_colour_limits(ds, fields_by_model)

    written = 0
    for name in names:
        # One directory per model.  Which estimator a surface came from is the
        # first thing a reader needs and the easiest to lose: a folder of
        # similar-looking maps distinguished only by a filename suffix invites
        # exactly the mix-up that makes a comparison worthless.
        fdir = out / "fields" / name
        fdir.mkdir(parents=True, exist_ok=True)
        fig = plots.plot_prediction_map(
            ds, model_name=name, raster=raster,
            fields=fields_by_model[name], limits=limits,
        )
        _save(fig, fdir, "prediction", dpi)
        written += 1

    print(f"\nwrote {written} field figures to {out / 'fields'}/<model>/prediction.png "
          f"({', '.join(names)})")
    print(f"  grid {raster.shape[1]}x{raster.shape[0]}, no presence mask; one row per unit —"
          f"\n  predicted / latent sd / lower 95% / upper 95%")
    print("  colour scales are shared across models per unit, so the same colour means"
          "\n  the same value in every model's figure")
    return written


#: Ordinary kriging: the honest default on real data (the mean is not known),
#: and unlike UK it fits on every unit however few CPTs hold it.
_DEFAULT_PROFILE_MODEL = "OK (fitted)"


def _normalisation(data_dir, args) -> Optional[tuple]:
    """``(profile, n, n_sd)`` for the ``qt`` profiles, or ``None`` if unknown.

    Read from the dataset's own ``prepare_metadata.json``, because the stress
    exponent and unit weights are properties of the supplied column and the
    prepared tables do not carry ``qt`` to re-derive them from.  Nothing is
    assumed when the file is silent: a guessed exponent is a depth-dependent
    error in ``qt`` that reads as stratigraphy, so the suite is skipped and the
    reason printed.  ``--stress-profile`` is the override for a dataset whose
    preparation step predates the metadata block.
    """
    n_sd = float(getattr(args, "stress_exponent_sd", 0.0) or 0.0)
    gradients = getattr(args, "stress_profile", None)
    if gradients:
        return (StressProfile.from_gradients(*gradients),
                float(getattr(args, "stress_exponent", 1.0) or 1.0), n_sd)
    meta_path = Path(data_dir) / "prepare_metadata.json" if data_dir else None
    if meta_path is None or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        profile, n = normalisation_from_metadata(meta)
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
    # A command-line n_sd overrides the recorded one: how much doubt to carry is
    # the caller's judgement, where n itself is a property of the data.
    return profile, n, (n_sd or exponent_uncertainty_from_metadata(meta))


def _profile_models(cv, spec) -> list:
    """Which estimators get a profile suite: one, a comma-separated set, or all.

    Order follows the cv table rather than the string, so ``all`` and an
    explicit list produce directories in the same sequence — increasing in what
    each model assumes about the mean, which is the order the estimators are
    meant to be compared in.
    """
    available = list(pd.unique(cv["model"]))
    if spec in ("all", "ALL"):
        return available
    wanted = {s.strip() for s in str(spec).split(",") if s.strip()}
    missing = sorted(wanted - set(available))
    if missing:
        print(f"\nno model(s) {missing} in the cv table; have {available}", file=sys.stderr)
    return [m for m in available if m in wanted]


def _depth_profiles(ds, cv, out: Path, dpi: int, how="all", model=_DEFAULT_PROFILE_MODEL,
                    data_dir=None, args=None) -> int:
    """Predicted resistance-with-depth against the measured trace, per hole.

    Leave-one-out throughout, so each hole is predicted as if it had never been
    drilled — an in-sample profile would be showing the model its own answer.

    ``model`` may name one estimator, several, or ``all``.  Each gets its own
    directory, because the profiles are the one output where the estimators
    genuinely differ hole by hole rather than in a summary statistic — the
    baseline draws the same band at every location, and how much a kriged one
    narrows it *at this hole* is only visible side by side.

    Two suites into two directories, ``Qtn/`` and ``qt/``.  The same prediction
    twice, on the axis each unit system is legible on: log for ``Qtn``, which is
    the space the model is Gaussian in, and linear MPa for ``qt``, which is how
    a CPT log is read.  ``profile_coverage.csv`` and the calibration figure stay
    at the model level and are *not* duplicated — the transform is monotone, so
    the calibration is one result about both suites, and writing it twice would
    invite the reading that they are two.
    """
    if how in (None, "none", 0, "0"):
        return 0
    models = _profile_models(cv, model)
    if not models:
        print("\nno usable model for depth profiles — skipped", file=sys.stderr)
        return 0
    return sum(
        _one_model_profiles(ds, cv, out, dpi, how, m, data_dir, args) for m in models
    )


def _one_model_profiles(ds, cv, out: Path, dpi: int, how, model, data_dir, args) -> int:
    """One estimator's profile suites.  See :func:`_depth_profiles`."""
    readings = reading_predictions(ds, cv, model=model)
    coverage = profile_coverage(readings)

    # Everything from one estimator lives under that estimator's name.  The
    # profiles carried no model in their filenames at all, so a directory of
    # 194 of them was silently the output of whichever model happened to be the
    # default — unrecoverable once the run had scrolled past.
    pdir = out / "profiles" / _safe_name(model)
    pdir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(pdir / "profile_coverage.csv", float_format="%.6g")

    print(f"\ndepth profiles ({model}, leave-one-out) — realised coverage of the "
          f"single-reading band:")
    print(coverage[["n_predicted", "within_sd", "coverage_read", "mssr_read"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))

    _save(plots.plot_profile_calibration(ds, readings), pdir, "profile_calibration", dpi)

    wanted = getattr(args, "profile_units", "both") or "both"
    suites = {"Qtn": readings} if wanted in ("both", "Qtn") else {}
    if wanted in ("both", "qt"):
        norm = _normalisation(data_dir, args)
        if norm is None:
            print("qt profiles skipped — no normalisation recorded for this dataset. "
                  "Add one in the preparation step, or pass "
                  "--stress-profile GAMMA_SAT GAMMA_W [--stress-exponent N].",
                  file=sys.stderr)
        else:
            profile, n, n_sd = norm
            suites["qt"] = qt_readings(readings, profile=profile, n=n, n_sd=n_sd)
            print(f"qt profiles use {profile.describe()}, "
                  f"{suites['qt'].attrs['exponent']}")
            if not suites["qt"].attrs["coverage_transfers"]:
                print("  bands include exponent uncertainty, so the realised coverage "
                      "above does not describe them", file=sys.stderr)

    cpt_ids = list(pd.unique(readings["cpt_id"]))
    if how not in ("all", "ALL"):
        cpt_ids = cpt_ids[: int(how)]

    written = 0
    for units, table in suites.items():
        udir = pdir / units
        udir.mkdir(parents=True, exist_ok=True)
        for cid in cpt_ids:
            _save(plots.plot_depth_profile(ds, table, cid, units=units),
                  udir, f"profile_{cid}", dpi)
        written += len(cpt_ids)
        print(f"wrote {len(cpt_ids)} {units} depth profiles to {udir}")
    return written


def _anisotropic_models(ds, out: Path, args) -> list:
    """Ordinary kriging with a *fitted axis*, where the axis survives the gates.

    Off by default because the gate is a simulated null and simulated nulls are
    minutes, not seconds.  The decision table is written whether or not any unit
    passes: "every unit stayed isotropic, and here is why each one did" is the
    result on a site with no fabric, and a run that produced it should say so
    rather than look like a run that never asked.
    """
    n_sim = getattr(args, "anisotropy", None)
    if not n_sim:
        return []

    from .models.anisotropy import anisotropic_kriging_factory, decide_anisotropy, decision_table

    print(f"\nfitting anisotropy ({n_sim} null simulations per unit) — this is the slow part")
    decisions = decide_anisotropy(ds, n_sim=int(n_sim),
                                  seed=getattr(args, "anisotropy_seed", None))
    table = decision_table(decisions)
    table.to_csv(out / "anisotropy.csv", float_format="%.6g")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    used = int(table["anisotropic"].sum()) if len(table) else 0
    print(f"{used} of {len(table)} units kriged with a fitted axis; the rest stay isotropic")
    if not used:
        # Worth saying plainly: the estimator is then OK by another name, and a
        # reader comparing the two figures should not go looking for the
        # difference that is not there.
        print("  -> 'OK (aniso)' is identical to 'OK (fitted)' on this dataset")
    return [("OK (aniso)", anisotropic_kriging_factory(ds, method="OK", decisions=decisions))]


def _cross_plots(ds, out: Path, dpi: int, args=None) -> int:
    """Leave-one-out predictions, scores and the predicted-vs-true cross plots.

    Predictions are leave-one-out, not in-sample: the baseline's fitted value
    *is* the mean of the points it would otherwise be plotted against, so an
    in-sample cross plot would flatter it for the one reason that cannot
    generalise.
    """
    vario = _fitted_variograms(ds)
    vario.to_csv(out / "variogram_fits.csv", float_format="%.6g")
    print("\nfitted variograms (isotropic, Matern 2.5):")
    print(vario.to_string(float_format=lambda v: f"{v:.3f}"))
    _variogram_figures(ds, out, dpi)

    # In increasing order of what they assume about the mean: known constant,
    # unknown constant, unknown linear drift.  All three run on real data.
    models = [
        ("baseline", baseline_factory(ds)),
        ("SK (fitted)", kriging_factory(ds, covariance="fit", method="SK")),
        ("OK (fitted)", kriging_factory(ds, covariance="fit", method="OK")),
        ("UK (fitted)", kriging_factory(ds, covariance="fit", method="UK")),
    ]
    if getattr(ds, "config", None) is not None:
        # Kriging handed the generating covariance: separates "is kriging
        # implemented right" from "can a variogram be fitted from 30 CPTs".
        from .synthetic.truth import truth_kriging_factory

        models.append(("SK (truth cov)", truth_kriging_factory(ds)))

    models += _anisotropic_models(ds, out, args)

    cv = pd.concat([loo_by_unit(ds, f, name) for name, f in models], ignore_index=True)
    cv.to_csv(out / "cv_predictions.csv", index=False, float_format="%.6g")

    _field_maps(ds, out, dpi,
                getattr(args, "fields", "default"),
                getattr(args, "field_res_km", None))

    _depth_profiles(ds, cv, out, dpi,
                    getattr(args, "profiles", "all"),
                    getattr(args, "profile_model", _DEFAULT_PROFILE_MODEL),
                    data_dir=getattr(args, "data", None), args=args)

    written = []
    for target, label, note in (
        ("observed", "the held-out observation", None),
        ("latent", "the true latent field (synthetic only)", "truth_points.csv"),
    ):
        if target == "latent" and not ("latent" in cv.columns and cv["latent"].notna().any()):
            print(f"\nno {note} — latent-target cross plot skipped")
            continue
        scores = score_by_unit(cv, target=target)
        scores.to_csv(out / f"cv_scores_{target}.csv", float_format="%.6g")
        print(f"\nleave-one-out vs. {label}:")
        print(
            scores.unstack(0)[["rmse", "coverage95", "mssr"]]
            .to_string(float_format=lambda v: f"{v:.3f}")
        )
        written.append(_save(plots.plot_prediction_vs_truth(ds, cv, target=target),
                             out, f"cross_plot_{target}", dpi))

    print(f"\nwrote {len(written)} cross plots to {out}")
    return 0


def cmd_run(args) -> int:
    out = Path(args.out)
    args.out = out / "data"
    rc = cmd_generate(args)
    if rc:
        return rc
    args.data = out / "data"
    args.out = out / "figures"
    rc = cmd_plot(args)
    if rc:
        return rc
    args.out = out / "models"
    return cmd_model(args)


def _add_output_args(parser) -> None:
    """Field-map and depth-profile options, shared by ``model`` and ``run``."""
    parser.add_argument(
        "--profiles", default="all", metavar="all|none|N",
        help="how many per-hole depth profiles to write (default: all — that is "
             "194 figures on IJmuiden, so 'none' or a count is the escape hatch)",
    )
    parser.add_argument(
        "--profile-model", default=_DEFAULT_PROFILE_MODEL, metavar="NAME|all|A,B",
        help="which cross-validated model(s) the profiles show, each into its own "
             f"directory (default: {_DEFAULT_PROFILE_MODEL}; 'all' does every "
             "estimator in the cv table)",
    )
    parser.add_argument(
        "--profile-units", default="both", choices=["both", "Qtn", "qt"],
        help="which profile suites to write (default: both, into profiles/<model>/Qtn "
             "and profiles/<model>/qt)",
    )
    parser.add_argument(
        "--stress-profile", type=float, nargs=2, default=None,
        metavar=("GAMMA_SAT", "GAMMA_W"),
        help="unit weights in kN/m3 for the qt conversion, overriding the dataset's "
             "prepare_metadata.json (which is where they belong)",
    )
    parser.add_argument(
        "--stress-exponent", type=float, default=None, metavar="N",
        help="stress exponent for the qt conversion; only read alongside "
             "--stress-profile (default: 1.0, i.e. Robertson's Qt)",
    )
    parser.add_argument(
        "--stress-exponent-sd", type=float, default=None, metavar="SD",
        help="uncertainty on the stress exponent, widening the qt bands (default: "
             "whatever the dataset records, else 0 — i.e. n treated as known)",
    )
    parser.add_argument(
        "--anisotropy", type=int, default=0, metavar="N_SIM",
        help="also fit an anisotropic OK, gating each unit's axis on N_SIM simulated "
             "isotropic nulls (0 = off; 30-60 is a usable gate and takes minutes)",
    )
    parser.add_argument(
        "--anisotropy-seed", type=int, default=None,
        help="seed for the anisotropy null simulations, for a reproducible gate",
    )
    parser.add_argument(
        "--fields", default="default", metavar="default|all|none|A,B",
        help="which models get field maps (default: "
             f"{','.join(_DEFAULT_FIELD_MODELS)}; 'all' adds UK)",
    )
    parser.add_argument(
        "--field-res-km", type=float, default=None,
        help="grid spacing for the field maps in km (default: the synthetic "
             "raster, or 0.15 km on real data)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpt-geostat", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    # -- synthetic-only: needs a generator config -----------------------------
    synth = sub.add_parser(
        "synth", help="synthetic data generation (needs a generator config)"
    )
    synth_sub = synth.add_subparsers(dest="synth_command", required=True)

    gen = synth_sub.add_parser("generate", help="generate a synthetic dataset")
    gen.add_argument("--config", default="config.yaml")
    gen.add_argument("--out", default="data")
    gen.add_argument("--seed", type=int, default=None, help="override the config seed")
    gen.set_defaults(func=cmd_generate)

    run = synth_sub.add_parser("run", help="generate, plot then model, into one directory")
    run.add_argument("--config", default="config.yaml")
    run.add_argument("--out", default="run")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--dpi", type=int, default=130)
    _add_output_args(run)
    run.set_defaults(func=cmd_run)

    # -- contract-driven: any dataset directory -------------------------------
    plot = sub.add_parser("plot", help="plot maps and diagnostics for any dataset")
    plot.add_argument("--data", default="data")
    plot.add_argument("--out", default="figures")
    plot.add_argument("--dpi", type=int, default=130)
    plot.set_defaults(func=cmd_plot)

    model = sub.add_parser("model", help="fit the per-unit baseline and cross-validate")
    model.add_argument("--data", default="data")
    model.add_argument("--out", default="models")
    model.add_argument("--dpi", type=int, default=130)
    _add_output_args(model)
    model.set_defaults(func=cmd_model)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
