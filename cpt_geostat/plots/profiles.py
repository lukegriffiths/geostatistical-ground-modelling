"""Predicted Qtn with depth, against the trace that was actually measured.

The most direct check the project has: each hole is predicted **leave-one-out**,
as if it had never been drilled, and then laid over its own readings.

Two nested bands, because they answer different questions and the wide one is
the only one comparable with a raw trace:

* the **unit-mean** band — what a new depth-average here would be;
* the **reading** band — where an individual 2 cm reading should fall, which
  adds the within-unit depth scatter.

On IJmuiden the second is much the wider, because ``within_sd`` runs 0.45-0.56
against a between-hole ``log_Q_sd`` of 0.31-0.71: the texture inside one hole is
as large as the variation between holes.  Drawing only the narrow band against a
raw trace would claim a precision the model does not have; drawing only the wide
one would hide how much of it a spatial model can actually reduce.

Two unit systems, drawn as two suites rather than two panels
------------------------------------------------------------

The same figure is produced in ``Qtn`` and in ``qt``, into separate directories.
They are not redundant, because the conventions that make each readable are
opposite ones:

``Qtn``
    **Log axis.**  The model is Gaussian in ``log Qtn``, so the bands are
    symmetric there and asymmetric in ``Qtn``; a log axis restores the symmetry
    and stops the lower band looking implausibly tight.  This is the figure for
    judging the *model*.
``qt``
    **Linear axis, MPa.**  How every CPT log in the industry is drawn, and the
    units a foundation is specified in.  The asymmetry of the band is left
    visible here rather than transformed away, because on this axis it is the
    honest shape of the answer.  The cost is that the wide single-reading band
    can run off the right-hand edge; the figure says so when it does, and the
    ``Qtn`` version is the one that always shows the whole band.

Everything else — the bands, the medians, the gaps, the realised coverage — is
the same object seen twice.  The coverage annotations are computed in log space
for both, which is not an approximation: the transform is monotone, so a reading
inside the band in ``log Qtn`` is inside it in ``qt``, exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.stats import norm

from ..models.profile import profile_coverage
from .style import figure_suptitle, unit_colours, unit_label

_MEAN_BAND = "#2A9D8F"
_READ_BAND = "#E9C46A"
_TRACE = "#333333"


@dataclass(frozen=True)
class ProfileUnits:
    """How one unit system is drawn, and which columns hold it.

    The column names differ only by prefix — ``Qtn``, ``Qtn_median``,
    ``Qtn_lo_read`` against ``qt``, ``qt_median``, ``qt_lo_read`` — so a prefix
    is the whole mapping and the figure code stays single-sourced.  What is
    genuinely per-system is the axis: scale, label and display factor.
    """

    prefix: str
    label: str
    scale: str
    factor: float = 1.0

    def col(self, suffix: str = "") -> str:
        return f"{self.prefix}_{suffix}" if suffix else self.prefix

    def required(self) -> list:
        return [self.col()] + [
            self.col(f"{side}_{band}") for band in ("mean", "read") for side in ("lo", "hi")
        ] + [self.col("median")]


#: The suites ``cli model`` writes, keyed by the directory each lands in.
PROFILE_UNITS = {
    "Qtn": ProfileUnits(prefix="Qtn", label=r"$Q_{tn}$  (log scale)", scale="log"),
    "qt": ProfileUnits(prefix="qt", label=r"$q_t$  (MPa)", scale="linear", factor=1e-3),
}


def _unit_runs(g: pd.DataFrame):
    """Contiguous ``(z_top, z_bot, unit_id)`` runs down one hole.

    Taken from the readings themselves rather than from ``layers``: a unit that
    re-enters lower down has one collapsed layer row spanning first-top to
    last-base, which would paint its band straight through the intervening
    material.  The readings know which unit each depth actually is.
    """
    unit = g["unit_id"].to_numpy()
    z = g["z"].to_numpy(dtype=float)
    if not len(unit):
        return []
    breaks = np.flatnonzero(unit[1:] != unit[:-1]) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(unit)]])
    runs = []
    for s, e in zip(starts, ends):
        # Boundaries at the midpoint of the gap, so consecutive runs meet.
        z_top = z[s] if s == 0 else 0.5 * (z[s - 1] + z[s])
        z_bot = z[e - 1] if e == len(unit) else 0.5 * (z[e - 1] + z[e])
        runs.append((float(z_top), float(z_bot), unit[s], slice(s, e)))
    return runs


def _units(units: str) -> ProfileUnits:
    try:
        return PROFILE_UNITS[units]
    except KeyError:
        raise ValueError(
            f"unknown units {units!r}; have {sorted(PROFILE_UNITS)}"
        ) from None


def _run_depths(sub: pd.DataFrame, z_top: float, z_bot: float) -> np.ndarray:
    """Reading depths, extended to the run's own boundaries.

    The first reading sits up to half a sample interval below the contact.
    Without the extension the band would stop short of it and leave a hairline
    of unshaded ground at every boundary.
    """
    z = sub["z"].to_numpy(dtype=float)
    return np.concatenate([[z_top], z, [z_bot]])


def _run_edge(sub: pd.DataFrame, u: ProfileUnits, suffix: str) -> np.ndarray:
    """One band edge down a run, in display units, matching :func:`_run_depths`.

    The boundary values repeat the nearest reading rather than extrapolating.
    The edge is linear in depth, so extrapolating would be exact — and would buy
    a correction of at most half a sample interval, on a quantity plotted at
    figure resolution.
    """
    v = sub[u.col(suffix)].to_numpy(dtype=float) * u.factor
    return np.concatenate([v[:1], v, v[-1:]])


def plot_depth_profile(
    ds,
    readings: pd.DataFrame,
    cpt_id: str,
    level: Optional[float] = None,
    show_coverage: bool = True,
    units: str = "Qtn",
) -> Figure:
    """One hole: the measured trace, the prediction, and both bands.

    ``readings`` is a :func:`cpt_geostat.models.profile.reading_predictions` table, or
    — for ``units="qt"`` — that table after
    :func:`cpt_geostat.models.profile.qt_readings`, which is where the stress profile
    and the stress exponent enter.  Runs where the unit had no fittable fold are
    left as **gaps**: the trace is still drawn, but no band is, because a band
    there would be an invention.
    """
    u = _units(units)
    g = readings[readings["cpt_id"] == cpt_id].sort_values("z")
    if not len(g):
        raise ValueError(f"no readings for cpt_id={cpt_id!r}")
    missing = [c for c in u.required() if c not in readings.columns]
    if missing:
        raise ValueError(
            f"readings table has no {units} columns (missing {missing}); "
            f"pass it through cpt_geostat.models.profile.qt_readings first"
        )
    level = float(level if level is not None else readings.attrs.get("level", 0.95))

    fig, ax = plt.subplots(figsize=(7.6, 9.4), constrained_layout=True)
    colours = unit_colours(ds)
    runs = _unit_runs(g)

    n_gap = 0
    for z_top, z_bot, uid, sl in runs:
        sub = g.iloc[sl]
        ax.axhline(z_top, color="0.75", lw=0.6, ls="-", zorder=1)
        if sub["pred"].isna().all():
            # Hatched, not shaded: a plain grey block is indistinguishable from
            # a unit whose own colour happens to be grey.
            n_gap += 1
            ax.axhspan(z_top, z_bot, facecolor="none", edgecolor="0.6", lw=0.0,
                       hatch="///", alpha=0.5, zorder=1)
            ax.annotate(" not predicted", xy=(0.02, 0.5 * (z_top + z_bot)),
                        xycoords=("axes fraction", "data"), fontsize=6.5,
                        va="center", color="0.45")
            continue

        # In qt the band edges vary down the run — the stress profile puts the
        # depth trend back — so they are drawn per reading rather than as the
        # one value a normalised unit prediction has.  In Qtn every reading in
        # the run carries the same number and this collapses to the old
        # two-point band.
        depths = _run_depths(sub, z_top, z_bot)
        for band, colour, alpha, zorder in (
            ("read", _READ_BAND, 0.45, 2), ("mean", _MEAN_BAND, 0.55, 3),
        ):
            ax.fill_betweenx(depths, _run_edge(sub, u, f"lo_{band}"),
                             _run_edge(sub, u, f"hi_{band}"),
                             color=colour, alpha=alpha, lw=0, zorder=zorder)
        ax.plot(_run_edge(sub, u, "median"), depths, color="k", lw=1.8, zorder=5)

    # The measured trace last, so it reads on top of every band.
    ax.plot(g[u.col()] * u.factor, g["z"], color=_TRACE, lw=0.45, zorder=6)

    # Unit labels and per-unit realised coverage down the right margin.
    for z_top, z_bot, uid, sl in runs:
        sub = g.iloc[sl]
        mid = 0.5 * (z_top + z_bot)
        ax.axhspan(z_top, z_bot, color=colours.get(uid, "#cccccc"), alpha=0.10,
                   lw=0, zorder=0)
        label = str(uid)
        if show_coverage and sub["pred"].notna().any():
            ok = sub["pred"].notna()
            inside = ((sub.loc[ok, "log_Q"] >= sub.loc[ok, "log_lo_read"])
                      & (sub.loc[ok, "log_Q"] <= sub.loc[ok, "log_hi_read"]))
            label = f"{uid}  {inside.mean():.0%}"
        ax.annotate(label, xy=(0.985, mid), xycoords=("axes fraction", "data"),
                    fontsize=7, ha="right", va="center", color="0.2",
                    bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.75))

    ok = g["pred"].notna()
    covered = float("nan")
    if ok.any():
        inside = ((g.loc[ok, "log_Q"] >= g.loc[ok, "log_lo_read"])
                  & (g.loc[ok, "log_Q"] <= g.loc[ok, "log_hi_read"]))
        covered = float(inside.mean())

    handles = [
        plt.Line2D([], [], color=_TRACE, lw=1.0, label="measured trace"),
        plt.Line2D([], [], color="k", lw=1.8, label="predicted median"),
        plt.Rectangle((0, 0), 1, 1, fc=_MEAN_BAND, alpha=0.55,
                      label=f"unit mean {level:.0%}"),
        plt.Rectangle((0, 0), 1, 1, fc=_READ_BAND, alpha=0.45,
                      label=f"single reading {level:.0%}"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left", framealpha=0.92)

    clipped: list = []
    if u.scale == "log":
        ax.set_xscale("log")
        # Plain numbers, not "2 x 10^0".  Qtn spans well under two decades, so
        # the default log formatter labels almost nothing legible.
        ax.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=14)
        )
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        # Framed on the measured trace and the predicted median — the two things
        # a reader compares.  Letting a band set the limit squeezes the trace
        # into the left quarter of the axis, because in qt the bands are several
        # times the width of the data: a log_Q_sd of 0.4 is a factor of 2.2 each
        # way, which on a linear axis is most of the page.  The bands are allowed
        # to overflow and the caption says which ones did — that they do not fit
        # is itself the finding, and the Qtn figure is where the whole width is
        # legible.
        upper = np.nanmax([_safe_max(g[u.col()]), _safe_max(g[u.col("median")])]) * u.factor
        if np.isfinite(upper) and upper > 0:
            upper *= 1.05
            ax.set_xlim(0.0, upper)
            clipped = [
                label for band, label in (("mean", "mean"), ("read", "reading"))
                if _safe_max(g[u.col(f"hi_{band}")]) * u.factor > upper
            ]
    ax.set_xlabel(u.label)
    ax.set_ylabel("depth below seabed (m)")
    # Framed on the sampled interval, not forced to start at the seabed: many
    # IJmuiden boreholes only log below 30 m, and anchoring at zero spends half
    # the figure on ground the hole never saw.  Clipped at zero so a hole that
    # does start shallow still shows the seabed.
    z0, z1 = float(g["z"].min()), float(g["z"].max())
    pad = max(0.02 * (z1 - z0), 0.1)
    ax.set_ylim(z1 + pad, max(z0 - pad, 0.0))
    ax.grid(ls=":", alpha=0.35, which="both")

    note = f"{covered:.1%} of readings inside the {level:.0%} band" if np.isfinite(covered) \
        else "no predictions available"
    if n_gap:
        note += f"   ·   {n_gap} run(s) not predicted"
    # A third line for what is specific to this unit system: the stress profile
    # is an input to the qt figure and not to the Qtn one, so it belongs on the
    # figure rather than in a run log, and clipping only ever happens on the
    # linear axis.  Kept off the coverage line, which is already long enough to
    # run off the page.
    footnotes = []
    if readings.attrs.get("stress_profile"):
        # `exponent` is a prepared label because n may be one number or one per
        # unit, with or without a spread — formatting it here would need the
        # figure to know all three shapes.
        footnotes.append(
            f"{readings.attrs['stress_profile']}, {readings.attrs.get('exponent', 'n = 1')}"
        )
    if readings.attrs.get("coverage_transfers") is False:
        # The band was widened for not knowing the exponent, but the trace it is
        # drawn against was de-normalised with the central n — so the coverage
        # printed above is not the band's nominal level and must not be read as
        # a calibration result.
        footnotes.append("band includes exponent uncertainty — coverage not comparable")
    if clipped:
        footnotes.append(f"clipped: {'+'.join(clipped)} band")
    if footnotes:
        # One per line rather than joined: the exponent line alone is ~50
        # characters and the caveat another 45, which together run off a 7.6 in
        # figure at any font size worth reading.
        note += "\n" + "\n".join(footnotes)
    figure_suptitle(
        fig,
        f"Depth profile ({units}) — {cpt_id}   ({len(g)} readings, "
        f"{readings.attrs.get('model', 'model')}, leave-one-out)\n{note}",
    )
    return fig


def _safe_max(s: pd.Series) -> float:
    """``max`` of a column that may be entirely nan, without the warning."""
    v = s.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def plot_profile_calibration(ds, readings: pd.DataFrame, level: Optional[float] = None) -> Figure:
    """Is the reading band the right width?  Measured, per unit and overall.

    The quantitative half of the profile comparison.  The band is built from a
    variance decomposition that knowingly double-counts the depth-averaging
    error, so whether it is right has to be observed rather than asserted — and
    a unit that comes out narrow is a finding about the estimator, not a reason
    to widen the band.
    """
    level = float(level if level is not None else readings.attrs.get("level", 0.95))
    cov = profile_coverage(readings, level=level)
    cov = cov[cov["n_predicted"] > 0]

    fig, (ax, ax_h) = plt.subplots(
        1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [3, 2]},
        constrained_layout=True,
    )

    # -- realised coverage per unit -------------------------------------------
    order = cov.index.tolist()
    ypos = np.arange(len(order))
    colours = unit_colours(ds, order)
    ax.barh(ypos + 0.19, cov["coverage_read"], height=0.36,
            color=[colours.get(u, "#888") for u in order], edgecolor="k", lw=0.4,
            label="single-reading band")
    ax.barh(ypos - 0.19, cov["coverage_mean"], height=0.36,
            color="0.72", edgecolor="k", lw=0.4, label="unit-mean band")
    ax.axvline(level, color="#B00020", ls="--", lw=1.5, zorder=4,
               label=f"nominal {level:.0%}")
    ax.set_yticks(ypos)
    ax.set_yticklabels([unit_label(ds, u) for u in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of real readings inside the band")
    ax.grid(axis="x", ls=":", alpha=0.4)
    # Opaque: every bar starts at x=0, so there is no empty corner to sit in.
    ax.legend(fontsize=7.5, loc="lower right", framealpha=1.0)
    ax.set_title(
        "The unit-mean band is *expected* to fall short here: it is the interval\n"
        "for a depth-average, scored against individual readings.",
        fontsize=8, loc="left", color="0.35",
    )

    # -- standardised residuals ----------------------------------------------
    z = readings.loc[readings["pred"].notna(), "resid_z"].to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    if z.size:
        ax_h.hist(z, bins=80, density=True, color="#264653", alpha=0.85,
                  edgecolor="none")
        grid = np.linspace(-5, 5, 400)
        ax_h.plot(grid, norm.pdf(grid), color="#B00020", lw=1.8,
                  label="N(0, 1) — a correctly sized band")
        ax_h.set_xlim(-5, 5)
        ax_h.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
        ax_h.annotate(
            f"mssr {np.mean(z**2):.2f}\nsd   {np.std(z):.2f}\nbias {np.mean(z):+.2f}",
            xy=(0.02, 0.97), xycoords="axes fraction", fontsize=7.5, va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="0.7", alpha=0.92),
        )
    ax_h.set_xlabel("(measured − predicted) / sd of a single reading\n"
                    "mssr 1.00 and sd 1.00 mean the band is exactly the right width")
    ax_h.set_ylabel("density")
    ax_h.grid(ls=":", alpha=0.4)

    figure_suptitle(
        fig,
        f"Profile calibration — {readings.attrs.get('model', 'model')}, leave-one-out, "
        f"{int(readings['pred'].notna().sum()):,} readings",
    )
    return fig
