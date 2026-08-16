"""Variogram figures — the fit, and what it is safe to say about it.

``models/variogram.py`` computes a careful identifiability verdict for every
unit — ``at_range_bound``, ``nugget_floored``, ``fit_failed``,
``structured_fraction``, ``why_not_resolved()`` — and until now all of it
arrived as a boolean and a sentence in a csv cell.  Each of those is really a
statement about the *shape* of a curve, so each one is drawn here: the reader
should be able to see why a range was refused, not take it on trust.

``VariogramFit`` has always retained ``lags``, ``gamma`` and ``counts`` for
exactly this purpose.  This module is what finally reads them.

Both figures follow the Part B contract: take a ``Dataset``, return a Figure,
touch no files, and run unchanged on real data.  Truth is overlaid only when a
config happens to be present, the same way
:func:`~cpt_geostat.plots.diagnostics.plot_trend_check` already does it, so nothing
here needs the synthetic package.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..covariance import build_model
from ..geometry import azimuth_unit_vector
from ..models.variogram import (
    DEFAULT_SECTOR_TOL_DEG,
    DEFAULT_SECTORS,
    MIN_SECTOR_PAIRS,
    directional_variogram,
    empirical_variogram,
    fit_unit_variogram,
    unit_block,
)
from .style import add_anisotropy_glyph, figure_suptitle, unit_colours, unit_label

_TRUTH = "#B00020"
_BOUND = "#E76F51"
_SECTOR_COLOURS = ["#264653", "#2A9D8F", "#E9C46A", "#6A4C93", "#F4A261", "#1D3557"]


def _require_fit(ds, unit_id: str):
    fit = fit_unit_variogram(ds, unit_id)
    if fit is None:
        n = len(unit_block(ds, unit_id))
        raise ValueError(
            f"{unit_id}: {n} CPT(s) is too few to fit a variogram — nothing to plot. "
            f"The unit baseline is the estimator here."
        )
    return fit


def _true_grf(ds, unit_id: str):
    """``(GrfConfig, nugget)`` the generator used, or ``None`` on real data."""
    cfg = getattr(ds, "config", None)
    if cfg is None or unit_id not in getattr(cfg, "units", {}):
        return None
    prop = cfg.units[unit_id].property
    return prop.grf, prop.nugget


def _truth_curves(grf, nugget, h):
    """True variogram along the major axis, and along the minor if anisotropic.

    An anisotropic model has no single variogram: the omnidirectional empirical
    estimate is a mixture of every bearing, so it should sit *between* the two
    axis curves rather than on either.  Drawing both as a band says that, where
    one curve would invite the reader to call the difference a fitting error.
    """
    major = build_model(replace(grf, aniso_ratio=1.0), nugget=nugget)
    if grf.aniso_ratio == 1.0:
        return major.variogram(h), None
    minor = build_model(
        replace(grf, aniso_ratio=1.0, range_km=grf.range_km / grf.aniso_ratio), nugget=nugget
    )
    return major.variogram(h), minor.variogram(h)


# --------------------------------------------------------------------------- #
# the isotropic fit
# --------------------------------------------------------------------------- #


def plot_variogram(ds, unit_id: str, model_curve_points: int = 300) -> Figure:
    """One unit's empirical variogram, its fitted model, and the verdict.

    Everything the fit is constrained by is drawn, because each constraint is
    the answer to a question a reader would otherwise have to trust:

    * the **sample variance** as a horizontal line — ``fix_sill`` pins
      ``sill + nugget`` to it, so the model cannot go above it and the split
      below it is the only free choice;
    * the **nugget** as an intercept, with the first bin highlighted in the
      count strip: those pairs are the only thing that identifies it;
    * the **practical range**, and the ``max_lag`` bound beyond which a range is
      not identifiable at all — so "the range ran to the bound" is visible as a
      curve pressed against a wall rather than asserted in a csv;
    * ``why_not_resolved()`` printed on the figure when the fit is refused.

    On synthetic data the generating covariance is overlaid, which turns the
    figure into a direct recovery check.
    """
    fit = _require_fit(ds, unit_id)
    colour = unit_colours(ds, [unit_id])[unit_id]
    n_cpt = len(unit_block(ds, unit_id))

    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(9.0, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
        constrained_layout=True,
    )

    h_max = max(float(fit.max_lag_km), float(fit.lags.max()) if fit.lags.size else 1.0)
    h = np.linspace(1e-6, h_max * 1.08, model_curve_points)

    # -- the constraint the fit works inside ---------------------------------
    # What each reference line *means* goes in the legend, not in floating text
    # on the axes: every free-floating label collided with either the data, the
    # stats box or the legend on one unit or another.
    if np.isfinite(fit.sample_var):
        ax.axhline(fit.sample_var, color="0.35", ls="--", lw=1.2, zorder=1,
                   label=f"sample variance {fit.sample_var:.3f} — sill+nugget pinned")

    # -- beyond half the maximum separation, a range is not identifiable ------
    if h[-1] > fit.max_lag_km:
        ax.axvspan(fit.max_lag_km, h[-1], color="0.85", alpha=0.55, lw=0, zorder=0)
        mid = 0.5 * (fit.max_lag_km + h[-1])
        ax.annotate(
            "beyond half the max separation",
            xy=(mid, 0.5), xycoords=("data", "axes fraction"),
            fontsize=7, rotation=90, va="center", ha="center", color="0.4",
        )

    # -- truth, where there is any --------------------------------------------
    truth = _true_grf(ds, unit_id)
    if truth is not None:
        g_major, g_minor = _truth_curves(truth[0], truth[1], h)
        if g_minor is None:
            ax.plot(h, g_major, color=_TRUTH, ls="--", lw=1.5, zorder=3,
                    label="true covariance")
        else:
            ax.fill_between(h, g_minor, g_major, color=_TRUTH, alpha=0.13, lw=0, zorder=2)
            ax.plot(h, g_major, color=_TRUTH, ls="--", lw=1.4, zorder=3,
                    label=f"true, major axis ({truth[0].range_km:g} km)")
            ax.plot(h, g_minor, color=_TRUTH, ls=":", lw=1.4, zorder=3,
                    label=f"true, minor axis ({truth[0].range_km / truth[0].aniso_ratio:.2g} km)")

    # -- the fit --------------------------------------------------------------
    ax.plot(h, fit.model.variogram(h), color="k", lw=1.8, zorder=4, label="fitted model")
    if fit.lags.size:
        # Marker area tracks pair count: a bin standing on six pairs should not
        # look as solid as one standing on six hundred.
        sizes = 18.0 + 90.0 * np.sqrt(fit.counts / max(fit.counts.max(), 1))
        ax.scatter(fit.lags, fit.gamma, s=sizes, color=colour, edgecolor="k", lw=0.4,
                   zorder=5, label="empirical")

    ax.axhline(fit.nugget, color="0.45", ls=":", lw=1.4, zorder=1,
               label=f"nugget {fit.nugget:.4f}")

    if np.isfinite(fit.range_km) and fit.range_km <= h[-1]:
        style = dict(color=_BOUND, lw=1.6) if fit.at_range_bound else dict(color="0.2", lw=1.2)
        ax.axvline(fit.range_km, ls="-." if fit.at_range_bound else "-", zorder=3, **style)
        # Horizontal along the bottom rather than rotated up the line: rotated,
        # it ran into whichever corner the legend was in.
        ax.annotate(
            f"range {fit.range_km:.2f} km" + (" — AT BOUND" if fit.at_range_bound else "") + " ",
            xy=(fit.range_km, 0.015), xycoords=("data", "axes fraction"),
            fontsize=7.5, va="bottom", ha="right", zorder=8,
            color=_BOUND if fit.at_range_bound else "0.2",
            bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.8),
        )

    _annotate_verdict(ax, fit, n_cpt)

    ax.set_ylabel(r"$\gamma(h)$  —  log $Q_{tn}$ variance")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(0.0, h[-1])
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    _count_strip(ax_n, fit)
    figure_suptitle(
        fig,
        f"Variogram — {unit_label(ds, unit_id)}   ({n_cpt} CPTs, isotropic, "
        f"{getattr(fit.model, 'name', 'model')})",
    )
    return fig


def _annotate_verdict(ax, fit, n_cpt: int) -> None:
    """The stats box, and the refusal notice when the fit is not resolved."""
    lines = [
        f"n CPT      {n_cpt}",
        f"sill       {fit.sill:.4f}",
        f"nugget     {fit.nugget:.4f}",
        f"range      {fit.range_km:.2f} km" if np.isfinite(fit.range_km) else "range      —",
        f"structured {fit.structured_fraction:.0%}",
        f"pairs      {fit.n_pairs} ({fit.n_short_pairs} in bin 1)",
    ]
    if fit.nugget_floored:
        lines.append("nugget floored at gamma(h_min)")
    ax.text(
        0.015, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
        va="top", ha="left", family="monospace", zorder=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="0.7", alpha=0.95),
    )

    if fit.resolved:
        return
    # Refused fits get the reason on the face of the figure.  A reader who only
    # sees a confident-looking curve has no way to know it was rejected.
    ax.patch.set_facecolor("#FFF4F4")
    ax.text(
        0.5, 0.5, f"NOT RESOLVED\n{fit.why_not_resolved()}",
        transform=ax.transAxes, fontsize=9.5, va="center", ha="center",
        color=_TRUTH, weight="bold", zorder=6, wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", fc="w", ec=_TRUTH, alpha=0.85),
    )


def _count_strip(ax, fit) -> None:
    """Pairs per bin.  The nugget lives or dies on the leftmost bar."""
    if not fit.lags.size:
        ax.set_visible(False)
        return
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    widths = np.diff(np.concatenate([[0.0], fit.lags]))
    # The short-lag bins are the narrowest *and* the most important, so a bar
    # drawn at true width disappears at exactly the moment it matters.
    widths = np.maximum(widths * 0.85, float(fit.lags.max()) * 0.012)
    colours = ["#E76F51"] + ["0.55"] * (len(fit.lags) - 1)
    ax.bar(fit.lags, fit.counts, width=widths, color=colours, edgecolor="k", lw=0.3)

    ax.set_ylabel("pairs", fontsize=8)
    ax.set_xlabel("separation h (km)")
    if fit.counts.max() / max(fit.counts.min(), 1) > 12:
        # Counts span orders of magnitude on a clustered layout; log keeps the
        # six-pair first bin visible next to a 900-pair one.  The default log
        # minor labels ("3x10^0") are noise at this size, so they go.
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", ls=":", alpha=0.4)
    # Anchored to its own bar — the shortest one — rather than to a corner of
    # the axes, so it cannot land on top of a taller bar on some other unit.
    ax.annotate(
        f" bin 1: {fit.n_short_pairs} pair(s) — all that identifies the nugget",
        xy=(fit.lags[0], fit.counts[0]), fontsize=7, va="bottom", ha="left",
        color="#E76F51", zorder=6,
        bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.85),
    )


# --------------------------------------------------------------------------- #
# directional
# --------------------------------------------------------------------------- #


def plot_directional_variogram(
    ds,
    unit_id: str,
    azimuths: Sequence[float] = DEFAULT_SECTORS,
    tol_deg: float = DEFAULT_SECTOR_TOL_DEG,
    bandwidth_km: Optional[float] = None,
) -> Figure:
    """Empirical variograms per azimuth sector — the anisotropy diagnostic.

    An isotropic fit averages every bearing together, so a unit whose minor axis
    is finer than the CPT spacing comes back as nearly pure nugget.  That is the
    correct answer to the omnidirectional question and a useless answer to the
    one actually being asked, which is whether there is structure *in some
    direction*.  Separating the lags by bearing is what distinguishes "no
    structure" from "structure I was averaging away".

    Sectors holding too few pairs are drawn faint and labelled with their count
    rather than dropped: a sector the layout cannot support is a fact about the
    survey, and hiding it would leave the reader comparing three curves without
    knowing the fourth was missing.

    Azimuths are degrees clockwise from north and describe an **axis** — 70 and
    250 select the same pairs.
    """
    fit = _require_fit(ds, unit_id)
    block = unit_block(ds, unit_id)
    x, y = block["x"].to_numpy(), block["y"].to_numpy()
    v = block["log_Q_mean"].to_numpy()

    per_sector = directional_variogram(
        x, y, v, azimuths=azimuths, tol_deg=tol_deg, bandwidth_km=bandwidth_km
    )

    # The compass gets its own column rather than an inset: overlaid on the
    # axes it sat on top of the data on every unit with a tall sector.
    fig, (ax, ax_c) = plt.subplots(
        1, 2, figsize=(11.4, 5.8), gridspec_kw={"width_ratios": [4, 1]},
        constrained_layout=True,
    )

    if np.isfinite(fit.sample_var):
        ax.axhline(fit.sample_var, color="0.35", ls="--", lw=1.1, zorder=1,
                   label=f"sample variance {fit.sample_var:.3f}")

    # The omnidirectional estimate and its fit, as the reference the sectors are
    # being compared against.
    iso_lags, iso_gamma, _ = empirical_variogram(x, y, v)
    ax.plot(iso_lags, iso_gamma, color="0.6", lw=1.4, marker="o", ms=4.5, zorder=2,
            label="omnidirectional")
    h = np.linspace(1e-6, float(fit.max_lag_km) * 1.05, 250)
    ax.plot(h, fit.model.variogram(h), color="k", lw=1.4, ls="--", zorder=2,
            label="isotropic fit")

    thin = []
    for k, az in enumerate(azimuths):
        lags, gamma, counts = per_sector[float(az)]
        n = int(counts.sum())
        weak = n < MIN_SECTOR_PAIRS
        if weak:
            thin.append((az, n))
        ax.plot(
            lags, gamma, color=_SECTOR_COLOURS[k % len(_SECTOR_COLOURS)],
            marker="s", ms=4.5, lw=1.9 if not weak else 1.0,
            alpha=0.95 if not weak else 0.45,
            ls="-" if not weak else ":", zorder=4 if not weak else 3,
            label=f"{az:g}° ({n} pairs)" + ("  — too few" if weak else ""),
        )

    _sector_compass(ax_c, ds, unit_id, azimuths)

    note = [
        "Curves separating means direction matters: the sector that stays",
        "lowest longest is the major axis.  Overlapping curves mean the",
        "layout cannot resolve anisotropy — not that there is none.",
    ]
    if thin:
        note.append(
            "Faint sectors hold under "
            f"{MIN_SECTOR_PAIRS} pairs: " + ", ".join(f"{a:g}° ({n})" for a, n in thin)
        )
    ax.text(
        0.015, 0.975, "\n".join(note), transform=ax.transAxes, fontsize=7.5,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="0.7", alpha=0.92),
    )

    ax.set_xlabel("separation h (km)")
    ax.set_ylabel(r"$\gamma(h)$  —  log $Q_{tn}$ variance")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0)
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9, ncol=2)

    figure_suptitle(
        fig,
        f"Directional variogram — {unit_label(ds, unit_id)}   "
        f"({len(block)} CPTs, ±{tol_deg:g}° sectors, azimuth CW from N)",
    )
    return fig


def _sector_compass(ax, ds, unit_id, azimuths) -> None:
    """The sectors as a compass, with the true anisotropy ellipse where known.

    The check that catches a 90-degree convention error: if the sector that
    reads as the major axis on the left-hand panel lies *across* the true
    ellipse rather than along it, the bearing bridge is inverted — and nothing
    about the variogram curves alone would reveal that.
    """
    ax.set_aspect("equal")
    r = 1.0
    for k, az in enumerate(azimuths):
        ux, uy = azimuth_unit_vector(az)
        ax.plot([-r * ux, r * ux], [-r * uy, r * uy],
                color=_SECTOR_COLOURS[k % len(_SECTOR_COLOURS)], lw=2.2, zorder=2)
        ax.annotate(f"{az:g}°", xy=(1.06 * r * ux, 1.06 * r * uy), fontsize=7,
                    ha="center", va="center",
                    color=_SECTOR_COLOURS[k % len(_SECTOR_COLOURS)])

    truth = _true_grf(ds, unit_id)
    if truth is not None and truth[0].aniso_ratio != 1.0:
        grf = truth[0]
        add_anisotropy_glyph(
            ax, range_km=r * 0.8, ratio=grf.aniso_ratio,
            azimuth_deg=grf.aniso_angle_deg, colour=_TRUTH,
        )
        ax.set_title(
            f"sectors, N up\ntrue major {grf.aniso_angle_deg:g}°, ratio {grf.aniso_ratio:g}",
            fontsize=7.5, color=_TRUTH,
        )
    else:
        ax.set_title("sectors, N up", fontsize=7.5)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.annotate("N", xy=(0, 1.2), fontsize=8, ha="center", va="bottom", weight="bold")
