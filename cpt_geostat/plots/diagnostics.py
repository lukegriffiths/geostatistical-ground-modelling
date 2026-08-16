"""B2 — depth traces, trend checks and lag coverage.

The lag-coverage figure is the one to read *before* fitting anything: it says
whether an anisotropic variogram is identifiable from this layout at all.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..geometry import pair_distances, project_on_azimuth
from ..trend import fit_linear_trend
from .style import figure_suptitle, unit_colours, unit_grid, unit_label


def fit_trend_azimuth(df: pd.DataFrame, value: str = "log_Q_mean"):
    """OLS ``value ~ 1 + x + y`` reported as ``(gradient per km, azimuth CW from N)``.

    This is what the trend check projects onto for real data, where no true
    azimuth exists; on synthetic data it doubles as a recovery readout.

    A thin wrapper over :func:`cpt_geostat.trend.fit_linear_trend`, which keeps the
    full coefficients.  The plotting side only needs the polar pair, but
    universal kriging and a detrended GP need the intercept as well, so the fit
    itself lives in one place rather than one per caller.
    """
    fit = fit_linear_trend(df["x"], df["y"], df[value])
    return fit.gradient, fit.azimuth_deg


def plot_trend_check(ds, unit_ids: Optional[Sequence[str]] = None) -> Figure:
    """``log Q`` projected onto an azimuth, with an OLS line.

    The fitted azimuth is always shown.  Where truth is available the true
    azimuth is shown alongside it, so a trend recovered on the wrong bearing is
    obvious rather than merely producing a slightly worse scatter.
    """
    unit_ids = list(unit_ids or ds.unit_ids)
    colours = unit_colours(ds, unit_ids)
    fig, axes = unit_grid(len(unit_ids), size=(4.2, 3.6))
    cfg = getattr(ds, "config", None)

    for ax, uid in zip(axes, unit_ids):
        sub = ds.unit_summary[ds.unit_summary["unit_id"] == uid]
        if len(sub) < 3:
            ax.set_title(f"{unit_label(ds, uid)} — too few CPTs", fontsize=9)
            continue

        fit_grad, fit_az = fit_trend_azimuth(sub)
        true = cfg.units[uid].property.trend if cfg is not None and uid in cfg.units else None
        use_az = true.azimuth_deg if (true is not None and true.grad != 0) else fit_az
        basis = "true" if (true is not None and true.grad != 0) else "fitted"

        s = project_on_azimuth(sub["x"], sub["y"], use_az)
        v = sub["log_Q_mean"].to_numpy()
        ax.scatter(s, v, s=24, color=colours[uid], edgecolor="k", lw=0.3)

        slope, intercept = np.polyfit(s, v, 1)
        xs = np.linspace(s.min(), s.max(), 2)
        ax.plot(xs, intercept + slope * xs, "k-", lw=1.4,
                label=f"OLS slope {slope:+.3f} / km")

        # With the projection on the true azimuth the generative trend is exactly
        # linear in s, so drawing it shows whether a mismatch is a broken
        # convention or simply a trend this unit has too few CPTs to resolve.
        if basis == "true":
            ax.plot(xs, v.mean() + true.grad * (xs - s.mean()), color="#B00020",
                    ls="--", lw=1.3, label=f"true slope {true.grad:+.3f} / km")

        lines = [f"fitted: {fit_grad:.3f} / km at {fit_az:.0f}°"]
        if true is not None:
            lines.append(f"true:   {true.grad:.3f} / km at {true.azimuth_deg:.0f}°")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", alpha=0.9))

        ax.set_xlabel(f"distance along {basis} azimuth {use_az:.0f}° (km)")
        ax.set_ylabel(r"depth-avg log $Q_{tn}$")
        ax.set_title(unit_label(ds, uid), fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(ls=":", alpha=0.4)

    figure_suptitle(fig, "Trend check — azimuths are degrees clockwise from north")
    return fig


def plot_depth_traces(ds, cpt_ids: Optional[Sequence[str]] = None, n: int = 5) -> Figure:
    """``Qtn`` vs depth for a handful of CPTs, unit intervals shaded.

    Confirms the within-unit scatter looks plausible — correlated wobble about a
    unit mean, not white noise and not a straight line.
    """
    if cpt_ids is None:
        counts = ds.layers.groupby("cpt_id").size().sort_values(ascending=False)
        cpt_ids = list(counts.index[:n])
    cpt_ids = list(cpt_ids)
    colours = unit_colours(ds)

    fig, axes = plt.subplots(1, len(cpt_ids), figsize=(2.6 * len(cpt_ids), 7.2),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, cid in zip(axes, cpt_ids):
        trace = ds.samples[ds.samples["cpt_id"] == cid].sort_values("z")
        for row in ds.layers[ds.layers["cpt_id"] == cid].itertuples(index=False):
            ax.axhspan(row.z_top, row.z_bot, color=colours.get(row.unit_id, "#cccccc"),
                       alpha=0.28, lw=0)
            ax.text(0.97, (row.z_top + row.z_bot) / 2, row.unit_id.replace("unit_", "U"),
                    transform=ax.get_yaxis_transform(), fontsize=7, ha="right", va="center")
        ax.plot(trace["Qtn"], trace["z"], color="k", lw=0.35)
        ax.set_xlabel(r"$Q_{tn}$")
        ax.set_title(cid, fontsize=9)
        ax.grid(ls=":", alpha=0.35)

    axes[0].set_ylabel("depth below seabed (m)")
    axes[0].invert_yaxis()
    figure_suptitle(fig, "Depth traces — within-unit scatter about each unit value")
    return fig


def plot_lag_coverage(ds, n_bins: int = 40) -> Figure:
    """Pairwise distance histogram and an azimuth rose.

    Read before fitting: if the shortest lags are empty the nugget and the range
    are not separately identifiable, and if the rose has a preferred direction
    the anisotropy estimate will inherit it.
    """
    dist, az = pair_distances(ds.layout)
    fig = plt.figure(figsize=(12.5, 5.2), constrained_layout=True)
    ax_hist = fig.add_subplot(1, 2, 1)
    ax_rose = fig.add_subplot(1, 2, 2, projection="polar")

    ax_hist.hist(dist, bins=n_bins, color="#2A9D8F", edgecolor="k", lw=0.3)
    ax_hist.set_xlabel("pair separation (km)")
    ax_hist.set_ylabel("number of pairs")
    ax_hist.set_yscale("log")
    ax_hist.grid(ls=":", alpha=0.4)

    short = dist < 0.5
    ax_hist.axvline(0.5, color="#E76F51", ls="--", lw=1.2)
    ax_hist.set_title(
        f"Lag coverage — {len(dist)} pairs, {int(short.sum())} below 0.5 km, "
        f"shortest {dist.min():.3f} km",
        fontsize=10,
    )

    # Rose: azimuths are already folded to [0, 180); mirror for a symmetric rose.
    bins = np.arange(0, 181, 10)
    counts, edges = np.histogram(az, bins=bins)
    centres = np.deg2rad(0.5 * (edges[:-1] + edges[1:]))
    width = np.deg2rad(10)
    for offset in (0.0, np.pi):
        ax_rose.bar(centres + offset, counts, width=width, bottom=0.0,
                    color="#264653", edgecolor="w", lw=0.4, alpha=0.9)
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)  # clockwise, matching the azimuth convention
    ax_rose.set_title("Pair azimuths (deg CW from north)", fontsize=10)

    return fig


def plot_within_unit_scatter(ds, unit_ids: Optional[Sequence[str]] = None) -> Figure:
    """Per-CPT within-unit sd against sample count.

    The depth-average is only as good as the effective sample size behind it;
    correlated residuals mean the raw ``n_samples`` overstates it, and this is
    where that shows up.
    """
    unit_ids = list(unit_ids or ds.unit_ids)
    colours = unit_colours(ds, unit_ids)
    fig, axes = unit_grid(len(unit_ids), size=(4.0, 3.4))
    cfg = getattr(ds, "config", None)

    for ax, uid in zip(axes, unit_ids):
        sub = ds.unit_summary[ds.unit_summary["unit_id"] == uid]
        if not len(sub):
            ax.set_title(f"{unit_label(ds, uid)} — absent", fontsize=9)
            continue
        ax.scatter(sub["n_samples"], sub["log_Q_sd"], s=22, color=colours[uid],
                   edgecolor="k", lw=0.3)
        if cfg is not None and uid in cfg.units:
            target = cfg.units[uid].property.within_unit.sd
            ax.axhline(target, color="k", ls="--", lw=1.2,
                       label=f"configured sd {target:g}")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xlabel("samples in unit")
        ax.set_ylabel(r"within-unit sd of log $Q_{tn}$")
        ax.set_title(unit_label(ds, uid), fontsize=9)
        ax.grid(ls=":", alpha=0.4)

    figure_suptitle(fig, "Within-unit scatter vs. samples per unit")
    return fig
