"""
Plots: a class overview for you, and one figure per student for their email.

Category colors come from a fixed, colorblind-checked order (category order in the config),
so a category keeps its color from plot to plot.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10,
})


def category_colors(cfg):
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(cfg["categories"])}


def _wrap(s, width):
    return textwrap.fill(s, width=width, break_long_words=False)


def _fit_ylim(ax, values):
    v = values[~np.isnan(values)]
    lo = min(0, v.min()) if v.size else 0
    hi = max(100, v.max()) if v.size else 100
    ax.set_ylim(lo - 4, hi + 4)


def _assignment_boxes(ax, res, wrap=28):
    """Box per assignment (class spread, in percent), colored by category. Returns x positions."""
    colors = category_colors(res.cfg)
    cols = list(res.assignments.index)
    x = np.arange(len(cols))
    for xi, col in zip(x, cols):
        v = 100 * res.pct[col].dropna().to_numpy()
        c = colors[res.assignments.at[col, "category"]]
        if len(v):
            ax.boxplot(v, positions=[xi], widths=0.55, patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=c, alpha=0.35, edgecolor=c, linewidth=1.2),
                       whiskerprops=dict(color=c, linewidth=1.2), capprops=dict(color=c, linewidth=1.2),
                       medianprops=dict(color=c, linewidth=2),
                       flierprops=dict(marker="o", markersize=4, markerfacecolor=c,
                                       markeredgecolor=SURFACE, alpha=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(res.assignments.at[c, "title"], wrap) for c in cols],
                       rotation=90, fontsize=8)
    ax.set_xlim(-0.7, len(cols) - 0.3)
    ax.set_ylabel("Score (%)")
    present = [k for k in res.cfg["categories"] if (res.assignments["category"] == k).any()]
    handles = [Patch(facecolor=colors[k], alpha=0.6, edgecolor=colors[k],
                     label=res.cfg["categories"][k]["label"]) for k in present]
    return x, handles


def class_overview(res, figsize=None):
    """Figure for your own inspection: assignment spreads, grade distribution,
    category spreads, and drops vs. no drops."""
    n = len(res.assignments)
    fig = plt.figure(figsize=figsize or (max(12, 0.45 * n + 4), 11), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])

    ax = fig.add_subplot(gs[0, :])
    _, handles = _assignment_boxes(ax, res)
    ax.set_title(f"{res.cfg['name']}: scores on each assignment ({len(res.students)} students)",
                 loc="left", fontsize=12, color=INK)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
              ncols=min(len(handles), 6), borderaxespad=0.2)
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=12, color=INK, pad=26)
    _fit_ylim(ax, 100 * res.pct.to_numpy(float))

    # Course grade distribution: equal 2.5-point bins, letter cutoffs as faint lines
    ax = fig.add_subplot(gs[1, 0])
    g = 100 * res.grade.dropna()
    lo = np.floor(min(g.min(), 60) / 5) * 5 if len(g) else 50
    hi = max(100, np.ceil(g.max())) if len(g) else 100
    ax.hist(g, bins=np.arange(lo, hi + 2.5, 2.5), color=PALETTE[0], edgecolor=SURFACE, linewidth=2)
    ax.yaxis.get_major_locator().set_params(integer=True)
    scale = [(c, l) for c, l in res.cfg["letter_scale"] if lo <= c <= hi]
    top = ax.get_ylim()[1]
    edges = [c for c, _ in scale]
    for (c, l), right in zip(scale, [hi] + edges[:-1]):
        ax.axvline(c, color=INK2, linewidth=0.6, alpha=0.4, zorder=0)
        ax.text((c + right) / 2, top, l, ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Course grade (%)")
    ax.set_ylabel("Students")
    ax.set_title("Course grades" + (" (with drops)" if res.cfg["apply_drops"] else ""),
                 loc="left", color=INK, pad=14)

    # Category spreads
    ax = fig.add_subplot(gs[1, 1])
    colors = category_colors(res.cfg)
    cats = [k for k in res.cfg["categories"] if res.categories[k].notna().any()]
    for i, k in enumerate(cats):
        v = 100 * res.categories[k].dropna()
        c = colors[k]
        ax.boxplot(v, positions=[i], widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor=c, alpha=0.35, edgecolor=c),
                   whiskerprops=dict(color=c), capprops=dict(color=c),
                   medianprops=dict(color=c, linewidth=2),
                   flierprops=dict(marker="o", markersize=4, markerfacecolor=c, markeredgecolor=SURFACE))
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([_wrap(res.cfg["categories"][k]["label"], 12) for k in cats], fontsize=8)
    ax.set_ylabel("Category average (%)")
    ax.set_title("By category", loc="left", color=INK)

    # Effect of drops
    ax = fig.add_subplot(gs[1, 2])
    a, b = 100 * res.grade_nodrops, 100 * res.grade_drops
    ax.scatter(a, b, s=40, color=PALETTE[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    lims = [min(a.min(), b.min()) - 3, max(a.max(), b.max()) + 3]
    ax.plot(lims, lims, color=INK2, linewidth=1, linestyle="--")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("No drops (%)"); ax.set_ylabel("With drops (%)")
    ax.set_title("Effect of dropping lowest scores", loc="left", color=INK)
    return fig


def student_figure(res, sid, figsize=None):
    """The figure attached to a student's email: class boxes, the student's scores,
    and their running course average."""
    s = res.students.loc[sid]
    n = len(res.assignments)
    fig, ax = plt.subplots(figsize=figsize or (max(10, 0.42 * n + 3), 7.5), layout="constrained")
    x, handles = _assignment_boxes(ax, res)

    p = 100 * res.pct.loc[sid].to_numpy(float)
    dropped = res.dropped.loc[sid].to_numpy() if res.cfg["apply_drops"] else np.zeros(n, bool)
    keep = ~np.isnan(p) & ~dropped
    ax.plot(x[keep], p[keep], linestyle="none", marker="*", markersize=15,
            markerfacecolor="#d03b3b", markeredgecolor=INK, markeredgewidth=0.8, zorder=5)
    if dropped.any():
        ax.plot(x[dropped], p[dropped], linestyle="none", marker="*", markersize=15,
                markerfacecolor="none", markeredgecolor="#d03b3b", markeredgewidth=1.2, zorder=5)
    ax.plot(x, 100 * res.running.loc[sid].to_numpy(float), color=INK, linewidth=2,
            linestyle="--", marker="o", markersize=5, zorder=4)

    grade = 100 * res.grade.loc[sid]
    handles += [
        Line2D([], [], marker="*", linestyle="none", markersize=13, markerfacecolor="#d03b3b",
               markeredgecolor=INK, label="Your score"),
        Line2D([], [], color=INK, linestyle="--", marker="o", markersize=5,
               label="Your running course average (before drops)"),
    ]
    if dropped.any():
        handles.append(Line2D([], [], marker="*", linestyle="none", markersize=13,
                              markerfacecolor="none", markeredgecolor="#d03b3b", label="Dropped"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
              ncols=3, fontsize=9, borderaxespad=0.2)
    ax.set_title(f"{s['first']} {s['last']}: course grade {grade:.1f}%   "
                 f"(boxes show the spread of the whole class)", loc="left", fontsize=12, color=INK,
                 pad=44)
    _fit_ylim(ax, 100 * res.pct.to_numpy(float))
    return fig
