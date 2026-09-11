# -*- coding: utf-8 -*-
"""Main-text Fig. 4 — what the structural prior buys, and whether samples are novel.

(a, b) Degree-preserving mask rewiring on CESC and KIRC: KEGG/family grouping
       versus random groups matched on group size and per-feature overlap degree.
(c)    Pathway/family identity encoding in the denoiser, isolated at matched
       architecture, parameter count and training budget.
(d)    Nearest-neighbour novelty audit against interpolation-based oversampling.

Effect sizes use the same independent-samples Cohen's d over the five seeds as
the main text. All inputs are stored experiment artifacts; nothing is retrained.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P  # noqa: E402

R = _P.RESULTS_DIR + os.sep
OUT = os.path.join(_P.FIGURES_DIR, "paper") + os.sep
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c7fb8"
RED = "#de2d26"
GREY = "#7f8c8d"
INK = "#2c3e50"
MUTED = "#5b6770"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": "#909aa0",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "savefig.dpi": 400,
})


def cohen_d(a, b):
    """Independent-samples Cohen's d, matching the convention in the text."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def paired(ax, ours, theirs, names, ylab, title):
    """Per-seed paired dot plot: ours (blue) versus the control (red)."""
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    x0, x1 = 0.0, 1.0
    for a, b in zip(ours, theirs):
        ax.plot([x0, x1], [a, b], color="#cbd0d4", lw=0.7, zorder=1)
    ax.scatter(np.full_like(ours, x0), ours, s=24, facecolor=BLUE,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(np.full_like(theirs, x1), theirs, s=24, facecolor=RED,
               edgecolor="white", linewidth=0.5, zorder=3)
    for x, v, c, ha in ((x0, ours, BLUE, "right"), (x1, theirs, RED, "left")):
        ax.plot([x - 0.3, x + 0.3], [v.mean()] * 2, color=c, lw=1.8, zorder=4)
        dx = -0.36 if ha == "right" else 0.36
        ax.text(x + dx, v.mean(), f"{v.mean():.1f}", ha=ha, va="center",
                fontsize=6.8, color=c, zorder=5)


    lo = min(ours.min(), theirs.min())
    hi = max(ours.max(), theirs.max())
    pad = 0.16 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.1 * pad)
    ax.set_xlim(-0.86, 1.86)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=6)
    ax.text(0.5, 0.965,
            f"$d$ = {cohen_d(ours, theirs):.2f},  "
            f"{int((ours < theirs).sum())}/{len(ours)} seeds",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.9,
            color=INK)
    ax.yaxis.grid(True, color="#e9ecee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def novelty(ax):
    """Distance to the nearest real training sample, four cohorts."""
    cohorts = ["CESC", "COAD", "HNSC", "KIRC"]
    gm, gs, sm, ss, gd, sd = [], [], [], [], [], []
    for c in cohorts:
        d = json.load(open(R + f"novelty_sensitivity_{c}.json", encoding="utf-8"))
        gm.append(d["GA-LDM"]["ratio_to_real"][0])
        gs.append(d["GA-LDM"]["ratio_to_real"][1])
        sm.append(d["SMOTE"]["ratio_to_real"][0])
        ss.append(d["SMOTE"]["ratio_to_real"][1])
        gd.append(d["GA-LDM"]["dup_rate"][0])
        sd.append(d["SMOTE"]["dup_rate"][0])

    x = np.arange(len(cohorts))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, gm, w, yerr=gs, color=BLUE, edgecolor="white",
           linewidth=0.8, capsize=2.0, error_kw=dict(lw=0.8, ecolor=INK),
           label="GA-LDM", zorder=3)
    ax.bar(x + w / 2 + 0.01, sm, w, yerr=ss, color=RED, edgecolor="white",
           linewidth=0.8, capsize=2.0, error_kw=dict(lw=0.8, ecolor=INK),
           label="SMOTE", zorder=3)
    ax.axhline(1.0, color=GREY, lw=0.9, ls="--", zorder=2)
    ax.text(x[-1] + 0.5, 1.015, "a real held-out sample", ha="right",
            va="bottom", fontsize=6.4, color=GREY)

    for xi in range(len(cohorts)):
        ax.text(xi - w / 2, gm[xi] + gs[xi] + 0.035, f"{100 * gd[xi]:.0f}%",
                ha="center", fontsize=6.6, color=BLUE)
        ax.text(xi + w / 2, sm[xi] + ss[xi] + 0.035, f"{100 * sd[xi]:.0f}%",
                ha="center", fontsize=6.6, color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, fontsize=7.5)
    ax.set_xlim(-0.55, len(cohorts) - 0.45)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("NN distance to training set\n(fraction of a real sample's)")
    ax.set_title("d  Novelty versus memorisation", loc="left", fontsize=8.5,
                 fontweight="bold", pad=6)
    ax.legend(loc="upper left", ncol=2, handletextpad=0.4, columnspacing=1.1)
    ax.yaxis.grid(True, color="#e9ecee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    pc = json.load(open(R + "prior_scarcity_CESC.json", encoding="utf-8"))
    pk = json.load(open(R + "prior_scarcity_KIRC.json", encoding="utf-8"))
    t6 = json.load(open(R + "t6_gadit_CESC.json", encoding="utf-8"))

    fig = plt.figure(figsize=(7.09, 2.75))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.85], wspace=0.55,
                          left=0.062, right=0.995, top=0.855, bottom=0.16)

    paired(fig.add_subplot(gs[0, 0]), pc["N199_KEGG"]["Bio_FID"],
           pc["N199_Random"]["Bio_FID"], ["KEGG /\nfamily", "matched\nrandom"],
           "Bio-FID (lower better)", "a  Grouping, CESC")
    paired(fig.add_subplot(gs[0, 1]), pk["N358_KEGG"]["Bio_FID"],
           pk["N358_Random"]["Bio_FID"], ["KEGG /\nfamily", "matched\nrandom"],
           "Bio-FID", "b  Grouping, KIRC")
    paired(fig.add_subplot(gs[0, 2]), t6["GA-DiT"]["Bio_FID"],
           t6["plain_DiT"]["Bio_FID"], ["GA-DiT", "plain\nDiT"],
           "Bio-FID", "c  Identity encoding, CESC")
    novelty(fig.add_subplot(gs[0, 3]))

    for ext in ("png", "pdf"):
        fig.savefig(OUT + f"fig4_prior_novelty.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT + "fig4_prior_novelty.png")


if __name__ == "__main__":
    main()
