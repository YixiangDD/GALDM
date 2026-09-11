# -*- coding: utf-8 -*-
"""Main-text Fig. 3 — biological fidelity of generated cohorts.

(a) Pathway-level stage effect: real versus generated Welch t (early vs late).
(b) Edge-wise cross-modal dependency on miRTarBase-validated edges, with the
    pairing-permuted null cloud behind it.
(c) XM-Corr across supervision/ablation conditions, including held-out edges.

All inputs are stored experiment artifacts; nothing is retrained here.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P  # noqa: E402

R = _P.RESULTS_DIR + os.sep
OUT = os.path.join(_P.FIGURES_DIR, "paper") + os.sep
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c7fb8"    # GA-LDM / real reference
RED = "#de2d26"     # comparator
GREY = "#7f8c8d"    # nulls and reference lines only, never a series
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


def load():
    bio = json.load(open(R + "bio_readout.json", encoding="utf-8"))
    ev = json.load(open(R + "edge_vectors_CESC.json", encoding="utf-8"))
    ho = json.load(open(R + "t2_heldout_CESC.json", encoding="utf-8"))
    xm = json.load(open(R + "crossmodal_CESC.json", encoding="utf-8"))
    return bio, ev, ho, xm


def panel_a(ax, bio):
    """Real vs generated pathway-level stage t-statistic (5-seed means)."""
    for cohort, colour, marker in (("CESC", BLUE, "o"), ("KIRC", RED, "^")):
        pp = bio[cohort]["per_pathway"]
        tr = np.asarray(pp["t_real_mean"])
        tg = np.asarray(pp["t_gen_mean"])
        names = bio[cohort]["pathway_names"]
        bg = np.array([n == "Background" for n in names])
        s = bio[cohort]["t_spearman"]
        ax.scatter(tr[~bg], tg[~bg], s=26, marker=marker, facecolor=colour,
                   edgecolor="white", linewidth=0.5, zorder=3,
                   label=f"{cohort}   " + r"$\rho$ = "
                         f"{s['mean']:.2f} $\\pm$ {s['std']:.2f}")
        ax.scatter(tr[bg], tg[bg], s=26, marker=marker, facecolor="none",
                   edgecolor=colour, linewidth=0.9, zorder=3)

    # Limits must cover BOTH cohorts: KIRC spans -3.49..2.89, so the old
    # (-2.95, 1.15) window silently clipped four KIRC pathways off the panel.
    lo, hi = -3.75, 3.15
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="#d5d8da", lw=0.6, zorder=0)
    ax.axvline(0, color="#d5d8da", lw=0.6, zorder=0)

    # Label the two pathways discussed in the text. Both sit just below the
    # diagonal in CESC, so they are placed on opposite sides (one above-left,
    # one below-right) with leader lines to keep the text clear of each other
    # and of the identity line.
    pp = bio["CESC"]["per_pathway"]
    names = bio["CESC"]["pathway_names"]
    for lab, off, ha in (("JAK-STAT signaling", (-14, 30), "center"),
                         ("Apoptosis", (44, -16), "center")):
        i = names.index(lab)
        ax.annotate(lab,
                    (pp["t_real_mean"][i], pp["t_gen_mean"][i]),
                    textcoords="offset points", xytext=off,
                    fontsize=6.5, color=MUTED, va="center", ha=ha,
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="#c3c9cc", lw=0.6,
                                    shrinkA=0, shrinkB=3))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Real cohort: Welch $t$ (late vs early)")
    ax.set_ylabel("Generated cohort: Welch $t$")
    ax.set_title("a  Stage-associated pathway programmes",
                 loc="left", fontsize=8.5, fontweight="bold", pad=6)
    leg = ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.2)
    leg.set_zorder(6)
    ax.text(0.98, 0.03,
            "5-seed means; $\\rho$ is the per-seed mean\nopen marker = background gene group",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            color=GREY, linespacing=1.5)



def panel_b(ax, ev):
    """Per-edge real vs generated correlation, with the permuted-pairing null."""
    real = np.asarray(ev["real"])
    gen = np.asarray(ev["gen"])
    shuf = np.asarray(ev["gen_shuffled"])

    ax.scatter(real, shuf, s=3, facecolor=GREY, alpha=0.18, linewidth=0,
               zorder=2, rasterized=True)
    ax.scatter(real, gen, s=3, facecolor=BLUE, alpha=0.35, linewidth=0,
               zorder=3, rasterized=True)
    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.5, mfc=BLUE, mec="none",
               label=f"GA-LDM   $r$ = {ev['pearson_real_gen']:.2f}"),
        Line2D([], [], marker="o", ls="none", ms=4.5, mfc=GREY, mec="none",
               label=f"pairing permuted   $r$ = {ev['pearson_real_shuffled']:.2f}"),
    ]

    lim = 0.92
    ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.8, ls="--", zorder=4)
    ax.axhline(0, color="#d5d8da", lw=0.6, zorder=1)
    ax.axvline(0, color="#d5d8da", lw=0.6, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Real: corr(miRNA, target mRNA)")
    ax.set_ylabel("Generated: corr(miRNA, target mRNA)")
    ax.set_title(f"b  {ev['n_edges']} miRTarBase-validated edges (CESC, seed 42)",
                 loc="left", fontsize=8.5, fontweight="bold", pad=6)
    ax.legend(handles=handles, loc="upper left", handletextpad=0.3,
              borderpad=0.2)
    ax.text(0.98, 0.04,
            "this seed: 84% of repressive edges keep a\nnegative sign; 5-seed mean 87 $\\pm$ 2%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=MUTED, linespacing=1.5)


def panel_c(ax, ho, xm):
    """Per-seed XM-Corr by condition on a zoomed axis; the null is annotated."""
    rows = [
        ("Training edges", ho["Full"]["train"], BLUE),
        ("Held-out edges", ho["Full"]["heldE"], BLUE),
        ("Held-out miRNAs\n(whole miRNAs removed)", ho["Full"]["heldN"], BLUE),
        ("Held-out edges,\nno cross-modal attention", ho["wo_CrossModal"]["heldE"], RED),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, vals, col) in zip(y, rows):
        v = np.asarray(vals)
        ax.scatter(v, np.full_like(v, yi), s=20, facecolor=col, alpha=0.45,
                   edgecolor="none", zorder=3)
        ax.plot([v.min(), v.max()], [yi, yi], color=col, lw=1.0, alpha=0.5,
                zorder=2)
        ax.scatter([v.mean()], [yi], s=52, marker="|", color=col,
                   linewidth=1.8, zorder=4)
        ax.text(v.mean(), yi + 0.30, f"{v.mean():.3f}", ha="center",
                fontsize=7, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.set_xlim(0.888, 0.948)
    ax.set_xlabel("XM-Corr on miRTarBase edges (5 seeds; bar = mean)")
    ax.set_title("c  Cross-modal dependency generalises to edges never seen "
                 "in training (CESC)",
                 loc="left", fontsize=8.5, fontweight="bold", pad=6)
    ax.xaxis.grid(True, color="#e9ecee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    null = np.asarray(ho["Full"]["heldE_null"])
    ax.text(0.995, -0.30,
            f"permuted pairing collapses to {abs(null.mean()):.2f} "
            f"(permutation $P$ < 0.001, {xm['n_edges']} edges) — off scale",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=6.8, color=GREY)




def main():
    bio, ev, ho, xm = load()
    fig = plt.figure(figsize=(7.09, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.78],
                          hspace=0.50, wspace=0.30,
                          left=0.085, right=0.985, top=0.945, bottom=0.10)
    panel_a(fig.add_subplot(gs[0, 0]), bio)
    panel_b(fig.add_subplot(gs[0, 1]), ev)
    panel_c(fig.add_subplot(gs[1, :]), ho, xm)
    for ext in ("png", "pdf"):
        fig.savefig(OUT + f"fig3_bioreadout.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT + "fig3_bioreadout.png")


if __name__ == "__main__":
    main()
