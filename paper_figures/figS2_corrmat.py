# -*- coding: utf-8 -*-
"""Fig. S2 for the supplement — within-pathway gene-correlation matrices, CESC.

Each panel is the correlation matrix of the 40 retained member genes of the first
KEGG pathway, on a shared -1 to +1 scale. Panel titles carry PCE (Frobenius
distance to the real matrix, lower better) and the RV coefficient with its Mantel
permutation P-value, so the visual impression is backed by numbers.

Reads the matrix and statistic dumps written by experiments/figs_shared.py, both
shipped, so this redraws on CPU without retraining anything.
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

FD = os.path.join(_P.FIGURES_DIR, "CESC")
SRC_MATS = os.path.join(FD, "_corr_mats_CESC.json")
SRC_RV = os.path.join(FD, "_rv_mantel_CESC.json")
OUT = os.path.join(_P.FIGURES_DIR, "paper") + os.sep
os.makedirs(OUT, exist_ok=True)

INK = "#2c3e50"
MUTED = "#5b6770"
GREY = "#7f8c8d"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": "#909aa0",
    "axes.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "savefig.dpi": 400,
})


def title_for(name, stat, letter):
    """Panel title; the real matrix is the reference, so it carries no statistic."""
    head = f"{letter}  {name}"
    if stat is None:
        return head + "\n(reference)"
    p = ("$P$ < 0.001" if stat["mantel_p"] < 1e-3
         else f"$P$ = {stat['mantel_p']:.3f}")
    return head + f"\nPCE = {stat['PCE']:.3f}   RV = {stat['RV']:.2f} ({p})"


def main():
    d = json.load(open(SRC_MATS, encoding="utf-8"))
    rv = json.load(open(SRC_RV, encoding="utf-8"))
    mats = d["mats"]

    names = list(mats.keys())
    ncol = 4
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.09, 4.1))
    axes = np.asarray(axes).ravel()

    im = None
    for i, (ax, name) in enumerate(zip(axes, names)):
        M = np.asarray(mats[name], float)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title_for(name, rv.get(name), "abcdefg"[i]), loc="left",
                     fontsize=7.2, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # the leftover grid cell hosts the colour bar, so no panel slot is left blank
    for ax in axes[len(names):]:
        ax.axis("off")
    spare = axes[len(names)] if len(names) < len(axes) else None
    if spare is not None:
        cax = spare.inset_axes([0.26, 0.12, 0.085, 0.72])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.set_label("Pearson correlation", fontsize=7.5)
        n = np.asarray(mats[names[0]]).shape[0]
        spare.text(0.26, 0.02, f"{n} x {n} member genes", fontsize=6.8,
                   color=GREY, transform=spare.transAxes, va="top")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT + f"figS2_corrmat.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT + "figS2_corrmat.png")


if __name__ == "__main__":
    main()
