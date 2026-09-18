# -*- coding: utf-8 -*-
"""Main-text Fig. 2 — shared-UMAP embedding of real versus generated CESC samples.

One UMAP is fitted on the real training data and every method is transformed into
those same coordinates, so the six panels share axes and are directly comparable.
TabDiff is omitted because its output tracks TabDDPM closely enough that the two
panels are indistinguishable.

Reads the coordinate dump written by experiments/figs_shared.py, which is shipped,
so this redraws on CPU without retraining anything.
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

SRC = os.path.join(_P.FIGURES_DIR, "CESC", "_umap_coords_CESC.json")
OUT = os.path.join(_P.FIGURES_DIR, "paper") + os.sep
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c7fb8"
RED = "#de2d26"
INK = "#2c3e50"
MUTED = "#5b6770"

ORDER = ["GA-LDM", "SMOTE", "CTGAN", "TabDDPM", "TabSyn", "scDiffusion"]

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
    "legend.frameon": False,
    "legend.fontsize": 7,
    "savefig.dpi": 400,
})


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    real = np.asarray(d["real"], float)
    gen = {m: np.asarray(d["gen"][m], float) for m in ORDER}

    # one window for every panel, otherwise the shared projection buys nothing
    allp = np.vstack([real] + list(gen.values()))
    xlim = (allp[:, 0].min() - 1, allp[:, 0].max() + 1)
    ylim = (allp[:, 1].min() - 1, allp[:, 1].max() + 1)

    fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.6))
    axes = axes.ravel()
    for i, (ax, m) in enumerate(zip(axes, ORDER)):
        ax.scatter(real[:, 0], real[:, 1], s=7, alpha=0.45, c=BLUE,
                   linewidths=0, label="Real", rasterized=True)
        ax.scatter(gen[m][:, 0], gen[m][:, 1], s=7, alpha=0.45, c=RED,
                   linewidths=0, label="Generated", rasterized=True)
        ax.set_title(f"{'abcdef'[i]}  {m}", loc="left", fontsize=8.5,
                     fontweight="bold", pad=4)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(loc="upper right", handletextpad=0.3, markerscale=1.6)

    fig.supxlabel("UMAP 1", fontsize=8, color=INK)
    fig.supylabel("UMAP 2", fontsize=8, color=INK)
    fig.tight_layout(rect=(0.015, 0.015, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT + f"fig2_shared_umap.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT + "fig2_shared_umap.png")


if __name__ == "__main__":
    main()
