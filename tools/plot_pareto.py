# -*- coding: utf-8 -*-
"""Plots the Pareto front of the cross-modal coupling weight lambda_cm: fidelity (Bio-FID)
against cross-modal dependence (miRNA Corr). Single-column size, writes
results/figures/pareto_crossmodal.png"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P

RES = _P.RESULTS_DIR
d = json.load(open(RES + "/pareto_crossmodal_CESC.json"))
lams = ["0", "0.25", "0.5", "1", "2", "4"]


def stat(lam, k):
    v = [x for x in d[lam].get(k, []) if not (isinstance(x, float) and np.isnan(x))]
    return np.mean(v), np.std(v)


bf = [stat(l, "Bio_FID") for l in lams]
mi = [stat(l, "miRNA_Corr") for l in lams]
bx, bxe = [x[0] for x in bf], [x[1] for x in bf]
my, mye = [x[0] for x in mi], [x[1] for x in mi]

fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
ax.errorbar(bx, my, xerr=bxe, yerr=mye, fmt="-o", color="#2c6fbb",
            ecolor="#9bb8d8", elinewidth=1, capsize=2.5, markersize=5,
            markerfacecolor="#2c6fbb", markeredgecolor="white", zorder=3)
# lambda annotations
offsets = {"0": (6, -10), "0.25": (6, 6), "0.5": (6, 6), "1": (6, -12),
           "2": (8, 4), "4": (-4, -14)}
for l, x, y in zip(lams, bx, my):
    dx, dy = offsets.get(l, (6, 6))
    ax.annotate(rf"$\lambda_{{cm}}$={l}", (x, y), textcoords="offset points",
                xytext=(dx, dy), fontsize=7, color="#333")
# mark the default operating point, lambda=1
i1 = lams.index("1")
ax.scatter([bx[i1]], [my[i1]], s=120, facecolors="none",
           edgecolors="#d9534f", linewidths=1.5, zorder=4, label=r"default ($\lambda_{cm}$=1)")

ax.set_xlabel("Bio-FID  (lower = better fidelity)", fontsize=8)
ax.set_ylabel("miRNA Corr  (higher = better\ncross-modal dependency)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=6.5, loc="lower right")
# arrow indicating the direction of the trade-off
ax.annotate("", xy=(15.8, 0.859), xytext=(19.6, 0.833),
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8, ls=":"))
fig.tight_layout()
os.makedirs(RES + "/figures", exist_ok=True)
out = RES + "/figures/pareto_crossmodal.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
