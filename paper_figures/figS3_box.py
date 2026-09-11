"""Fig S3 for the supplement: per-sample HIF-1 activity as a real box plot.

The original interpretability script drew two histograms of group densities.
The rewritten caption reports quartiles and a rank-biserial effect size, so the
figure has to show the per-sample distribution that those statistics come from.
Reads the per-sample dump produced by experiments/interpretability_dump.py.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P  # noqa: E402

SRC = os.path.join(_P.RESULTS_DIR, "interpretability_KIRC_persample.json")
OUT = os.path.join(_P.FIGURES_DIR, "paper", "figS3_hif1_box.png")

BLUE = "#2C7FB8"
RED = "#DE2D26"
INK = "#2C3E50"
GREY = "#7F8C8D"


def rank_biserial(a, b):
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (a.size * b.size) - 1


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    ps = {k: np.asarray(v, float) for k, v in d["per_sample"].items()}
    order = [("Early\n(I/II)", "real_early", BLUE), ("Late\n(III/IV)", "real_late", RED),
             ("Early\n(I/II)", "gen_early", BLUE), ("Late\n(III/IV)", "gen_late", RED)]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    rng = np.random.default_rng(0)
    pos = [1, 1.8, 3.4, 4.2]

    for (lab, key, colour), x in zip(order, pos):
        v = ps[key]
        bp = ax.boxplot([v], positions=[x], widths=0.5, showfliers=False,
                        patch_artist=True, medianprops=dict(color=INK, lw=1.8),
                        whiskerprops=dict(color=INK, lw=1.0),
                        capprops=dict(color=INK, lw=1.0),
                        boxprops=dict(edgecolor=INK, lw=1.0))
        bp["boxes"][0].set_facecolor(colour)
        bp["boxes"][0].set_alpha(0.22)
        ax.scatter(x + rng.uniform(-0.14, 0.14, v.size), v, s=7, color=colour,
                   alpha=0.45, linewidths=0, zorder=3)
        ax.text(x, ax.get_ylim()[0], "", ha="center")

    ax.set_xticks(pos)
    ax.set_xticklabels([o[0] for o in order], fontsize=9)
    ax.set_ylabel("HIF-1 pathway activity", fontsize=10)
    for x, name, n in ((1.4, "Real cohort", ps["real_early"].size + ps["real_late"].size),
                       (3.8, "Generated (GA-LDM)",
                        ps["gen_early"].size + ps["gen_late"].size)):
        ax.text(x, 1.02, f"{name}   (n = {n})", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, color=INK, weight="bold")

    # the two statistics the caption reports
    for x, key_e, key_l in ((1.4, "real_early", "real_late"),
                            (3.8, "gen_early", "gen_late")):
        e, l = ps[key_e], ps[key_l]
        p = stats.mannwhitneyu(e, l, alternative="two-sided").pvalue
        r = rank_biserial(e, l)
        ax.text(x, 0.045, f"$P$ = {p:.3f},  $r$ = {r:.3f}",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5,
                color=GREY)

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0.4, 4.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT, os.path.getsize(OUT), "bytes")

    for tag in ("real", "gen"):
        e, l = ps[f"{tag}_early"], ps[f"{tag}_late"]
        q = np.percentile(e, [25, 50, 75]), np.percentile(l, [25, 50, 75])
        print(f"{tag:5s} early q={np.round(q[0],4)} late q={np.round(q[1],4)} "
              f"r={rank_biserial(e,l):+.4f} "
              f"P={stats.mannwhitneyu(e,l,alternative='two-sided').pvalue:.4f}")


if __name__ == "__main__":
    main()
