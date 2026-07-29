# -*- coding: utf-8 -*-
"""Re-expresses the existing 5-seed results as split-level statistics.

Each seed is an independent stratified patient split, so the split is the right unit of
analysis: this reports 95% CIs over splits plus a paired comparison of GA-LDM against the
strongest baseline, rather than seed-level standard deviations alone. Input is
results/cells/{cancer}_s{seed}_{method}.json, one split per seed. Headline metrics are
Bio-FID, PCE and TSTR (all seven are computed), tested with a paired t-test on matched splits
and with Wilcoxon."""
import sys
import os, json, glob
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P

RES = _P.RESULTS_DIR
CELLS = os.path.join(RES, "cells")
CANCERS = ["CESC", "COAD", "HNSC", "KIRC"]
SEEDS = [42, 123, 456, 789, 1024]
METRICS = ["Bio_FID", "PCE", "TSTR", "MMD", "KS", "miRNA_Corr", "DE"]
LOWER_BETTER = {"Bio_FID", "PCE", "MMD"}  # metrics where lower is better


def load(cancer, method, metric):
    vals = []
    for s in SEEDS:
        f = os.path.join(CELLS, f"{cancer}_s{s}_{method}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            if metric in d and d[metric] is not None:
                vals.append(float(d[metric]))
    return np.array(vals)


def ci95(a):
    """Student-t 95% CI, appropriate at this sample size."""
    if len(a) < 2:
        return (np.nan, np.nan)
    m, se = a.mean(), stats.sem(a)
    h = se * stats.t.ppf(0.975, len(a) - 1)
    return m - h, m + h


def run():
    out = {}
    # strongest generative baseline is CTGAN; SMOTE is an interpolation reference, listed apart
    strongest = "CTGAN"
    print("=== T4 split-level statistics (5 patient-level splits) ===\n")
    for cancer in CANCERS:
        out[cancer] = {}
        print(f"--- {cancer} ---")
        for metric in ["Bio_FID", "PCE", "TSTR"]:  # headline metrics
            ga = load(cancer, "GA-LDM", metric)
            bl = load(cancer, strongest, metric)
            if len(ga) < 2 or len(bl) < 2:
                continue
            lo, hi = ci95(ga)
            # paired difference on matched splits: GA-LDM - baseline
            n = min(len(ga), len(bl))
            diff = ga[:n] - bl[:n]
            # sign convention: for lower-better metrics, GA-LDM better means diff < 0
            better = (diff < 0).sum() if metric in LOWER_BETTER else (diff > 0).sum()
            try:
                t_p = stats.ttest_rel(ga[:n], bl[:n]).pvalue
            except Exception:
                t_p = np.nan
            try:
                w_p = stats.wilcoxon(ga[:n], bl[:n]).pvalue if n >= 5 else np.nan
            except Exception:
                w_p = np.nan
            out[cancer][metric] = {
                "GA-LDM_mean": round(float(ga.mean()), 4),
                "GA-LDM_95CI": [round(float(lo), 4), round(float(hi), 4)],
                f"{strongest}_mean": round(float(bl.mean()), 4),
                "paired_diff_mean": round(float(diff.mean()), 4),
                "GA-LDM_better_splits": f"{int(better)}/{n}",
                "paired_t_p": round(float(t_p), 4),
                "wilcoxon_p": round(float(w_p), 4) if not np.isnan(w_p) else None,
            }
            print(f"  {metric:8s} GA-LDM {ga.mean():.3f} [95%CI {lo:.3f},{hi:.3f}] vs "
                  f"{strongest} {bl.mean():.3f} | diff {diff.mean():+.3f} | "
                  f"better {int(better)}/{n} | t_p={t_p:.4f}")
        print()
    with open(os.path.join(RES, "t4_split_level_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/t4_split_level_stats.json")
    return out


if __name__ == "__main__":
    run()
