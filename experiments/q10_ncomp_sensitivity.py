# -*- coding: utf-8 -*-
"""Robustness of Bio-FID to the number of PCA components.

Bio-FID is a Frechet-type metric, so its absolute value necessarily moves with n_comp: more
dimensions mean more covariance terms, and in a small-sample regime the higher components are
badly estimated, which inflates the distance. The question worth answering is therefore not
whether the absolute value is stable but whether the ranking between methods is.

This computes Bio-FID for GA-LDM, the strongest generative baseline (CTGAN) and SMOTE over 5
seeds each, at n_comp in {10, 20, 50}, and checks that the ordering does not change. Original
log2 mRNA space, matching the main comparison table."""
import os, sys, json, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.comparison import run_baseline, run_galdm, set_seed
from evaluation.metrics import bio_fid
from configs import paths as _P

RAW = _P.RAW_ROOT
PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR
NCOMP_GRID = (10, 20, 50)
METHODS = ["GA-LDM", "CTGAN", "SMOTE"]


def run(cancer="CESC", seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # agg[method][nc] = list over seeds
    agg = {m: {nc: [] for nc in NCOMP_GRID} for m in METHODS}

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        norm = ds["norm"]
        real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        n = ds["train"]["n_samples"]

        for m in METHODS:
            set_seed(seed)
            if m == "GA-LDM":
                gm, gmi, gy = run_galdm(cfg, bp, ds, device)
            else:
                gm, gmi, gy = run_baseline(m, ds, n, device)
            for nc in NCOMP_GRID:
                agg[m][nc].append(float(bio_fid(real_m, gm, n_comp=nc)))
            print(f"[{cancer} s{seed}] {m:8s} " +
                  " ".join(f"nc{nc}={agg[m][nc][-1]:.2f}" for nc in NCOMP_GRID), flush=True)

    out = {m: {str(nc): [float(np.mean(v)), float(np.std(v))] for nc, v in d.items()}
           for m, d in agg.items()}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"q10_ncomp_{cancer}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n=== Q10 Bio-FID vs n_comp (mean+/-std over seeds) ===")
    print("%-8s " % "n_comp" + " ".join("%12s" % m for m in METHODS))
    for nc in NCOMP_GRID:
        row = "%-8d " % nc + " ".join("%6.1f+/-%-4.1f" % (out[m][str(nc)][0], out[m][str(nc)][1]) for m in METHODS)
        print(row)
    # ranking check
    print("\n=== ranking check (GA-LDM best at every n_comp?) ===")
    for nc in NCOMP_GRID:
        vals = {m: out[m][str(nc)][0] for m in METHODS}
        best = min(vals, key=vals.get)
        print("  n_comp=%2d: GA-LDM=%.1f  CTGAN=%.1f  SMOTE=%.1f  -> best=%s" %
              (nc, vals["GA-LDM"], vals["CTGAN"], vals["SMOTE"], best))
    return out


if __name__ == "__main__":
    t0 = time.time()
    run("CESC")
    print(f"\n[done] {time.time()-t0:.0f}s")
