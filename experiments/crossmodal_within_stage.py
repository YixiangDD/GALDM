# -*- coding: utf-8 -*-
"""Within-stage cross-modal permutation test: addresses reviewer W2.

The concern: shuffling mRNA-miRNA pairing ACROSS classes destroys both the generator's
coupling AND the shared early/late stage signal, so a near-zero shuffled XM-Corr could be a
stage artifact rather than genuine cross-modal coupling. Fix: permute pairing ONLY within the
same stage label, so the class signal is preserved and only the sample-level pairing is broken.
If XM-Corr still collapses under within-stage permutation, the coupling is genuinely
sample-level, not a class-conditional-mean artifact.

Runs on saved GA-LDM samples (results/generated_<C>.xlsx). No retraining."""
import os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.crossmodal_eval import load_mirtarbase_edges, edge_crosscorr

import pickle
from configs import paths as _P

PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def within_stage_perm(real_mi, real_m, gen_mi, gen_m, gy, edges, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    cr = edge_crosscorr(real_mi, real_m, edges)
    cg_obs = edge_crosscorr(gen_mi, gen_m, edges)
    obs = float(np.corrcoef(cr, cg_obs)[0, 1])
    gy = np.asarray(gy)
    classes = np.unique(gy)
    null = np.empty(B)
    for b in range(B):
        perm = np.arange(len(gy))
        for c in classes:                       # permute pairing only within each stage
            idx = np.where(gy == c)[0]
            perm[idx] = rng.permutation(idx)
        cg = edge_crosscorr(gen_mi[perm], gen_m, edges)
        null[b] = np.corrcoef(cr, cg)[0, 1] if np.std(cg) > 1e-8 else 0.0
    p = float((np.sum(null >= obs) + 1) / (B + 1))
    return obs, float(null.mean()), float(null.std()), p


def main(cancers=("CESC", "COAD", "HNSC", "KIRC")):
    out = {}
    for c in cancers:
        npz = np.load(os.path.join(PROC, f"{c}_seed42.npz"), allow_pickle=True)
        bp = pickle.load(open(os.path.join(PROC, f"{c}_seed42_priors.pkl"), "rb"))
        real_m = npz["train_mrna"] * npz["mrna_sd"] + npz["mrna_mu"]
        real_mi = npz["train_mirna"] * npz["mirna_sd"] + npz["mirna_mu"]
        edges = load_mirtarbase_edges([str(g) for g in npz["gene_ids"]],
                                      [str(m) for m in npz["mirna_ids"]], bp.ens2sym)
        xl = pd.ExcelFile(os.path.join(RES, f"generated_{c}.xlsx"))
        gdf = xl.parse("mRNA")
        gm = gdf.iloc[:, 2:].values.astype(np.float64)
        gy = gdf["label"].values.astype(int)
        gmi = xl.parse("miRNA").iloc[:, 2:].values.astype(np.float64)
        obs, nmean, nstd, p = within_stage_perm(real_mi, real_m, gmi, gm, gy, edges)
        out[c] = {"n_edges": len(edges), "XM_Corr": round(obs, 4),
                  "within_stage_null_mean": round(nmean, 4),
                  "within_stage_null_std": round(nstd, 4), "p_value": p}
        print(f"[{c}] XM-Corr={obs:.3f} within-stage null={nmean:.3f}±{nstd:.3f} p={p:.4f}")
    with open(os.path.join(RES, "crossmodal_within_stage.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
