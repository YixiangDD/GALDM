# -*- coding: utf-8 -*-
"""Baseline cross-modal correlation reference row (CPU only).

SMOTE interpolates real paired vectors in the concatenated mRNA||miRNA space, so it preserves
sample-level cross-modal pairing by construction and its XM-Corr should sit close to real. The
price is that it copies real pairings and adds no novelty, which is what the novelty analysis
measures. Computes SMOTE's XM-Corr per cohort, reusing the edge and correlation logic from
crossmodal_eval. Writes results/xmcorr_baseline.json.
"""
import os, sys, json, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.crossmodal_eval import load_mirtarbase_edges, xm_corr
from baselines.classical import SMOTEBaseline
from configs import paths as _P

PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def run(cancers=("CESC", "COAD", "HNSC", "KIRC")):
    out = {}
    for cancer in cancers:
        npz = np.load(os.path.join(PROC, f"{cancer}_seed42.npz"), allow_pickle=True)
        bp = pickle.load(open(os.path.join(PROC, f"{cancer}_seed42_priors.pkl"), "rb"))
        gene_ids = [str(g) for g in npz["gene_ids"]]
        mirna_ids = [str(m) for m in npz["mirna_ids"]]
        # standardized train (SMOTE fits here, same as comparison.py)
        Xm_s = npz["train_mrna"]; Xmi_s = npz["train_mirna"]
        y = npz["train_y"]
        mu_m, sd_m = npz["mrna_mu"], npz["mrna_sd"]
        mu_mi, sd_mi = npz["mirna_mu"], npz["mirna_sd"]
        real_m = Xm_s * sd_m + mu_m
        real_mi = Xmi_s * sd_mi + mu_mi
        split = Xm_s.shape[1]

        edges = load_mirtarbase_edges(gene_ids, mirna_ids, bp.ens2sym)

        # SMOTE generates in standardised space; de-standardise back to the original
        np.random.seed(42)
        X = np.concatenate([Xm_s, Xmi_s], 1)
        sm = SMOTEBaseline(split_dim=split, device="cpu")
        sm.fit(X, y)
        n = Xm_s.shape[0]
        gs, _ = sm.generate(n)
        sm_m = gs[:, :split] * sd_m + mu_m
        sm_mi = gs[:, split:] * sd_mi + mu_mi

        xc, _, _ = xm_corr(real_mi, real_m, sm_mi, sm_m, edges)
        out[cancer] = {"n_edges": len(edges), "SMOTE_XM_Corr": round(float(xc), 4)}
        print(f"[{cancer}] edges={len(edges)} SMOTE XM-Corr={xc:.4f}", flush=True)

    with open(os.path.join(RES, "xmcorr_baseline.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
