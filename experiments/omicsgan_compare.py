# -*- coding: utf-8 -*-
"""Compare GA-LDM against omicsGAN (Ahmed et al., Bioinformatics 2022).

omicsGAN transforms observed profiles (output row i <-> input patient i) rather
than sampling new patients, so it is evaluated in the only way that is
meaningful for it: its output is scored with the identical 7-metric harness plus
XM-Corr, and its per-sample proximity to the training data is measured with the
same novelty audit used for SMOTE.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from evaluation.metrics import compute_all_metrics
from experiments.comparison import set_seed, RAW_ROOT, PROC_DIR, RESULTS_DIR
from experiments.crossmodal_eval import load_mirtarbase_edges, xm_corr
from baselines.omicsgan import OmicsGAN, INFO
from models.postprocess import postprocess


def build_bipartite(bp, gene_ids, mirna_ids):
    """(n_mirna, n_gene) binary miRNA-target indicator from miRTarBase."""
    edges = load_mirtarbase_edges(gene_ids, mirna_ids, bp.ens2sym)
    A = np.zeros((len(mirna_ids), len(gene_ids)), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
    return A, edges


def novelty_stats(gen, real_train, real_test):
    """Median NN distance to train, as a fraction of a held-out real sample's own
    NN distance (same definition as the paper's novelty audit)."""
    from scipy.spatial.distance import cdist
    mu, sd = real_train.mean(0), real_train.std(0) + 1e-8
    g = (gen - mu) / sd
    tr = (real_train - mu) / sd
    te = (real_test - mu) / sd
    d_gen = cdist(g, tr).min(1)
    d_real = cdist(te, tr).min(1)
    ref = float(np.median(d_real))
    return {"median_frac": float(np.median(d_gen) / ref),
            "min_dist": float(d_gen.min()),
            "real_ref_median": ref,
            "dup_rate_half": float((d_gen < 0.5 * ref).mean())}


def run(cancers, seeds, K=3, epochs=600):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"_info": INFO}
    for cancer in cancers:
        rows = {}
        for seed in seeds:
            set_seed(seed)
            ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
            nm = ds["norm"]
            real_m = ds["train"]["mrna"] * nm["mrna_sd"] + nm["mrna_mu"]
            real_mi = ds["train"]["mirna"] * nm["mirna_sd"] + nm["mirna_mu"]
            real_y = ds["train"]["labels"]
            test_m = ds["test"]["mrna"] * nm["mrna_sd"] + nm["mrna_mu"]
            test_y = ds["test"]["labels"]

            gene_ids = [str(g) for g in ds["gene_ids"]]
            mirna_ids = [str(m) for m in ds["mirna_ids"]]
            A, edges = build_bipartite(bp, gene_ids, mirna_ids)
            print(f"[{cancer} s{seed}] bipartite edges among selected features: {int(A.sum())}")

            model = OmicsGAN(A, K=K, epochs=epochs, device=device, seed=seed)
            model.fit_transform(real_m, real_mi)

            # Evaluate EVERY update k and keep the best by Bio-FID. The authors
            # themselves select k by downstream AUC; giving omicsGAN an oracle
            # choice of k on the target metric is deliberately generous.
            best, best_k = None, None
            for k_i, (cm, cmi) in enumerate(model.per_update, start=1):
                gy = real_y.copy()          # row-aligned by construction
                gm, gmi, gy = postprocess(cm.copy(), cmi.copy(), real_m, real_mi,
                                          do_filter=False,
                                          calib_strength=get_config().eval.calib_strength,
                                          gen_labels=gy, real_labels=real_y)
                mk = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                                         test_m, test_y, bp.pathway_mask,
                                         bp.mirna_family_mask)
                if len(edges) >= 5:
                    xc, _, _ = xm_corr(real_mi, real_m, gmi, gm, edges)
                    mk["XM_Corr"] = float(xc)
                mk.update({"novelty_" + kk: vv for kk, vv in
                           novelty_stats(gm, real_m, test_m).items()})
                print(f"    k={k_i}: BioFID={mk['Bio_FID']:.1f} KS={mk['KS']:.3f} "
                      f"TSTR={mk['TSTR']:.3f}")
                if best is None or mk["Bio_FID"] < best["Bio_FID"]:
                    best, best_k = mk, k_i
            met = best
            met["best_k"] = best_k
            for k, v in met.items():
                rows.setdefault(k, []).append(v)
            print(f"[{cancer} s{seed}] omicsGAN " +
                  " ".join(f"{k}={met[k]:.3f}" for k in
                           ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR"]) +
                  f" BioFID={met['Bio_FID']:.2f}" +
                  (f" XM={met.get('XM_Corr', float('nan')):.3f}" if 'XM_Corr' in met else "") +
                  f" novelty={met['novelty_median_frac']:.3f}")

        out[cancer] = {k: {"vals": v, "mean": float(np.mean(v)),
                           "std": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                       for k, v in rows.items()}
        p = os.path.join(RESULTS_DIR, "omicsgan_compare.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {p}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", default="CESC,COAD,HNSC,KIRC")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    run(a.cancers.split(","), [int(s) for s in a.seeds.split(",")], a.K, a.epochs)
