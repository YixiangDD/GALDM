"""Extreme small-sample ablation: does the inductive bias of GDVD and PCA-CTG show up when
data is scarce?

The CESC training split is stratified-downsampled to N=40/60/100/full and Full is compared
against w/o_both (GDVD and PCA-CTG removed). A prior should matter most when data is scarce, so
if Full beats w/o_both at small N the two modules earn their place in the regime this work
targets. Each size runs over several seeds, since downsampling is noisy.
"""
import os, sys, json, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from bio_priors.assemble import load_bio_priors
from data_pipeline.data_loader import load_cached
from experiments.trainer import GALDMTrainer
from evaluation.metrics import compute_all_metrics, regulatory_direction_consistency
from configs import paths as _P

PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def subsample_train(ds, n_target, seed):
    """Stratified downsampling of the training split to about n_target; val/test unchanged.
    Returns a shallow copy of the dataset."""
    rng = np.random.RandomState(seed)
    y = ds["train"]["labels"]
    classes = np.unique(y)
    keep = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        n_c = max(2, int(round(n_target * len(idx_c) / len(y))))  # at least 2 per class
        n_c = min(n_c, len(idx_c))
        keep.extend(rng.choice(idx_c, n_c, replace=False).tolist())
    keep = np.array(sorted(keep))
    new = copy.copy(ds)
    new["train"] = {
        "mrna": ds["train"]["mrna"][keep],
        "mirna": ds["train"]["mirna"][keep],
        "labels": ds["train"]["labels"][keep],
        "n_samples": len(keep),
    }
    return new


def run(cancer="CESC", sizes=(40, 60, 100, 0), seeds=(42, 123, 456),
        vae_ep=300, dit_ep=600):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bp = load_bio_priors(PROC, cancer, 42)
    base = load_cached(PROC, cancer, 42)
    norm = base["norm"]
    real_mi_full = base["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
    test_m = base["test"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
    test_y = base["test"]["labels"]
    results = {}

    for N in sizes:
        tag = f"N={N if N else 'full'}"
        results[tag] = {"Full": {}, "w/o_both": {}}
        for variant, flags in [("Full", {}), ("w/o_both", {"use_gdvd": False, "use_pcactg": False})]:
            for seed in seeds:
                set_seed(seed)
                ds = base if N == 0 else subsample_train(base, N, seed)
                real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
                real_mi = ds["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
                try:
                    tr = GALDMTrainer(get_config(), bp, ds, device=device, ablation_flags=flags)
                    tr.train_vae(epochs=vae_ep, verbose=False)
                    tr.train_dit(epochs=dit_ep, verbose=False)
                    gm, gmi, gy = tr.generate(max(ds["train"]["n_samples"], 150))
                    met = compute_all_metrics(real_m, real_mi, ds["train"]["labels"],
                                              gm, gmi, gy, test_m, test_y,
                                              bp.pathway_mask, bp.mirna_family_mask)
                    rdc = regulatory_direction_consistency(gm, gmi, bp.grn,
                                                           bp.pathway_mask, bp.mirna_family_mask)
                    met.update(rdc)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    met = {}
                for k, v in met.items():
                    results[tag][variant].setdefault(k, []).append(v)
                print(f"[{tag} {variant} s{seed}] BioFID={met.get('Bio_FID',float('nan')):.1f} "
                      f"TSTR={met.get('TSTR',float('nan')):.3f} RDC={met.get('RDC_acc',float('nan')):.3f}",
                      flush=True)
        # print the Full vs w/o_both means at this size
        print(f"\n--- {tag} means ---")
        for mt in ["Bio_FID", "TSTR", "miRNA_Corr", "PCE", "RDC_acc"]:
            fa = np.nanmean(results[tag]["Full"].get(mt, [np.nan]))
            wo = np.nanmean(results[tag]["w/o_both"].get(mt, [np.nan]))
            print(f"  {mt}: Full={fa:.3f}  w/o_both={wo:.3f}  diff={fa-wo:+.3f}")
        print()

    with open(os.path.join(RES, f"smallsample_ablation_{cancer}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", default="CESC")
    ap.add_argument("--sizes", default="40,60,100,0")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--vae_ep", type=int, default=300)
    ap.add_argument("--dit_ep", type=int, default=600)
    args = ap.parse_args()
    sizes = tuple(int(x) for x in args.sizes.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    run(args.cancer, sizes, seeds, args.vae_ep, args.dit_ep)
