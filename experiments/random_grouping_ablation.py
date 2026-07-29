# -*- coding: utf-8 -*-
"""Matched random-grouping ablation — answers reviewer W4.

Tests whether the KEGG *biology* matters, or only the *fact* of grouping. We build a random
grouping with the SAME group count and SAME per-group sizes as the real KEGG pathway mask
(and the same for miRNA families), then retrain PA-VAE+GA-DiT end-to-end and compare.
If KEGG ≈ random, the contribution is 'structured grouping', not 'pathway biology'.

We evaluate on the pathway-agnostic metrics (Bio-FID, MMD, KS, DE, TSTR) plus XM-Corr.
NOTE: PCE and miRNA Corr are computed on the *real* KEGG/family masks for both conditions
(a fixed external yardstick), so they do not give the real-grouping model a structural
home-field advantage. CESC, 3 seeds."""
import os, sys, json, random, copy
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from evaluation.metrics import compute_all_metrics
from experiments.crossmodal_eval import load_mirtarbase_edges, xm_corr
from configs import paths as _P

RAW = _P.RAW_ROOT; PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def randomize_mask(mask, rng):
    """Build a random binary mask with identical per-row sizes but shuffled column membership.
    Each column (gene/miRNA) is assigned to preserve row sums; sampling without replacement
    per row over a random permutation of columns, allowing overlap like the real mask."""
    n_rows, n_cols = mask.shape
    sizes = mask.sum(1).astype(int)
    new = np.zeros_like(mask)
    for r in range(n_rows):
        cols = rng.choice(n_cols, size=sizes[r], replace=False)
        new[r, cols] = 1.0
    return new


def run(cancer="CESC", seeds=(42, 123, 456)):
    cfg = get_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"KEGG": {}, "Random": {}}
    mkeys = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID", "XM_Corr"]

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
        real_y = ds["train"]["labels"]
        test_m = ds["test"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        test_y = ds["test"]["labels"]
        n = ds["train"]["n_samples"]
        edges = load_mirtarbase_edges([str(g) for g in ds["gene_ids"]],
                                      [str(m) for m in ds["mirna_ids"]], bp.ens2sym)
        # fixed external yardstick masks (real KEGG/family) for PCE & miRNA_Corr
        eval_pm, eval_fm = bp.pathway_mask, bp.mirna_family_mask

        for cond in ["KEGG", "Random"]:
            set_seed(seed)
            bp_use = bp
            if cond == "Random":
                rng = np.random.default_rng(seed + 7000)
                bp_use = copy.copy(bp)
                bp_use.pathway_mask = randomize_mask(bp.pathway_mask, rng)
                bp_use.mirna_family_mask = randomize_mask(bp.mirna_family_mask, rng)
            tr = GALDMTrainer(cfg, bp_use, ds, device=dev)
            tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
            tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
            gm, gmi, gy = tr.generate(n)
            met = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                                      test_m, test_y, eval_pm, eval_fm)
            met["XM_Corr"] = xm_corr(real_mi, real_m, gmi, gm, edges)[0]
            for k in mkeys:
                out[cond].setdefault(k, []).append(met[k])
            print(f"[{cancer} s{seed}] {cond:7s} BioFID={met['Bio_FID']:.1f} PCE={met['PCE']:.3f} "
                  f"KS={met['KS']:.3f} miRNA={met['miRNA_Corr']:.3f} XM={met['XM_Corr']:.3f}", flush=True)
        with open(os.path.join(RES, f"random_grouping_{cancer}.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)

    print("\n===== random-grouping ablation (mean) =====")
    for cond in ["KEGG", "Random"]:
        r = out[cond]
        def mn(k):
            v = [x for x in r[k] if not (isinstance(x, float) and np.isnan(x))]
            return np.mean(v) if v else float("nan")
        print("%-7s BioFID=%.2f PCE=%.3f KS=%.3f miRNA=%.3f DE=%.3f XM=%.3f" % (
            cond, mn("Bio_FID"), mn("PCE"), mn("KS"), mn("miRNA_Corr"), mn("DE"), mn("XM_Corr")))
    return out


if __name__ == "__main__":
    run()
