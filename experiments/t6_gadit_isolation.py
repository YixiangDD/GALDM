# -*- coding: utf-8 -*-
"""Parameter-matched single-factor control isolating the contribution of pathway/family
identity encoding in GA-DiT.

GA-DiT (use_pathway_identity=True) vs a plain DiT (False: no pathway identity in the position
encoding, sinusoidal only). joint_dim, n_tokens, depth, heads, training, sampling and
calibration are all identical, and parameter counts match because the identity embedding
weights still exist, they are simply unused.

GDVD and PCA-CTG are switched off so no unvalidated module confounds the comparison.
CESC, 5 seeds, compute_all_metrics plus XM-Corr."""
import os, sys, json, random, time
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


def run(cancer="CESC", seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mkeys = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID", "XM_Corr"]
    # both arms: PA-VAE + GA-DiT + cross-modal on, GDVD/PCA-CTG off; only pathway identity varies
    conds = {"GA-DiT": dict(use_gdvd=False, use_pcactg=False, use_consistency=False,
                            use_pathway_identity=True),
             "plain_DiT": dict(use_gdvd=False, use_pcactg=False, use_consistency=False,
                               use_pathway_identity=False)}
    agg = {c: {k: [] for k in mkeys} for c in conds}

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        norm = ds["norm"]
        real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        real_mi = ds["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
        real_y = ds["train"]["labels"]
        test_m = ds["test"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        test_y = ds["test"]["labels"]
        n = ds["train"]["n_samples"]
        edges = load_mirtarbase_edges([str(g) for g in ds["gene_ids"]],
                                      [str(m) for m in ds["mirna_ids"]], bp.ens2sym)
        for cname, flags in conds.items():
            set_seed(seed)
            tr = GALDMTrainer(cfg, bp, ds, device=dev, ablation_flags=flags)
            tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
            tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
            gm, gmi, gy = tr.generate(n)
            met = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                                      test_m, test_y, bp.pathway_mask, bp.mirna_family_mask)
            met["XM_Corr"] = xm_corr(real_mi, real_m, gmi, gm, edges)[0]
            for k in mkeys:
                agg[cname][k].append(float(met[k]))
            print("[%s s%d] %-9s BioFID=%.2f PCE=%.3f KS=%.3f miRNA=%.3f XM=%.3f"
                  % (cancer, seed, cname, met["Bio_FID"], met["PCE"], met["KS"],
                     met["miRNA_Corr"], met["XM_Corr"]), flush=True)
        with open(os.path.join(RES, "t6_gadit_%s.json" % cancer), "w") as f:
            json.dump(agg, f, indent=2)

    print("\n===== T6 GA-DiT vs plain DiT (mean+/-std) =====")
    for c in conds:
        r = agg[c]
        print("%-9s BioFID=%.2f+/-%.2f PCE=%.3f miRNA=%.3f KS=%.3f XM=%.3f"
              % (c, np.mean(r["Bio_FID"]), np.std(r["Bio_FID"]), np.mean(r["PCE"]),
                 np.mean(r["miRNA_Corr"]), np.mean(r["KS"]), np.mean(r["XM_Corr"])))
    fa = np.array(agg["GA-DiT"]["Bio_FID"]); pa = np.array(agg["plain_DiT"]["Bio_FID"])
    print("GA-DiT - plain on Bio-FID: mean %+.2f, GA-DiT better %d/%d seeds"
          % ((fa - pa).mean(), int((fa < pa).sum()), len(fa)))
    return agg


if __name__ == "__main__":
    t0 = time.time()
    c = sys.argv[1] if len(sys.argv) > 1 else "CESC"
    run(c)
    print("\n[done] %.0fs" % (time.time() - t0))
