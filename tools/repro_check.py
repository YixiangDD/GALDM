"""Reproducibility check: trains the full GA-LDM on one cohort/seed with the current code and
config, then prints the seven metrics next to the published reference values.

    python tools/repro_check.py                       # CESC, seed 42, full epoch budget
    python tools/repro_check.py --cancer KIRC --seed 123
    python tools/repro_check.py --vae_ep 60 --dit_ep 60   # fast sanity check, not comparable

Uses the cached dataset in data_pipeline/processed/ when present, so it runs without the raw
TCGA files. Falls back to CPU when no GPU is available, though the full budget is impractical
there -- pass --vae_ep/--dit_ep to shorten it.
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from evaluation.metrics import compute_all_metrics
from experiments.trainer import GALDMTrainer

# published Bio-FID for CESC/seed42, for orientation only
REFERENCE = "comparison table = 20.0 ; ablation table = 31.5  (CESC, full budget)"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser(description="GA-LDM reproducibility check")
    ap.add_argument("--cancer", default="CESC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vae_ep", type=int, default=0, help="override VAE epochs (0 = config)")
    ap.add_argument("--dit_ep", type=int, default=0, help="override DiT epochs (0 = config)")
    args = ap.parse_args()

    cfg = get_config()
    vae_ep = args.vae_ep or cfg.train.vae_epochs
    dit_ep = args.dit_ep or cfg.train.dit_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no GPU detected, running on CPU; the full epoch budget will be slow")

    set_seed(args.seed)
    ds, bp = prepare_cancer(_P.RAW_ROOT, _P.PROCESSED_DIR, args.cancer, args.seed, verbose=False)
    nrm = ds["norm"]
    test_m = ds["test"]["mrna"] * nrm["mrna_sd"] + nrm["mrna_mu"]
    test_y = ds["test"]["labels"]
    real_m = ds["train"]["mrna"] * nrm["mrna_sd"] + nrm["mrna_mu"]
    real_mi = ds["train"]["mirna"] * nrm["mirna_sd"] + nrm["mirna_mu"]
    real_y = ds["train"]["labels"]

    set_seed(args.seed)
    t0 = time.time()
    tr = GALDMTrainer(cfg, bp, ds, device=device)   # no ablation_flags: same path as comparison
    tr.train_vae(epochs=vae_ep, verbose=False)
    tr.train_dit(epochs=dit_ep, verbose=False)
    gm, gmi, gy = tr.generate(ds["train"]["n_samples"])
    met = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                              test_m, test_y, bp.pathway_mask, bp.mirna_family_mask)

    print("=" * 68)
    print("REPRO %s seed%d | %s | VAE %d / DiT %d" %
          (args.cancer, args.seed, device, vae_ep, dit_ep))
    for k, v in met.items():
        print("  %-12s %.4f" % (k, v))
    print("  elapsed      %.0fs" % (time.time() - t0))
    if args.cancer == "CESC" and args.seed == 42 and not (args.vae_ep or args.dit_ep):
        print("REFERENCE Bio-FID: " + REFERENCE)
    else:
        print("(no reference value for this setting)")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
