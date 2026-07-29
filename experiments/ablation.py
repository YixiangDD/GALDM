"""Ablations: progressive accumulation, and leave-one-out from the full model.

Table A, progressive accumulation (M0-M5):
  M0: plain VAE plus MLP denoiser (baseline)
  M1: + PA-VAE (pathway/family-aware encoding)
  M2: + GA-DiT (replaces the MLP, single global SNR)
  M3: + GDVD (per-pathway SNR from the GRN causal-depth prior)
  M4: + PCA-CTG (causal attention mask and hinge loss)
  M5: + empirical consistency losses = full GA-LDM

Table B, leave-one-out from the full model:
  Full:                 complete GA-LDM
  w/o PA-VAE:           back to a plain VAE encoder
  w/o GDVD:             back to a single global SNR
  w/o PCA-CTG:          drop the causal mask and hinge loss
  w/o cross-modal attn: encode mRNA and miRNA independently, no interaction
  w/o consistency loss: drop top-K correlation and pathway-activity matching
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from evaluation.metrics import compute_all_metrics
from configs import paths as _P

RAW_ROOT = _P.RAW_ROOT
PROC_DIR = _P.PROCESSED_DIR
RESULTS_DIR = _P.RESULTS_DIR


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# progressive-accumulation variants
PROGRESSIVE_VARIANTS = {
    "M0": dict(use_pavae=False, use_gadit=False, use_gdvd=False, use_pcactg=False,
               use_consistency=False, use_crossmodal=False),
    "M1": dict(use_pavae=True, use_gadit=False, use_gdvd=False, use_pcactg=False,
               use_consistency=False, use_crossmodal=True),
    "M2": dict(use_pavae=True, use_gadit=True, use_gdvd=False, use_pcactg=False,
               use_consistency=False, use_crossmodal=True),
    "M3": dict(use_pavae=True, use_gadit=True, use_gdvd=True, use_pcactg=False,
               use_consistency=False, use_crossmodal=True),
    "M4": dict(use_pavae=True, use_gadit=True, use_gdvd=True, use_pcactg=True,
               use_consistency=False, use_crossmodal=True),
    "M5": dict(use_pavae=True, use_gadit=True, use_gdvd=True, use_pcactg=True,
               use_consistency=True, use_crossmodal=True),  # Full
}

# leave-one-out from the full model
LEAVE_ONE_OUT_VARIANTS = {
    "Full": dict(),
    "w/o_PA-VAE": dict(use_pavae=False),
    "w/o_GDVD": dict(use_gdvd=False),
    "w/o_PCA-CTG": dict(use_pcactg=False),
    "w/o_CrossModal": dict(use_crossmodal=False),
    "w/o_Consistency": dict(use_consistency=False),
}


def run_ablation(cancers, seeds, mode="progressive", quick=False,
                 vae_ep=None, dit_ep=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    variants = PROGRESSIVE_VARIANTS if mode == "progressive" else LEAVE_ONE_OUT_VARIANTS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for cancer in cancers:
        all_results[cancer] = {v: {} for v in variants}
        for seed in seeds:
            set_seed(seed)
            ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
            test_m = ds["test"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
            test_y = ds["test"]["labels"]
            real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
            real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
            real_y = ds["train"]["labels"]

            for vname, flags in variants.items():
                set_seed(seed)
                t0 = time.time()
                try:
                    tr = GALDMTrainer(cfg, bp, ds, device=device, ablation_flags=flags)
                    tr.train_vae(epochs=vae_ep or (20 if quick else cfg.train.vae_epochs),
                                 verbose=False)
                    # M0 sets use_gadit=False, so the trainer swaps the DiT for a plain MLP;
                    # the same code path is used and the flags decide the behaviour
                    tr.train_dit(epochs=dit_ep or (30 if quick else cfg.train.dit_epochs),
                                 verbose=False)
                    gm, gmi, gy = tr.generate(ds["train"]["n_samples"])
                    met = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                                              test_m, test_y, bp.pathway_mask,
                                              bp.mirna_family_mask)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    met = {k: float("nan") for k in
                           ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]}
                for k, v in met.items():
                    all_results[cancer][vname].setdefault(k, []).append(v)
                print(f"[{cancer} s{seed}] {vname:15s} "
                      f"MMD={met['MMD']:.3f} BioFID={met['Bio_FID']:.1f} ({time.time()-t0:.0f}s)")

        out = os.path.join(RESULTS_DIR, f"ablation_{mode}_{cancer}{'_quick' if quick else ''}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results[cancer], f, indent=2, default=str)
    return all_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", default="CESC")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--mode", default="progressive", choices=["progressive", "leave_one_out", "both"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vae_ep", type=int, default=0)
    ap.add_argument("--dit_ep", type=int, default=0)
    args = ap.parse_args()
    cancers = [c.strip() for c in args.cancers.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    modes = ["progressive", "leave_one_out"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"\n{'='*50} ABLATION: {mode} {'='*50}")
        run_ablation(cancers, seeds, mode, args.quick,
                     vae_ep=args.vae_ep or None, dit_ep=args.dit_ep or None)
