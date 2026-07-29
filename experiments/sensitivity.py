"""Hyper-parameter sensitivity analysis.

Sweeps the key hyper-parameters on CESC and records the effect on Bio-FID, TSTR and
miRNA_Corr:
- latent width latent_dim: {128, 256, 512}
- CFG guidance strength w: {1.0, 1.5, 2.0, 2.5}
- GDVD depth-offset scale: {0.5, 1.0, 2.0}
- consistency loss weight lambda_corr: {0.02, 0.05, 0.1, 0.2}
"""
import os
import sys
import json
import copy
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
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


SENSITIVITY_SWEEPS = {
    "latent_dim": [128, 256, 512],
    "cfg_scale": [1.0, 1.5, 2.0, 2.5],
    "gdvd_depth_scale": [0.5, 1.0, 2.0],
    "lambda_corr": [0.02, 0.05, 0.1, 0.2],
}


def run_sensitivity(cancer="CESC", seed=42, quick=False, vae_ep=None, dit_ep=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    for param_name, values in SENSITIVITY_SWEEPS.items():
        results[param_name] = {}
        for val in values:
            set_seed(seed)
            cfg = get_config()
            # set the hyper-parameter under test
            if param_name == "latent_dim":
                # the mRNA latent and the cross-modal shared space must have equal width for
                # bidirectional attention, so they are set together
                cfg.pavae.mrna_latent_dim = val
                cfg.pavae.cross_modal_dim = val
            elif param_name == "cfg_scale":
                cfg.diffusion.cfg_scale = val
            elif param_name == "gdvd_depth_scale":
                cfg.gdvd.depth_offset_scale = val
            elif param_name == "lambda_corr":
                cfg.consistency.lambda_corr = val

            ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
            try:
                tr = GALDMTrainer(cfg, bp, ds, device=device)
                tr.train_vae(epochs=vae_ep or (20 if quick else 200), verbose=False)
                tr.train_dit(epochs=dit_ep or (30 if quick else 400), verbose=False)
                gm, gmi, gy = tr.generate(ds["train"]["n_samples"])
                real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
                real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
                test_m = ds["test"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
                test_y = ds["test"]["labels"]
                met = compute_all_metrics(real_m, real_mi, ds["train"]["labels"],
                                          gm, gmi, gy, test_m, test_y,
                                          bp.pathway_mask, bp.mirna_family_mask)
            except Exception as e:
                import traceback; traceback.print_exc()
                met = {k: float("nan") for k in ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]}
            results[param_name][str(val)] = met
            print(f"  [{param_name}={val}] Bio-FID={met['Bio_FID']:.1f} TSTR={met['TSTR']:.3f} "
                  f"miRNA={met['miRNA_Corr']:.3f}")

    out = os.path.join(RESULTS_DIR, f"sensitivity_{cancer}{'_quick' if quick else ''}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", default="CESC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vae_ep", type=int, default=0)
    ap.add_argument("--dit_ep", type=int, default=0)
    args = ap.parse_args()
    run_sensitivity(args.cancer, args.seed, args.quick,
                    vae_ep=args.vae_ep or None, dit_ep=args.dit_ep or None)
