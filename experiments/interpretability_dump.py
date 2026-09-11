"""Per-sample variant of the KIRC HIF-1 interpretability case study.

Same computation as interpretability.py, with two differences:

  1. the per-sample pathway activities are written out, so the box plot in the
     supplement can be redrawn without another training run;
  2. the output goes to interpretability_<COHORT>_persample.json, so the
     published interpretability_<COHORT>.json is never overwritten.

The pathway masks used here are the same ones the encoder sees, so this is a
consistency check rather than independent biological validation.

Note: this retrains from scratch on every call and does not cache the generated
samples. KIRC at the published setting (VAE 500 / DiT 800 epochs) takes about
12 minutes on an RTX 4070 Laptop. Seed 42 reproduces bit-identical statistics.

Output: results/interpretability_<COHORT>_persample.json
        results/interpretability_<COHORT>_HIF1_persample.png
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from bio_priors.kegg_pathways import KEGG_15_PATHWAYS
from configs import paths as _P

RAW_ROOT = _P.RAW_ROOT
PROC_DIR = _P.PROCESSED_DIR
RESULTS_DIR = _P.RESULTS_DIR


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def pathway_activity(mrna, pathway_mask, pw_idx):
    """Activity of one pathway: the mean over its member genes."""
    genes = np.where(pathway_mask[pw_idx] > 0)[0]
    if len(genes) == 0:
        return np.zeros(mrna.shape[0])
    return mrna[:, genes].mean(axis=1)


def run_interpretability(cancer="KIRC", seed=42, quick=False, vae_ep=None, dit_ep=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(seed)
    ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)

    # train GA-LDM
    tr = GALDMTrainer(cfg, bp, ds, device=device)
    tr.train_vae(epochs=vae_ep or (20 if quick else cfg.train.vae_epochs), verbose=False)
    tr.train_dit(epochs=dit_ep or (30 if quick else cfg.train.dit_epochs), verbose=False)
    gm, gmi, gy = tr.generate(ds["train"]["n_samples"])
    real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
    real_y = ds["train"]["labels"]

    # locate the HIF-1 pathway index
    hsa_ids = list(KEGG_15_PATHWAYS.keys())
    hif_idx = None
    for i, h in enumerate(hsa_ids):
        if "HIF" in KEGG_15_PATHWAYS[h]:
            hif_idx = i
            break
    if hif_idx is None:
        print("HIF-1 pathway not found"); return

    # compute pathway activity
    real_act = pathway_activity(real_m, bp.pathway_mask, hif_idx)
    gen_act = pathway_activity(gm, bp.pathway_mask, hif_idx)

    # split by class
    real_early = real_act[real_y == 0]
    real_late = real_act[real_y == 1]
    gen_early = gen_act[gy == 0]
    gen_late = gen_act[gy == 1]

    # statistics
    from scipy import stats
    stat_real = stats.mannwhitneyu(real_early, real_late, alternative="two-sided")
    stat_gen = stats.mannwhitneyu(gen_early, gen_late, alternative="two-sided") if len(gen_early) > 1 and len(gen_late) > 1 else (0, 1.0)

    results = {
        "cancer": cancer,
        "pathway": "HIF-1 signaling",
        "real_activity_mean": [float(real_early.mean()), float(real_late.mean())],
        "gen_activity_mean": [float(gen_early.mean()), float(gen_late.mean())],
        "real_mannwhitney_p": float(stat_real.pvalue),
        "gen_mannwhitney_p": float(stat_gen[1]) if hasattr(stat_gen, '__len__') else float(stat_gen.pvalue),
        "activity_distribution_distance": float(np.abs(real_act.mean() - gen_act.mean())),
        # per-sample activities, so the box plot can be rebuilt without retraining
        "per_sample": {
            "real_early": [float(v) for v in real_early],
            "real_late": [float(v) for v in real_late],
            "gen_early": [float(v) for v in gen_early],
            "gen_late": [float(v) for v in gen_late],
        },
        "n": {"real_early": int(real_early.size), "real_late": int(real_late.size),
              "gen_early": int(gen_early.size), "gen_late": int(gen_late.size)},
    }
    print(f"  HIF-1 activity (real): early={real_early.mean():.3f} late={real_late.mean():.3f} p={stat_real.pvalue:.4f}")
    print(f"  HIF-1 activity (gen):  early={gen_early.mean():.3f} late={gen_late.mean():.3f}")
    print(f"  activity distance: |mean_real - mean_gen| = {results['activity_distribution_distance']:.4f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (data_e, data_l, title) in zip(axes, [
        (real_early, real_late, "Real"),
        (gen_early, gen_late, "Generated (GA-LDM)")
    ]):
        ax.hist(data_e, bins=20, alpha=0.6, label="Early (I/II)", color="#2c7fb8", density=True)
        ax.hist(data_l, bins=20, alpha=0.6, label="Late (III/IV)", color="#de2d26", density=True)
        ax.set_title(f"{title} — HIF-1 Pathway Activity")
        ax.set_xlabel("Activity"); ax.legend()
    fig.suptitle(f"{cancer}: HIF-1 Pathway Activity Distribution")
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f"interpretability_{cancer}_HIF1_persample.png")
    plt.savefig(fig_path, dpi=150); plt.close()

    # separate file: never overwrite the published interpretability_{cancer}.json
    out = os.path.join(RESULTS_DIR,
                       f"interpretability_{cancer}_persample"
                       f"{'_quick' if quick else ''}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out}, {fig_path}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", default="KIRC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vae_ep", type=int, default=0)
    ap.add_argument("--dit_ep", type=int, default=0)
    args = ap.parse_args()
    run_interpretability(args.cancer, args.seed, args.quick,
                         vae_ep=args.vae_ep or None, dit_ep=args.dit_ep or None)
