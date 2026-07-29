"""Comparison figures for every method (GA-LDM and all baselines).

Standalone script: retrains and regenerates each method's samples for a given cohort, then
saves per-method t-SNE (2D and 3D), per-feature KDE and within-pathway correlation heatmaps,
plus side-by-side multi-panel grids (t-SNE grid and pathway heatmap grid) showing how the
methods differ: SMOTE's interpolation artefacts, CTGAN's mode collapse, the over-smoothing of
the diffusion baselines, and GA-LDM.

Usage:
  python experiments/visualize_all.py --cancer CESC --seed 42
  python experiments/visualize_all.py --cancer CESC,KIRC --seed 42 --quick
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.comparison import run_galdm, run_baseline, set_seed
from evaluation.visualization import (plot_tsne, plot_tsne_3d, plot_kde, plot_corr_heatmap)
from configs import paths as _P

RAW_ROOT = _P.RAW_ROOT
PROC_DIR = _P.PROCESSED_DIR
FIG_DIR = _P.FIGURES_DIR

ALL_METHODS = ["SMOTE", "CTGAN", "TabDDPM", "TabSyn", "TabDiff", "scDiffusion", "GA-LDM"]


def _gen_for_method(method, cfg, bp, ds, device, quick, vae_ep, dit_ep, base_ep):
    n_gen = ds["train"]["n_samples"]
    if method == "GA-LDM":
        return run_galdm(cfg, bp, ds, device, quick, vae_ep, dit_ep)
    return run_baseline(method, ds, n_gen, device, quick, base_ep)


def _tsne_embed(real_m, gen_m, max_n=400, n_comp=2):
    from sklearn.manifold import TSNE
    if len(real_m) > max_n:
        real_m = real_m[np.random.choice(len(real_m), max_n, replace=False)]
    if len(gen_m) > max_n:
        gen_m = gen_m[np.random.choice(len(gen_m), max_n, replace=False)]
    X = np.vstack([real_m, gen_m])
    lab = np.array([0] * len(real_m) + [1] * len(gen_m))
    emb = TSNE(n_components=n_comp, perplexity=min(30, max(5, len(X) // 4)),
               init="pca", random_state=42).fit_transform(X)
    return emb, lab


def _corr_pw(X, idx):
    Xc = X[:, idx] - X[:, idx].mean(0, keepdims=True)
    std = Xc.std(0) + 1e-8
    Xn = Xc / std
    return (Xn.T @ Xn) / Xn.shape[0]


def visualize_cancer(cancer, seed=42, quick=False, vae_ep=None, dit_ep=None, base_ep=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    ve = vae_ep or (20 if quick else cfg.train.vae_epochs)
    de = dit_ep or (30 if quick else cfg.train.dit_epochs)
    be = base_ep or (30 if quick else 300)
    fd = os.path.join(FIG_DIR, cancer)
    os.makedirs(fd, exist_ok=True)

    set_seed(seed)
    ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
    norm = ds["norm"]
    real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]

    # collect generated samples per method
    gens = {}
    for method in ALL_METHODS:
        set_seed(seed)
        try:
            gm, gmi, gy = _gen_for_method(method, cfg, bp, ds, device, quick, ve, de, be)
            gens[method] = gm
            # single-method figures
            plot_tsne(real_m, gm, os.path.join(fd, f"tsne2d_{method}.png"),
                      f"{cancer} t-SNE: {method}")
            plot_tsne_3d(real_m, gm, os.path.join(fd, f"tsne3d_{method}.png"),
                         f"{cancer} 3D t-SNE: {method}")
            plot_kde(real_m, gm, os.path.join(fd, f"kde_{method}.png"),
                     title=f"{cancer} KDE: {method}")
            plot_corr_heatmap(real_m, gm, bp.pathway_mask,
                              os.path.join(fd, f"heatmap_{method}.png"),
                              pw_idx=0, title=f"{cancer} Pathway Corr: {method}")
            print(f"  [{cancer}] {method} figures saved", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [{cancer}] {method} failed: {e}")

    # side-by-side multi-panel figures
    _combined_tsne_grid(cancer, real_m, gens, fd)
    _combined_heatmap_grid(cancer, real_m, gens, bp.pathway_mask, fd)
    print(f"[{cancer}] all figures done -> {fd}")


def _combined_tsne_grid(cancer, real_m, gens, fd):
    """Side-by-side t-SNE for every method, with the real data as reference."""
    methods = list(gens.keys())
    n = len(methods)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
    axes = np.array(axes).ravel()
    for ax, method in zip(axes, methods):
        emb, lab = _tsne_embed(real_m, gens[method])
        ax.scatter(emb[lab == 0, 0], emb[lab == 0, 1], s=8, alpha=0.5, c="#2c7fb8", label="Real")
        ax.scatter(emb[lab == 1, 0], emb[lab == 1, 1], s=8, alpha=0.5, c="#de2d26", label="Gen")
        ax.set_title(method, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{cancer}: t-SNE of Real vs Generated (all methods)", fontsize=13)
    fig.tight_layout()
    plt.savefig(os.path.join(fd, f"_combined_tsne_{cancer}.png"), dpi=150); plt.close()


def _combined_heatmap_grid(cancer, real_m, gens, pathway_mask, fd, pw_idx=0):
    """Side-by-side within-pathway gene correlation heatmaps: real plus every method."""
    idx = np.where(pathway_mask[pw_idx] > 0)[0][:40]
    panels = [("Real", real_m)] + [(m, g) for m, g in gens.items()]
    n = len(panels)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow))
    axes = np.array(axes).ravel()
    for ax, (name, X) in zip(axes, panels):
        C = _corr_pw(X, idx)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(name, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{cancer}: Within-Pathway Gene Correlation (Real vs all methods)", fontsize=12)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02)
    plt.savefig(os.path.join(fd, f"_combined_heatmap_{cancer}.png"), dpi=150); plt.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", default="CESC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vae_ep", type=int, default=0)
    ap.add_argument("--dit_ep", type=int, default=0)
    ap.add_argument("--base_ep", type=int, default=0)
    args = ap.parse_args()
    for c in args.cancer.split(","):
        visualize_cancer(c.strip(), args.seed, args.quick,
                         vae_ep=args.vae_ep or None, dit_ep=args.dit_ep or None,
                         base_ep=args.base_ep or None)
