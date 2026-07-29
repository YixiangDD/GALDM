# -*- coding: utf-8 -*-
"""Two comparison figures.

Shared-projection 2D grid: UMAP is fitted once on the real training data and every method's
samples are transformed into that same coordinate system, so all panels share axes and can be
compared directly. Fitting a separate projection per panel, as is often done, makes the panels
incomparable.

Within-pathway correlation heatmap grid: each panel is annotated with its PCE against real
data, so the visual impression is backed by a number.

CESC seed 42, VAE 500 / DiT 800 epochs, matching the main experiments. Writes
results/figures/CESC/_shared_umap_CESC.png and _heatmap_pce_CESC.png.
"""
import os, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.comparison import run_galdm, run_baseline, set_seed
from evaluation.metrics import pce
from configs import paths as _P

RAW = _P.RAW_ROOT
PROC = _P.PROCESSED_DIR
FIG = _P.FIGURES_DIR
METHODS = ["SMOTE", "CTGAN", "TabDDPM", "TabSyn", "scDiffusion", "GA-LDM"]  # TabDiff tracks TabDDPM closely here, omitted


def shared_umap_grid(cancer, real_m, gens, fd, real_ref=None):
    import umap
    # fitted on real training data; every panel shares this coordinate system
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    real_emb = reducer.fit_transform(real_m)
    # common axis limits
    all_pts = [real_emb]
    proj = {}
    for m, g in gens.items():
        e = reducer.transform(g)
        proj[m] = e
        all_pts.append(e)
    allp = np.vstack(all_pts)
    xlim = (allp[:, 0].min() - 1, allp[:, 0].max() + 1)
    ylim = (allp[:, 1].min() - 1, allp[:, 1].max() + 1)

    methods = list(gens.keys())
    ncol = 3
    nrow = (len(methods) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.4 * nrow))
    axes = np.array(axes).ravel()
    for ax, m in zip(axes, methods):
        ax.scatter(real_emb[:, 0], real_emb[:, 1], s=9, alpha=0.45, c="#2c7fb8", label="Real", edgecolors="none")
        ax.scatter(proj[m][:, 0], proj[m][:, 1], s=9, alpha=0.45, c="#de2d26", label="Generated", edgecolors="none")
        ax.set_title(m, fontsize=11); ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.suptitle(f"{cancer}: shared-UMAP embedding (fitted once on real data; all panels share coordinates)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(fd, f"_shared_umap_{cancer}.png")
    plt.savefig(out, dpi=160); plt.close()
    print("saved", out, flush=True)


def _corr_pw(X, idx):
    Xc = X[:, idx] - X[:, idx].mean(0, keepdims=True)
    Xn = Xc / (Xc.std(0) + 1e-8)
    return (Xn.T @ Xn) / Xn.shape[0]


def rv_mantel(Cr, Cg, n_perm=2000, seed=0):
    """Similarity of two correlation matrices; returns (RV coefficient, Mantel p-value).
    RV = <Cr,Cg>_F / (||Cr||_F ||Cg||_F) in [-1,1], where 1 means identical structure.
    Mantel: permute rows and columns together on the upper-triangular vector; p is the
    fraction of permutations whose correlation exceeds the observed one."""
    iu = np.triu_indices_from(Cr, k=1)
    vr, vg = Cr[iu], Cg[iu]
    rv = float((Cr * Cg).sum() / (np.sqrt((Cr ** 2).sum()) * np.sqrt((Cg ** 2).sum()) + 1e-12))
    obs = float(np.corrcoef(vr, vg)[0, 1]) if vr.std() > 1e-8 and vg.std() > 1e-8 else 0.0
    rng = np.random.default_rng(seed)
    p_ct, k = 1, Cr.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(k)
        vp = Cg[perm][:, perm][iu]
        rp = np.corrcoef(vr, vp)[0, 1] if vp.std() > 1e-8 else 0.0
        if rp >= obs:
            p_ct += 1
    return rv, p_ct / (n_perm + 1)


def heatmap_pce_grid(cancer, real_m, gens, pathway_mask, fd, pw_idx=0):
    idx = np.where(pathway_mask[pw_idx] > 0)[0][:40]
    Creal = _corr_pw(real_m, idx)
    panels = [("Real", real_m, None)]
    for m, g in gens.items():
        panels.append((m, g, pce(real_m, g, pathway_mask[:15])))
    n = len(panels); ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.3 * nrow))
    axes = np.array(axes).ravel()
    im = None
    rv_out = {}
    for ax, (name, X, pceval) in zip(axes, panels):
        C = _corr_pw(X, idx)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        if pceval is None:
            ttl = name
        else:
            rv, p = rv_mantel(Creal, C)
            rv_out[name] = {"PCE": round(float(pceval), 4), "RV": round(rv, 3),
                            "mantel_p": p}
            pstr = "p<0.001" if p < 1e-3 else ("p=%.3f" % p)
            ttl = f"{name}\nPCE = {pceval:.3f}  RV = {rv:.2f} ({pstr})"
        ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{cancer}: within-pathway gene-correlation "
                 f"(PCE = Frobenius distance to real, lower better; RV = matrix similarity, higher better)",
                 fontsize=10.5)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02)
    out = os.path.join(fd, f"_heatmap_pce_{cancer}.png")
    plt.savefig(out, dpi=160); plt.close()
    print("saved", out, flush=True)
    import json
    with open(os.path.join(fd, f"_rv_mantel_{cancer}.json"), "w", encoding="utf-8") as f:
        json.dump(rv_out, f, indent=2, ensure_ascii=False)
    print("RV/Mantel:", rv_out, flush=True)


def run(cancer="CESC", seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    fd = os.path.join(FIG, cancer); os.makedirs(fd, exist_ok=True)
    set_seed(seed)
    ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
    norm = ds["norm"]
    real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
    gens = {}
    for m in METHODS:
        set_seed(seed)
        if m == "GA-LDM":
            gm, gmi, gy = run_galdm(cfg, bp, ds, device, False, cfg.train.vae_epochs, cfg.train.dit_epochs)
        else:
            gm, gmi, gy = run_baseline(m, ds, ds["train"]["n_samples"], device, False, 300)
        gens[m] = gm
        print(f"  [{cancer}] {m} generated", flush=True)
    shared_umap_grid(cancer, real_m, gens, fd)
    heatmap_pce_grid(cancer, real_m, gens, bp.pathway_mask, fd)
    print(f"[{cancer}] done -> {fd}")


if __name__ == "__main__":
    run("CESC")
