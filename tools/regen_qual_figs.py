# -*- coding: utf-8 -*-
"""Regenerates samples for all 7 methods on CESC with the current code (VAE 500 / DiT 800) and
redraws the per-method qualitative figures, so they stay consistent with the main comparison
table. Overwrites results/figures/CESC/{tsne2d,tsne3d,heatmap}_<method>.png. Seed fixed at 42,
the first seed of the main run."""
import os, sys, random
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from baselines.modern import BASELINE_REGISTRY
from evaluation.visualization import plot_tsne, plot_tsne_3d, plot_corr_heatmap
from models.postprocess import postprocess
from configs import paths as _P

RAW = _P.RAW_ROOT
PROC = _P.PROCESSED_DIR
FD = os.path.join(_P.FIGURES_DIR, "CESC")
os.makedirs(FD, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


cfg = get_config()
seed = 42
set_seed(seed)
ds, bp = prepare_cancer(RAW, PROC, "CESC", seed, verbose=False)
real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
n_gen = ds["train"]["n_samples"]
device = "cuda" if torch.cuda.is_available() else "cpu"


def gen_baseline(name):
    Xm, Xmi = ds["train"]["mrna"], ds["train"]["mirna"]
    y = ds["train"]["labels"]
    X = np.concatenate([Xm, Xmi], 1); split = Xm.shape[1]
    cls = BASELINE_REGISTRY[name]; kw = {}
    if name in ("CTGAN", "TabDDPM", "TabDiff"): kw["epochs"] = 300
    m = cls(split_dim=split, device=device, **kw)
    if name in ("TabSyn", "scDiffusion"):
        m.vae_epochs = 200; m.diff_epochs = 300
    m.fit(X, y)
    g, gy = m.generate(n_gen)
    gm = g[:, :split] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
    gmi = g[:, split:] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
    gm, gmi, gy = postprocess(gm, gmi, real_m, real_mi, do_filter=False,
                              calib_strength=cfg.eval.calib_strength,
                              gen_labels=gy, real_labels=ds["train"]["labels"])
    return gm


def gen_galdm():
    set_seed(seed)
    tr = GALDMTrainer(cfg, bp, ds, device=device)
    tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
    tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
    gm, gmi, gy = tr.generate(n_gen)
    return gm


METHODS = ["GA-LDM", "SMOTE", "CTGAN", "TabDDPM", "TabSyn", "TabDiff", "scDiffusion"]
for meth in METHODS:
    set_seed(seed)
    print(f"[gen] {meth} ...", flush=True)
    gm = gen_galdm() if meth == "GA-LDM" else gen_baseline(meth)
    plot_tsne(real_m, gm, os.path.join(FD, f"tsne2d_{meth}.png"), f"{meth}")
    plot_tsne_3d(real_m, gm, os.path.join(FD, f"tsne3d_{meth}.png"), f"{meth}")
    plot_corr_heatmap(real_m, gm, bp.pathway_mask, os.path.join(FD, f"heatmap_{meth}.png"),
                      title=f"{meth}")
    print(f"[done] {meth}", flush=True)

print("ALL qualitative figs regenerated (current code, CESC seed42)")
