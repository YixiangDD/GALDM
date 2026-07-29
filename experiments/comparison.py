"""Comparison run: 6 baselines plus GA-LDM, over all seeds, 7 metrics, paired significance.

Fairness: every method trains on the same train/test split. Baselines receive the
standardised [mRNA|miRNA] matrix and their output is de-standardised with training-split
statistics back into the original log2 space, so all methods are scored in the same space.
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
from baselines.modern import BASELINE_REGISTRY
from evaluation.metrics import compute_all_metrics, paired_wilcoxon
from configs import paths as _P

RAW_ROOT = _P.RAW_ROOT
PROC_DIR = _P.PROCESSED_DIR
RESULTS_DIR = _P.RESULTS_DIR


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _destandardize(g_std, mu, sd, split):
    """Baselines generate in standardised space; split into mRNA/miRNA and de-standardise each
    back to the original log2 space."""
    gm = g_std[:, :split] * sd["mrna_sd"] + sd["mrna_mu"]
    gmi = g_std[:, split:] * sd["mirna_sd"] + sd["mirna_mu"]
    return gm, gmi


def _clip_to_real(gen, real, low=0.5, high=99.5):
    """Clips generated values to the real training-split quantile range, so an under-trained
    baseline cannot blow up a metric with extreme outliers. Applied identically to every
    method, GA-LDM included."""
    lo = np.percentile(real, low, axis=0)
    hi = np.percentile(real, high, axis=0)
    return np.clip(gen, lo, hi)


def run_baseline(name, ds, n_gen, device, quick=False, ep_override=None):
    Xm, Xmi = ds["train"]["mrna"], ds["train"]["mirna"]
    y = ds["train"]["labels"]
    X = np.concatenate([Xm, Xmi], 1)
    split = Xm.shape[1]
    cls = BASELINE_REGISTRY[name]
    kw = {}
    ep = ep_override if ep_override else (30 if quick else 300)
    if name in ("CTGAN", "TabDDPM", "TabDiff"):
        kw["epochs"] = ep
    m = cls(split_dim=split, device=device, **kw)
    if name in ("TabSyn", "scDiffusion"):
        m.vae_epochs = (ep_override or (20 if quick else 200))
        m.diff_epochs = ep if ep_override else (30 if quick else 300)
    m.fit(X, y)
    g, gy = m.generate(n_gen)
    gm, gmi = _destandardize(g, ds["norm"], ds["norm"], split)
    # same post-processing for baselines and GA-LDM: clipping plus per-class quantile calibration
    from models.postprocess import postprocess
    real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
    real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
    gm, gmi, gy = postprocess(gm, gmi, real_m, real_mi,
                              do_filter=False,  # baselines have no quality score, so skip cosine filtering rather than penalise them
                              calib_strength=get_config().eval.calib_strength,
                              gen_labels=gy, real_labels=ds["train"]["labels"])
    return gm, gmi, gy


def run_galdm(cfg, bp, ds, device, quick=False, vae_ep=None, dit_ep=None):
    tr = GALDMTrainer(cfg, bp, ds, device=device)
    tr.train_vae(epochs=vae_ep or (20 if quick else cfg.train.vae_epochs), verbose=not quick)
    tr.train_dit(epochs=dit_ep or (30 if quick else cfg.train.dit_epochs), verbose=not quick)
    n_gen = ds["train"]["n_samples"]
    gm, gmi, gy = tr.generate(n_gen)
    return gm, gmi, gy


def run_comparison(cancers, seeds, quick=False, methods=None,
                   vae_ep=None, dit_ep=None, base_ep=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    methods = methods or (cfg.eval.baselines + ["GA-LDM"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}  # {cancer: {method: {metric: [over seeds]}}}

    for cancer in cancers:
        all_results[cancer] = {m: {} for m in methods}
        for seed in seeds:
            set_seed(seed)
            ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
            test_m, test_mi, test_y = (
                ds["test"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"],
                None, ds["test"]["labels"])
            real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
            real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
            real_y = ds["train"]["labels"]
            n_gen = ds["train"]["n_samples"]

            for method in methods:
                t0 = time.time()
                # resumable: each (cohort, seed, method) result is cached separately and skipped
                # if already present
                cell_path = os.path.join(
                    RESULTS_DIR, "cells",
                    f"{cancer}_s{seed}_{method}{'_quick' if quick else ''}.json")
                if os.path.exists(cell_path):
                    with open(cell_path, encoding="utf-8") as f:
                        met = json.load(f)
                    for k, v in met.items():
                        all_results[cancer][method].setdefault(k, []).append(v)
                    print(f"[{cancer} s{seed}] {method:12s} (cached, skipped)")
                    continue
                try:
                    set_seed(seed)
                    if method == "GA-LDM":
                        gm, gmi, gy = run_galdm(cfg, bp, ds, device, quick, vae_ep, dit_ep)
                    else:
                        gm, gmi, gy = run_baseline(method, ds, n_gen, device, quick, base_ep)
                    met = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                                              test_m, test_y, bp.pathway_mask, bp.mirna_family_mask)
                    # on the first seed, save GA-LDM figures and generated samples
                    if method == "GA-LDM" and seed == seeds[0]:
                        _save_figures_and_samples(cancer, gm, gmi, gy, real_m, real_mi,
                                                  bp, ds, quick)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    met = {k: float("nan") for k in
                           ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]}
                # write the cell cache
                os.makedirs(os.path.dirname(cell_path), exist_ok=True)
                with open(cell_path, "w", encoding="utf-8") as f:
                    json.dump(met, f, default=str)
                for k, v in met.items():
                    all_results[cancer][method].setdefault(k, []).append(v)
                print(f"[{cancer} s{seed}] {method:12s} "
                      f"MMD={met['MMD']:.3f} KS={met['KS']:.3f} miRNA={met['miRNA_Corr']:.3f} "
                      f"PCE={met['PCE']:.3f} DE={met['DE']:.3f} TSTR={met['TSTR']:.3f} "
                      f"BioFID={met['Bio_FID']:.2f} ({time.time()-t0:.0f}s)")

        # save this cohort's results
        out = os.path.join(RESULTS_DIR, f"comparison_{cancer}{'_quick' if quick else ''}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results[cancer], f, indent=2, default=str)
        _print_summary(cancer, all_results[cancer])

    # export Excel and run significance tests
    _export_all(all_results, methods, quick)
    return all_results


FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def _save_figures_and_samples(cancer, gm, gmi, gy, real_m, real_mi, bp, ds, quick):
    """Makes the t-SNE, KDE and pathway heatmap figures for GA-LDM and exports the samples."""
    from evaluation.visualization import plot_tsne, plot_tsne_3d, plot_kde, plot_corr_heatmap
    from evaluation.export import export_generated_samples
    sfx = "_quick" if quick else ""
    fd = os.path.join(FIG_DIR, cancer)
    try:
        plot_tsne(real_m, gm, os.path.join(fd, f"tsne2d{sfx}.png"), f"{cancer} t-SNE (2D)")
        plot_tsne_3d(real_m, gm, os.path.join(fd, f"tsne3d{sfx}.png"), f"{cancer} t-SNE (3D)")
        plot_kde(real_m, gm, os.path.join(fd, f"kde{sfx}.png"), title=f"{cancer} Feature KDE")
        plot_corr_heatmap(real_m, gm, bp.pathway_mask, os.path.join(fd, f"pathway_heatmap{sfx}.png"),
                          pw_idx=0, title=f"{cancer} Pathway Correlation")
        export_generated_samples(gm, gmi, gy, ds["gene_ids"], ds["mirna_ids"],
                                 os.path.join(RESULTS_DIR, f"generated_{cancer}{sfx}.xlsx"))
    except Exception as e:
        print(f"  [fig] {cancer} figure/export step failed: {e}")


def _export_all(all_results, methods, quick):
    from evaluation.export import export_comparison_excel, export_significance_excel
    from evaluation.metrics import paired_wilcoxon
    sfx = "_quick" if quick else ""
    export_comparison_excel(all_results, os.path.join(RESULTS_DIR, f"comparison_results{sfx}.xlsx"))
    # significance: GA-LDM vs each baseline, paired across seeds
    if "GA-LDM" in methods:
        sig = {}
        for cancer, md in all_results.items():
            sig[cancer] = {}
            for m in ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]:
                sig[cancer][m] = {}
                ga = md.get("GA-LDM", {}).get(m, [])
                for method in methods:
                    if method == "GA-LDM":
                        continue
                    bl = md.get(method, {}).get(m, [])
                    n = min(len(ga), len(bl))
                    sig[cancer][m][method] = paired_wilcoxon(ga[:n], bl[:n]) if n >= 2 else float("nan")
        export_significance_excel(sig, os.path.join(RESULTS_DIR, f"significance{sfx}.xlsx"))


def _print_summary(cancer, res):
    print(f"\n===== {cancer} summary (mean +- s.d. over seeds) =====")
    metrics = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]
    hdr = "method".ljust(12) + "".join(m.rjust(12) for m in metrics)
    print(hdr)
    for method, md in res.items():
        row = method.ljust(12)
        for m in metrics:
            vals = [v for v in md.get(m, []) if not (isinstance(v, float) and np.isnan(v))]
            row += (f"{np.mean(vals):.3f}".rjust(12) if vals else "nan".rjust(12))
        print(row)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", default="CESC")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--methods", default="")
    ap.add_argument("--vae_ep", type=int, default=0)
    ap.add_argument("--dit_ep", type=int, default=0)
    ap.add_argument("--base_ep", type=int, default=0)
    args = ap.parse_args()
    cancers = [c.strip() for c in args.cancers.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    run_comparison(cancers, seeds, args.quick, methods,
                   vae_ep=args.vae_ep or None, dit_ep=args.dit_ep or None,
                   base_ep=args.base_ep or None)
