"""Cross-modal coupling sweep: vary lambda_cm to trace the Pareto front between fidelity
(Bio-FID) and cross-modal dependence (miRNA_Corr).

This turns cross-modal attention from an on/off switch into a continuous scalar: lambda_cm=1.0
is the full model and lambda_cm=0 disables the coupling. Note that lambda_cm=0 differs from the
w/o_CrossModal ablation by one LayerNorm, so it is annotated separately in the figure. Every
other component stays at its full-model setting.

Usage: python -m experiments.pareto_crossmodal --cancer CESC --seeds 42,123,456,789,1024

"""
import os, sys, json, time, random
import numpy as np, torch

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
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run_pareto(cancer="CESC", seeds=(42, 123, 456, 789, 1024),
               lambdas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
               vae_ep=None, dit_ep=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    results = {f"{lam:g}": {} for lam in lambdas}
    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
        test_m = ds["test"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        test_y = ds["test"]["labels"]
        real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
        real_y = ds["train"]["labels"]
        for lam in lambdas:
            set_seed(seed)
            t0 = time.time()
            try:
                tr = GALDMTrainer(cfg, bp, ds, device=device,
                                  ablation_flags={"lambda_cm": float(lam)})
                tr.train_vae(epochs=vae_ep or cfg.train.vae_epochs, verbose=False)
                tr.train_dit(epochs=dit_ep or cfg.train.dit_epochs, verbose=False)
                gm, gmi, gy = tr.generate(ds["train"]["n_samples"])
                met = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                                          test_m, test_y, bp.pathway_mask, bp.mirna_family_mask)
            except Exception:
                import traceback; traceback.print_exc()
                met = {k: float("nan") for k in
                       ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]}
            for k, v in met.items():
                results[f"{lam:g}"].setdefault(k, []).append(v)
            print(f"[{cancer} s{seed}] lambda_cm={lam:<4g} "
                  f"BioFID={met['Bio_FID']:.2f} miRNA={met['miRNA_Corr']:.4f} "
                  f"PCE={met['PCE']:.4f} ({time.time()-t0:.0f}s)")
        # flush after each seed, so an interruption loses at most one
        out = os.path.join(RESULTS_DIR, f"pareto_crossmodal_{cancer}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    print("\n===== Pareto summary (mean over seeds) =====")
    print("%-10s %10s %10s %10s" % ("lambda_cm", "BioFID", "miRNA", "PCE"))
    for lam in lambdas:
        r = results[f"{lam:g}"]
        def m(k):
            v = [x for x in r.get(k, []) if not (isinstance(x, float) and np.isnan(x))]
            return np.mean(v) if v else float("nan")
        print("%-10g %10.2f %10.4f %10.4f" % (lam, m("Bio_FID"), m("miRNA_Corr"), m("PCE")))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", default="CESC")
    ap.add_argument("--seeds", default="42,123,456,789,1024")
    ap.add_argument("--lambdas", default="0,0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--vae_ep", type=int, default=0)
    ap.add_argument("--dit_ep", type=int, default=0)
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    lambdas = tuple(float(x) for x in args.lambdas.split(","))
    run_pareto(args.cancer, seeds, lambdas, args.vae_ep or None, args.dit_ep or None)
