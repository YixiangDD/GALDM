# -*- coding: utf-8 -*-
"""Calibration ablation — answers reviewer P0-3.

Decomposes the two post-hoc calibration stages of GA-LDM:
  1) latent whitening-recoloring (2nd-order latent covariance calibration), in generate()
  2) expression-space quantile calibration (marginal alignment), in postprocess()
Both are fit ONLY on the training split (no test leakage).

Four settings on one trained full model (so any difference is purely the calibration):
  raw            : recolor=F, postproc=F
  +latent-cov    : recolor=T, postproc=F
  +quantile      : recolor=F, postproc=T
  full           : recolor=T, postproc=T
Reports all 7 metrics + the genuine cross-modal XM-Corr, to show how much of Bio-FID/PCE
comes from generation vs calibration, and that XM-Corr (pairing) is NOT a calibration artifact.
CESC, 3 seeds (cheap)."""
import os, sys, json, random
import numpy as np, torch, pandas as pd

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


def run(cancer="CESC", seeds=(42, 123, 456)):
    cfg = get_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    settings = {"raw": dict(do_recolor=False, postproc=False),
                "latent_cov": dict(do_recolor=True, postproc=False),
                "quantile": dict(do_recolor=False, postproc=True),
                "full": dict(do_recolor=True, postproc=True)}
    out = {s: {} for s in settings}
    mkeys = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
        real_y = ds["train"]["labels"]
        test_m = ds["test"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        test_y = ds["test"]["labels"]
        n = ds["train"]["n_samples"]
        gene_ids = [str(g) for g in ds["gene_ids"]]
        mirna_ids = [str(m) for m in ds["mirna_ids"]]
        edges = load_mirtarbase_edges(gene_ids, mirna_ids, bp.ens2sym)

        set_seed(seed)
        tr = GALDMTrainer(cfg, bp, ds, device=dev)
        tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
        tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)

        for s, kw in settings.items():
            set_seed(seed)
            gm, gmi, gy = tr.generate(n, **kw)
            met = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                                      test_m, test_y, bp.pathway_mask, bp.mirna_family_mask)
            if edges:
                met["XM_Corr"] = xm_corr(real_mi, real_m, gmi, gm, edges)[0]
            for k in met:
                out[s].setdefault(k, []).append(met[k])
            print(f"[{cancer} s{seed}] {s:11s} BioFID={met['Bio_FID']:.1f} PCE={met['PCE']:.3f} "
                  f"KS={met['KS']:.3f} miRNA={met['miRNA_Corr']:.3f} "
                  f"XM={met.get('XM_Corr', float('nan')):.3f}", flush=True)
        with open(os.path.join(RES, f"calib_ablation_{cancer}.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)

    print("\n===== calibration ablation (mean over seeds) =====")
    for s in settings:
        r = out[s]
        def mn(k):
            v = [x for x in r.get(k, []) if not (isinstance(x, float) and np.isnan(x))]
            return np.mean(v) if v else float("nan")
        print("%-11s BioFID=%.2f PCE=%.3f KS=%.3f miRNA=%.3f DE=%.3f XM=%.3f" % (
            s, mn("Bio_FID"), mn("PCE"), mn("KS"), mn("miRNA_Corr"), mn("DE"), mn("XM_Corr")))
    return out


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "CESC"
    run(c)
