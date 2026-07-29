# -*- coding: utf-8 -*-
"""Raw (uncalibrated) vs calibrated cross-modal correlation, to show what calibration
contributes, across all four cohorts.

Measured exactly as the paired column of the cross-modal table: crossmodal's xm_corr at
seed 42.
- calibrated: tr.generate(...) with defaults (do_recolor=True, postproc=True), which should
  reproduce XM_Corr_joint in crossmodal_{c}.json as a self-check
- raw:        tr.generate(do_recolor=False, postproc=False)
Each cohort is trained once and both variants are generated from that same model, so
calibration is the only variable."""
import os, sys, json, pickle, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from experiments.comparison import set_seed
from experiments.crossmodal_eval import load_mirtarbase_edges, xm_corr
from configs import paths as _P

RAW = _P.RAW_ROOT
PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def run(cancers=("CESC", "COAD", "HNSC", "KIRC"), seed=42):
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for cancer in cancers:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        norm = ds["norm"]
        real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        real_mi = ds["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
        gene_ids = [str(g) for g in ds["gene_ids"]]
        mirna_ids = [str(m) for m in ds["mirna_ids"]]
        edges = load_mirtarbase_edges(gene_ids, mirna_ids, bp.ens2sym)

        set_seed(seed)
        tr = GALDMTrainer(cfg, bp, ds, device=device)
        tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
        tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
        n = ds["train"]["n_samples"]

        # calibrated (defaults) vs raw (both calibration stages off)
        set_seed(seed)
        gm_c, gmi_c, _ = tr.generate(n, do_recolor=True, postproc=True)
        set_seed(seed)
        gm_r, gmi_r, _ = tr.generate(n, do_recolor=False, postproc=False)

        calib = xm_corr(real_mi, real_m, gmi_c, gm_c, edges)[0]
        raw = xm_corr(real_mi, real_m, gmi_r, gm_r, edges)[0]
        out[cancer] = {"n_edges": len(edges), "raw_XM_Corr": round(float(raw), 4),
                       "calibrated_XM_Corr": round(float(calib), 4)}
        print("[%s] edges=%d  raw=%.4f  calibrated=%.4f" % (cancer, len(edges), raw, calib), flush=True)

    with open(os.path.join(RES, "q6_raw_calib_xmcorr.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n=== Q6 raw vs calibrated XM-Corr (seed=42, same pipeline as Table IV) ===")
    for c, v in out.items():
        print("  %-5s raw %.3f -> calibrated %.3f" % (c, v["raw_XM_Corr"], v["calibrated_XM_Corr"]))
    return out


if __name__ == "__main__":
    t0 = time.time()
    run()
    print("\n[done] %.0fs" % (time.time() - t0))
