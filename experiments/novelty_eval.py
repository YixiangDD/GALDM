# -*- coding: utf-8 -*-
"""Novelty and memorisation analysis, separating GA-LDM (generates new samples) from SMOTE
(interpolates between real points).

All metrics are computed in the standardised mRNA||miRNA space so scales match:
  1. NN-dist: mean Euclidean distance from each generated sample to its nearest real training
     sample. SMOTE produces convex combinations of real points, so it lands on the segments
     between them and its NN-dist is small (near-memorisation). GA-LDM generates from noise,
     so its NN-dist is larger.
  2. real_ref: the NN-dist from the real test set to the real training set, i.e. how novel a
     genuinely new real sample looks. A good generator should sit near real_ref rather than
     far below it (copying) or far above it (drifting off-distribution).
  3. dup_rate: the fraction of samples with NN-dist below half the real_ref median, i.e. near
     duplicates of training points.
CESC, 5 seeds. Writes results/novelty_CESC.json."""
import os, sys, json, time, random
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from baselines.modern import BASELINE_REGISTRY
from evaluation.metrics import bio_fid
from configs import paths as _P

# near-duplicate thresholds (x the real_ref median), and Bio-FID n_comp values to sweep
DUP_FRACS = (0.3, 0.5, 0.7)
NCOMP_GRID = (10, 20, 50)

RAW = _P.RAW_ROOT
PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def nn_dist(gen_std, real_std):
    """Nearest-neighbour distance from each gen row to real; chunked to bound memory."""
    d = np.empty(len(gen_std))
    for i in range(0, len(gen_std), 256):
        blk = gen_std[i:i + 256]
        # (b, n_real): ||g||^2 + ||r||^2 - 2 g·r
        dist2 = (blk ** 2).sum(1, keepdims=True) + (real_std ** 2).sum(1) - 2 * blk @ real_std.T
        d[i:i + len(blk)] = np.sqrt(np.maximum(dist2.min(1), 0))
    return d


def to_std_concat(m, mi, norm):
    """Original log2 space -> standardised concatenation using training statistics."""
    ms = (m - norm["mrna_mu"]) / (norm["mrna_sd"] + 1e-8)
    mis = (mi - norm["mirna_mu"]) / (norm["mirna_sd"] + 1e-8)
    return np.concatenate([ms, mis], 1)


def run(cancer="CESC", seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ["GA-LDM", "SMOTE"]
    agg = {mname: {"nn_mean": [], "nn_median": [], "nn_min": [], "dup_rate": [],
                   "ratio_to_real": [],
                   **{"dup_rate_%.1f" % fr: [] for fr in DUP_FRACS}} for mname in methods}
    agg["real_ref"] = {"nn_mean": [], "nn_median": []}
    # Bio-FID vs number of principal components (GA-LDM only, same generated batch)
    biofid_ncomp = {nc: [] for nc in NCOMP_GRID}

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        norm = ds["norm"]
        # real train/test in the original space
        real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        real_mi = ds["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
        test_m = ds["test"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        test_mi = ds["test"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
        n = ds["train"]["n_samples"]

        real_std = to_std_concat(real_m, real_mi, norm)  # the set that could be copied from
        # reference: NN-dist from the real test set to the training set
        test_std = to_std_concat(test_m, test_mi, norm)
        ref_d = nn_dist(test_std, real_std)
        ref_med = float(np.median(ref_d))
        agg["real_ref"]["nn_mean"].append(float(ref_d.mean()))
        agg["real_ref"]["nn_median"].append(ref_med)
        dup_thr = 0.5 * ref_med  # near-duplicate threshold

        # GA-LDM
        set_seed(seed)
        tr = GALDMTrainer(cfg, bp, ds, device=device)
        tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
        tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
        gm, gmi, gy = tr.generate(n)
        gstd = to_std_concat(gm, gmi, norm)
        dg = nn_dist(gstd, real_std)
        agg["GA-LDM"]["nn_mean"].append(float(dg.mean()))
        agg["GA-LDM"]["nn_median"].append(float(np.median(dg)))
        agg["GA-LDM"]["nn_min"].append(float(dg.min()))
        agg["GA-LDM"]["dup_rate"].append(float((dg < dup_thr).mean()))
        agg["GA-LDM"]["ratio_to_real"].append(float(np.median(dg) / ref_med))
        for fr in DUP_FRACS:
            agg["GA-LDM"]["dup_rate_%.1f" % fr].append(float((dg < fr * ref_med).mean()))
        # Bio-FID n_comp sensitivity (original log2 mRNA space, as in the main experiments)
        real_m_train = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        for nc in NCOMP_GRID:
            biofid_ncomp[nc].append(float(bio_fid(real_m_train, gm, n_comp=nc)))

        # SMOTE: fitted in standardised space, then de-standardised back
        set_seed(seed)
        Xm, Xmi = ds["train"]["mrna"], ds["train"]["mirna"]
        X = np.concatenate([Xm, Xmi], 1); split = Xm.shape[1]
        sm = BASELINE_REGISTRY["SMOTE"](split_dim=split, device=device)
        sm.fit(X, ds["train"]["labels"])
        gs, _ = sm.generate(n)
        sm_m = gs[:, :split] * norm["mrna_sd"] + norm["mrna_mu"]
        sm_mi = gs[:, split:] * norm["mirna_sd"] + norm["mirna_mu"]
        sstd = to_std_concat(sm_m, sm_mi, norm)
        dsm = nn_dist(sstd, real_std)
        agg["SMOTE"]["nn_mean"].append(float(dsm.mean()))
        agg["SMOTE"]["nn_median"].append(float(np.median(dsm)))
        agg["SMOTE"]["nn_min"].append(float(dsm.min()))
        agg["SMOTE"]["dup_rate"].append(float((dsm < dup_thr).mean()))
        agg["SMOTE"]["ratio_to_real"].append(float(np.median(dsm) / ref_med))
        for fr in DUP_FRACS:
            agg["SMOTE"]["dup_rate_%.1f" % fr].append(float((dsm < fr * ref_med).mean()))

        print(f"[{cancer} s{seed}] real_ref med={ref_med:.2f} | "
              f"GA-LDM med={np.median(dg):.2f}(dup{(dg<dup_thr).mean():.1%}) | "
              f"SMOTE med={np.median(dsm):.2f}(dup{(dsm<dup_thr).mean():.1%})", flush=True)

    # aggregate mean +- s.d.
    out = {}
    for k, v in agg.items():
        out[k] = {mk: [float(np.mean(mv)), float(np.std(mv))] for mk, mv in v.items() if mv}
    out["biofid_ncomp"] = {str(nc): [float(np.mean(vs)), float(np.std(vs))]
                           for nc, vs in biofid_ncomp.items()}
    os.makedirs(RES, exist_ok=True)
    # write the sensitivity json, leaving novelty_{cancer}.json untouched
    with open(os.path.join(RES, f"novelty_sensitivity_{cancer}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n=== SUMMARY (mean±std over seeds) ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== Q7 threshold sensitivity (near-duplicate fraction) ===")
    for fr in DUP_FRACS:
        g = out["GA-LDM"]["dup_rate_%.1f" % fr]; s = out["SMOTE"]["dup_rate_%.1f" % fr]
        print("  thr=%.1f x ref: GA-LDM %.3f%%  SMOTE %.1f%%" % (fr, g[0]*100, s[0]*100))
    print("  GA-LDM nn_min (min over samples, mean+/-std over seeds): %.2f +/- %.2f  (dup_thr@0.5=%.2f-ish)"
          % (out["GA-LDM"]["nn_min"][0], out["GA-LDM"]["nn_min"][1], 0.5*out["real_ref"]["nn_median"][0]))
    print("\n=== Q10 Bio-FID vs n_comp (GA-LDM, CESC 5-seed) ===")
    for nc in NCOMP_GRID:
        b = out["biofid_ncomp"][str(nc)]
        print("  n_comp=%2d: Bio-FID %.2f +/- %.2f" % (nc, b[0], b[1]))
    return out


if __name__ == "__main__":
    t0 = time.time()
    run("CESC")
    print(f"\n[done] {time.time()-t0:.0f}s")
