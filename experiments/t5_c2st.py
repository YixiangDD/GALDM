# -*- coding: utf-8 -*-
"""Classifier two-sample test (C2ST): a non-moment-matching metric.

A classifier is trained to tell real from generated samples and scored by 5-fold CV ROC-AUC.
AUC near 0.5 means the two distributions are indistinguishable (good generation); near 1 means
easily separable (poor).

C2ST is sensitive to higher-order and non-linear differences and cannot be gamed by
mean/covariance post-processing, which is exactly what Bio-FID and PCE (both moment-matching)
are vulnerable to. Both GA-LDM and the strongest baseline (CTGAN) are measured raw and
calibrated. CESC, 5 seeds.

Reading it: if raw GA-LDM already beats calibrated CTGAN, the advantage comes from the
generator rather than from calibration."""
import os, sys, json, random, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from experiments.comparison import run_baseline
from models.postprocess import postprocess
from configs import paths as _P

RAW = _P.RAW_ROOT; PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def c2st_auc(real, gen, seed=0):
    """Mean 5-fold CV ROC-AUC of a classifier separating real from generated, in standardised
    mRNA space. 0.5 means indistinguishable."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    try:
        from xgboost import XGBClassifier
        mk = lambda: XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                   subsample=0.8, eval_metric="logloss", verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        mk = lambda: RandomForestClassifier(n_estimators=200)
    X = np.vstack([real, gen])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(gen))])
    # standardise using real-data statistics
    mu, sd = real.mean(0), real.std(0) + 1e-8
    X = (X - mu) / sd
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr_i, te_i in skf.split(X, y):
        clf = mk(); clf.fit(X[tr_i], y[tr_i])
        p = clf.predict_proba(X[te_i])[:, 1]
        aucs.append(roc_auc_score(y[te_i], p))
    return float(np.mean(aucs))


def run(cancer="CESC", seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    keys = ["GA-LDM_raw", "GA-LDM_calib", "CTGAN_raw", "CTGAN_calib"]
    agg = {k: [] for k in keys}

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        norm = ds["norm"]
        real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        real_mi = ds["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
        y = ds["train"]["labels"]; n = ds["train"]["n_samples"]

        # GA-LDM raw (calibration off) and calibrated
        set_seed(seed)
        tr = GALDMTrainer(cfg, bp, ds, device=dev)
        tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
        tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
        set_seed(seed)
        gm_r, _, _ = tr.generate(n, do_recolor=False, postproc=False)
        set_seed(seed)
        gm_c, _, _ = tr.generate(n, do_recolor=True, postproc=True)
        agg["GA-LDM_raw"].append(c2st_auc(real_m, gm_r, seed))
        agg["GA-LDM_calib"].append(c2st_auc(real_m, gm_c, seed))

        # CTGAN raw, and with the same expression calibration for fairness
        set_seed(seed)
        gm_ct_c, gmi_ct_c, _ = run_baseline("CTGAN", ds, n, dev)   # already post-processed
        # raw CTGAN: regenerate without calibration
        from baselines.modern import BASELINE_REGISTRY
        set_seed(seed)
        Xcat = np.concatenate([ds["train"]["mrna"], ds["train"]["mirna"]], 1)
        split = ds["train"]["mrna"].shape[1]
        m = BASELINE_REGISTRY["CTGAN"](split_dim=split, device=dev, epochs=300)
        m.fit(Xcat, y)
        g, _ = m.generate(n)
        gm_ct_r = g[:, :split] * norm["mrna_sd"] + norm["mrna_mu"]
        agg["CTGAN_raw"].append(c2st_auc(real_m, gm_ct_r, seed))
        agg["CTGAN_calib"].append(c2st_auc(real_m, gm_ct_c, seed))

        print("[%s s%d] GA-LDM raw=%.3f calib=%.3f | CTGAN raw=%.3f calib=%.3f"
              % (cancer, seed, agg["GA-LDM_raw"][-1], agg["GA-LDM_calib"][-1],
                 agg["CTGAN_raw"][-1], agg["CTGAN_calib"][-1]), flush=True)
        with open(os.path.join(RES, "t5_c2st_%s.json" % cancer), "w") as f:
            json.dump(agg, f, indent=2)

    print("\n===== T5 C2ST-AUC (0.5=indistinguishable, mean+/-std) =====")
    for k in keys:
        a = np.array(agg[k]); print("%-14s %.3f +/- %.3f" % (k, a.mean(), a.std()))
    return agg


if __name__ == "__main__":
    t0 = time.time()
    c = sys.argv[1] if len(sys.argv) > 1 else "CESC"
    run(c)
    print("\n[done] %.0fs" % (time.time() - t0))
