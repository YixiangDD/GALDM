# -*- coding: utf-8 -*-
"""Augmentation learning curve: does synthetic data add value on top of real data?

TSTR measures replacement, not augmentation, so this measures augmentation directly:
real-only vs real+GA-LDM vs real+SMOTE, scored by balanced accuracy, macro-F1 and AUROC.

To avoid re-indexing bugs from retraining per subset, GA-LDM is trained once on the full
training split, a synthetic pool is generated from it, and that pool augments real subsets of
varying size. The classifier only ever sees the real training subset plus synthetic samples;
the real test set is fixed. CESC, 5 seeds."""
import os, sys, json, random, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from baselines.modern import BASELINE_REGISTRY as CLASSICAL
from configs import paths as _P

RAW = _P.RAW_ROOT; PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR
FRACS = [0.10, 0.25, 0.50, 1.0]


def set_seed(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def clf_eval(Xtr, ytr, Xte, yte):
    """XGBoost; returns balanced_acc, macro_f1, auroc."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                            subsample=0.8, eval_metric="logloss", verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200)
    if len(np.unique(ytr)) < 2:
        return 0.0, 0.0, 0.5
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    ba = balanced_accuracy_score(yte, pred)
    mf = f1_score(yte, pred, average="macro")
    try:
        proba = clf.predict_proba(Xte)[:, 1]
        au = roc_auc_score(yte, proba)
    except Exception:
        au = 0.5
    return float(ba), float(mf), float(au)


def strat_subset(y, frac, seed):
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = max(2, int(round(len(idx) * frac)))
        keep.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
    return np.array(sorted(keep))


def pool_sample(pool_X, pool_y, n, seed):
    """Draws n samples from the synthetic pool, balanced across classes."""
    rng = np.random.RandomState(seed + 3)
    classes = np.unique(pool_y); per = max(1, n // len(classes)); idx = []
    for c in classes:
        ci = np.where(pool_y == c)[0]
        idx.extend(rng.choice(ci, min(per, len(ci)), replace=len(ci) < per).tolist())
    return pool_X[idx], pool_y[idx]


def run(cancer="CESC", seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ["real_only", "real+GA-LDM", "real+SMOTE"]
    agg = {f: {m: {"ba": [], "f1": [], "au": []} for m in methods} for f in FRACS}

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        Xtr_full = ds["train"]["mrna"]; ytr_full = ds["train"]["labels"]
        Xte = ds["test"]["mrna"]; yte = ds["test"]["labels"]
        split = Xtr_full.shape[1]
        n_full = ds["train"]["n_samples"]

        # train GA-LDM once on the full split, then generate a 3x synthetic pool
        set_seed(seed)
        tr = GALDMTrainer(cfg, bp, ds, device=dev)
        tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
        tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
        gm, gmi, gy = tr.generate(n_full * 3)
        # GA-LDM generates in the original space; classification happens in the standardised
        # mRNA space, same as for real data
        norm = ds["norm"]
        gm_std = (gm - norm["mrna_mu"]) / (norm["mrna_sd"] + 1e-8)
        pool_X, pool_y = gm_std, np.asarray(gy)

        # SMOTE pool, fitted on the full split
        set_seed(seed)
        Xcat = np.concatenate([Xtr_full, ds["train"]["mirna"]], 1)
        sm = CLASSICAL["SMOTE"](split_dim=split, device=dev)
        sm.fit(Xcat, ytr_full)
        sm_g, sm_y = sm.generate(n_full * 3)
        sm_pool_X = sm_g[:, :split]; sm_pool_y = np.asarray(sm_y)

        for frac in FRACS:
            idx = strat_subset(ytr_full, frac, seed)
            Xr, yr = Xtr_full[idx], ytr_full[idx]
            n_add = len(idx)  # 1x augmentation
            # real_only
            ba, f1, au = clf_eval(Xr, yr, Xte, yte)
            agg[frac]["real_only"]["ba"].append(ba); agg[frac]["real_only"]["f1"].append(f1); agg[frac]["real_only"]["au"].append(au)
            # real+GA-LDM
            sx, sy = pool_sample(pool_X, pool_y, n_add, seed)
            ba, f1, au = clf_eval(np.vstack([Xr, sx]), np.concatenate([yr, sy]), Xte, yte)
            agg[frac]["real+GA-LDM"]["ba"].append(ba); agg[frac]["real+GA-LDM"]["f1"].append(f1); agg[frac]["real+GA-LDM"]["au"].append(au)
            # real+SMOTE
            sx, sy = pool_sample(sm_pool_X, sm_pool_y, n_add, seed)
            ba, f1, au = clf_eval(np.vstack([Xr, sx]), np.concatenate([yr, sy]), Xte, yte)
            agg[frac]["real+SMOTE"]["ba"].append(ba); agg[frac]["real+SMOTE"]["f1"].append(f1); agg[frac]["real+SMOTE"]["au"].append(au)
            print("[%s s%d f%.2f] real=%.3f  +GA-LDM=%.3f  +SMOTE=%.3f (balanced acc)"
                  % (cancer, seed, frac, agg[frac]["real_only"]["ba"][-1],
                     agg[frac]["real+GA-LDM"]["ba"][-1], agg[frac]["real+SMOTE"]["ba"][-1]), flush=True)
        with open(os.path.join(RES, "augment_curve_%s.json" % cancer), "w") as f:
            json.dump(agg, f, indent=2)

    print("\n===== augmentation learning curve (balanced acc, mean over seeds) =====")
    for frac in FRACS:
        r = agg[frac]
        print("frac=%.2f: real=%.3f | +GA-LDM=%.3f (gain %+.3f) | +SMOTE=%.3f" % (
            frac, np.mean(r["real_only"]["ba"]), np.mean(r["real+GA-LDM"]["ba"]),
            np.mean(r["real+GA-LDM"]["ba"]) - np.mean(r["real_only"]["ba"]),
            np.mean(r["real+SMOTE"]["ba"])))
    return agg


if __name__ == "__main__":
    t0 = time.time()
    c = sys.argv[1] if len(sys.argv) > 1 else "CESC"
    run(c)
    print("\n[done] %.0fs" % (time.time() - t0))
