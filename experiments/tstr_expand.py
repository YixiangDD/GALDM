# -*- coding: utf-8 -*-
"""Expanded downstream utility (TSTR) — answers reviewer P1-6.

The paper reports TSTR as XGBoost accuracy, which can sit near the majority-class rate on
imbalanced cohorts. Here we report, on the held-out real test set:
  - Majority-class baseline accuracy (the floor)
  - TRTR upper bound: train on real train, test on real test (XGBoost)
  - TSTR: train on GA-LDM synthetic, test on real test -> Accuracy, Balanced Acc, Macro-F1, AUROC
  - Augmentation (real + synthetic) vs real-only, to test whether synthetic data helps.
Uses saved GA-LDM samples (results/generated_<C>.xlsx) + processed test split. No retraining."""
import os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P
PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def _clf():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             subsample=0.8, eval_metric="logloss", verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200)


def _scores(y_true, y_pred, y_prob):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    acc = float((y_pred == y_true).mean())
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")
    return dict(Acc=round(acc, 3), BalAcc=round(bacc, 3), MacroF1=round(f1, 3), AUROC=round(auroc, 3))


def fit_eval(Xtr, ytr, Xte, yte):
    clf = _clf()
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    try:
        prob = clf.predict_proba(Xte)[:, 1]
    except Exception:
        prob = pred
    return _scores(yte, pred, prob)


def run(cancers=("CESC", "COAD", "HNSC", "KIRC")):
    allout = {}
    for c in cancers:
        npz = np.load(os.path.join(PROC, f"{c}_seed42.npz"), allow_pickle=True)
        real_tr = npz["train_mrna"] * npz["mrna_sd"] + npz["mrna_mu"]
        ytr = npz["train_y"]
        real_te = npz["test_mrna"] * npz["mrna_sd"] + npz["mrna_mu"]
        yte = npz["test_y"]
        maj = float(max(np.mean(yte == 0), np.mean(yte == 1)))

        xl = pd.ExcelFile(os.path.join(RES, f"generated_{c}.xlsx"))
        gdf = xl.parse("mRNA")
        gy = gdf["label"].values.astype(int)
        gm = gdf.iloc[:, 2:].values.astype(np.float64)

        res = {
            "majority_acc": round(maj, 3),
            "TRTR_real_only": fit_eval(real_tr, ytr, real_te, yte),
            "TSTR_synth_only": fit_eval(gm, gy, real_te, yte),
            "augment_real+synth": fit_eval(np.vstack([real_tr, gm]),
                                           np.concatenate([ytr, gy]), real_te, yte),
        }
        allout[c] = res
        print(f"[{c}] majority={maj:.3f}")
        for k in ["TRTR_real_only", "TSTR_synth_only", "augment_real+synth"]:
            print(f"   {k:20s} {res[k]}")
    with open(os.path.join(RES, "tstr_expanded.json"), "w") as f:
        json.dump(allout, f, indent=2)
    return allout


if __name__ == "__main__":
    run()
