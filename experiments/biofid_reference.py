"""Bio-FID real-vs-real reference, i.e. the in-sample floor.

The problem this addresses: reporting "disjoint real subsets 60-175 vs calibrated synthetic
6-19" makes synthetic data look more real than real data. It is an artefact of sample size:
the synthetic set has n_train samples while the real reference subset has only n_test, and
the Frechet distance blows up as n falls in high dimensions.

Sample-size-matched references computed here:
  ref_test_vs_train    real test (n_test) vs real train        the unfair original comparison
  ref_test_up_vs_train real test resampled with replacement to n_train (n matched, but
                       samples repeat)
  ref_half_vs_half     train split randomly in half (n_train/2 each), same distribution and
                       same n, i.e. the achievable floor
  ref_sub_vs_train     train subsampled without replacement to n_test vs train, isolating the
                       pure sample-size effect
  galdm_full           GA-LDM synthetic at n_train vs train, as reported in the main table
  galdm_sub            GA-LDM synthetic subsampled to n_test vs train, directly comparable
                       with ref_test_vs_train

How to read it: only galdm_sub against ref_test_vs_train is a fair noise-floor comparison,
since those two share both n and the reference set.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from evaluation.metrics import bio_fid
from experiments.comparison import run_galdm, set_seed, RAW_ROOT, PROC_DIR, RESULTS_DIR


def real_space(ds, split):
    m = ds[split]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
    return m


def run(cancers, seeds, with_galdm=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    out = {}
    for cancer in cancers:
        rows = {k: [] for k in ["ref_test_vs_train", "ref_test_up_vs_train",
                                "ref_half_vs_half", "ref_sub_vs_train",
                                "galdm_full", "galdm_sub", "n_train", "n_test"]}
        for seed in seeds:
            set_seed(seed)
            ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
            tr = real_space(ds, "train")
            te = real_space(ds, "test")
            n_tr, n_te = tr.shape[0], te.shape[0]
            rng = np.random.default_rng(seed)

            rows["n_train"].append(n_tr)
            rows["n_test"].append(n_te)
            # 1) original comparison: held-out real test vs train
            rows["ref_test_vs_train"].append(bio_fid(tr, te))
            # 2) test upsampled with replacement to n_train
            idx = rng.integers(0, n_te, n_tr)
            rows["ref_test_up_vs_train"].append(bio_fid(tr, te[idx]))
            # 3) train split in half (same distribution, n=n_train/2 each)
            perm = rng.permutation(n_tr)
            h1, h2 = perm[: n_tr // 2], perm[n_tr // 2:]
            rows["ref_half_vs_half"].append(bio_fid(tr[h1], tr[h2]))
            # 4) train subsampled without replacement to n_test, scored against train: the
            #    pure sample-size effect, and the in-sample floor
            sub = rng.choice(n_tr, n_te, replace=False)
            rows["ref_sub_vs_train"].append(bio_fid(tr, tr[sub]))

            if with_galdm:
                set_seed(seed)
                gm, gmi, gy = run_galdm(cfg, bp, ds, device)
                rows["galdm_full"].append(bio_fid(tr, gm))
                gsub = rng.choice(gm.shape[0], n_te, replace=False)
                rows["galdm_sub"].append(bio_fid(tr, gm[gsub]))

            print(f"[{cancer} s{seed}] n_tr={n_tr} n_te={n_te} "
                  f"test_vs_train={rows['ref_test_vs_train'][-1]:.1f} "
                  f"test_up={rows['ref_test_up_vs_train'][-1]:.1f} "
                  f"half={rows['ref_half_vs_half'][-1]:.1f} "
                  f"sub={rows['ref_sub_vs_train'][-1]:.1f} "
                  + (f"galdm_full={rows['galdm_full'][-1]:.1f} "
                     f"galdm_sub={rows['galdm_sub'][-1]:.1f}" if with_galdm else ""))

        out[cancer] = {k: {"vals": v,
                           "mean": float(np.mean(v)) if v else None,
                           "std": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                       for k, v in rows.items()}
        p = os.path.join(RESULTS_DIR, "biofid_reference.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {p}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", default="CESC,COAD,HNSC,KIRC")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--no_galdm", action="store_true")
    a = ap.parse_args()
    run(a.cancers.split(","), [int(s) for s in a.seeds.split(",")],
        with_galdm=not a.no_galdm)
