# -*- coding: utf-8 -*-
"""Biological readout of generated cohorts (for the Results case-study subsection).

Two questions, both answered against real data from the same split:
  (1) Stage-associated pathway programmes. For each KEGG pathway we compute a
      per-sample activity score (mean standardised expression of member genes) and
      a Welch t-statistic for early versus late stage. We then ask whether the
      generated cohort reproduces the real direction and ranking of these
      programmes, and which pathways are significant in both.
  (2) miRNA-target repression axes. For miRTarBase-validated miRNA-target pairs we
      compute the sample-level correlation in real and generated data, and report
      how many validated pairs keep a negative (repressive) correlation.

Nothing here is used for training or model selection; it is a post-hoc readout.
"""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.comparison import run_galdm, set_seed, RAW_ROOT, PROC_DIR, RESULTS_DIR
from experiments.crossmodal_eval import load_mirtarbase_edges


def pathway_activity(X, mask, mu, sd):
    """(N, K) activity = mean standardised expression of each pathway's members."""
    Z = (X - mu) / (sd + 1e-8)
    out = np.zeros((X.shape[0], mask.shape[0]), dtype=np.float64)
    for k in range(mask.shape[0]):
        idx = np.where(mask[k] > 0)[0]
        out[:, k] = Z[:, idx].mean(1) if len(idx) else 0.0
    return out


def stage_t(act, y):
    """Welch t per pathway, early(0) versus late(1)."""
    a, b = act[y == 0], act[y == 1]
    if len(a) < 3 or len(b) < 3:
        return np.full(act.shape[1], np.nan), np.full(act.shape[1], np.nan)
    t, p = stats.ttest_ind(b, a, axis=0, equal_var=False)
    return t, p


def edge_corr(mi, m, edges):
    out = np.empty(len(edges))
    for k, (i, j) in enumerate(edges):
        a, b = mi[:, i], m[:, j]
        if a.std() < 1e-8 or b.std() < 1e-8:
            out[k] = 0.0
        else:
            out[k] = np.corrcoef(a, b)[0, 1]
    return out


def run(cancers, seeds):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    out = {}
    for cancer in cancers:
        acc = {}
        names = None
        for seed in seeds:
            set_seed(seed)
            ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
            nm = ds["norm"]
            real_m = ds["train"]["mrna"] * nm["mrna_sd"] + nm["mrna_mu"]
            real_mi = ds["train"]["mirna"] * nm["mirna_sd"] + nm["mirna_mu"]
            real_y = ds["train"]["labels"]
            names = bp.pathway_info["pathway_names"]

            set_seed(seed)
            gm, gmi, gy = run_galdm(cfg, bp, ds, device)

            mu, sd = real_m.mean(0), real_m.std(0)
            mask = bp.pathway_mask
            ar = pathway_activity(real_m, mask, mu, sd)
            ag = pathway_activity(gm, mask, mu, sd)
            tr, pr = stage_t(ar, real_y)
            tg, pg = stage_t(ag, gy)
            ok = ~(np.isnan(tr) | np.isnan(tg))
            acc.setdefault("t_spearman", []).append(
                float(stats.spearmanr(tr[ok], tg[ok]).statistic))
            acc.setdefault("sign_agree", []).append(
                float((np.sign(tr[ok]) == np.sign(tg[ok])).mean()))
            acc.setdefault("t_real", []).append(tr.tolist())
            acc.setdefault("t_gen", []).append(tg.tolist())
            acc.setdefault("p_real", []).append(pr.tolist())
            acc.setdefault("p_gen", []).append(pg.tolist())

            edges = load_mirtarbase_edges([str(g) for g in ds["gene_ids"]],
                                          [str(x) for x in ds["mirna_ids"]], bp.ens2sym)
            cr, cg = edge_corr(real_mi, real_m, edges), edge_corr(gmi, gm, edges)
            neg = cr < 0
            acc.setdefault("n_edges", []).append(len(edges))
            acc.setdefault("frac_neg_real", []).append(float(neg.mean()))
            acc.setdefault("neg_preserved", []).append(float((cg[neg] < 0).mean()))
            acc.setdefault("edge_spearman", []).append(
                float(stats.spearmanr(cr, cg).statistic))
            print(f"[{cancer} s{seed}] pathway t rho={acc['t_spearman'][-1]:.3f} "
                  f"sign={acc['sign_agree'][-1]:.3f} | edges={len(edges)} "
                  f"negPreserved={acc['neg_preserved'][-1]:.3f} "
                  f"edgeRho={acc['edge_spearman'][-1]:.3f}")

        res = {"pathway_names": names}
        for k in ["t_spearman", "sign_agree", "frac_neg_real", "neg_preserved",
                  "edge_spearman", "n_edges"]:
            v = acc[k]
            res[k] = {"mean": float(np.mean(v)),
                      "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                      "vals": v}
        tr = np.array(acc["t_real"]); tg = np.array(acc["t_gen"])
        pr = np.array(acc["p_real"]); pg = np.array(acc["p_gen"])
        res["per_pathway"] = {
            "t_real_mean": np.nanmean(tr, 0).tolist(),
            "t_gen_mean": np.nanmean(tg, 0).tolist(),
            "n_seeds_sig_real": (pr < 0.05).sum(0).tolist(),
            "n_seeds_sig_gen": (pg < 0.05).sum(0).tolist(),
        }
        out[cancer] = res
        p = os.path.join(RESULTS_DIR, "bio_readout.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("[saved]", p)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", default="CESC,KIRC")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    a = ap.parse_args()
    run(a.cancers.split(","), [int(s) for s in a.seeds.split(",")])
