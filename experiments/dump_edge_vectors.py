# -*- coding: utf-8 -*-
"""Dump the per-edge cross-modal correlation vectors used by main-text Fig. 3b.

Reuses crossmodal_eval on the SAVED generated samples (no retraining): for every
miRTarBase-validated miRNA->target edge present among the selected features we
store the real and the generated sample-level correlation, plus one
pairing-shuffled generated vector as the null cloud.

Output: results/edge_vectors_<COHORT>.json
"""
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from experiments.crossmodal_eval import (  # noqa: E402
    PROC, RES, load_mirtarbase_edges, edge_crosscorr,
)


def dump(cohort):
    npz = np.load(os.path.join(PROC, f"{cohort}_seed42.npz"), allow_pickle=True)
    bp = pickle.load(open(os.path.join(PROC, f"{cohort}_seed42_priors.pkl"), "rb"))
    gene_ids = [str(g) for g in npz["gene_ids"]]
    mirna_ids = [str(m) for m in npz["mirna_ids"]]
    real_m = npz["train_mrna"] * npz["mrna_sd"] + npz["mrna_mu"]
    real_mi = npz["train_mirna"] * npz["mirna_sd"] + npz["mirna_mu"]

    edges = load_mirtarbase_edges(gene_ids, mirna_ids, bp.ens2sym)
    gen_xlsx = os.path.join(RES, f"generated_{cohort}.xlsx")
    if not os.path.exists(gen_xlsx):
        raise SystemExit(
            f"missing {gen_xlsx}\n"
            "The generated-sample workbooks are too large to ship. Produce it with\n"
            f"    python -m experiments.comparison --cancers {cohort} --seeds 42\n"
            "or just use the shipped results/edge_vectors_*.json, which this script wrote."
        )
    xl = pd.ExcelFile(gen_xlsx)
    gm = xl.parse("mRNA").iloc[:, 2:].values.astype(np.float64)
    gmi = xl.parse("miRNA").iloc[:, 2:].values.astype(np.float64)
    assert gm.shape[1] == real_m.shape[1] and gmi.shape[1] == real_mi.shape[1]

    cr = edge_crosscorr(real_mi, real_m, edges)
    cg = edge_crosscorr(gmi, gm, edges)
    rng = np.random.default_rng(1)
    cs = edge_crosscorr(gmi[rng.permutation(gmi.shape[0])], gm, edges)

    out = {
        "cohort": cohort,
        "n_edges": len(edges),
        "real": [round(float(v), 5) for v in cr],
        "gen": [round(float(v), 5) for v in cg],
        "gen_shuffled": [round(float(v), 5) for v in cs],
        "pearson_real_gen": round(float(np.corrcoef(cr, cg)[0, 1]), 4),
        "pearson_real_shuffled": round(float(np.corrcoef(cr, cs)[0, 1]), 4),
        "frac_real_negative": round(float((cr < 0).mean()), 4),
        "frac_neg_preserved": round(float((cg[cr < 0] < 0).mean()), 4),
    }
    path = os.path.join(RES, f"edge_vectors_{cohort}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[{cohort}] edges={out['n_edges']} r(real,gen)={out['pearson_real_gen']} "
          f"r(real,shuf)={out['pearson_real_shuffled']} "
          f"neg_preserved={out['frac_neg_preserved']} -> {path}")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["CESC", "KIRC"]):
        dump(c)
