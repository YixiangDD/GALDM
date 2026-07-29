# -*- coding: utf-8 -*-
"""Directed edge-sign satisfaction (RDC) on saved GA-LDM samples — supports reviewer P1-1.

For every directed GRN edge, checks whether the sign of the correlation between the two
node activities in the GENERATED data matches the regulatory sign (activating -> positive,
repressive -> negative). Reports the fraction satisfied (RDC_acc) and mean signed margin.
Compared against the same statistic on REAL data as reference. This is the internal,
directed metric the PCA-CTG objective targets (as opposed to undirected fidelity metrics)."""
import os, sys, json, pickle
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import regulatory_direction_consistency
from configs import paths as _P
PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def run(cancers=("CESC", "COAD", "HNSC", "KIRC")):
    out = {}
    for c in cancers:
        npz = np.load(os.path.join(PROC, f"{c}_seed42.npz"), allow_pickle=True)
        bp = pickle.load(open(os.path.join(PROC, f"{c}_seed42_priors.pkl"), "rb"))
        real_m = npz["train_mrna"] * npz["mrna_sd"] + npz["mrna_mu"]
        real_mi = npz["train_mirna"] * npz["mirna_sd"] + npz["mirna_mu"]
        xl = pd.ExcelFile(os.path.join(RES, f"generated_{c}.xlsx"))
        gm = xl.parse("mRNA").iloc[:, 2:].values.astype(np.float64)
        gmi = xl.parse("miRNA").iloc[:, 2:].values.astype(np.float64)
        rr = regulatory_direction_consistency(real_m, real_mi, bp.grn,
                                              bp.pathway_mask, bp.mirna_family_mask)
        rg = regulatory_direction_consistency(gm, gmi, bp.grn,
                                              bp.pathway_mask, bp.mirna_family_mask)
        out[c] = {"real_RDC_acc": round(rr["RDC_acc"], 3), "gen_RDC_acc": round(rg["RDC_acc"], 3),
                  "real_margin": round(rr["RDC_margin"], 3), "gen_margin": round(rg["RDC_margin"], 3),
                  "n_edges": len(bp.grn.activating_edges) + len(bp.grn.suppressive_edges)}
        print(f"[{c}] real RDC={rr['RDC_acc']:.3f} gen RDC={rg['RDC_acc']:.3f} "
              f"(edges={out[c]['n_edges']})")
    with open(os.path.join(RES, "rdc_generated.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run()
