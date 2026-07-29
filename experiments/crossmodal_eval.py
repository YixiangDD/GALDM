# -*- coding: utf-8 -*-
"""Genuine sample-level cross-modal (mRNA-miRNA) evaluation — answers reviewer P0-1.

The paper's `miRNA_Corr` metric measures ONLY within-miRNA correlation structure and
therefore cannot certify that sample-level mRNA<->miRNA pairing is preserved. This script
adds a metric that DOES depend on pairing, evaluated on miRTarBase-validated miRNA->target
mRNA edges, plus a pairing permutation test.

Definitions
-----------
Cross-correlation matrix C[i,j] = corr(miRNA_i, mRNA_j) over samples, restricted to the set
E of miRTarBase-validated (miRNA_i -> target mRNA_j) pairs present in the selected features.

XM-Corr (cross-modal correlation agreement):
    Pearson r between the vector of edge cross-correlations {C^real_e} and {C^gen_e}, e in E.
    This is destroyed if you shuffle the mRNA<->miRNA sample pairing of the generated data,
    because C^gen_e then collapses toward 0. A single-modality (independent) generator that
    reproduces each marginal perfectly but has NO shared sampling still scores ~0.

Pairing permutation test:
    Recompute XM-Corr after randomly permuting the sample order of generated miRNA relative
    to generated mRNA (B permutations). p = P(XM-Corr_perm >= XM-Corr_observed). A model that
    truly preserves pairing should give observed >> permuted (small p).

Runs on the SAVED generated samples (results/generated_<C>.xlsx) — no retraining.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P

CACHE = _P.CACHE_DIR
PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def _norm_mir(name):
    """hsa-miR-21-5p / hsa-mir-21 -> mir-21 root (align miRTarBase to our miRNA ids)."""
    s = str(name).lower().replace("hsa-", "")
    s = s.replace("mir-", "mir").replace("let-", "let")
    # keep leading token like mir21 / mir21a; drop -5p/-3p arm suffix
    for suf in ["-5p", "-3p", "_5p", "_3p"]:
        s = s.replace(suf, "")
    return s


def load_mirtarbase_edges(gene_syms, mirna_ids, ens2sym):
    """Return list of (mirna_col_idx, gene_col_idx) for validated miRNA->target edges
    that are present among our selected features."""
    df = pd.read_excel(os.path.join(CACHE, "mirtarbase_se_wr.xlsx"))
    df = df.rename(columns={"miRNA": "mirna", "Target Gene": "target"})
    df = df[["mirna", "target"]].dropna()
    # symbol -> our mRNA column index
    sym2col = {}
    for j, ens in enumerate(gene_syms):
        sym = ens2sym.get(ens, None)
        if sym:
            sym2col.setdefault(sym.upper(), j)
    # normalized mir root -> our miRNA column index
    mir2col = {}
    for i, m in enumerate(mirna_ids):
        mir2col.setdefault(_norm_mir(m), i)
    edges = set()
    for _, row in df.iterrows():
        mi_root = _norm_mir(row["mirna"])
        tgt = str(row["target"]).upper()
        if mi_root in mir2col and tgt in sym2col:
            edges.add((mir2col[mi_root], sym2col[tgt]))
    return sorted(edges)


def edge_crosscorr(mi, m, edges):
    """Vector of corr(miRNA_i, mRNA_j) over samples for each (i,j) in edges."""
    mi_z = (mi - mi.mean(0)) / (mi.std(0) + 1e-8)
    m_z = (m - m.mean(0)) / (m.std(0) + 1e-8)
    n = mi.shape[0]
    out = np.empty(len(edges))
    for k, (i, j) in enumerate(edges):
        out[k] = float((mi_z[:, i] * m_z[:, j]).mean())
    return out


def xm_corr(real_mi, real_m, gen_mi, gen_m, edges):
    cr = edge_crosscorr(real_mi, real_m, edges)
    cg = edge_crosscorr(gen_mi, gen_m, edges)
    if np.std(cr) < 1e-8 or np.std(cg) < 1e-8:
        return 0.0, cr, cg
    return float(np.corrcoef(cr, cg)[0, 1]), cr, cg


def permutation_test(real_mi, real_m, gen_mi, gen_m, edges, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs, cr, _ = xm_corr(real_mi, real_m, gen_mi, gen_m, edges)
    n = gen_mi.shape[0]
    ge = np.empty(B)
    for b in range(B):
        perm = rng.permutation(n)
        cg = edge_crosscorr(gen_mi[perm], gen_m, edges)
        ge[b] = np.corrcoef(cr, cg)[0, 1] if np.std(cg) > 1e-8 else 0.0
    p = float((np.sum(ge >= obs) + 1) / (B + 1))
    return obs, float(ge.mean()), float(ge.std()), p


def main(cancer="CESC"):
    npz = np.load(os.path.join(PROC, f"{cancer}_seed42.npz"), allow_pickle=True)
    bp = pickle.load(open(os.path.join(PROC, f"{cancer}_seed42_priors.pkl"), "rb"))
    gene_ids = [str(g) for g in npz["gene_ids"]]
    mirna_ids = [str(m) for m in npz["mirna_ids"]]
    # real (train) in original space
    real_m = npz["train_mrna"] * npz["mrna_sd"] + npz["mrna_mu"]
    real_mi = npz["train_mirna"] * npz["mirna_sd"] + npz["mirna_mu"]

    edges = load_mirtarbase_edges(gene_ids, mirna_ids, bp.ens2sym)
    print(f"[{cancer}] validated miRNA->target edges among selected features: {len(edges)}")
    if len(edges) < 5:
        print("Too few edges; abort.")
        return

    # generated samples (GA-LDM joint), saved with sample-level pairing intact
    xl = pd.ExcelFile(os.path.join(RES, f"generated_{cancer}.xlsx"))
    gm = xl.parse("mRNA").iloc[:, 2:].values.astype(np.float64)
    gmi = xl.parse("miRNA").iloc[:, 2:].values.astype(np.float64)
    assert gm.shape[1] == real_m.shape[1] and gmi.shape[1] == real_mi.shape[1]

    obs, perm_mean, perm_std, p = permutation_test(real_mi, real_m, gmi, gm, edges, B=2000)
    # independent control: shuffle generated pairing once (matches m10 spirit)
    rng = np.random.default_rng(1)
    indep, _, _ = xm_corr(real_mi, real_m, gmi[rng.permutation(gmi.shape[0])], gm, edges)

    out = {
        "cancer": cancer,
        "n_edges": len(edges),
        "XM_Corr_joint": round(obs, 4),
        "XM_Corr_shuffled_pairing": round(indep, 4),
        "perm_null_mean": round(perm_mean, 4),
        "perm_null_std": round(perm_std, 4),
        "perm_p_value": p,
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(RES, f"crossmodal_{cancer}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "CESC"
    main(c)
