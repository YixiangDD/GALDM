# -*- coding: utf-8 -*-
"""Value of the biological prior under data scarcity: KEGG/family grouping vs a
degree-preserving random grouping, at N=40 (very small) and at full size.

Hypothesis: the prior only pays off when data is scarce. With enough samples the model learns
the structure from data and the prior becomes redundant.

Fairness: the degree-preserving edge swap keeps group sizes and each feature's overlap degree
intact, randomising only the biological identity of the groups. PCE and miRNA_Corr are always
scored against the fixed real KEGG/family masks, so the random grouping is not evaluated on its
own terms. CESC, 5 seeds."""
import os, sys, json, random, copy, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from evaluation.metrics import compute_all_metrics
from experiments.crossmodal_eval import load_mirtarbase_edges, xm_corr
from configs import paths as _P

RAW = _P.RAW_ROOT; PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def degree_preserving_swap(mask, rng, factor=20):
    """Bipartite edge swap: preserves row sums (group sizes) and column sums (members per
    feature, i.e. overlap degree) while randomising biological identity."""
    M = mask.copy().astype(int)
    edges = list(zip(*np.where(M > 0)))
    E = len(edges)
    for _ in range(E * factor):
        i, j = rng.integers(0, E), rng.integers(0, E)
        (r1, c1), (r2, c2) = edges[i], edges[j]
        if r1 == r2 or c1 == c2:
            continue
        if M[r1, c2] == 0 and M[r2, c1] == 0:
            M[r1, c1] = 0; M[r2, c2] = 0; M[r1, c2] = 1; M[r2, c1] = 1
            edges[i] = (r1, c2); edges[j] = (r2, c1)
    return M.astype(float)


def subsample_train(ds, n_target, seed):
    """Stratified downsampling of the training split to about n_target; val/test untouched."""
    rng = np.random.RandomState(seed)
    y = ds["train"]["labels"]; keep = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        n_c = min(len(idx_c), max(2, int(round(n_target * len(idx_c) / len(y)))))
        keep.extend(rng.choice(idx_c, n_c, replace=False).tolist())
    keep = np.array(sorted(keep))
    new = copy.copy(ds)
    new["train"] = {"mrna": ds["train"]["mrna"][keep], "mirna": ds["train"]["mirna"][keep],
                    "labels": ds["train"]["labels"][keep], "n_samples": len(keep)}
    return new


def run(cancer="CESC", sizes=(40, 0), seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mkeys = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID", "XM_Corr"]
    out = {}

    for seed in seeds:
        set_seed(seed)
        base, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        norm = base["norm"]
        eval_pm, eval_fm = bp.pathway_mask, bp.mirna_family_mask  # fixed real yardstick
        edges = load_mirtarbase_edges([str(g) for g in base["gene_ids"]],
                                      [str(m) for m in base["mirna_ids"]], bp.ens2sym)
        for N in sizes:
            ds = base if N == 0 else subsample_train(base, N, seed)
            tag = "N%d" % (ds["train"]["n_samples"])
            real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
            real_mi = ds["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
            real_y = ds["train"]["labels"]
            test_m = ds["test"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
            test_y = ds["test"]["labels"]
            n_gen = ds["train"]["n_samples"]
            for cond in ["KEGG", "Random"]:
                set_seed(seed)
                bp_use = bp
                if cond == "Random":
                    rng = np.random.default_rng(seed + 7000)
                    bp_use = copy.copy(bp)
                    bp_use.pathway_mask = degree_preserving_swap(bp.pathway_mask, rng)
                    bp_use.mirna_family_mask = degree_preserving_swap(bp.mirna_family_mask, rng)
                tr = GALDMTrainer(cfg, bp_use, ds, device=dev)
                tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
                tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
                gm, gmi, gy = tr.generate(n_gen)
                met = compute_all_metrics(real_m, real_mi, real_y, gm, gmi, gy,
                                          test_m, test_y, eval_pm, eval_fm)
                met["XM_Corr"] = xm_corr(real_mi, real_m, gmi, gm, edges)[0]
                key = "%s_%s" % (tag, cond)
                for k in mkeys:
                    out.setdefault(key, {}).setdefault(k, []).append(float(met[k]))
                print("[%s s%d] %-6s BioFID=%.2f PCE=%.3f KS=%.3f miRNA=%.3f XM=%.3f"
                      % (tag, seed, cond, met["Bio_FID"], met["PCE"], met["KS"],
                         met["miRNA_Corr"], met["XM_Corr"]), flush=True)
            with open(os.path.join(RES, "prior_scarcity_%s.json" % cancer), "w") as f:
                json.dump(out, f, indent=2)

    print("\n===== prior value vs data scarcity (mean+/-std over seeds) =====")
    def cohens_d(a, b):
        a, b = np.array(a), np.array(b); na, nb = len(a), len(b)
        sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
        return (b.mean() - a.mean()) / sp if sp > 0 else 0.0  # positive = KEGG better
    tags = sorted({k.rsplit("_", 1)[0] for k in out})
    for tag in tags:
        k, r = "%s_KEGG" % tag, "%s_Random" % tag
        if k in out and r in out:
            kb, rb = out[k]["Bio_FID"], out[r]["Bio_FID"]
            d = cohens_d(rb, kb)  # random - kegg; lower Bio-FID is better, hence the order
            print("%-5s KEGG BioFID=%.2f+/-%.2f | Random=%.2f+/-%.2f | gap=%.2f | Cohen_d=%.2f"
                  % (tag, np.mean(kb), np.std(kb), np.mean(rb), np.std(rb),
                     np.mean(rb)-np.mean(kb), d))
    return out


if __name__ == "__main__":
    t0 = time.time()
    cancer = sys.argv[1] if len(sys.argv) > 1 else "CESC"
    run(cancer)
    print("\n[done] %.0fs" % (time.time() - t0))
