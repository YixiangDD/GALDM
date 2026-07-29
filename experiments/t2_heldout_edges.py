# -*- coding: utf-8 -*-
"""Held-out edge test of the cross-modal contribution.

PCA-CTG is switched off (use_pcactg=False) so the model never touches miRTarBase edges at all;
the held-out cross-modal correlation is therefore not circular. Full (cross-modal on) is
compared against w/o-CrossModal (off), and both report XM-Corr.

miRTarBase edges are split into train and held-out two ways:
  - edge-held-out: half the edges held out at random;
  - node-held-out (stricter): held out by miRNA, so no miRNA of a held-out edge appears in any
    training edge, which rules out node-level leakage.
XM-Corr is computed separately on training and held-out edges, against a permutation null that
shuffles the generated pairing. CESC, 5 seeds.

Reading it: if Full beats w/o-CM on held-out edges, cross-modal attention preserves paired
dependence that generalises to unseen edges without ever being supervised on them."""
import os, sys, json, random, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from experiments.crossmodal_eval import load_mirtarbase_edges, xm_corr, edge_crosscorr
from configs import paths as _P

RAW = _P.RAW_ROOT; PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def split_edges(edges, seed):
    """Returns (train_e, heldE_e, heldN_e): edge-held-out holds out half the edges at random;
    node-held-out holds out by miRNA so held-out miRNAs never appear in a training edge."""
    rng = np.random.default_rng(seed + 555)
    edges = list(edges)
    # edge-held-out 50/50
    perm = rng.permutation(len(edges))
    half = len(edges) // 2
    train_e = [edges[i] for i in perm[:half]]
    heldE_e = [edges[i] for i in perm[half:]]
    # node-held-out: hold out roughly half the miRNAs
    mirnas = sorted({e[0] for e in edges})
    rng.shuffle(mirnas)
    held_mi = set(mirnas[: max(1, len(mirnas) // 2)])
    trainN = [e for e in edges if e[0] not in held_mi]
    heldN_e = [e for e in edges if e[0] in held_mi]
    return train_e, heldE_e, heldN_e, trainN


def perm_null(real_mi, real_m, gen_mi, gen_m, edges, B=1000, seed=0):
    """Mean XM-Corr under a shuffled generated pairing; lower means the observed value depends
    more on the real pairing."""
    if len(edges) < 3:
        return float("nan")
    cr = edge_crosscorr(real_mi, real_m, edges)
    rng = np.random.default_rng(seed)
    n = gen_mi.shape[0]; vals = []
    for _ in range(B):
        p = rng.permutation(n)
        cg = edge_crosscorr(gen_mi[p], gen_m, edges)
        vals.append(np.corrcoef(cr, cg)[0, 1] if np.std(cg) > 1e-8 else 0.0)
    return float(np.mean(vals))


def run(cancer="CESC", seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    conds = {"Full": dict(use_pcactg=False, use_crossmodal=True),
             "wo_CrossModal": dict(use_pcactg=False, use_crossmodal=False)}
    agg = {c: {"train": [], "heldE": [], "heldN": [], "heldE_null": []} for c in conds}

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        norm = ds["norm"]
        real_m = ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        real_mi = ds["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
        edges = load_mirtarbase_edges([str(g) for g in ds["gene_ids"]],
                                      [str(m) for m in ds["mirna_ids"]], bp.ens2sym)
        train_e, heldE_e, heldN_e, _ = split_edges(edges, seed)
        n = ds["train"]["n_samples"]
        for cname, flags in conds.items():
            set_seed(seed)
            tr = GALDMTrainer(cfg, bp, ds, device=dev, ablation_flags=flags)
            tr.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
            tr.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
            gm, gmi, gy = tr.generate(n)
            xt = xm_corr(real_mi, real_m, gmi, gm, train_e)[0]
            xe = xm_corr(real_mi, real_m, gmi, gm, heldE_e)[0]
            xn = xm_corr(real_mi, real_m, gmi, gm, heldN_e)[0]
            null_e = perm_null(real_mi, real_m, gmi, gm, heldE_e, B=1000, seed=seed)
            agg[cname]["train"].append(float(xt)); agg[cname]["heldE"].append(float(xe))
            agg[cname]["heldN"].append(float(xn)); agg[cname]["heldE_null"].append(null_e)
            print("[%s s%d] %-13s train=%.3f heldE=%.3f heldN=%.3f (heldE_null=%.3f) |E|=%d/%d/%d"
                  % (cancer, seed, cname, xt, xe, xn, null_e, len(train_e), len(heldE_e), len(heldN_e)), flush=True)
        with open(os.path.join(RES, "t2_heldout_%s.json" % cancer), "w") as f:
            json.dump(agg, f, indent=2)

    print("\n===== T2-1 held-out edge XM-Corr (mean+/-std) =====")
    for c in conds:
        r = agg[c]
        def ms(k): v = np.array(r[k]); return v.mean(), v.std()
        print("%-13s train %.3f+/-%.3f | heldE %.3f+/-%.3f | heldN %.3f+/-%.3f | heldE_null %.3f"
              % (c, *ms("train"), *ms("heldE"), *ms("heldN"), np.mean(r["heldE_null"])))
    # paired difference, Full - wo_CM, on held-out edges
    for tag in ["heldE", "heldN"]:
        fa = np.array(agg["Full"][tag]); wa = np.array(agg["wo_CrossModal"][tag])
        diff = fa - wa
        print("Full - wo_CM on %s: mean %+.3f, win %d/%d seeds" % (tag, diff.mean(), int((diff > 0).sum()), len(diff)))
    return agg


if __name__ == "__main__":
    t0 = time.time()
    c = sys.argv[1] if len(sys.argv) > 1 else "CESC"
    run(c)
    print("\n[done] %.0fs" % (time.time() - t0))
