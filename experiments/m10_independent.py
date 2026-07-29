# -*- coding: utf-8 -*-
"""Independent-generation control: what does joint modelling actually buy?

The control trains two fully independent joint models from different initialisations, takes
mRNA from model A and miRNA from model B, and pairs them at random within each class, so the
two modalities share no sample-level correspondence. This is compared against the joint model.
What matters is cross-modal dependence: the joint model should preserve sample-level
mRNA-miRNA coordination (visible in downstream TSTR/DE and cross-modal consistency) whereas
independent generation destroys it. CESC, 5 seeds."""
import os, sys, json, time, random
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from evaluation.metrics import compute_all_metrics
from configs import paths as _P

RAW = _P.RAW_ROOT
PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run(cancer="CESC", seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"Joint": {}, "Independent": {}}
    mkeys = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
        real_y = ds["train"]["labels"]
        test_m = ds["test"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        test_y = ds["test"]["labels"]
        n = ds["train"]["n_samples"]

        # --- joint: full GA-LDM ---
        set_seed(seed)
        trJ = GALDMTrainer(cfg, bp, ds, device=device)
        trJ.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
        trJ.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
        gmJ, gmiJ, gyJ = trJ.generate(n)
        metJ = compute_all_metrics(real_m, real_mi, real_y, gmJ, gmiJ, gyJ,
                                   test_m, test_y, bp.pathway_mask, bp.mirna_family_mask)

        # --- independent: mRNA from model A (seed), miRNA from model B (seed+5000),
        #     paired at random within class ---
        # model A output is reused from the joint run above (gmJ, gyJ); model B is trained on
        # an independent seed and its miRNA output is shuffled within class, which cuts the
        # sample-level correspondence with A
        set_seed(seed + 5000)
        dsB, bpB = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
        trB = GALDMTrainer(cfg, bpB, dsB, device=device)
        trB.train_vae(epochs=cfg.train.vae_epochs, verbose=False)
        trB.train_dit(epochs=cfg.train.dit_epochs, verbose=False)
        _, gmiB, gyB = trB.generate(n)
        # within-class random pairing: for each A sample draw a same-class miRNA from B
        perm = np.arange(n)
        for c in np.unique(gyJ):
            idxA = np.where(gyJ == c)[0]
            poolB = np.where(gyB == c)[0]
            if len(poolB) == 0:
                poolB = np.arange(n)
            perm[idxA] = np.random.choice(poolB, size=len(idxA), replace=True)
        gmi_indep = gmiB[perm]
        metI = compute_all_metrics(real_m, real_mi, real_y, gmJ, gmi_indep, gyJ,
                                   test_m, test_y, bp.pathway_mask, bp.mirna_family_mask)

        for k in mkeys:
            results["Joint"].setdefault(k, []).append(metJ[k])
            results["Independent"].setdefault(k, []).append(metI[k])
        print(f"[{cancer} s{seed}] Joint miRNA={metJ['miRNA_Corr']:.4f} BioFID={metJ['Bio_FID']:.1f} "
              f"| Indep miRNA={metI['miRNA_Corr']:.4f} BioFID={metI['Bio_FID']:.1f}", flush=True)
        with open(os.path.join(RES, f"m10_independent_{cancer}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("\n===== M10 summary (mean over seeds) =====")
    for grp in ["Joint", "Independent"]:
        r = results[grp]
        def m(k):
            v = [x for x in r[k] if not (isinstance(x, float) and np.isnan(x))]
            return np.mean(v), np.std(v)
        print("%-12s miRNA=%.4f±%.4f PCE=%.4f DE=%.4f TSTR=%.4f BioFID=%.2f" % (
            grp, *m("miRNA_Corr"), m("PCE")[0], m("DE")[0], m("TSTR")[0], m("Bio_FID")[0]))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", default="CESC")
    ap.add_argument("--seeds", default="42,123,456,789,1024")
    a = ap.parse_args()
    run(a.cancer, tuple(int(s) for s in a.seeds.split(",")))
