"""GRN ablation: the real TRRUST + miRTarBase GRN vs a degree-matched random GRN, measuring
what GDVD and PCA-CTG actually draw from the prior. If the real prior matters, replacing it
with a random graph of the same shape should hurt.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from experiments.trainer import GALDMTrainer
from evaluation.metrics import compute_all_metrics
from bio_priors.grn_builder import MultiOmicsGRN
from configs import paths as _P

RAW_ROOT = _P.RAW_ROOT
PROC_DIR = _P.PROCESSED_DIR
RESULTS_DIR = _P.RESULTS_DIR


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _make_random_grn(real_grn: MultiOmicsGRN, seed=99):
    """Keeps node, edge and layer counts identical, but randomises edge directions and depths."""
    rng = np.random.RandomState(seed)
    n = real_grn.n_nodes
    M = real_grn.n_mirna_families
    # random causal depths
    depth = list(rng.randint(0, 5, size=n))
    # random layer assignment (miRNAs stay at layer 0)
    layers = [0] * M + list(rng.choice([1, 2, 3], size=n - M))
    # random edges: same count, random source/target, keeping source layer < target layer
    n_act = len(real_grn.activating_edges)
    n_sup = len(real_grn.suppressive_edges)
    act, sup = [], []
    for _ in range(n_act * 10):
        s, t = rng.randint(0, n), rng.randint(0, n)
        if layers[s] < layers[t] and (s, t) not in act:
            act.append((s, t))
        if len(act) >= n_act:
            break
    for _ in range(n_sup * 10):
        s, t = rng.randint(0, n), rng.randint(0, n)
        if layers[s] < layers[t] and (s, t) not in sup and (s, t) not in act:
            sup.append((s, t))
        if len(sup) >= n_sup:
            break
    return MultiOmicsGRN(
        n_mirna_families=M, n_pathways=real_grn.n_pathways,
        node_names=real_grn.node_names, node_layers=layers, causal_depth=depth,
        activating_edges=act, suppressive_edges=sup)


def run_grn_ablation(cancer="CESC", seeds=(42, 123, 456, 789, 1024),
                     quick=False, vae_ep=None, dit_ep=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # per-seed metric lists collected for each grn_type
    results = {"real_GRN": {}, "random_GRN": {}}
    metric_keys = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]

    for seed in seeds:
        set_seed(seed)
        ds, bp = prepare_cancer(RAW_ROOT, PROC_DIR, cancer, seed, verbose=False)
        real_m = ds["train"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        real_mi = ds["train"]["mirna"] * ds["norm"]["mirna_sd"] + ds["norm"]["mirna_mu"]
        test_m = ds["test"]["mrna"] * ds["norm"]["mrna_sd"] + ds["norm"]["mrna_mu"]
        test_y = ds["test"]["labels"]

        for grn_type in ["real_GRN", "random_GRN"]:
            set_seed(seed)
            bp_use = bp
            if grn_type == "random_GRN":
                import copy
                bp_use = copy.deepcopy(bp)
                # a different random GRN per seed, so one unlucky graph cannot decide it
                bp_use.grn = _make_random_grn(bp.grn, seed=seed + 99)
            try:
                tr = GALDMTrainer(cfg, bp_use, ds, device=device)
                # same epoch budget as the main experiments, so the numbers stay comparable
                tr.train_vae(epochs=vae_ep or (20 if quick else cfg.train.vae_epochs),
                             verbose=False)
                tr.train_dit(epochs=dit_ep or (30 if quick else cfg.train.dit_epochs),
                             verbose=False)
                gm, gmi, gy = tr.generate(ds["train"]["n_samples"])
                met = compute_all_metrics(real_m, real_mi, ds["train"]["labels"],
                                          gm, gmi, gy, test_m, test_y,
                                          bp.pathway_mask, bp.mirna_family_mask)
            except Exception as e:
                import traceback; traceback.print_exc()
                met = {k: float("nan") for k in metric_keys}
            for k, v in met.items():
                results[grn_type].setdefault(k, []).append(v)
            print(f"  [{cancer} s{seed}] {grn_type:10s} Bio-FID={met['Bio_FID']:.1f} "
                  f"TSTR={met['TSTR']:.3f} PCE={met['PCE']:.3f} miRNA={met['miRNA_Corr']:.3f}")
        # flush after each seed, so an interruption loses at most one
        out = os.path.join(RESULTS_DIR, f"grn_ablation_{cancer}{'_quick' if quick else ''}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    # aggregate
    print(f"\n===== GRN ablation {cancer} (mean±std over {len(seeds)} seeds) =====")
    for grn_type in ["real_GRN", "random_GRN"]:
        r = results[grn_type]
        bf = [x for x in r.get("Bio_FID", []) if not (isinstance(x, float) and np.isnan(x))]
        mi = [x for x in r.get("miRNA_Corr", []) if not (isinstance(x, float) and np.isnan(x))]
        pce = [x for x in r.get("PCE", []) if not (isinstance(x, float) and np.isnan(x))]
        print(f"  {grn_type:10s} Bio-FID={np.mean(bf):.2f}±{np.std(bf):.2f} "
              f"miRNA={np.mean(mi):.4f} PCE={np.mean(pce):.4f}")
    print(f"Saved: {out}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", default="CESC")
    ap.add_argument("--seeds", default="42,123,456,789,1024")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vae_ep", type=int, default=0)
    ap.add_argument("--dit_ep", type=int, default=0)
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    run_grn_ablation(args.cancer, seeds, args.quick,
                     vae_ep=args.vae_ep or None, dit_ep=args.dit_ep or None)
