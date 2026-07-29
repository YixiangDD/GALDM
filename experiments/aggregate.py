"""Aggregates the per-cell results in results/cells/ into mean +- s.d. tables per cohort."""
import os, json, glob, re
import numpy as np
from collections import defaultdict

CELLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cells")
METRICS = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]
ARROW = {"MMD": "↓", "KS": "↑", "miRNA_Corr": "↑", "PCE": "↓", "DE": "↑", "TSTR": "↑", "Bio_FID": "↓"}
ORDER = ["SMOTE", "CTGAN", "TabDDPM", "TabSyn", "TabDiff", "scDiffusion", "GA-LDM"]


def aggregate():
    # filename: <COHORT>_s<seed>_<method>[_quick].json
    pat = re.compile(r"^([A-Z]+)_s(\d+)_(.+?)(_quick)?\.json$")
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(os.path.join(CELLS, "*.json")):
        name = os.path.basename(f)
        m = pat.match(name)
        if not m:
            continue
        cancer, seed, method, quick = m.groups()
        if quick:
            continue
        with open(f, encoding="utf-8") as fh:
            met = json.load(fh)
        for k, v in met.items():
            data[cancer][method][k].append(v)
    return data


def main():
    data = aggregate()
    for cancer in sorted(data):
        methods = data[cancer]
        nseed = max(len(v.get("MMD", [])) for v in methods.values())
        print(f"\n===== {cancer} (n_seed={nseed}) =====")
        print("method".ljust(12) + "".join((mm + ARROW[mm]).rjust(16) for mm in METRICS))
        for method in ORDER:
            if method not in methods:
                continue
            row = method.ljust(12)
            for mm in METRICS:
                vals = [x for x in methods[method].get(mm, [])
                        if not (isinstance(x, float) and np.isnan(x))]
                row += (f"{np.mean(vals):.3f}±{np.std(vals):.3f}".rjust(16) if vals
                        else "nan".rjust(16))
            print(row)


if __name__ == "__main__":
    main()
