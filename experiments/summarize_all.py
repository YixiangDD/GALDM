"""Collects every experiment result into one multi-sheet Excel workbook.

Covers the comparison (4 cohorts x 5 seeds), both ablations (progressive M0-M5 and
leave-one-out), the GRN ablation, the sensitivity sweep and the interpretability case study.
"""
import os, json, glob, re
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
METRICS = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]
AR = {"MMD": "↓", "KS": "↑", "miRNA_Corr": "↑", "PCE": "↓", "DE": "↑", "TSTR": "↑", "Bio_FID": "↓"}
ORDER = ["SMOTE", "CTGAN", "TabDDPM", "TabSyn", "TabDiff", "scDiffusion", "GA-LDM"]
CANCERS = ["CESC", "COAD", "HNSC", "KIRC"]


def _ms(vals):
    v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
    return (np.mean(v), np.std(v)) if v else (float("nan"), float("nan"))


def load_comparison():
    pat = re.compile(r"^([A-Z]+)_s(\d+)_(.+?)\.json$")
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(os.path.join(RES, "cells", "*.json")):
        m = pat.match(os.path.basename(f))
        if not m or "quick" in f:
            continue
        c, s, method = m.groups()
        with open(f, encoding="utf-8") as fh:
            met = json.load(fh)
        for k, v in met.items():
            data[c][method][k].append(v)
    return data


def main():
    out = os.path.join(RES, "all_experiment_results.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # ===== 1. comparison =====
        comp = load_comparison()
        rows = []
        for c in CANCERS:
            for method in ORDER:
                if method not in comp.get(c, {}):
                    continue
                row = {"dataset": c, "method": method}
                for mt in METRICS:
                    mean, std = _ms(comp[c][method].get(mt, []))
                    row[f"{mt}{AR[mt]}"] = f"{mean:.4f}±{std:.4f}"
                rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name="1_comparison", index=False)

        # ===== 2. ablations: progressive and leave-one-out =====
        for mode, sheet in [("progressive", "2_ablation_progressive"),
                            ("leave_one_out", "3_ablation_leave_one_out")]:
            p = os.path.join(RES, f"ablation_{mode}_CESC.json")
            if os.path.exists(p):
                d = json.load(open(p, encoding="utf-8"))
                rows = []
                for variant, met in d.items():
                    row = {"variant": variant}
                    for mt in METRICS:
                        mean, std = _ms(met.get(mt, []))
                        row[f"{mt}{AR[mt]}"] = f"{mean:.4f}±{std:.4f}"
                    rows.append(row)
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)

        # ===== 4. GRN ablation =====
        p = os.path.join(RES, "grn_ablation_CESC.json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            rows = []
            for variant, met in d.items():
                row = {"grn_type": variant}
                row.update({f"{k}{AR.get(k,'')}": round(met[k], 4) for k in METRICS if k in met})
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name="4_grn_real_vs_random", index=False)

        # ===== 5. hyper-parameter sensitivity =====
        p = os.path.join(RES, "sensitivity_CESC.json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            rows = []
            for param, vals in d.items():
                for v, met in vals.items():
                    row = {"parameter": param, "value": v}
                    row.update({k: round(met[k], 4) for k in METRICS if k in met})
                    rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name="5_sensitivity", index=False)

        # ===== 6. interpretability =====
        p = os.path.join(RES, "interpretability_KIRC.json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            rows = [{"item": k, "value": str(v)} for k, v in d.items()]
            pd.DataFrame(rows).to_excel(writer, sheet_name="6_interpretability_KIRC_HIF1", index=False)

    print(f"wrote summary workbook: {out}  ({os.path.getsize(out)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
