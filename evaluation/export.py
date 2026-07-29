"""Exports results to multi-sheet Excel workbooks, plus the generated samples themselves.

- export_comparison_excel: comparison results (mean +- s.d. and per-seed) as one sheet per
  metric (method x cohort), plus a summary sheet and a significance sheet.
- export_ablation_excel: ablation results.
- export_generated_samples: generated mRNA/miRNA samples and labels, with gene/miRNA names
  as column headers.
"""
import os
import numpy as np
import pandas as pd

METRICS = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID"]
METRIC_DIR = {"MMD": "↓", "KS": "↑", "miRNA_Corr": "↑", "PCE": "↓",
              "DE": "↑", "TSTR": "↑", "Bio_FID": "↓"}


def _mean_std(vals):
    v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def export_comparison_excel(all_results, out_path):
    """all_results: {cancer: {method: {metric: [over seeds]}}}"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # summary sheet: mean +- s.d. per cohort, method and metric
        rows = []
        for cancer, methods in all_results.items():
            for method, md in methods.items():
                row = {"cohort": cancer, "method": method}
                for m in METRICS:
                    mean, std = _mean_std(md.get(m, []))
                    row[f"{m}{METRIC_DIR[m]}"] = f"{mean:.4f}±{std:.4f}"
                rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name="summary", index=False)

        # one sheet per metric: rows = methods, columns = cohorts (mean)
        for m in METRICS:
            data = {}
            methods_order = list(next(iter(all_results.values())).keys())
            for cancer, methods in all_results.items():
                col = []
                for method in methods_order:
                    mean, _ = _mean_std(methods[method].get(m, []))
                    col.append(round(mean, 4))
                data[cancer] = col
            df = pd.DataFrame(data, index=methods_order)
            df.index.name = f"{m}{METRIC_DIR[m]}"
            df.to_excel(writer, sheet_name=m[:31])

        # per-seed detail
        detail = []
        for cancer, methods in all_results.items():
            for method, md in methods.items():
                n_seed = max((len(v) for v in md.values()), default=0)
                for s in range(n_seed):
                    row = {"cohort": cancer, "method": method, "seed": s}
                    for m in METRICS:
                        vals = md.get(m, [])
                        row[m] = round(vals[s], 4) if s < len(vals) else None
                    detail.append(row)
        pd.DataFrame(detail).to_excel(writer, sheet_name="per_seed", index=False)
    print(f"  [export] comparison table -> {out_path}")


def export_significance_excel(sig_results, out_path):
    """sig_results: {cancer: {metric: {method: p_value}}}, paired tests of GA-LDM vs each baseline."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for cancer, mm in sig_results.items():
        for metric, methods in mm.items():
            for method, p in methods.items():
                rows.append({"cohort": cancer, "metric": metric, "baseline": method,
                             "p_value": round(p, 5) if not np.isnan(p) else None,
                             "significant_p<0.05": "yes" if (not np.isnan(p) and p < 0.05) else "no"})
    pd.DataFrame(rows).to_excel(out_path, index=False)
    print(f"  [export] significance tests -> {out_path}")


def export_ablation_excel(all_results, out_path, mode="progressive"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for cancer, variants in all_results.items():
            rows = []
            for vname, md in variants.items():
                row = {"variant": vname}
                for m in METRICS:
                    mean, std = _mean_std(md.get(m, []))
                    row[f"{m}{METRIC_DIR[m]}"] = f"{mean:.4f}±{std:.4f}"
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name=f"{cancer}_{mode}"[:31], index=False)
    print(f"  [export] ablation table -> {out_path}")


def export_generated_samples(gen_mrna, gen_mirna, gen_labels, gene_ids, mirna_ids,
                             out_path, max_rows=2000):
    """Writes the generated samples to xlsx: an mRNA sheet, a miRNA sheet and the labels."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = min(len(gen_mrna), max_rows)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        dfm = pd.DataFrame(gen_mrna[:n], columns=gene_ids)
        dfm.insert(0, "label", gen_labels[:n])
        dfm.insert(0, "sample_id", [f"GEN_{i:05d}" for i in range(n)])
        dfm.to_excel(writer, sheet_name="mRNA", index=False)
        dfmi = pd.DataFrame(gen_mirna[:n], columns=mirna_ids)
        dfmi.insert(0, "label", gen_labels[:n])
        dfmi.insert(0, "sample_id", [f"GEN_{i:05d}" for i in range(n)])
        dfmi.to_excel(writer, sheet_name="miRNA", index=False)
    print(f"  [export] generated samples -> {out_path}")


def export_sensitivity_excel(results, out_path):
    """results: {param: {value: {metric: v}}}"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for param, vals in results.items():
            rows = []
            for v, met in vals.items():
                row = {param: v}
                row.update({k: round(met[k], 4) for k in METRICS if k in met})
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name=param[:31], index=False)
    print(f"  [export] sensitivity -> {out_path}")
