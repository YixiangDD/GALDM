"""TCGA multi-omics loading and preprocessing.

Protocol:
  - primary tumour samples only (4th barcode field starting with a tumour code, 01/03)
  - keep only samples that have both mRNA and miRNA
  - mRNA: drop genes that are zero in every sample, then take the 2000 highest-variance
    genes as measured on the training split
  - miRNA: the 200 highest-variance miRNAs on the training split
  - label: AJCC/FIGO stage I/II = 0 (early), III/IV = 1 (late)
  - stratified 70/15/15 split; feature selection happens on the training split only, so no
    test-set information leaks into it
Gene IDs are versioned Ensembl; mapping to symbols for KEGG/TRRUST happens in bio_priors.
"""
import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split


def _raw_candidates(raw_root: str, cancer: str) -> Dict[str, list]:
    """Candidate paths per modality, highest priority first.

    UCSC Xena and the GDC ship tab-separated files in feature x sample orientation, named
    e.g. TCGA-CESC.star_fpkm.tsv. Some mirrors provide a comma-separated copy, and a
    pre-transposed sample x feature form (mRNA_com.csv) may exist from earlier processing.
    All of these are accepted: the separator is taken from the extension and the orientation
    is detected from the first column header, so no manual conversion step is needed.

    Both layouts are searched: the files directly under <raw_root>/<cancer>/, and the same
    files inside a directory named after the download (which is how Xena bundles them).
    """
    base = os.path.join(raw_root, cancer)
    stem = {"mrna": f"TCGA-{cancer}.star_fpkm",
            "mirna": f"TCGA-{cancer}.mirna",
            "clinical": f"TCGA-{cancer}.clinical"}
    pre = {"mrna": "mRNA_com.csv", "mirna": "miRNA_com.csv", "clinical": "clinical_com.csv"}
    plain = {"mrna": None, "mirna": None, "clinical": "clinical.csv"}

    out = {}
    for mod in ("mrna", "mirna", "clinical"):
        names = [pre[mod]]                      # pre-transposed, if a previous run made one
        if plain[mod]:
            names.append(plain[mod])
        names += [stem[mod] + ".csv", stem[mod] + ".tsv",
                  stem[mod] + ".csv.gz", stem[mod] + ".tsv.gz"]
        cands = []
        for d in (os.path.join(base, stem[mod] + ".tsv"), base):
            for n in names:
                cands.append(os.path.join(d, n))
        out[mod] = cands
    return out


def _first_existing(cands: list, what: str) -> str:
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "no %s file found for this cohort. Tried:\n  %s\n"
        "Download the cohort from https://xenabrowser.net/datapages/ and place the files "
        "under <raw_root>/<COHORT>/ (see the Data section of the README)."
        % (what, "\n  ".join(cands)))


def _read_table(path: str, **kw) -> pd.DataFrame:
    """Reads csv/tsv (optionally gzipped), taking the separator from the extension."""
    low = path.lower()
    if low.endswith(".gz"):
        low = low[:-3]
    sep = "\t" if low.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep, **kw)


# feature-id column names used by the raw feature x sample files
_FEAT_ID_COLS = {"ensembl_id", "mirna_id", "gene_id", "sample_id", "id"}


def _load_expr_matrix(cands: list, what: str) -> pd.DataFrame:
    """Returns an expression matrix with rows = samples, columns = features.

    Orientation is detected from the header rather than assumed from the filename: if the
    first column is a feature id (Ensembl_ID, miRNA_ID, ...) the file is feature x sample and
    gets transposed; otherwise it is already sample x feature.
    """
    path = _first_existing(cands, what)
    df = _read_table(path)
    df = df.set_index(df.columns[0])
    if str(df.index.name).strip().lower() in _FEAT_ID_COLS:
        return df.T          # feature x sample -> sample x feature
    return df                # already sample x feature


_STAGE_COLS = [
    "ajcc_pathologic_stage.diagnoses",
    "figo_stage.diagnoses",
    "ajcc_clinical_stage.diagnoses",
]


def _stage_to_binary(val) -> Optional[int]:
    """Stage string or code -> 0 (early, I/II), 1 (late, III/IV), None (unknown).

    Handles the TCGA spellings 'Stage IB1', 'Stage IIIA', 'Stage IVB' and similar by
    stripping the 'stage' prefix, taking the leading run of Roman numerals, then testing
    for III/IV.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "not reported", "unknown",
                                 "[not available]", "[unknown]", "stage x"):
        return None
    # the reduced KIRC file already encodes 0/1/2/3 = stage I/II/III/IV
    if s in ("0", "1", "2", "3"):
        return 0 if int(s) <= 1 else 1
    # take the leading Roman numerals after 'stage' (case-insensitive), then drop A/B/C and
    # numeric suffixes
    m = re.search(r"stage\s*([ivx]+)", s, re.IGNORECASE)
    if not m:
        return None
    roman = m.group(1).upper()
    if roman in ("III", "IV"):
        return 1
    if roman in ("I", "II"):
        return 0
    return None


def _is_primary_tumor(barcode: str) -> bool:
    """In a TCGA barcode the 4th field is the sample type; 01-09 are tumour, 01 = primary solid."""
    parts = barcode.split("-")
    if len(parts) < 4:
        return True
    code = parts[3][:2]
    return code in ("01", "03")  # 01 primary solid tumour, 03 primary blood tumour


def _load_clinical_labels(paths: Dict[str, list]) -> pd.Series:
    """Returns a Series indexed by patient prefix (TCGA-XX-XXXX) holding the binary label."""
    path = _first_existing(paths["clinical"], "clinical")
    cl = _read_table(path, low_memory=False)
    # id column: prefer `sample`, else `submitter_id`
    id_col = "sample" if "sample" in cl.columns else (
        "submitter_id" if "submitter_id" in cl.columns else cl.columns[0])
    # stage column, by priority; tolerates column names mangled by an i->3 substitution
    stage_col = None
    for c in _STAGE_COLS:
        if c in cl.columns:
            stage_col = c
            break
    if stage_col is None:
        for c in cl.columns:
            cl_low = c.lower().replace("3", "i")
            if "stage" in cl_low and ("diagnoses" in cl_low or "pathologic" in cl_low):
                stage_col = c
                break
    if stage_col is None:
        raise ValueError(f"no stage column in {path}; columns look like {list(cl.columns)[:10]}")

    labels = {}
    for _, row in cl.iterrows():
        raw_id = str(row[id_col])
        pid = "-".join(raw_id.split("-")[:3])  # patient prefix
        lab = _stage_to_binary(row[stage_col])
        if lab is not None and pid not in labels:
            labels[pid] = lab
    return pd.Series(labels, name="label")


def _patient_prefix(barcode: str) -> str:
    return "-".join(barcode.split("-")[:3])


def load_raw_cancer(raw_root: str, cancer: str) -> Dict:
    """Loads the raw paired data for one cohort and labels it, without feature selection."""
    paths = _raw_candidates(raw_root, cancer)
    mrna = _load_expr_matrix(paths["mrna"], "mRNA expression")
    mirna = _load_expr_matrix(paths["mirna"], "miRNA expression")

    # keep primary tumour samples only (samples are on the row index)
    mrna = mrna.loc[[i for i in mrna.index if _is_primary_tumor(i)]]
    mirna = mirna.loc[[i for i in mirna.index if _is_primary_tumor(i)]]

    # pair up: samples that have both mRNA and miRNA
    common = sorted(set(mrna.index) & set(mirna.index))
    mrna = mrna.loc[common]
    mirna = mirna.loc[common]

    # label by patient prefix, and deduplicate at patient level (one aliquot per patient) so
    # the same patient cannot land in two splits
    labels = _load_clinical_labels(paths)
    sample_labels = []
    keep = []
    seen_patients = set()
    for s in common:                       # common is sorted, so the dedup rule is deterministic: keep the first aliquot
        pt = _patient_prefix(s)
        lab = labels.get(pt)
        if lab is not None and pt not in seen_patients:
            sample_labels.append(lab)
            keep.append(s)
            seen_patients.add(pt)
    mrna = mrna.loc[keep]
    mirna = mirna.loc[keep]
    y = np.array(sample_labels, dtype=np.int64)

    return {
        "cancer": cancer,
        "mrna_df": mrna,                       # (N, ~14k) Ensembl IDs
        "mirna_df": mirna,                     # (N, ~400)
        "labels": y,                           # (N,)
        "sample_ids": keep,
        "gene_ids": list(mrna.columns),
        "mirna_ids": list(mirna.columns),
    }


def _log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """log2(x+1). The raw _com.csv files are mostly linear FPKM/RPM values."""
    arr = df.values.astype(np.float64)
    # if the data already looks log-scaled (small maximum), skip; otherwise apply log2(x+1)
    if np.nanmax(arr) > 30:
        arr = np.log2(arr + 1.0)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def select_features_train(
    mrna_tr: pd.DataFrame, mirna_tr: pd.DataFrame, n_genes: int, n_mirna: int,
) -> Tuple[list, list]:
    """Selects features on the training split only, to avoid leakage. Returns column names."""
    # mRNA: drop all-zero genes, then take the top-n_genes by variance
    nz = (mrna_tr != 0).any(axis=0)
    mrna_nz = mrna_tr.loc[:, nz]
    gene_var = mrna_nz.var(axis=0)
    top_genes = gene_var.sort_values(ascending=False).head(n_genes).index.tolist()
    # miRNA: top-n_mirna by variance
    mi_var = mirna_tr.var(axis=0)
    top_mirna = mi_var.sort_values(ascending=False).head(n_mirna).index.tolist()
    return top_genes, top_mirna


def build_dataset(
    raw_root: str, cancer: str, n_genes: int, n_mirna: int,
    val_split: float, test_split: float, seed: int,
    gene_ids: Optional[list] = None, mirna_ids: Optional[list] = None,
    return_raw_train: bool = False,
) -> Dict:
    """Builds one cohort end to end: load -> log2 -> stratified split -> feature selection on
    the training split -> standardisation.

    If gene_ids/mirna_ids are supplied (from the variance-plus-pathway-union selection in
    bio_priors), they are used instead of the internal variance-only selection, which keeps
    the data columns strictly aligned with the pathway mask.
    With return_raw_train=True the unfiltered training mRNA frame is also returned, for
    building the biological priors.
    """
    raw = load_raw_cancer(raw_root, cancer)
    mrna = _log2_transform(raw["mrna_df"])
    mirna = _log2_transform(raw["mirna_df"])
    y = raw["labels"]
    N = len(y)

    idx = np.arange(N)
    # stratified 70/15/15 split
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=(val_split + test_split), stratify=y, random_state=seed)
    rel_test = test_split / (val_split + test_split)
    idx_val, idx_te, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed)

    # features: use the externally supplied union if given, else variance on the training split
    if gene_ids is not None and mirna_ids is not None:
        top_genes, top_mirna = gene_ids, mirna_ids
    else:
        top_genes, top_mirna = select_features_train(
            mrna.iloc[idx_tr], mirna.iloc[idx_tr], n_genes, n_mirna)

    mrna_sel = mrna[top_genes]
    mirna_sel = mirna[top_mirna]

    # z-score using training-split statistics; the diffusion stage standardises the latent again
    mu_m, sd_m = mrna_sel.iloc[idx_tr].mean(0), mrna_sel.iloc[idx_tr].std(0).replace(0, 1)
    mu_mi, sd_mi = mirna_sel.iloc[idx_tr].mean(0), mirna_sel.iloc[idx_tr].std(0).replace(0, 1)
    mrna_z = (mrna_sel - mu_m) / sd_m
    mirna_z = (mirna_sel - mu_mi) / sd_mi

    def pack(ix):
        return {
            "mrna": mrna_z.iloc[ix].values.astype(np.float32),
            "mirna": mirna_z.iloc[ix].values.astype(np.float32),
            "labels": y[ix].astype(np.int64),
            "n_samples": len(ix),
        }

    result = {
        "cancer": cancer,
        "train": pack(idx_tr),
        "val": pack(idx_val),
        "test": pack(idx_te),
        "gene_ids": top_genes,
        "mirna_ids": top_mirna,
        "norm": {  # kept for de-standardisation, needed when post-processing clips to real quantiles
            "mrna_mu": mu_m.values.astype(np.float32), "mrna_sd": sd_m.values.astype(np.float32),
            "mirna_mu": mu_mi.values.astype(np.float32), "mirna_sd": sd_mi.values.astype(np.float32),
        },
        "meta": {
            "n_total": N, "n_genes": len(top_genes), "n_mirna": len(top_mirna),
            "n_train": len(idx_tr), "n_val": len(idx_val), "n_test": len(idx_te),
            "class_balance": [int((y == 0).sum()), int((y == 1).sum())],
        },
    }
    if return_raw_train:
        # unfiltered training mRNA (all Ensembl IDs as columns), for gene selection in bio_priors
        result["raw_train_mrna_df"] = mrna.iloc[idx_tr]
        result["all_gene_ids"] = list(mrna.columns)
        result["all_mirna_ids"] = list(mirna.columns)
    return result


def cache_path(processed_dir: str, cancer: str, seed: int) -> str:
    return os.path.join(processed_dir, f"{cancer}_seed{seed}.npz")


def save_dataset(ds: Dict, processed_dir: str, seed: int):
    os.makedirs(processed_dir, exist_ok=True)
    p = cache_path(processed_dir, ds["cancer"], seed)
    np.savez_compressed(
        p,
        train_mrna=ds["train"]["mrna"], train_mirna=ds["train"]["mirna"], train_y=ds["train"]["labels"],
        val_mrna=ds["val"]["mrna"], val_mirna=ds["val"]["mirna"], val_y=ds["val"]["labels"],
        test_mrna=ds["test"]["mrna"], test_mirna=ds["test"]["mirna"], test_y=ds["test"]["labels"],
        gene_ids=np.array(ds["gene_ids"]), mirna_ids=np.array(ds["mirna_ids"]),
        mrna_mu=ds["norm"]["mrna_mu"], mrna_sd=ds["norm"]["mrna_sd"],
        mirna_mu=ds["norm"]["mirna_mu"], mirna_sd=ds["norm"]["mirna_sd"],
        meta=json.dumps(ds["meta"]),
    )
    # also write meta as standalone json for inspection
    with open(p.replace(".npz", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"cancer": ds["cancer"], "seed": seed, **ds["meta"],
                   "gene_ids_head": ds["gene_ids"][:10], "mirna_ids_head": ds["mirna_ids"][:10]},
                  f, indent=2, ensure_ascii=False)
    return p


def load_cached(processed_dir: str, cancer: str, seed: int) -> Optional[Dict]:
    p = cache_path(processed_dir, cancer, seed)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    def pack(pre):
        return {"mrna": d[f"{pre}_mrna"], "mirna": d[f"{pre}_mirna"],
                "labels": d[f"{pre}_y"], "n_samples": len(d[f"{pre}_y"])}
    return {
        "cancer": cancer,
        "train": pack("train"), "val": pack("val"), "test": pack("test"),
        "gene_ids": list(d["gene_ids"]), "mirna_ids": list(d["mirna_ids"]),
        "norm": {"mrna_mu": d["mrna_mu"], "mrna_sd": d["mrna_sd"],
                 "mirna_mu": d["mirna_mu"], "mirna_sd": d["mirna_sd"]},
        "meta": json.loads(str(d["meta"])),
    }
