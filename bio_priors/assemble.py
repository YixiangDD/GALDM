"""Assembles every biological prior PA-VAE / GDVD / PCA-CTG needs for one cohort.

Produces BioPriors: pathway masks, miRNA family masks and the real multi-omics GRN
(causal depth, layers, activating/suppressive edges), cached to
"""
import os
import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Dict

from .pathway_mask import select_genes_variance_pathway_union, build_pathway_mask
from .mirna_families import assign_mirna_families
from .grn_builder import build_multiomics_grn, MultiOmicsGRN


@dataclass
class BioPriors:
    pathway_mask: np.ndarray          # (K+1, n_genes), last row is background
    pathway_info: Dict
    mirna_family_mask: np.ndarray     # (M, n_mirna)
    family_info: Dict
    grn: MultiOmicsGRN
    selected_gene_ids: list           # selected Ensembl IDs, aligned with mRNA columns
    ens2sym: Dict


def build_bio_priors(mrna_train_df, all_gene_ids, mirna_ids,
                     n_genes=2000, n_pathways=15, n_mirna_families=8,
                     verbose=True) -> BioPriors:
    """Builds all priors on the training split. mrna_train_df columns must be Ensembl IDs."""
    sel, e2s = select_genes_variance_pathway_union(
        mrna_train_df, all_gene_ids, n_genes, verbose=verbose)
    pmask, pinfo = build_pathway_mask(sel, e2s, verbose=verbose)
    fmask, finfo = assign_mirna_families(mirna_ids, n_mirna_families, verbose=verbose)
    grn = build_multiomics_grn(pinfo, finfo, set(e2s.values()), verbose=verbose)
    return BioPriors(pmask, pinfo, fmask, finfo, grn, sel, e2s)


def save_bio_priors(bp: BioPriors, processed_dir: str, cancer: str, seed: int):
    os.makedirs(processed_dir, exist_ok=True)
    p = os.path.join(processed_dir, f"{cancer}_seed{seed}_priors.pkl")
    with open(p, "wb") as f:
        pickle.dump(bp, f)
    return p


def load_bio_priors(processed_dir: str, cancer: str, seed: int):
    p = os.path.join(processed_dir, f"{cancer}_seed{seed}_priors.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)
