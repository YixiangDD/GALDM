"""Pathway mask construction plus variance/pathway-union gene selection.

Selection strategy: take the top-(n_genes - P) highest-variance genes and force in the
members of the 15 KEGG pathways (those that map to a symbol and exist in the data, P in
total). This lifts pathway coverage enough for the pathway-aware encoder to be meaningful.
The pathway mask M in {0,1}^(K+1, n_genes) has K=15 rows for the KEGG pathways (a gene may
belong to several) and a final background row holding genes in no selected pathway.
"""
import numpy as np
from typing import Dict, List, Tuple

from .kegg_pathways import get_kegg_pathway_genes, KEGG_15_PATHWAYS
from .gene_mapping import map_ensembl_to_symbol


def select_genes_variance_pathway_union(
    mrna_train_df, ensembl_ids: List[str], n_genes: int,
    verbose: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """On the training split: top-(n_genes-P) by variance, unioned with KEGG members.

    Args:
        mrna_train_df: training mRNA frame (samples x all genes), columns are Ensembl IDs
        ensembl_ids: all gene column names (versioned Ensembl)
        n_genes: target gene count (2000)
    Returns:
        selected_ids: selected Ensembl IDs (length n_genes)
        ens2sym: {ensembl_id -> symbol} for the selected genes only
    """
    # 1) map every gene to a symbol
    ens2sym_all = map_ensembl_to_symbol(ensembl_ids, verbose=verbose)
    sym2ens = {}
    for eid, sym in ens2sym_all.items():
        sym2ens.setdefault(sym, eid)  # one representative Ensembl ID per symbol

    # 2) all member symbols of the 15 KEGG pathways -> Ensembl IDs present in the data
    kegg = get_kegg_pathway_genes(verbose=False)
    pathway_syms = set()
    for syms in kegg.values():
        pathway_syms.update(syms)
    pathway_ens = [sym2ens[s] for s in pathway_syms if s in sym2ens]
    pathway_ens = sorted(set(pathway_ens))
    if verbose:
        print(f"  [select] {len(pathway_ens)} KEGG pathway members found in the data")

    # 3) rank by variance, dropping all-zero genes
    nz = (mrna_train_df != 0).any(axis=0)
    mrna_nz = mrna_train_df.loc[:, nz]
    gene_var = mrna_nz.var(axis=0).sort_values(ascending=False)
    var_ranked = gene_var.index.tolist()

    # 4) union: pathway genes first, then top-variance genes up to n_genes
    selected = list(dict.fromkeys(pathway_ens))  # order-preserving dedup
    for eid in var_ranked:
        if len(selected) >= n_genes:
            break
        if eid not in selected:
            selected.append(eid)
    selected = selected[:n_genes]

    ens2sym = {eid: ens2sym_all[eid] for eid in selected if eid in ens2sym_all}
    if verbose:
        n_pw_in = sum(1 for e in selected if e in set(pathway_ens))
        print(f"  [select] {len(selected)} genes kept, {n_pw_in} of them in a pathway "
              f"({100*n_pw_in/len(selected):.1f}%)")
    return selected, ens2sym


def build_pathway_mask(
    selected_ids: List[str], ens2sym: Dict[str, str],
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Builds the pathway mask M in {0,1}^((K+1), n_genes).

    K=15 KEGG pathways plus one background group; genes may appear in several pathways.
    Returns the mask and pathway_info (pathway names, gene counts per pathway, ...).
    """
    kegg = get_kegg_pathway_genes(verbose=False)
    hsa_ids = list(KEGG_15_PATHWAYS.keys())  # fixed order
    K = len(hsa_ids)
    n_genes = len(selected_ids)

    sym2idx = {}
    for j, eid in enumerate(selected_ids):
        sym = ens2sym.get(eid)
        if sym:
            sym2idx.setdefault(sym, []).append(j)

    mask = np.zeros((K + 1, n_genes), dtype=np.float32)
    for i, hsa in enumerate(hsa_ids):
        for sym in kegg[hsa]:
            for j in sym2idx.get(sym, []):
                mask[i, j] = 1.0

    # background group: genes in none of the KEGG pathways
    assigned = mask[:K].sum(axis=0) > 0
    mask[K, ~assigned] = 1.0

    pathway_info = {
        "hsa_ids": hsa_ids + ["background"],
        "pathway_names": [KEGG_15_PATHWAYS[h] for h in hsa_ids] + ["Background"],
        "genes_per_pathway": mask.sum(axis=1).astype(int).tolist(),
        "n_pathways": K + 1,
        "n_background": int((~assigned).sum()),
    }
    if verbose:
        print(f"  [mask] shape {mask.shape}, genes per pathway {pathway_info['genes_per_pathway']}")
    return mask, pathway_info
