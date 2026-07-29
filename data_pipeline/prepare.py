"""Prepares one cohort: dataset plus biological priors, strictly column-aligned and cached.

Order of work: get the unfiltered training mRNA via return_raw_train, run the
variance-plus-pathway-union gene selection and build every prior on it, rebuild the dataset
with the selected gene_ids/mirna_ids so columns line up, then cache both.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.data_loader import build_dataset, save_dataset, load_cached
from bio_priors.assemble import build_bio_priors, save_bio_priors, load_bio_priors
from configs import paths as _P


def prepare_cancer(raw_root, processed_dir, cancer, seed,
                   n_genes=2000, n_mirna=200, n_pathways=15, n_mirna_families=8,
                   use_cache=True, verbose=True):
    """Returns (dataset, bio_priors) and caches both."""
    if use_cache:
        ds = load_cached(processed_dir, cancer, seed)
        bp = load_bio_priors(processed_dir, cancer, seed)
        if ds is not None and bp is not None:
            if verbose:
                print(f"[prepare] {cancer} seed{seed} cache hit")
            return ds, bp

    if verbose:
        print(f"[prepare] {cancer} seed{seed} building ...")
    # 1) unfiltered training mRNA plus the variance-selected top-200 miRNAs, same seed split
    ds0 = build_dataset(raw_root, cancer, n_genes, n_mirna, 0.15, 0.15, seed,
                        return_raw_train=True)
    mirna_ids = ds0["mirna_ids"]  # variance-selected top-n_mirna, the miRNA columns actually used
    # 2) build the priors on the unfiltered training data: genes by variance-plus-pathway union,
    #    miRNA families over the 200 selected miRNAs
    bp = build_bio_priors(
        ds0["raw_train_mrna_df"], ds0["all_gene_ids"], mirna_ids,
        n_genes=n_genes, n_pathways=n_pathways, n_mirna_families=n_mirna_families,
        verbose=verbose)
    # 3) rebuild the aligned dataset from the selected genes and miRNAs
    ds = build_dataset(raw_root, cancer, n_genes, n_mirna, 0.15, 0.15, seed,
                       gene_ids=bp.selected_gene_ids, mirna_ids=mirna_ids)
    # 4) cache
    save_dataset(ds, processed_dir, seed)
    save_bio_priors(bp, processed_dir, cancer, seed)
    return ds, bp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default=_P.RAW_ROOT)
    ap.add_argument("--processed_dir", default=_P.PROCESSED_DIR)
    ap.add_argument("--cancers", default="CESC,COAD,HNSC,KIRC")
    ap.add_argument("--seeds", default="42")
    args = ap.parse_args()
    for c in args.cancers.split(","):
        for s in [int(x) for x in args.seeds.split(",")]:
            ds, bp = prepare_cancer(args.raw_root, args.processed_dir, c.strip(), s)
            n_in_pathway = int((bp.pathway_mask[:15].sum(axis=0) > 0).sum())
            print(f"  {c} seed{s}: train={ds['train']['n_samples']} "
                  f"genes={ds['meta']['n_genes']} mirna={ds['meta']['n_mirna']} "
                  f"pathway coverage={n_in_pathway}/{ds['meta']['n_genes']} "
                  f"GRN(act {len(bp.grn.activating_edges)}/sup {len(bp.grn.suppressive_edges)})")
