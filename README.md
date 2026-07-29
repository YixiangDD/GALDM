# GA-LDM

Gene-Adaptive Latent Diffusion for joint mRNA–miRNA cohort augmentation.

GA-LDM generates paired mRNA and miRNA expression profiles for small TCGA
cohorts. A pathway-aware VAE (PA-VAE) compresses the two modalities into a
shared 256-d latent space using KEGG pathway groupings, and a gene-adaptive
diffusion transformer (GA-DiT) denoises that latent under class conditioning,
with auxiliary losses that hold co-expression structure and validated
miRNA–target repression in place.

This repository is the reference implementation for the paper and reproduces
every table and figure in it.

## What is here

```
configs/          hyper-parameters (config.py) and path resolution (paths.py)
data_pipeline/    TCGA loading, filtering, log2 transform, stratified splits
bio_priors/       KEGG pathway masks, Ensembl->symbol mapping, GRN assembly
models/           PA-VAE, miRNA VAE, cross-modal attention, GA-DiT, diffusion,
                  post-processing (latent recolouring + quantile calibration)
losses/           structural consistency losses (co-expression, pathway, causal)
baselines/        SMOTE, CTGAN, TabDDPM, TabSyn, TabDiff, scDiffusion, omicsGAN
evaluation/       the 7 metrics, cross-modal correlation, plotting, xlsx export
experiments/      one script per experiment in the paper
scripts/          fetch_priors.py — pulls the third-party prior files
tools/            helper scripts (repro check, GPU queue, figure regeneration)
data_pipeline/processed/   the preprocessed dataset, 4 cohorts x 5 seeds (shipped)
results/          the metric JSONs behind every reported number
results/cells/    per-(cohort, seed, method) results, the split-level unit of analysis
run_all.py        single entry point
```

`results/*.json` are the actual outputs from the runs reported in the paper, so
you can inspect or re-plot any number without spending GPU time first.
`results/cells/` holds the same results before aggregation, which is what
`t4_split_level_stats.py` uses to report confidence intervals over splits.

## Install

Python 3.10 or 3.11.

```bash
git clone https://github.com/<user>/GA-LDM.git
cd GA-LDM
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Reported results were produced with PyTorch 2.5.1 + CUDA 12.1 on a single
RTX 4090 (24 GB). The `--quick` preset runs on CPU; the full protocol does not.

## Data

**The preprocessed dataset is included**, in `data_pipeline/processed/` (70 MB): all four
cohorts at all five protocol seeds, already split, feature-selected and standardised, with
the matching biological priors. Nothing else is needed to reproduce the published numbers —
every experiment below runs straight from this cache:

```
data_pipeline/processed/
  CESC_seed42.npz          train/val/test mRNA+miRNA matrices, labels, gene/miRNA ids,
                           standardisation statistics
  CESC_seed42_priors.pkl   pathway mask, miRNA family mask, GRN (edges, layers, depths)
  CESC_seed42_meta.json    split sizes and class balance
  ... x {CESC,COAD,HNSC,KIRC} x {42,123,456,789,1024}
```

These arrays contain no patient identifiers — only expression values, feature IDs and summary
statistics.

### Starting from raw TCGA instead

Only needed if you want to change the cohorts, the seeds or the feature count. Download from
[UCSC Xena](https://xenabrowser.net/datapages/) and drop the files in unchanged:

```
data/
  CESC/
    TCGA-CESC.star_fpkm.tsv     mRNA, genes x samples (FPKM)
    TCGA-CESC.mirna.tsv         miRNA, miRNAs x samples (RPM)
    TCGA-CESC.clinical.tsv      clinical table with an AJCC/FIGO stage column
  COAD/ ... HNSC/ ... KIRC/ ...
```

The loader takes the separator from the file extension and detects orientation from the header,
so Xena `.tsv`, a `.csv` copy, a `.gz` of either, and a pre-transposed `samples x features`
form are all accepted, in a flat layout or inside a directory named after the download. No
conversion step. Point elsewhere with `GALDM_RAW_ROOT=/path/to/tcga`; every directory is
resolved in `configs/paths.py` and overridable by a `GALDM_*` variable, and no absolute path is
baked into the code.

Then fetch the biological priors:

```bash
python scripts/fetch_priors.py
```

None of the prior resources are redistributed here. KEGG pathway membership (15 pathways) and
TRRUST v2 are fetched automatically; miRTarBase 9.0 requires accepting its terms on the
download page, so the script prints the exact URL, filter settings and target filename for a
manual download; `ensembl2symbol.json` is built on first use from mygene.info. Note that KEGG
content changes over time, so a fresh fetch may differ slightly from the snapshot behind the
published numbers — the shipped cache is what reproduces those exactly.

### One inconsistency in the published preprocessing

Rebuilding from raw will not reproduce the paper's feature sets exactly, and the reason is
worth stating plainly. For CESC, HNSC and KIRC the raw matrices were passed through an
expression prefilter (~60,660 genes down to ~14,000) before the top-2000-by-variance
selection; for COAD that prefilter was not applied, so its variance ranking was computed over
all 60,660 genes. 174 of COAD's 2000 selected genes would not have survived the prefilter used
for the other three cohorts, one of them zero in every sample (high variance from sparse
outliers). The prefilter itself is not reproduced in this repository.

This affects which genes enter COAD's feature set, not the protocol applied to them: splits,
training, generation and evaluation are identical across cohorts, and every method within a
cohort sees exactly the same features. The shipped cache preserves the selections actually
used, so the published numbers reproduce; a rebuild from raw is internally consistent but will
differ from them.

## Run

```bash
python run_all.py --quick                     # ~10 min, 1 cohort, 1 seed — smoke test
python run_all.py --medium                    # ~30 min
python run_all.py --full                      # full protocol, ~60-80 GPU-hours

python run_all.py --exp comparison --cancers CESC --seeds 42
python run_all.py --exp ablation --cancers CESC,KIRC --mode both
python run_all.py --exp sensitivity
python run_all.py --exp interpretability
```

Preprocessing is cached per cohort and seed in `data_pipeline/processed/`, so
only the first run pays for it. Results land in `results/` as JSON keyed by
method and metric; `python -m experiments.summarize_all` collects them into
`results/all_experiment_results.xlsx`.

The protocol is 4 cohorts x 5 seeds (42, 123, 456, 789, 1024) with 70/15/15
stratified splits. Metrics are reported as mean +- s.d. over seeds, and every
baseline gets the same splits, the same generated-sample budget and the same
post-processing calibration strength.

Two scripts default to a different seed set: `bio_readout.py` and `omicsgan_compare.py` use
42-46 rather than the protocol seeds. Both take `--seeds`, so pass the protocol seeds to line
them up with the main tables. Note that the shipped cache covers only the five protocol seeds;
any other seed is rebuilt from raw data, which requires the raw files.

## Metrics

Seven metrics, all in `evaluation/metrics.py`:

| Metric | What it measures | Direction |
|---|---|---|
| MMD | maximum mean discrepancy, real vs generated | lower |
| KS | fraction of genes passing a Kolmogorov–Smirnov test | higher |
| miRNA_Corr | within-family miRNA correlation preservation | higher |
| PCE | pathway co-expression error | lower |
| DE | recovery of differentially expressed genes | higher |
| TSTR | train-on-synthetic, test-on-real AUC | higher |
| Bio_FID | Fréchet distance in a biology-aware feature space | lower |

CESC, mean over 5 seeds, from `results/comparison_CESC.json`:

| Method | MMD | KS | miRNA_Corr | PCE | DE | TSTR | Bio_FID |
|---|---|---|---|---|---|---|---|
| SMOTE | 0.009 | 0.975 | 0.851 | 0.091 | 0.859 | 0.749 | 70.0 |
| CTGAN | 0.008 | 0.891 | 0.838 | 0.099 | 0.804 | 0.688 | 71.8 |
| TabDDPM | 0.050 | 0.000 | 0.546 | 0.201 | 0.407 | 0.558 | 1107.0 |
| TabSyn | 0.037 | 0.014 | 0.784 | 0.130 | 0.603 | 0.651 | 231.5 |
| TabDiff | 0.050 | 0.000 | 0.546 | 0.201 | 0.402 | 0.577 | 1105.3 |
| scDiffusion | 0.040 | 0.007 | 0.793 | 0.123 | 0.498 | 0.665 | 235.7 |
| **GA-LDM** | **0.005** | 0.925 | **0.857** | **0.080** | **0.874** | **0.772** | **18.1** |

SMOTE keeps a higher KS score because it interpolates between real samples, which
reproduces marginals almost exactly while adding little that is new — the novelty
and Bio-FID columns are where that shows. Numbers for the other three cohorts are
in `results/comparison_{COAD,HNSC,KIRC}.json`.

## Two things worth knowing before you trust a number

**Bio-FID is biased at small n.** It is a Fréchet distance estimated from a
few dozen samples, so part of any gap is estimator bias rather than sample
quality. `experiments/biofid_reference.py` measures the in-sample floor by
bootstrapping real data at matched n; compare against that floor, not against 0.

**The pathway readout is a consistency check, not independent validation.** The
pathway masks used in `experiments/bio_readout.py` are the same ones the encoder
sees, so pathway-level agreement is partly built in. The miRTarBase repression
edges are external and the reported model has the edge-consistency loss off,
which is why that half of the readout is the informative half. Synthetic cohorts
reproduce the *ranking* of stage effects more reliably than their significance,
and should not be used in place of real samples for hypothesis testing.

## Reproducing a specific result

```bash
python tools/repro_check.py                   # verify env + priors + one seed end to end
python -m experiments.comparison --cancers CESC --seeds 42,123,456,789,1024
python -m experiments.bio_readout             # pathway + miRNA-target readout
python -m experiments.omicsgan_compare        # omicsGAN comparison
python -m experiments.ablation --mode both    # progressive + leave-one-out
python -m experiments.visualize_all           # regenerate figures into results/figures/
```

## Citation

```bibtex
@article{ding2026galdm,
  title   = {Gene-adaptive latent diffusion for joint mRNA-miRNA cohort augmentation},
  author  = {Ding, Yixiang},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {under review}
}
```

## Licence

Code released under the MIT Licence (see `LICENSE`).

The MIT licence covers the code only. The shipped data in `data_pipeline/processed/` is derived
from TCGA open-access expression and clinical data, redistributed under the NIH GDC data-use
policy for open-access tiers; it carries no patient identifiers. If you use it, cite TCGA as
well as this work.

No third-party prior resource is redistributed here, and each keeps its own terms: KEGG (free
for academic use, redistribution not permitted, commercial use requires a licence from Pathway
Solutions), TRRUST v2 (CC BY-SA 4.0), miRTarBase 9.0 (academic use). `scripts/fetch_priors.py`
retrieves them from source so you obtain each under its own terms. Confirm you are entitled to
each before use.
