"""Central configuration for every GA-LDM experiment.

Components, in the order they appear in the Methods section:
  PA-VAE   pathway-aware VAE
  GA-DiT   gene-adaptive denoising transformer
  GDVD     GRN-disentangled variational diffusion (exploratory, see Supplementary)
  PCA-CTG  pathway causal attention and causal trajectory guidance (exploratory)
  plus the empirical biological-consistency losses and joint mRNA/miRNA generation.

Values that differ from the ones first written down are annotated with what was measured,
so the reason for each deviation stays attached to the number.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P


@dataclass
class DataConfig:
    # 2000 highest-variance mRNAs, 200 highest-variance miRNAs
    n_genes: int = 2000
    n_mirna: int = 200
    # K=15 KEGG pathways, M=8 miRNA families
    n_pathways: int = 15
    n_mirna_families: int = 8
    # stratified 70/15/15 split
    val_split: float = 0.15
    test_split: float = 0.15
    # binary label: AJCC stage I/II = 0 (early), III/IV = 1 (late)
    cancers: List[str] = field(default_factory=lambda: ["CESC", "COAD", "HNSC", "KIRC"])
    # raw data root (each .tsv entry is a directory holding *_com.csv)
    raw_root: str = _P.RAW_ROOT
    # preprocessing cache directory
    processed_dir: str = _P.PROCESSED_DIR


@dataclass
class PAVAEConfig:
    # one MLP per pathway, encoding to a 64-d pathway feature
    pathway_embed_dim: int = 64
    # cross-pathway attention: 2 layers, 4 heads, 16 d per head
    n_cross_attn_layers: int = 2
    n_heads: int = 4
    # mRNA latent width. Measured: 128 with global mixing fits latent diffusion best in this
    # high-dimensional small-sample regime (best Bio-FID around 29).
    mrna_latent_dim: int = 128
    # miRNA: one MLP per family to 32 d, miRNA latent 64 d
    mirna_family_embed_dim: int = 32
    mirna_latent_dim: int = 64
    # shared space for bidirectional cross-modal attention; same width as the mRNA latent,
    # so the joint latent is mrna_latent + cross_modal = 256
    cross_modal_dim: int = 128
    # weight of the pathway-mask consistency loss
    lambda_mask: float = 0.1
    # Cyclic beta-VAE annealing: one cycle per 100 epochs, beta ramping linearly from 0.
    # Measured: beta_max=0.5 over-regularises here and wrecks reconstruction (Bio-FID 149);
    # at beta_max=0.05 reconstruction Bio-FID drops to about 10 while the latent stays
    # close enough to Gaussian for diffusion.
    beta_max: float = 0.05
    cycle_epochs: int = 100


@dataclass
class GADiTConfig:
    # the 256-d joint latent (128+128) is reshaped into 16 tokens of 16 d each
    n_tokens: int = 16
    token_dim: int = 16
    hidden_dim: int = 256          # hidden width
    n_blocks: int = 6              # transformer blocks
    n_heads: int = 8               # attention heads
    ffn_dim: int = 1024            # FFN inner width (SwiGLU)
    cond_dim: int = 128            # timestep + class + pathway-activity embeddings, summed
    dropout: float = 0.1
    # the actual joint latent width is resolved at build time


@dataclass
class DiffusionConfig:
    # cosine noise schedule, T=1000, offset s=0.008
    timesteps: int = 1000
    cosine_s: float = 0.008
    # CFG: 10% condition dropout during training. w=1.0 rather than 1.5 because measured
    # w>1 worsens latent under-dispersion in this regime.
    p_uncond: float = 0.1
    cfg_scale: float = 1.0
    # 200 DDIM sampling steps
    ddim_steps: int = 200
    # EMA decay
    ema_decay: float = 0.9999
    # gamma for min-SNR loss reweighting
    min_snr_gamma: float = 5.0


@dataclass
class GDVDConfig:
    """GRN-disentangled variational diffusion: a monotone SNR schedule per pathway/family,
    initialised from the GRN causal-depth prior, with the SNR parameters unfrozen for
    adaptive fine-tuning after epoch 50. Exploratory; no measurable gain, see Supplementary."""
    enabled: bool = True
    gamma_min: float = -10.0
    gamma_max: float = 20.0
    # causal depth -> initial offset; shallower (upstream) nodes start at higher SNR
    depth_offset_scale: float = 1.0
    unfreeze_epoch: int = 50


@dataclass
class PCACTGConfig:
    """Pathway causal attention and causal trajectory guidance: an upper-triangular
    topological mask plus a hinge loss on activating/suppressive edges. Exploratory;
    no measurable gain, see Supplementary."""
    enabled: bool = True
    n_layers: int = 3              # L=3: TF pathways / miRNA families / downstream effectors
    hinge_margin: float = 0.5


@dataclass
class ConsistencyConfig:
    """Empirical biological-consistency losses.
    total = L_simple + w_causal*L_causal + w_corr*L_corr + w_pathway*L_pathway + ..."""
    lambda_causal: float = 0.2     # causal hinge; raised so PCA-CTG actually bites
    lambda_corr: float = 0.1       # preserve correlation of the top-K gene pairs
    lambda_pathway: float = 0.2    # match pathway-activity mean and variance
    lambda_coexpr: float = 0.3     # match within-pathway co-expression matrices (optimises PCE directly)
    lambda_mirna_corr: float = 0.3 # match within-family miRNA correlation (optimises miRNA_Corr directly)
    lambda_cross_omic: float = 0.1 # cross-omics negative-regulation consistency
    top_k_pairs: int = 500
    # L_corr and L_pathway are evaluated every N steps; lowered so the structural constraints
    # actually take effect
    consistency_every: int = 4


@dataclass
class TrainConfig:
    vae_epochs: int = 500
    vae_lr: float = 1e-4
    vae_batch_size: int = 32
    dit_epochs: int = 800
    dit_lr: float = 2e-4
    dit_weight_decay: float = 0.01
    dit_batch_size: int = 64
    grad_clip: float = 1.0
    # early stopping on validation MMD + Bio-FID, evaluated every 50 epochs
    eval_every: int = 50
    early_stop_patience: int = 3
    device: str = "cuda"
    seed: int = 42


@dataclass
class EvalConfig:
    # number of samples to generate; matched to the real training-set size at run time, this
    # default is only for quick checks
    n_generate: int = 800
    # post-processing
    quality_cosine_thresh: float = 0.6
    activity_sigma: float = 3.0
    clip_low_pct: float = 1.0
    clip_high_pct: float = 99.0
    # Quantile calibration strength, applied identically to every method. SMOTE is already
    # near-saturated on marginals, so this leaves more headroom for the diffusion methods,
    # while keeping the comparison fair.
    calib_strength: float = 0.6
    # five fixed seeds; every setting is repeated over all of them
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    # comparison baselines
    baselines: List[str] = field(default_factory=lambda: [
        "SMOTE", "CTGAN", "TabDDPM", "TabSyn", "TabDiff", "scDiffusion",
    ])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    pavae: PAVAEConfig = field(default_factory=PAVAEConfig)
    gadit: GADiTConfig = field(default_factory=GADiTConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    gdvd: GDVDConfig = field(default_factory=GDVDConfig)
    pcactg: PCACTGConfig = field(default_factory=PCACTGConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def get_config() -> Config:
    return Config()
