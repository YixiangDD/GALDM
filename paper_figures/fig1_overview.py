"""Figure 1: overall architecture of GA-LDM.

A schematic, so it reads no experiment artifacts and runs on CPU in seconds.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P  # noqa: E402

OUT = os.path.join(_P.FIGURES_DIR, "paper")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 14,
    'mathtext.fontset': 'dejavusans',
})

C = {
    'mrna': '#4A90D9', 'mrna_light': '#D6EAF8',
    'mirna': '#27AE60', 'mirna_light': '#D5F5E3',
    'gdvd': '#F39C12', 'gdvd_light': '#FEF5E7',
    'ctg': '#E74C3C', 'ctg_light': '#FDEDEC',
    'cross': '#F1C40F', 'cross_light': '#FEF9E7',
    'dark': '#2C3E50', 'gray': '#7F8C8D', 'note': '#41525E', 'sub': '#33454F',
    'bg1': '#F8F9FA', 'bg2': '#FFFFFF', 'purple': '#8E44AD', 'white': '#FFFFFF',
}

fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(-0.5, 21.6)
ax.set_ylim(-0.8, 15.5)
ax.set_aspect('equal')
ax.axis('off')

def rbox(xy, w, h, fc, ec=None, lw=1.2, alpha=1.0, zo=2, r=0.15):
    ec = ec or fc
    p = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zo)
    ax.add_patch(p)

def T(x, y, s, fs=14, c='#2C3E50', ha='center', va='center', w='normal', zo=5, **kw):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, fontweight=w, zorder=zo, **kw)

def A(p1, p2, c='#2C3E50', lw=1.5, sty='->', ls='-', zo=3):
    ax.annotate('', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle=sty, color=c, lw=lw, linestyle=ls), zorder=zo)


# ════════════════════════════════════════════════════════════
#  STAGE 1  (y ~ 10–15)
# ════════════════════════════════════════════════════════════
rbox((-0.3, 10.0), 21.7, 5.3, C['bg1'], '#BDC3C7', lw=1, alpha=.45, zo=0)
T(0.6, 15.0, 'Stage 1', fs=18, w='bold')
T(4.4, 15.0, 'training: multi-modal PA-VAE', fs=16, c=C['sub'], w='bold')

# — Inputs —
rbox((0, 13.3), 2.0, 1.1, C['mrna_light'], C['mrna'], lw=1.5)
T(1.0, 13.95, '$X_{mRNA}$', fs=15, w='bold', c=C['mrna'])
T(1.0, 13.6, '(N x 2000)', fs=13, c=C['note'])

rbox((0, 11.3), 2.0, 1.1, C['mirna_light'], C['mirna'], lw=1.5)
T(1.0, 11.95, '$X_{miRNA}$', fs=15, w='bold', c=C['mirna'])
T(1.0, 11.6, '(N x 200)', fs=13, c=C['note'])

# — mRNA pathway encoding —
A((2.0, 13.85), (3.0, 13.85), C['mrna'])
rbox((3.0, 13.0), 2.0, 1.7, C['mrna_light'], C['mrna'], lw=1.2)
T(4.0, 14.55, 'Pathway Groups (K=15 + bg)', fs=10, c=C['mrna'], w='bold')
for i, lab in enumerate(['$MLP_1$', '$MLP_2$', '$\\cdots$', '$MLP_K$']):
    by = 14.15 - i * 0.32
    rbox((3.15, by), 1.7, 0.24, C['mrna'], C['mrna'], alpha=0.25, r=0.04)
    T(4.0, by + 0.12, lab, fs=12, c=C['mrna'])

# mRNA MHSA
A((5.0, 13.85), (5.7, 13.85), C['mrna'])
rbox((5.7, 13.3), 1.5, 1.1, C['mrna_light'], C['mrna'], lw=1.5)
T(6.45, 13.95, 'MHSA', fs=14, w='bold', c=C['mrna'])
T(6.45, 13.6, 'Intra-modal', fs=12, c=C['note'])

# — miRNA family encoding (shifted down by ~0.3) —
A((2.0, 11.85), (3.0, 11.85), C['mirna'])
rbox((3.0, 10.9), 2.0, 1.6, C['mirna_light'], C['mirna'], lw=1.2)
T(4.0, 12.35, "Family Groups (M'=8)", fs=10, c=C['mirna'], w='bold')
for i, lab in enumerate(["$MLP'_1$", "$MLP'_2$", "$MLP'_{M'}$"]):
    by = 11.88 - i * 0.38
    rbox((3.15, by), 1.7, 0.28, C['mirna'], C['mirna'], alpha=0.25, r=0.04)
    T(4.0, by + 0.14, lab, fs=12, c=C['mirna'])

# miRNA MHSA (shifted down to match)
A((5.0, 11.85), (5.7, 11.85), C['mirna'])
rbox((5.7, 11.3), 1.5, 1.1, C['mirna_light'], C['mirna'], lw=1.5)
T(6.45, 11.95, 'MHSA', fs=14, w='bold', c=C['mirna'])
T(6.45, 11.6, 'Intra-modal', fs=12, c=C['note'])

# — Cross-Modal Attention —
A((7.2, 13.7), (8.0, 13.2), C['cross'], lw=1.8)
A((7.2, 12.0), (8.0, 12.6), C['cross'], lw=1.8)

rbox((8.0, 12.3), 2.2, 1.4, C['cross_light'], C['cross'], lw=2.0)
T(9.1, 13.15, 'Cross-Modal', fs=14, w='bold', c='#B7950B')
T(9.1, 12.8, 'Attention', fs=14, w='bold', c='#B7950B')
T(9.1, 12.45, '(bidirectional, weight $\lambda_{cm}$)', fs=12, c=C['note'])

# — Reparameterization —
A((10.2, 13.0), (11.0, 13.0), C['dark'])
rbox((11.0, 12.5), 1.8, 1.0, '#EBF5FB', C['dark'], lw=1.2)
T(11.9, 13.1, '$\\mu,\\ \\log\\sigma^2$', fs=14, c=C['dark'])
T(11.9, 12.75, '$z_0 = \\mu + \\sigma \\cdot \\epsilon$', fs=13, c=C['note'])

A((12.8, 13.0), (13.5, 13.0), C['dark'])

# — Joint latent z₀ —
rbox((13.5, 12.3), 2.0, 1.4, '#EBF5FB', C['dark'], lw=2.0)
T(14.5, 13.2, '$z_0 \\in \\mathbb{R}^{256}$', fs=15, w='bold', c=C['dark'])
T(14.5, 12.85, '[mRNA 128d', fs=12, c=C['mrna'])
T(14.5, 12.55, ' | miRNA 128d]', fs=12, c=C['mirna'])

# — Dual-branch decoder —
A((15.5, 13.5), (16.5, 13.85), C['mrna'])
A((15.5, 12.5), (16.5, 12.1), C['mirna'])

rbox((16.5, 13.5), 2.0, 0.9, C['mrna_light'], C['mrna'], lw=1.2)
T(17.5, 14.0, 'Pathway Dec.', fs=13, w='bold', c=C['mrna'])
T(17.5, 13.7, '$\\rightarrow \\hat{X}_{mRNA}$', fs=13, c=C['mrna'])

rbox((16.5, 11.7), 2.0, 0.9, C['mirna_light'], C['mirna'], lw=1.2, zo=4)
T(17.5, 12.2, 'Family Dec.', fs=13, w='bold', c=C['mirna'])
T(17.5, 11.9, '$\\rightarrow \\hat{X}_{miRNA}$', fs=13, c=C['mirna'])

# Loss box (centered under decoders)
rbox((15.5, 10.2), 4.0, 0.8, '#F8F9FA', C['gray'], lw=0.8)
T(17.5, 10.72, '$\\mathcal{L}_{VAE} = \\|X - \\hat{X}\\|^2 + \\|m - \\hat{m}\\|^2$', fs=12, c=C['dark'])
T(17.5, 10.40, '$+\\ \\beta\\,(KL_{mRNA} + KL_{miRNA})$', fs=12, c=C['dark'])
# Both decoder branches feed the reconstruction loss. Route the mRNA link down the
# outside (x=19.7) so it clears the Family Dec. box (zo=4) instead of being hidden
# behind it; the miRNA branch gets its own link for the |m - m_hat|^2 term.
ax.plot([18.5, 19.7], [13.95, 13.95], color=C['gray'], lw=1.0, ls=':', zorder=3)
ax.plot([19.7, 19.7], [13.95, 11.0], color=C['gray'], lw=1.0, ls=':', zorder=3)
A((19.7, 11.0), (19.0, 11.0), C['gray'], lw=1.0, sty='->', ls=':')
A((17.5, 11.7), (17.5, 11.0), C['gray'], lw=1.0, sty='->', ls=':')


# ============================================================
#  STAGE 2  (GA-DiT latent diffusion; exploratory GRN priors -> Supplement)
# ============================================================
rbox((-0.3, 3.8), 21.7, 5.5, C['bg2'], '#BDC3C7', lw=1, alpha=.45, zo=0)
T(0.6, 9.0, 'Stage 2', fs=18, w='bold')
T(4.9, 9.0, 'training: class-conditional latent diffusion', fs=16, c=C['sub'], w='bold')

# z0 hand-off: a single arrow-headed path routed through the gap between the two
# panels, in an accent colour distinct from the grey dotted reconstruction link.
HAND = '#B7950B'
HY = 9.52
ax.plot([14.5, 14.5], [12.28, HY], color=HAND, lw=2.0, solid_capstyle='butt',
        zorder=6)
ax.plot([14.5, 1.7], [HY, HY], color=HAND, lw=2.0, solid_capstyle='butt',
        zorder=6)
A((1.7, HY), (1.7, 8.22), HAND, lw=2.0, sty='-|>', zo=6)
A((8.6, HY), (7.7, HY), HAND, lw=2.0, sty='-|>', zo=6)
T(9.9, 9.52, 'clean latent $z_0$, frozen Stage-1 encoder',
  fs=12.5, c='#8A6D0B', w='bold', ha='left', zo=8,
  bbox=dict(boxstyle='round,pad=0.24', fc='#FFFFFF', ec='none'))


# z0 (clean)
rbox((1.0, 7.3), 1.4, 0.9, '#EBF5FB', C['dark'], lw=1.5)
T(1.7, 7.85, '$z_0$', fs=16, w='bold', c=C['dark'])
T(1.7, 7.55, '(clean)', fs=12, c=C['note'])

# --- Forward diffusion ---
A((2.4, 7.75), (3.0, 7.75), C['dark'])
rbox((3.0, 7.3), 1.7, 0.9, '#EBF5FB', C['dark'], lw=1.2)
T(3.85, 7.9, 'Forward', fs=13, w='bold', c=C['dark'])
T(3.85, 7.55, '$q(z_t|z_0)$', fs=12, c=C['dark'])
A((4.7, 7.75), (5.4, 7.75), C['dark'])
rbox((5.4, 7.3), 1.4, 0.9, '#EBF5FB', C['dark'], lw=1.5)
T(6.1, 7.9, '$z_t$', fs=16, w='bold', c=C['dark'])
T(6.1, 7.55, '(noisy)', fs=12, c=C['note'])

# --- Pathway/family token identity panel (what makes GA-DiT gene-adaptive) ---
pl, pb, pw, ph = 2.9, 4.9, 4.1, 2.0
rbox((pl, pb), pw, ph, C['white'], C['mrna'], lw=1.2, r=0.08)
T(pl + pw / 2, pb + ph - 0.22, 'Pathway / family token identity', fs=12, w='bold', c=C['mrna'])
# The 256-d latent is split into 16 uniform tokens (config n_tokens=16); the
# K+M' = 23 pathway/family segments are a separate partition mapped onto those
# tokens, so tokens are NOT in 1:1 correspondence with groups.
tok_lab = ['$t_1$', '$t_2$', '$t_3$', '$\\cdots$', '$t_{16}$']
tok_grp = [C['mrna'], C['mrna'], C['mrna'], None, C['mirna']]
for i, (lb, gc) in enumerate(zip(tok_lab, tok_grp)):
    cx_ = pl + 0.35 + i * 0.72
    rbox((cx_, pb + 0.85), 0.6, 0.5, '#EBF5FB', C['dark'], lw=0.9, r=0.06)
    T(cx_ + 0.3, pb + 1.16, lb, fs=11.5, c=C['dark'])
    if gc is not None:                      # chip = the group this token is assigned
        rbox((cx_ + 0.12, pb + 0.92), 0.36, 0.14, gc, gc, alpha=0.55, r=0.03)
T(pl + pw / 2, pb + 0.55,
  '16 uniform tokens; the colour chip marks the pathway/', fs=10.5, c=C['note'])
T(pl + pw / 2, pb + 0.28,
  'family group each token is assigned (positional encoding)', fs=10.5, c=C['note'])

# --- GA-DiT ---
A((6.8, 7.75), (8.0, 7.75), C['dark'])
dit_x, dit_y, dit_w, dit_h = 8.0, 4.8, 4.6, 4.0
rbox((dit_x, dit_y), dit_w, dit_h, '#EBF5FB', C['dark'], lw=2.4, r=0.2)
T(dit_x + dit_w / 2, dit_y + dit_h + 0.25, 'GA-DiT (Transformer denoiser)',
  fs=15, w='bold', c=C['dark'])
for i in range(4):
    by = dit_y + dit_h - 0.8 - i * 0.8
    rbox((dit_x + 0.3, by), dit_w - 0.6, 0.6, '#D6EAF8', C['mrna'], lw=0.8, r=0.08)
    T(dit_x + dit_w / 2, by + 0.3, f'DiT Block {i+1}:  AdaLN-Zero (LN+scale/shift) '
                                   f'-> MHSA -> SwiGLU FFN',
      fs=9.8, c=C['dark'])
T(dit_x + dit_w / 2, dit_y + 0.62, '... (x6 blocks total)', fs=13, c=C['note'])
rbox((dit_x + 0.25, dit_y + 0.1), 1.15, 0.42, '#D5D8DC', C['gray'], lw=0.8, r=0.06)
T(dit_x + 0.825, dit_y + 0.31, '$e_t$ time', fs=11, c=C['dark'])
rbox((dit_x + 1.6, dit_y + 0.1), 1.25, 0.42, '#E8DAEF', C['purple'], lw=0.8, r=0.06)
T(dit_x + 2.225, dit_y + 0.31, '$e_c$ stage', fs=11, c=C['purple'])
rbox((dit_x + 3.05, dit_y + 0.1), 1.3, 0.42, C['mrna_light'], C['mrna'], lw=0.8, r=0.06)
T(dit_x + 3.7, dit_y + 0.31, '$e_{pos}$ group', fs=11, c=C['mrna'])

# --- output + objective ---
A((dit_x + dit_w, 7.75), (13.2, 7.75), C['dark'])
rbox((13.2, 7.3), 1.5, 0.9, '#EBF5FB', C['dark'], lw=1.5)
T(13.95, 7.9, '$\\hat{\\epsilon}$', fs=16, w='bold', c=C['dark'])
T(13.95, 7.58, '(noise pred.)', fs=11, c=C['note'])

ol, ob, ow, oh = 15.1, 5.4, 5.0, 2.8
rbox((ol, ob), ow, oh, '#F8F9FA', C['gray'], lw=1.2, r=0.1)
T(ol + ow / 2, ob + oh - 0.28, 'Diffusion objective', fs=14, w='bold', c=C['dark'])
T(ol + ow / 2, ob + oh - 0.78,
  '$\\mathcal{L} = \\mathcal{L}_{simple}$  (min-SNR-$\\gamma$, $\\gamma$=5)', fs=12.5, c=C['dark'])
T(ol + ow / 2, ob + oh - 1.26, '$+\\ \\lambda_{corr}\\mathcal{L}_{corr} + \\lambda_{path}\\mathcal{L}_{path}'
                               ' + \\lambda_{coexpr}\\mathcal{L}_{coexpr}$',
  fs=12, c=C['dark'])
T(ol + ow / 2, ob + oh - 1.66, '$+\\ \\lambda_{mifam}\\mathcal{L}_{mifam}'
                               ' + \\lambda_{causal}\\mathcal{L}_{causal}$',
  fs=12, c=C['dark'])
T(ol + ow / 2, ob + 0.34, 'structure-preserving terms (co-expression, pathway', fs=10.5, c=C['note'])
T(ol + ow / 2, ob + 0.10, 'activity, within-family miRNA, causal hinge); every 4th step, t < 0.3T',
  fs=10.5, c=C['note'])
A((14.7, 7.75), (ol + ow / 2, ob + oh), C['dark'], lw=1.0, ls='--')

# --- exploratory GRN priors: demoted to a dashed side note ---
el, eb, ew, eh = 15.1, 4.05, 5.0, 1.05
p_ = FancyBboxPatch((el, eb), ew, eh, boxstyle='round,pad=0.02,rounding_size=0.08',
                    fc='#FDFEFE', ec=C['gray'], lw=1.0, ls='--', alpha=0.9, zorder=2)
ax.add_patch(p_)
T(el + ew / 2, eb + eh - 0.28, 'Exploratory GRN-informed priors (GDVD, PCA-CTG)',
  fs=11.5, w='bold', c=C['note'])
T(el + ew / 2, eb + eh - 0.60, 'enabled in all reported runs; ablation shows',
  fs=11, c=C['note'])
T(el + ew / 2, eb + eh - 0.86, 'no measurable fidelity gain (Suppl. Section S6)',
  fs=11, c=C['note'])

# ════════════════════════════════════════════════════════════
#  GENERATION PIPELINE  (y ~ -0.3–3.5)
# ════════════════════════════════════════════════════════════
rbox((-0.3, -0.5), 21.7, 4.0, C['bg1'], '#BDC3C7', lw=1, alpha=.45, zo=0)
T(0.95, 3.2, 'Inference', fs=18, w='bold')
T(4.6, 3.2, 'sampling and decoding (no training)', fs=16, c=C['sub'], w='bold')

# Order follows trainer.generate(): DDIM -> latent whiten-recolour -> inverse latent
# scaling -> PA-VAE decode -> inverse log2 scaling -> quality filter + 1-99% clip ->
# per-class quantile calibration. Recolouring is in LATENT space, i.e. before decoding.
pipe_y, pipe_h = 1.15, 1.15
PW_, PGAP = 1.85, 0.32
pipe_labels = [
    ('$z_T \\sim N(0, I)$',                 '#D5D8DC', C['dark']),
    ('DDIM 200 steps\n$\\eta$ = 0',          '#D6EAF8', C['mrna']),
    ('latent whiten\n$-$recolour',           '#DDE1E4', C['note']),
    ('inverse latent\nscaling',              '#DDE1E4', C['note']),
    ('PA-VAE\nDecoder',                      '#EBF5FB', C['dark']),
    ('inverse $\\log_2$\nscaling',           '#DDE1E4', C['note']),
    ('quality filter\n+ 1$-$99% clip',       '#DDE1E4', C['note']),
    ('per-class\nquantile calib.',           '#DDE1E4', C['note']),
]
prev_right = None
for i, (label, bg, col) in enumerate(pipe_labels):
    px = 0.15 + i * (PW_ + PGAP)
    rbox((px, pipe_y), PW_, pipe_h, bg, col, lw=1.2)
    T(px + PW_ / 2, pipe_y + pipe_h / 2, label, fs=11, c=col, w='bold')
    if prev_right is not None:
        A((prev_right, pipe_y + pipe_h / 2), (px, pipe_y + pipe_h / 2), C['dark'])
    prev_right = px + PW_

# Output branches: the calibrated paired sample
A((prev_right, pipe_y + 0.75), (17.62, 2.30), C['mrna'])
A((prev_right, pipe_y + 0.40), (17.62, 1.05), C['mirna'])

rbox((17.62, 1.95), 2.55, 0.80, C['mrna_light'], C['mrna'], lw=1.5)
T(18.90, 2.35, '$\\hat{X}_{mRNA}$ (2000 genes)', fs=12, w='bold', c=C['mrna'])

rbox((17.62, 0.65), 2.55, 0.80, C['mirna_light'], C['mirna'], lw=1.5)
T(18.90, 1.05, '$\\hat{X}_{miRNA}$ (200 miRNAs)', fs=12, w='bold', c=C['mirna'])

# brace-style tie marking the two branches as one paired sample
ax.plot([20.28, 20.28], [1.05, 2.35], color=C['gdvd'], lw=2.0, zorder=6)
ax.plot([20.17, 20.28], [2.35, 2.35], color=C['gdvd'], lw=2.0, zorder=6)
ax.plot([20.17, 20.28], [1.05, 1.05], color=C['gdvd'], lw=2.0, zorder=6)
T(20.42, 1.70, 'paired\n$X_{gen}$', fs=11.5, w='bold', c=C['dark'], ha='left')

# — Legend —
leg_y = -0.4
leg = [
    (C['mrna_light'], C['mrna'], 'mRNA branch'),
    (C['mirna_light'], C['mirna'], 'miRNA branch'),
    (C['cross_light'], C['cross'], 'Cross-modal attention'),
]
for i, (fc, ec, lab) in enumerate(leg):
    lx = 0.2 + i * 3.2
    rbox((lx, leg_y), 0.35, 0.25, fc, ec, lw=1.2, r=0.04)
    T(lx + 1.8, leg_y + 0.12, lab, fs=12, c=C['dark'])

plt.tight_layout()
try:
    plt.savefig(OUT + '/fig1_overview.pdf', dpi=300, bbox_inches='tight')
except PermissionError:
    print("PDF skipped (file locked)")
plt.savefig(OUT + '/fig1_overview.png', dpi=300, bbox_inches='tight')
print("Fig 1 overview saved.")
plt.close()
