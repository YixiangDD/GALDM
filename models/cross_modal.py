"""Bidirectional cross-modal attention and the joint multi-omics VAE.

The miRNA latent (64) is linearly mapped into the 256-d shared space; the mRNA latent (256) is
unchanged. Bidirectional cross-attention lets mRNA and miRNA serve as each other's Q/K/V, and
the concatenation forms the joint latent representation the diffusion model works on. At
generation time the two halves are split back apart and decoded separately.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pa_vae import PAVAE
from .mirna_vae import MiRNAVAE


class BidirectionalCrossModalAttention(nn.Module):
    def __init__(self, dim: int = 256, n_heads: int = 4, dropout: float = 0.1,
                 lambda_cm: float = 1.0):
        super().__init__()
        self.m2mi = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.mi2m = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.norm_mi = nn.LayerNorm(dim)
        self.lambda_cm = lambda_cm   # cross-modal coupling strength for the Pareto sweep; 1.0 = default

    def forward(self, z_mrna, z_mirna):  # both (B, dim)
        m = z_mrna.unsqueeze(1)          # (B,1,dim)
        mi = z_mirna.unsqueeze(1)
        # mRNA as Q, miRNA as K/V
        a_m, _ = self.m2mi(m, mi, mi)
        z_m = self.norm_m(z_mrna + self.lambda_cm * a_m.squeeze(1))
        # miRNA as Q, mRNA as K/V
        a_mi, _ = self.mi2m(mi, m, m)
        z_mi = self.norm_mi(z_mirna + self.lambda_cm * a_mi.squeeze(1))
        return z_m, z_mi


class JointMultiOmicsVAE(nn.Module):
    """PA-VAE, miRNA-VAE and cross-modal attention combined into the joint latent."""
    def __init__(self, pathway_mask: np.ndarray, family_mask: np.ndarray,
                 mrna_latent: int = 256, mirna_latent: int = 64,
                 common_dim: int = 256, n_heads: int = 4, use_cross_modal: bool = True,
                 lambda_cm: float = 1.0):
        super().__init__()
        self.pa_vae = PAVAE(pathway_mask, latent_dim=mrna_latent)
        self.mirna_vae = MiRNAVAE(family_mask, latent_dim=mirna_latent)
        self.mirna_to_common = nn.Linear(mirna_latent, common_dim)
        self.common_to_mirna = nn.Linear(common_dim, mirna_latent)
        self.use_cross_modal = use_cross_modal
        self.cross_modal = BidirectionalCrossModalAttention(common_dim, n_heads, lambda_cm=lambda_cm)
        self.mrna_latent = mrna_latent
        self.mirna_latent = mirna_latent
        self.common_dim = common_dim
        self.joint_dim = mrna_latent + common_dim   # 256 + 256 = 512

    def encode_joint(self, x_mrna, x_mirna):
        mu_m, lv_m = self.pa_vae.encode(x_mrna)
        z_m = self.pa_vae.reparameterize(mu_m, lv_m)
        mu_mi, lv_mi = self.mirna_vae.encode(x_mirna)
        z_mi = self.mirna_vae.reparameterize(mu_mi, lv_mi)
        z_mi_common = self.mirna_to_common(z_mi)
        if self.use_cross_modal:
            z_m2, z_mi2 = self.cross_modal(z_m, z_mi_common)
        else:
            # w/o CrossModal ablation: encode both modalities independently, no attention, concat
            z_m2, z_mi2 = z_m, z_mi_common
        z_joint = torch.cat([z_m2, z_mi2], dim=-1)   # (B, joint_dim)
        return z_joint, (mu_m, lv_m, mu_mi, lv_mi)

    def split_joint(self, z_joint):
        z_m = z_joint[:, :self.mrna_latent]
        z_mi_common = z_joint[:, self.mrna_latent:]
        z_mi = self.common_to_mirna(z_mi_common)
        return z_m, z_mi

    def decode_joint(self, z_joint):
        z_m, z_mi = self.split_joint(z_joint)
        recon_m = self.pa_vae.decode(z_m)
        recon_mi = self.mirna_vae.decode(z_mi)
        return recon_m, recon_mi

    def forward(self, x_mrna, x_mirna):
        z_joint, stats = self.encode_joint(x_mrna, x_mirna)
        recon_m, recon_mi = self.decode_joint(z_joint)
        return recon_m, recon_mi, z_joint, stats

    def loss(self, x_mrna, x_mirna, recon_m, recon_mi, stats, beta: float,
             lambda_cross: float = 0.1, neg_pairs=None):
        mu_m, lv_m, mu_mi, lv_mi = stats
        rec_m = F.mse_loss(recon_m, x_mrna)
        rec_mi = F.mse_loss(recon_mi, x_mirna)
        kl_m = -0.5 * torch.mean(1 + lv_m - mu_m.pow(2) - lv_m.exp())
        kl_mi = -0.5 * torch.mean(1 + lv_mi - mu_mi.pow(2) - lv_mi.exp())
        total = rec_m + rec_mi + beta * (kl_m + kl_mi)
        logd = {"rec_m": rec_m.item(), "rec_mi": rec_mi.item(),
                "kl_m": kl_m.item(), "kl_mi": kl_mi.item()}
        # cross-omics negative-regulation consistency: a known miRNA-mRNA pair should anti-correlate
        if neg_pairs is not None and len(neg_pairs) > 0:
            mi_idx = neg_pairs[:, 0].to(x_mirna.device)
            m_idx = neg_pairs[:, 1].to(x_mrna.device)
            # push generated miRNA and its target mRNA towards anti-correlation by penalising a
            # positive mean per-sample product
            mi_v = recon_mi[:, mi_idx]
            m_v = recon_m[:, m_idx]
            mi_c = mi_v - mi_v.mean(0, keepdim=True)
            m_c = m_v - m_v.mean(0, keepdim=True)
            cross = (mi_c * m_c).mean(0)             # covariance per pair
            cross_loss = F.relu(cross).mean()        # penalise positive correlation
            total = total + lambda_cross * cross_loss
            logd["cross"] = cross_loss.item()
        return total, logd
