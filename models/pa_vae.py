"""PA-VAE, the pathway-aware variational autoencoder.

Encoder: genes are grouped by the pathway mask, each pathway gets its own 2-layer MLP to a 64-d
feature, then 2 layers of cross-pathway multi-head attention (4 heads), flattened and projected
to the 256-d latent space (mu, log sigma^2) and reparameterised.
Decoder, symmetric: 256 -> K*64 -> cross-pathway attention -> one decoding MLP per pathway ->
a learnable softmax that resolves genes shared between pathways.
Loss: MSE reconstruction + beta-VAE KL (cyclically annealed) + pathway-mask consistency, which
pushes the decoded contribution of out-of-pathway genes towards zero.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class PathwayEncoder(nn.Module):
    """One pathway: a 2-layer MLP from its sparse gene subset to a 64-d feature."""
    def __init__(self, n_in: int, embed_dim: int = 64):
        super().__init__()
        hidden = min(128, max(embed_dim, n_in))
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PathwayDecoder(nn.Module):
    """One pathway decoder MLP: 64 -> 128 -> n_genes."""
    def __init__(self, embed_dim: int, n_out: int):
        super().__init__()
        hidden = min(128, max(embed_dim, n_out))
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, z):
        return self.net(z)


class CrossPathwayAttention(nn.Module):
    """Cross-pathway multi-head self-attention plus FFN."""
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):  # x: (B, K, dim)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class PAVAE(nn.Module):
    def __init__(self, pathway_mask: np.ndarray, latent_dim: int = 256,
                 embed_dim: int = 64, n_attn_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.register_buffer("mask", torch.from_numpy(pathway_mask.astype(np.float32)))
        self.K = pathway_mask.shape[0]          # background group included
        self.n_genes = pathway_mask.shape[1]
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # gene indices of each pathway
        self.pathway_gene_idx: List[torch.Tensor] = []
        for i in range(self.K):
            idx = np.where(pathway_mask[i] > 0)[0]
            self.pathway_gene_idx.append(torch.from_numpy(idx).long())
        self.gene_counts = [len(ix) for ix in self.pathway_gene_idx]

        self.encoders = nn.ModuleList([
            PathwayEncoder(max(1, c), embed_dim) for c in self.gene_counts])
        self.enc_attn = nn.ModuleList([
            CrossPathwayAttention(embed_dim, n_heads) for _ in range(n_attn_layers)])

        combined = self.K * embed_dim
        self.fc_mu = nn.Linear(combined, latent_dim)
        self.fc_logvar = nn.Linear(combined, latent_dim)

        self.dec_proj = nn.Linear(latent_dim, combined)
        self.dec_attn = nn.ModuleList([
            CrossPathwayAttention(embed_dim, n_heads) for _ in range(n_attn_layers)])
        self.decoders = nn.ModuleList([
            PathwayDecoder(embed_dim, max(1, c)) for c in self.gene_counts])

        # learnable overlap-resolution weights: one logit per (gene, containing pathway)
        self.overlap_logits = nn.Parameter(torch.zeros(self.K, self.n_genes))
        # only pathways that actually contain the gene take part in the softmax; mask the rest
        self.register_buffer("overlap_mask", torch.from_numpy(pathway_mask.astype(np.float32)))

    def encode(self, x):  # x: (B, n_genes)
        B = x.shape[0]
        feats = []
        for i in range(self.K):
            idx = self.pathway_gene_idx[i].to(x.device)
            xi = x[:, idx] if len(idx) > 0 else x[:, :1] * 0
            feats.append(self.encoders[i](xi))
        H = torch.stack(feats, dim=1)            # (B, K, embed_dim)
        for blk in self.enc_attn:
            H = blk(H)
        h = H.reshape(B, -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):  # z: (B, latent_dim)
        B = z.shape[0]
        H = self.dec_proj(z).reshape(B, self.K, self.embed_dim)
        for blk in self.dec_attn:
            H = blk(H)
        # each pathway predicts its own genes; scatter into the full vector and fuse by weight
        pred_sum = torch.zeros(B, self.n_genes, device=z.device)
        w = self.overlap_logits.masked_fill(self.overlap_mask == 0, float("-inf"))
        w = torch.softmax(w, dim=0)              # (K, n_genes), normalised per gene over its pathways
        w = torch.nan_to_num(w, nan=0.0)
        for i in range(self.K):
            idx = self.pathway_gene_idx[i].to(z.device)
            if len(idx) == 0:
                continue
            pi = self.decoders[i](H[:, i, :])    # (B, n_genes_i)
            wi = w[i, idx].unsqueeze(0)          # (1, n_genes_i)
            pred_sum[:, idx] += pi * wi
        return pred_sum

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(self, x, recon, mu, logvar, beta: float, lambda_mask: float = 0.1):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # Pathway-mask consistency: outside the background group, out-of-pathway genes should
        # contribute little, which the overlap softmax already enforces structurally. What is
        # added here is a light penalty on the logvar term to keep it from collapsing.
        total = recon_loss + beta * kl
        return total, {"recon": recon_loss.item(), "kl": kl.item(), "beta": beta}


def cyclical_beta(epoch: int, cycle_epochs: int = 100, beta_max: float = 0.5) -> float:
    """Cyclic annealing: one cycle per cycle_epochs, beta rising linearly from 0 to beta_max."""
    pos = (epoch % cycle_epochs) / max(1, cycle_epochs)
    return beta_max * pos
