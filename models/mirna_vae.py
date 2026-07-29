"""miRNA family-aware variational autoencoder.

Each miRNA family is encoded by its own MLP to 32 dimensions, followed by cross-family
attention and a projection to a 64-d latent space (mu, log sigma^2). Structurally similar to
PA-VAE but smaller: miRNA carries less information, so a 64-d latent guards against overfitting.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pa_vae import CrossPathwayAttention  # the same cross-group attention block


class FamilyEncoder(nn.Module):
    def __init__(self, n_in: int, embed_dim: int = 32):
        super().__init__()
        hidden = min(64, max(embed_dim, n_in))
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FamilyDecoder(nn.Module):
    def __init__(self, embed_dim: int, n_out: int):
        super().__init__()
        hidden = min(64, max(embed_dim, n_out))
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, z):
        return self.net(z)


class MiRNAVAE(nn.Module):
    def __init__(self, family_mask: np.ndarray, latent_dim: int = 64,
                 embed_dim: int = 32, n_attn_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.register_buffer("mask", torch.from_numpy(family_mask.astype(np.float32)))
        self.M = family_mask.shape[0]
        self.n_mirna = family_mask.shape[1]
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.family_idx = [torch.from_numpy(np.where(family_mask[i] > 0)[0]).long()
                           for i in range(self.M)]
        self.counts = [len(ix) for ix in self.family_idx]

        self.encoders = nn.ModuleList([FamilyEncoder(max(1, c), embed_dim) for c in self.counts])
        self.enc_attn = nn.ModuleList([CrossPathwayAttention(embed_dim, n_heads)
                                       for _ in range(n_attn_layers)])
        combined = self.M * embed_dim
        self.fc_mu = nn.Linear(combined, latent_dim)
        self.fc_logvar = nn.Linear(combined, latent_dim)

        self.dec_proj = nn.Linear(latent_dim, combined)
        self.dec_attn = nn.ModuleList([CrossPathwayAttention(embed_dim, n_heads)
                                       for _ in range(n_attn_layers)])
        self.decoders = nn.ModuleList([FamilyDecoder(embed_dim, max(1, c)) for c in self.counts])

    def encode(self, x):
        B = x.shape[0]
        feats = []
        for i in range(self.M):
            idx = self.family_idx[i].to(x.device)
            xi = x[:, idx] if len(idx) > 0 else x[:, :1] * 0
            feats.append(self.encoders[i](xi))
        H = torch.stack(feats, dim=1)
        for blk in self.enc_attn:
            H = blk(H)
        h = H.reshape(B, -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        B = z.shape[0]
        H = self.dec_proj(z).reshape(B, self.M, self.embed_dim)
        for blk in self.dec_attn:
            H = blk(H)
        out = torch.zeros(B, self.n_mirna, device=z.device)
        for i in range(self.M):
            idx = self.family_idx[i].to(z.device)
            if len(idx) == 0:
                continue
            out[:, idx] = self.decoders[i](H[:, i, :])
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(self, x, recon, mu, logvar, beta: float):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, {"recon": recon_loss.item(), "kl": kl.item()}
