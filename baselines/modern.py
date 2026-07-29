"""Recent SOTA baselines: TabSyn (ICLR'24), TabDiff (ICLR'25), scDiffusion (Bioinformatics'24).

To compare fairly and controllably on this task (high-dimensional expression, small
cohorts, mRNA-miRNA concatenation), these are compact reimplementations faithful to
each method's core idea:
- TabSyn: VAE down to a low-dimensional latent, then EDM-style score-based diffusion there.
- TabDiff: single continuous diffusion, deeper MLP denoiser with adaptive noise (approximated
  here by a learned per-dimension noise scale).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classical import BaselineModel


def _cosine_abar(T, device):
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 0.9999).to(device)


class _Enc(nn.Module):
    def __init__(self, dim, c_dim, latent, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + c_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2 * latent))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], 1)).chunk(2, 1)


class _Dec(nn.Module):
    def __init__(self, latent, c_dim, dim, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent + c_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, dim))

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))


class _LatentDenoiser(nn.Module):
    def __init__(self, latent, c_dim, h=256, t_dim=128, depth=3):
        super().__init__()
        self.t_dim = t_dim
        layers = [nn.Linear(latent + c_dim + t_dim, h), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.SiLU()]
        layers += [nn.Linear(h, latent)]
        self.net = nn.Sequential(*layers)

    def temb(self, t):
        half = self.t_dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        a = t.float().unsqueeze(1) * f.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, z, t, c):
        return self.net(torch.cat([z, c, self.temb(t)], 1))


class _VAELatentDiffusion(BaselineModel):
    """Shared skeleton for TabSyn and scDiffusion: VAE, then latent diffusion.
    Subclasses differ only in hyper-parameters."""
    latent = 32
    vae_epochs = 200
    diff_epochs = 300
    depth = 3
    T = 1000

    def fit(self, X, y):
        self.dim = X.shape[1]; self.n_cls = len(np.unique(y))
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        self.enc = _Enc(self.dim, self.n_cls, self.latent).to(self.device)
        self.dec = _Dec(self.latent, self.n_cls, self.dim).to(self.device)
        # Phase 1: VAE
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), 1e-3)
        n = len(X); bs = min(64, n)
        for ep in range(self.vae_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; xb = Xt[idx]; cb = yt[idx]
                mu, lv = self.enc(xb, cb)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
                rec = self.dec(z, cb)
                loss = F.mse_loss(rec, xb) - 0.005 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                opt.zero_grad(); loss.backward(); opt.step()
        # standardise the latent
        with torch.no_grad():
            mu, _ = self.enc(Xt, yt)
            self.z_mu = mu.mean(0, keepdim=True); self.z_sd = mu.std(0, keepdim=True).clamp(min=1e-4)
            Z = (mu - self.z_mu) / self.z_sd
        # Phase 2: latent diffusion
        self.abar = _cosine_abar(self.T, self.device)
        self.den = _LatentDenoiser(self.latent, self.n_cls, depth=self.depth).to(self.device)
        opt2 = torch.optim.Adam(self.den.parameters(), 1e-3)
        for ep in range(self.diff_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; zb = Z[idx]; cb = yt[idx]
                t = torch.randint(1, self.T, (len(idx),), device=self.device)
                ab = self.abar[t].unsqueeze(1); noise = torch.randn_like(zb)
                zn = torch.sqrt(ab) * zb + torch.sqrt(1 - ab) * noise
                loss = F.mse_loss(self.den(zn, t, cb), noise)
                opt2.zero_grad(); loss.backward(); opt2.step()
        return self

    @torch.no_grad()
    def generate(self, n, steps=100):
        y = np.random.randint(0, self.n_cls, n)
        c = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        z = torch.randn(n, self.latent, device=self.device)
        ts = torch.linspace(self.T - 1, 1, steps, device=self.device).long()
        for k in range(len(ts)):
            t = ts[k]; tb = torch.full((n,), int(t), device=self.device, dtype=torch.long)
            ab = self.abar[t]; eps = self.den(z, tb, c)
            z0 = (z - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            z = (torch.sqrt(self.abar[ts[k + 1]]) * z0 + torch.sqrt(1 - self.abar[ts[k + 1]]) * eps) \
                if k < len(ts) - 1 else z0
        z = z * self.z_sd + self.z_mu
        return self.dec(z, c).cpu().numpy(), y


class TabSynBaseline(_VAELatentDiffusion):
    name = "TabSyn"
    latent = 32; vae_epochs = 200; diff_epochs = 300; depth = 4


class scDiffusionBaseline(_VAELatentDiffusion):
    name = "scDiffusion"
    latent = 64; vae_epochs = 250; diff_epochs = 300; depth = 3   # expression-specific, wider latent


class TabDiffBaseline(BaselineModel):
    """TabDiff (ICLR'25): one continuous diffusion with a per-dimension adaptive noise
    schedule (a learned per-dimension log-SNR offset), denoising directly in the
    high-dimensional data space with a deeper MLP."""
    name = "TabDiff"

    def __init__(self, split_dim, device="cuda", T=1000, epochs=300, **kw):
        super().__init__(split_dim, device)
        self.T = T; self.epochs = epochs

    def fit(self, X, y):
        from .classical import _MLPDenoiser
        self.dim = X.shape[1]; self.n_cls = len(np.unique(y))
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        self.abar = _cosine_abar(self.T, self.device)
        self.model = _MLPDenoiser(self.dim, self.n_cls, h=512).to(self.device)
        # learned per-dimension log-SNR offset, TabDiff's adaptive-noise idea
        self.dim_offset = nn.Parameter(torch.zeros(self.dim, device=self.device))
        opt = torch.optim.Adam(list(self.model.parameters()) + [self.dim_offset], 1e-3)
        n = len(X); bs = min(64, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; xb = Xt[idx]; cb = yt[idx]
                t = torch.randint(1, self.T, (len(idx),), device=self.device)
                ab = self.abar[t].unsqueeze(1)
                logit = torch.log(ab / (1 - ab)) + self.dim_offset.unsqueeze(0)
                ab2 = torch.sigmoid(logit)
                noise = torch.randn_like(xb)
                xn = torch.sqrt(ab2) * xb + torch.sqrt(1 - ab2) * noise
                loss = F.mse_loss(self.model(xn, t, cb), noise)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def generate(self, n, steps=100):
        y = np.random.randint(0, self.n_cls, n)
        c = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        x = torch.randn(n, self.dim, device=self.device)
        ts = torch.linspace(self.T - 1, 1, steps, device=self.device).long()
        for k in range(len(ts)):
            t = ts[k]; tb = torch.full((n,), int(t), device=self.device, dtype=torch.long)
            ab = self.abar[t]; eps = self.model(x, tb, c)
            x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            x = (torch.sqrt(self.abar[ts[k + 1]]) * x0 + torch.sqrt(1 - self.abar[ts[k + 1]]) * eps) \
                if k < len(ts) - 1 else x0
        return x.cpu().numpy(), y


BASELINE_REGISTRY = {}
def _register():
    from .classical import SMOTEBaseline, CTGANBaseline, TVAEBaseline, TabDDPMBaseline
    for cls in [SMOTEBaseline, CTGANBaseline, TVAEBaseline, TabDDPMBaseline,
                TabSynBaseline, TabDiffBaseline, scDiffusionBaseline]:
        BASELINE_REGISTRY[cls.name] = cls
_register()
