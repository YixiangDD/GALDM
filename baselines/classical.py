"""Comparison baselines behind one interface, BaselineModel.

Every baseline takes the concatenated [mRNA|miRNA] matrix (N, n_genes+n_mirna)
and labels y. After fit(), generate(n) returns synthetic samples plus labels,
which evaluation splits back at split_dim = n_genes (first n_genes columns = mRNA).

Classical baselines (this file): SMOTE, CTGAN, TVAE, TabDDPM.
Recent SOTA (modern.py): TabSyn, TabDiff, scDiffusion.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel:
    name = "base"

    def __init__(self, split_dim, device="cuda", **kw):
        self.split_dim = split_dim
        self.device = device

    def fit(self, X, y):
        raise NotImplementedError

    def generate(self, n):
        raise NotImplementedError

    def split(self, X):
        return X[:, :self.split_dim], X[:, self.split_dim:]


# ---------------- SMOTE ----------------
class SMOTEBaseline(BaselineModel):
    name = "SMOTE"

    def fit(self, X, y):
        self.X, self.y = X, np.asarray(y)
        return self

    def generate(self, n):
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        y = self.y
        cnt = Counter(y)
        # oversample each class until roughly n new samples exist
        target = {c: cnt[c] + n // len(cnt) for c in cnt}
        k = min(5, min(cnt.values()) - 1)
        if k < 1:
            # too few samples in this class, fall back to noisy duplication
            idx = np.random.choice(len(self.X), n)
            return self.X[idx] + np.random.randn(n, self.X.shape[1]) * 0.01, y[idx]
        sm = SMOTE(sampling_strategy=target, k_neighbors=k, random_state=42)
        Xr, yr = sm.fit_resample(self.X, y)
        # keep only the newly created part
        Xnew = Xr[len(self.X):]; ynew = yr[len(self.y):]
        if len(Xnew) >= n:
            return Xnew[:n], ynew[:n]
        return Xnew, ynew


# ---------------- CTGAN (own implementation, WGAN-GP for tabular) ----------------
class _Gen(nn.Module):
    def __init__(self, z_dim, c_dim, out_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, out_dim))

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))


class _Disc(nn.Module):
    def __init__(self, in_dim, c_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + c_dim, h), nn.LeakyReLU(0.2),
            nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], 1))


class CTGANBaseline(BaselineModel):
    name = "CTGAN"

    def __init__(self, split_dim, device="cuda", z_dim=128, epochs=300, **kw):
        super().__init__(split_dim, device)
        self.z_dim = z_dim; self.epochs = epochs

    def fit(self, X, y):
        self.dim = X.shape[1]; self.n_cls = len(np.unique(y))
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        self.G = _Gen(self.z_dim, self.n_cls, self.dim).to(self.device)
        self.D = _Disc(self.dim, self.n_cls).to(self.device)
        og = torch.optim.Adam(self.G.parameters(), 2e-4, betas=(0.5, 0.9))
        od = torch.optim.Adam(self.D.parameters(), 2e-4, betas=(0.5, 0.9))
        n = len(X); bs = min(64, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; xb = Xt[idx]; cb = yt[idx]
                if len(idx) < 4:
                    continue
                # D step (WGAN-GP)
                for _ in range(2):
                    z = torch.randn(len(idx), self.z_dim, device=self.device)
                    xf = self.G(z, cb).detach()
                    eps = torch.rand(len(idx), 1, device=self.device)
                    xh = (eps * xb + (1 - eps) * xf).requires_grad_(True)
                    dh = self.D(xh, cb)
                    g = torch.autograd.grad(dh.sum(), xh, create_graph=True)[0]
                    gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
                    ld = self.D(xf, cb).mean() - self.D(xb, cb).mean() + 10 * gp
                    od.zero_grad(); ld.backward(); od.step()
                # G step
                z = torch.randn(len(idx), self.z_dim, device=self.device)
                lg = -self.D(self.G(z, cb), cb).mean()
                og.zero_grad(); lg.backward(); og.step()
        return self

    @torch.no_grad()
    def generate(self, n):
        self.G.eval()
        y = np.random.randint(0, self.n_cls, n)
        c = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        z = torch.randn(n, self.z_dim, device=self.device)
        return self.G(z, c).cpu().numpy(), y


# ---------------- TVAE (tabular VAE) ----------------
class TVAEBaseline(BaselineModel):
    name = "TVAE"

    def __init__(self, split_dim, device="cuda", latent=128, epochs=300, **kw):
        super().__init__(split_dim, device)
        self.latent = latent; self.epochs = epochs

    def fit(self, X, y):
        self.dim = X.shape[1]; self.n_cls = len(np.unique(y))
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        h = 256
        self.enc = nn.Sequential(nn.Linear(self.dim + self.n_cls, h), nn.ReLU(),
                                 nn.Linear(h, 2 * self.latent)).to(self.device)
        self.dec = nn.Sequential(nn.Linear(self.latent + self.n_cls, h), nn.ReLU(),
                                 nn.Linear(h, self.dim)).to(self.device)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), 1e-3)
        n = len(X); bs = min(64, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; xb = Xt[idx]; cb = yt[idx]
                mu, lv = self.enc(torch.cat([xb, cb], 1)).chunk(2, 1)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
                rec = self.dec(torch.cat([z, cb], 1))
                loss = F.mse_loss(rec, xb) - 0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()) * 0.01
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def generate(self, n):
        y = np.random.randint(0, self.n_cls, n)
        c = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        z = torch.randn(n, self.latent, device=self.device)
        return self.dec(torch.cat([z, c], 1)).cpu().numpy(), y


# ---------------- TabDDPM (tabular DDPM, MLP denoiser) ----------------
class _MLPDenoiser(nn.Module):
    def __init__(self, dim, c_dim, h=512, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(dim + c_dim + t_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))

    def temb(self, t):
        import math
        half = self.t_dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        a = t.float().unsqueeze(1) * f.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, x, t, c):
        return self.net(torch.cat([x, c, self.temb(t)], 1))


class TabDDPMBaseline(BaselineModel):
    name = "TabDDPM"

    def __init__(self, split_dim, device="cuda", T=1000, epochs=300, **kw):
        super().__init__(split_dim, device)
        self.T = T; self.epochs = epochs

    def _abar(self, T):
        import math
        t = torch.linspace(0, T, T + 1) / T
        f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        return (f / f[0]).clamp(1e-5, 0.9999)

    def fit(self, X, y):
        self.dim = X.shape[1]; self.n_cls = len(np.unique(y))
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = F.one_hot(torch.tensor(y, dtype=torch.long), self.n_cls).float().to(self.device)
        self.abar = self._abar(self.T).to(self.device)
        self.model = _MLPDenoiser(self.dim, self.n_cls).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), 1e-3)
        n = len(X); bs = min(64, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; xb = Xt[idx]; cb = yt[idx]
                t = torch.randint(1, self.T, (len(idx),), device=self.device)
                ab = self.abar[t].unsqueeze(1)
                noise = torch.randn_like(xb)
                xn = torch.sqrt(ab) * xb + torch.sqrt(1 - ab) * noise
                pred = self.model(xn, t, cb)
                loss = F.mse_loss(pred, noise)
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
            ab = self.abar[t]
            eps = self.model(x, tb, c)
            x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            if k < len(ts) - 1:
                abn = self.abar[ts[k + 1]]
                x = torch.sqrt(abn) * x0 + torch.sqrt(1 - abn) * eps
            else:
                x = x0
        return x.cpu().numpy(), y
