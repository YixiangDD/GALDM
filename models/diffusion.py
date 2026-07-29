"""The diffusion process: cosine-schedule DDPM, GDVD heterogeneous SNR, DDIM sampling.

- Cosine noise schedule (T=1000, s=0.008) giving alpha_bar_t.
- GDVD: the joint latent is divided into G=K+M segments (one per pathway or miRNA family) and
  each learns an independent monotone SNR offset, initialised from the GRN causal-depth prior
  (shallower upstream nodes start at higher SNR and decay more slowly).
- min-SNR loss reweighting; CFG (condition dropout during training); deterministic DDIM sampling.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """alpha_bar_t of the cosine schedule."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    return abar.clamp(1e-5, 0.9999)  # (T+1,)


class GDVDSchedule(nn.Module):
    """Per-segment SNR offsets for GRN-disentangled variational diffusion.

    On top of the standard cosine alpha_bar_t, each joint-latent dimension gets a learnable
    log-SNR offset delta_g according to the segment (pathway or family) it belongs to, initialised
    from causal depth. This shifts that segment's alpha_bar towards higher or lower SNR.
    Shallower (upstream) segments get a larger initial offset, so their signal decays more
    slowly and survives longer.
    """
    def __init__(self, joint_dim: int, seg_of_dim: np.ndarray, causal_depth: np.ndarray,
                 enabled: bool = True, depth_scale: float = 1.0):
        super().__init__()
        self.joint_dim = joint_dim
        self.enabled = enabled
        # which segment (0..G-1) each latent dimension belongs to
        self.register_buffer("seg_of_dim", torch.from_numpy(seg_of_dim.astype(np.int64)))
        G = int(seg_of_dim.max()) + 1
        self.G = G
        # causal depth -> initial offset: shallower (upstream) means larger offset (higher SNR),
        # normalised to [-1,1] * scale
        d = causal_depth.astype(np.float32)
        if d.max() > d.min():
            d_norm = 1.0 - 2.0 * (d - d.min()) / (d.max() - d.min())  # shallow -> +1, deep -> -1
        else:
            d_norm = np.zeros_like(d)
        init = torch.from_numpy(d_norm * depth_scale)
        # one learnable log-SNR offset per segment, frozen for the first 50 epochs (the training
        # loop toggles requires_grad)
        self.delta = nn.Parameter(init.clone())

    def dim_offset(self):
        """Returns the log-SNR offset of every latent dimension, shape (joint_dim,)."""
        if not self.enabled:
            return torch.zeros(self.joint_dim, device=self.delta.device)
        return self.delta[self.seg_of_dim]

    def freeze(self):
        self.delta.requires_grad_(False)

    def unfreeze(self):
        self.delta.requires_grad_(True)


class Diffusion(nn.Module):
    def __init__(self, joint_dim: int, timesteps: int = 1000, s: float = 0.008,
                 min_snr_gamma: float = 5.0, gdvd: GDVDSchedule = None):
        super().__init__()
        self.T = timesteps
        self.min_snr_gamma = min_snr_gamma
        self.gdvd = gdvd
        abar = cosine_alpha_bar(timesteps, s)
        self.register_buffer("abar", abar)               # (T+1,)

    def _abar_t(self, t):
        return self.abar[t]                              # (B,)

    def abar_gdvd(self, t_scalar, joint_dim, device):
        """Effective alpha_bar of shape (joint_dim,) at scalar timestep t, after the GDVD
        per-dimension offset. Without GDVD this degenerates to broadcasting the scalar
        alpha_bar. Used at sampling time so the schedule matches training's q_sample."""
        abar = self.abar[t_scalar].clamp(1e-5, 0.9999)
        if self.gdvd is not None and self.gdvd.enabled:
            offset = self.gdvd.dim_offset().to(device)   # (D,)
            logit = torch.log(abar / (1 - abar)) + offset
            return torch.sigmoid(logit).clamp(1e-5, 0.9999)
        return abar.expand(joint_dim) if torch.is_tensor(abar) else abar

    def q_sample(self, z0, t, noise=None):
        """Forward noising z_t = sqrt(a)*z0 + sqrt(1-a)*eps, with the GDVD per-dim SNR offset."""
        if noise is None:
            noise = torch.randn_like(z0)
        abar = self._abar_t(t).unsqueeze(1)              # (B,1)
        if self.gdvd is not None and self.gdvd.enabled:
            # apply the log-SNR offset per dimension: logit(a') = logit(a) + delta
            offset = self.gdvd.dim_offset().unsqueeze(0)  # (1,D)
            logit = torch.log(abar / (1 - abar)) + offset
            abar = torch.sigmoid(logit)
        sqrt_ab = torch.sqrt(abar)
        sqrt_1mab = torch.sqrt(1 - abar)
        return sqrt_ab * z0 + sqrt_1mab * noise, noise, abar

    def min_snr_weight(self, t):
        """min-SNR-gamma loss reweighting."""
        abar = self._abar_t(t)
        snr = abar / (1 - abar)
        w = torch.clamp(snr, max=self.min_snr_gamma) / snr
        return w                                          # (B,)

    def p_loss(self, model, z0, t, class_label=None, activity=None, p_uncond=0.1):
        """Simple epsilon-prediction loss L_simple, with CFG condition dropout and min-SNR weighting."""
        B = z0.shape[0]
        z_t, noise, _ = self.q_sample(z0, t)
        # CFG: drop the condition with probability p_uncond
        cl = class_label
        act = activity
        if class_label is not None:
            drop = torch.rand(B, device=z0.device) < p_uncond
            cl = class_label.clone()
            cl[drop] = model.class_embed.num_embeddings - 1  # the unconditional token
        eps_hat = model(z_t, t, cl, act)
        w = self.min_snr_weight(t).unsqueeze(1)
        loss = (w * (eps_hat - noise) ** 2).mean()
        return loss, eps_hat, z_t

    @torch.no_grad()
    def ddim_sample(self, model, n_samples, joint_dim, steps=200, device="cuda",
                    class_label=None, activity=None, cfg_scale=1.5, eta=0.0):
        """Deterministic DDIM sampling. Returns the joint latent z0_hat."""
        ts = torch.linspace(self.T, 1, steps, device=device).long()
        z = torch.randn(n_samples, joint_dim, device=device)
        for i in range(steps):
            t = ts[i]
            t_b = torch.full((n_samples,), int(t), device=device, dtype=torch.long)
            # matching training q_sample: use the GDVD-adjusted effective alpha_bar, shape (1,D)
            abar_t = self.abar_gdvd(t, joint_dim, device).unsqueeze(0)
            if class_label is not None and cfg_scale != 1.0:
                eps_c = model(z, t_b, class_label, activity)
                eps_u = model(z, t_b, None, activity)
                eps = eps_u + cfg_scale * (eps_c - eps_u)
            else:
                eps = model(z, t_b, class_label, activity)
            z0 = (z - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)
            z0 = z0.clamp(-5.0, 5.0)   # the latent is standardised to about [-4,4]; clamping z0
                                       # keeps sampling from diverging
            if i < steps - 1:
                t_next = ts[i + 1]
                abar_next = self.abar_gdvd(t_next, joint_dim, device).unsqueeze(0)
                z = torch.sqrt(abar_next) * z0 + torch.sqrt(1 - abar_next) * eps
            else:
                z = z0
        return z
