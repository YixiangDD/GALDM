"""GA-DiT, the gene-adaptive denoising transformer.

The joint latent vector is reshaped into n_tokens tokens and patch-embedded to hidden_dim.
Position encoding is pathway-aware (sinusoidal plus a learnable pathway-ID embedding), followed
by 6 DiT blocks that inject the timestep, class and pathway-activity conditions through
AdaLN-Zero, with SwiGLU FFNs and an optional causal attention mask from PCA-CTG. The output is
a noise estimate with the same shape as the input.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT block, with support for a causal attention mask."""
    def __init__(self, dim: int, cond_dim: int, n_heads: int, ffn_dim: int, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(dim, ffn_dim, dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond, attn_mask: Optional[torch.Tensor] = None):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + g1.unsqueeze(1) * a
        h = self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class GADiT(nn.Module):
    def __init__(self, joint_dim: int, n_tokens: int = 16, hidden_dim: int = 256,
                 n_blocks: int = 6, n_heads: int = 8, ffn_dim: int = 1024,
                 cond_dim: int = 128, n_classes: int = 2, n_pathways: int = 16,
                 n_activity: int = 16, dropout: float = 0.1,
                 use_pathway_identity: bool = True):
        super().__init__()
        assert joint_dim % n_tokens == 0, f"joint_dim {joint_dim} is not divisible by n_tokens {n_tokens}"
        self.joint_dim = joint_dim
        self.n_tokens = n_tokens
        self.token_dim = joint_dim // n_tokens
        self.hidden_dim = hidden_dim
        # single-factor control: with False the pathway/family identity encoding is dropped
        # (weights remain, unused, so parameter counts match) giving a plain DiT
        self.use_pathway_identity = use_pathway_identity

        self.token_in = nn.Linear(self.token_dim, hidden_dim)
        self.token_out = nn.Linear(hidden_dim, self.token_dim)

        # pathway-aware position encoding: sinusoidal plus a learnable pathway-ID embedding
        self.register_buffer("sin_pos", self._sinusoid(n_tokens, hidden_dim))
        self.pathway_embed = nn.Embedding(n_pathways + 1, hidden_dim)
        # token -> pathway-ID map, set at run time by set_token_pathway; identity by default
        self.register_buffer("token_pathway", torch.zeros(n_tokens, dtype=torch.long))

        # conditions: timestep, class and pathway activity, each cond_dim wide, summed
        self.time_embed = nn.Sequential(SinusoidalEmbedding(cond_dim),
                                        nn.Linear(cond_dim, cond_dim), nn.SiLU(),
                                        nn.Linear(cond_dim, cond_dim))
        self.class_embed = nn.Embedding(n_classes + 1, cond_dim)  # +1 for the CFG null token
        self.activity_embed = nn.Sequential(nn.Linear(n_activity, cond_dim), nn.SiLU(),
                                            nn.Linear(cond_dim, cond_dim))
        self.cond_dim = cond_dim
        self.n_activity = n_activity

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, cond_dim, n_heads, ffn_dim, dropout) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self._causal_mask: Optional[torch.Tensor] = None

    @staticmethod
    def _sinusoid(n, d):
        pos = torch.arange(n).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000) / d))
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def set_token_pathway(self, token_pathway: np.ndarray):
        self.token_pathway = torch.from_numpy(token_pathway.astype(np.int64))

    def set_causal_mask(self, mask: Optional[torch.Tensor]):
        """PCA-CTG topological attention mask (n_tokens, n_tokens); -inf forbids attention."""
        self._causal_mask = mask

    def _build_cond(self, t, class_label, activity):
        B = t.shape[0]
        c = self.time_embed(t)
        if class_label is not None:
            c = c + self.class_embed(class_label)
        else:
            null = torch.full((B,), self.class_embed.num_embeddings - 1,
                              dtype=torch.long, device=t.device)
            c = c + self.class_embed(null)
        if activity is not None:
            c = c + self.activity_embed(activity)
        else:
            c = c + self.activity_embed(torch.zeros(B, self.n_activity, device=t.device))
        return c

    def forward(self, z_t, t, class_label=None, activity=None):
        B = z_t.shape[0]
        cond = self._build_cond(t, class_label, activity)
        x = z_t.reshape(B, self.n_tokens, self.token_dim)
        x = self.token_in(x)
        # position encoding: GA-DiT uses sinusoidal + pathway identity, plain DiT sinusoidal only
        if self.use_pathway_identity:
            pe = self.sin_pos + self.pathway_embed(self.token_pathway.to(z_t.device)).unsqueeze(0)
        else:
            pe = self.sin_pos
        x = x + pe
        mask = self._causal_mask.to(z_t.device) if self._causal_mask is not None else None
        for blk in self.blocks:
            x = blk(x, cond, attn_mask=mask)
        s, sc = self.final_adaLN(cond).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + sc.unsqueeze(1)) + s.unsqueeze(1)
        x = self.token_out(x)
        return x.reshape(B, -1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class MLPDenoiser(nn.Module):
    """Plain MLP denoiser, used by the M0/M1 ablations in place of GA-DiT.

    Same interface as GADiT, forward(z_t, t, class_label, activity), but with no tokenisation,
    no pathway position encoding, no AdaLN-Zero and no causal attention: purely dense layers
    with the timestep, class and activity conditions concatenated. Isolates what the GA-DiT
    backbone contributes.
    """
    def __init__(self, joint_dim, hidden_dim=512, cond_dim=128, n_classes=2,
                 n_activity=16, depth=4, dropout=0.1):
        super().__init__()
        self.joint_dim = joint_dim
        self.cond_dim = cond_dim
        self.n_activity = n_activity
        self.time_embed = nn.Sequential(SinusoidalEmbedding(cond_dim),
                                        nn.Linear(cond_dim, cond_dim), nn.SiLU(),
                                        nn.Linear(cond_dim, cond_dim))
        self.class_embed = nn.Embedding(n_classes + 1, cond_dim)
        self.activity_embed = nn.Sequential(nn.Linear(n_activity, cond_dim), nn.SiLU(),
                                            nn.Linear(cond_dim, cond_dim))
        layers = [nn.Linear(joint_dim + cond_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, joint_dim)]
        self.net = nn.Sequential(*layers)
        # kept so trainer/diffusion can still read class_embed.num_embeddings
        self.set_token_pathway = lambda *a, **k: None
        self.set_causal_mask = lambda *a, **k: None

    def _build_cond(self, t, class_label, activity):
        B = t.shape[0]
        c = self.time_embed(t)
        if class_label is not None:
            c = c + self.class_embed(class_label)
        else:
            null = torch.full((B,), self.class_embed.num_embeddings - 1,
                              dtype=torch.long, device=t.device)
            c = c + self.class_embed(null)
        if activity is not None:
            c = c + self.activity_embed(activity)
        else:
            c = c + self.activity_embed(torch.zeros(B, self.n_activity, device=t.device))
        return c

    def forward(self, z_t, t, class_label=None, activity=None):
        cond = self._build_cond(t, class_label, activity)
        return self.net(torch.cat([z_t, cond], dim=-1))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

