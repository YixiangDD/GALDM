"""PCA-CTG causal constraints and the empirical biological-consistency losses.

PCA-CTG:
  - Topological attention mask: an upper-triangular mask over GA-DiT's token sequence derived
    from the GRN layers (L=3), so a downstream token can only attend to upstream ones
    (layer <= its own). Applied before the self-attention softmax.
  - Causal consistency hinge: activating edges (TF -> mRNA/miRNA) are pushed towards positive
    correlation and suppressive edges (miRNA -> mRNA) towards negative, margin 0.5. Applied to
    the one-step prediction z0_hat.

Empirical consistency:
  - Top-K gene correlation preservation (the 500 most correlated gene pairs).
  - Pathway activity distribution matching (mean and variance of each KEGG pathway score).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_token_causal_mask(token_layer: np.ndarray) -> torch.Tensor:
    """token_layer: (n_tokens,) the topological layer of each token.
    Returns an (n_tokens, n_tokens) mask where -inf forbids attention: token i may only attend
    to tokens with layer <= layer_i.
    """
    n = len(token_layer)
    mask = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if token_layer[j] > token_layer[i]:
                mask[i, j] = float("-inf")
    return mask


class CausalHingeLoss(nn.Module):
    """Causal consistency hinge loss over activating and suppressive edges.

    Segment-level activation signals are taken as per-segment (pathway/family) means of the
    one-step prediction z0_hat. For each activating edge (s->t) corr(s,t) is pushed above zero
    and for each suppressive edge below, using the within-batch covariance as the estimate.
    """
    def __init__(self, seg_of_dim: np.ndarray, activating_edges, suppressive_edges,
                 margin: float = 0.5):
        super().__init__()
        self.register_buffer("seg_of_dim", torch.from_numpy(seg_of_dim.astype(np.int64)))
        self.G = int(seg_of_dim.max()) + 1
        self.act = activating_edges
        self.sup = suppressive_edges
        self.margin = margin

    def _segment_means(self, z):  # z:(B,D)->(B,G)
        B = z.shape[0]
        seg = self.seg_of_dim.to(z.device)
        out = torch.zeros(B, self.G, device=z.device)
        out.index_add_(1, seg, z)
        counts = torch.bincount(seg, minlength=self.G).clamp(min=1).float()
        return out / counts.unsqueeze(0)

    def forward(self, z0_hat):
        if not self.act and not self.sup:
            return z0_hat.new_tensor(0.0)
        seg_act = self._segment_means(z0_hat)            # (B,G)
        sc = seg_act - seg_act.mean(0, keepdim=True)
        std = sc.std(0, keepdim=True).clamp(min=1e-4)
        norm = sc / std
        loss = z0_hat.new_tensor(0.0)
        cnt = 0
        for (s, t) in self.act:
            corr = (norm[:, s] * norm[:, t]).mean()
            loss = loss + F.relu(self.margin - corr)     # push towards positive correlation
            cnt += 1
        for (s, t) in self.sup:
            corr = (norm[:, s] * norm[:, t]).mean()
            loss = loss + F.relu(corr + self.margin)     # push towards negative correlation
            cnt += 1
        return loss / max(1, cnt)


class TopKCorrLoss(nn.Module):
    """Top-K gene correlation preservation loss.

    The Ktop most correlated gene pairs are chosen on the real training split and their real
    Pearson correlations recorded; generated (decoded) samples are then required to match those
    correlations.
    """
    def __init__(self, top_pairs: np.ndarray, target_corr: np.ndarray):
        super().__init__()
        self.register_buffer("pairs", torch.from_numpy(top_pairs.astype(np.int64)))  # (P,2)
        self.register_buffer("target", torch.from_numpy(target_corr.astype(np.float32)))

    def forward(self, x_gen):  # x_gen: (B, n_genes), decoded mRNA
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        a = x_gen[:, i]; b = x_gen[:, j]
        a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
        num = (a * b).mean(0)
        den = (a.std(0) * b.std(0)).clamp(min=1e-4)
        corr = num / den
        return F.mse_loss(corr, self.target)


class PathwayActivityLoss(nn.Module):
    """Pathway activity distribution matching: mean and variance of each pathway score (the
    mean over its member genes) are pushed towards the real values."""
    def __init__(self, pathway_mask: np.ndarray, target_mean: np.ndarray, target_var: np.ndarray):
        super().__init__()
        self.register_buffer("mask", torch.from_numpy(pathway_mask.astype(np.float32)))
        self.register_buffer("t_mean", torch.from_numpy(target_mean.astype(np.float32)))
        self.register_buffer("t_var", torch.from_numpy(target_var.astype(np.float32)))

    def activity(self, x):  # (B, n_genes) -> (B, K), mean over pathway members
        m = self.mask.to(x.device)
        counts = m.sum(1).clamp(min=1)
        return (x @ m.T) / counts.unsqueeze(0)

    def forward(self, x_gen):
        a = self.activity(x_gen)
        return F.mse_loss(a.mean(0), self.t_mean) + F.mse_loss(a.var(0), self.t_var)


class PathwayCoexpressionLoss(nn.Module):
    """Within-pathway co-expression preservation, which optimises exactly what PCE measures.

    For each KEGG pathway, the within-pathway gene correlation matrix of generated samples is
    matched to the real one. This supervises the pathway structure that PA-VAE, GDVD and PCA-CTG
    are meant to preserve explicitly, compensating for the fact that global latent compression
    leaves the per-segment constraints with little to act on.
    """
    def __init__(self, pathway_mask: np.ndarray, real_mrna: np.ndarray, max_genes=60):
        super().__init__()
        # precompute member indices and the real correlation matrix per pathway
        self.pw_idx = []
        self.target_corrs = []
        for p in range(pathway_mask.shape[0]):
            idx = np.where(pathway_mask[p] > 0)[0]
            if len(idx) < 2:
                self.pw_idx.append(None); self.target_corrs.append(None); continue
            if len(idx) > max_genes:        # keep the highest-variance genes, to bound cost
                v = real_mrna[:, idx].var(0)
                idx = idx[np.argsort(-v)[:max_genes]]
            self.pw_idx.append(torch.from_numpy(idx).long())
            self.target_corrs.append(torch.from_numpy(
                self._corr(real_mrna[:, idx]).astype(np.float32)))

    @staticmethod
    def _corr(X):
        Xc = X - X.mean(0, keepdims=True)
        std = Xc.std(0) + 1e-8
        Xn = Xc / std
        return (Xn.T @ Xn) / Xn.shape[0]

    @staticmethod
    def _corr_t(X):
        Xc = X - X.mean(0, keepdim=True)
        std = Xc.std(0) + 1e-8
        Xn = Xc / std
        return (Xn.T @ Xn) / Xn.shape[0]

    def forward(self, x_gen):  # x_gen: (B, n_genes), decoded mRNA (standardised is fine too)
        loss = x_gen.new_tensor(0.0); cnt = 0
        for idx, tgt in zip(self.pw_idx, self.target_corrs):
            if idx is None:
                continue
            idx = idx.to(x_gen.device); tgt = tgt.to(x_gen.device)
            cg = self._corr_t(x_gen[:, idx])
            loss = loss + F.mse_loss(cg, tgt)
            cnt += 1
        return loss / max(1, cnt)


class MiRNAFamilyCorrLoss(nn.Module):
    """Within-family miRNA correlation preservation, which optimises the miRNA_Corr metric.

    For each miRNA family, the within-family member correlation matrix of generated samples is
    matched to the real one.
    """
    def __init__(self, family_mask: np.ndarray, real_mirna: np.ndarray):
        super().__init__()
        self.fam_idx = []
        self.target_corrs = []
        for f in range(family_mask.shape[0]):
            idx = np.where(family_mask[f] > 0)[0]
            if len(idx) < 2:
                self.fam_idx.append(None); self.target_corrs.append(None); continue
            self.fam_idx.append(torch.from_numpy(idx).long())
            self.target_corrs.append(torch.from_numpy(
                PathwayCoexpressionLoss._corr(real_mirna[:, idx]).astype(np.float32)))

    def forward(self, x_gen_mirna):
        loss = x_gen_mirna.new_tensor(0.0); cnt = 0
        for idx, tgt in zip(self.fam_idx, self.target_corrs):
            if idx is None:
                continue
            idx = idx.to(x_gen_mirna.device); tgt = tgt.to(x_gen_mirna.device)
            cg = PathwayCoexpressionLoss._corr_t(x_gen_mirna[:, idx])
            loss = loss + F.mse_loss(cg, tgt)
            cnt += 1
        return loss / max(1, cnt)


def compute_topk_pairs(mrna_train: np.ndarray, k: int = 500):
    """Selects the top-K most correlated gene pairs on the real training split, with values."""
    X = mrna_train - mrna_train.mean(0, keepdims=True)
    std = X.std(0) + 1e-8
    Xn = X / std
    n = Xn.shape[1]
    C = (Xn.T @ Xn) / Xn.shape[0]
    iu = np.triu_indices(n, k=1)
    corrs = C[iu]
    order = np.argsort(-np.abs(corrs))[:k]
    pairs = np.stack([iu[0][order], iu[1][order]], axis=1)
    target = corrs[order]
    return pairs.astype(np.int64), target.astype(np.float32)


def compute_pathway_targets(mrna_train: np.ndarray, pathway_mask: np.ndarray):
    """Mean and variance of each pathway activity on the real training split."""
    counts = pathway_mask.sum(1).clip(min=1)
    act = (mrna_train @ pathway_mask.T) / counts[None, :]
    return act.mean(0).astype(np.float32), act.var(0).astype(np.float32)
