# -*- coding: utf-8 -*-
"""omicsGAN baseline (Ahmed et al., Bioinformatics 2022; compbiolabucf/omicsGAN).

Faithful re-implementation of the published architecture and optimiser settings,
adapted to our train-split-only protocol.

IMPORTANT — what omicsGAN actually is (verified against the authors' code):
  * The generator input is NOT noise. omics1.py line ~250 computes
        latent_value = miRNA_batch.T @ adj          (adj = normalised bipartite
                                                     miRNA x gene target matrix,
                                                     +1/-1 signed)
    and then generated = generator(latent_value).
  * The generator loss anchors the output to the OBSERVED matrix X_0:
        loss_G = -mean(D(fake)) + lambda * ||X_0 - fake||_2
  * Therefore output row i corresponds to input sample i: omicsGAN produces an
    N x G *enhanced view* of the same N patients, not N new patients. It is a
    supervised-selected multi-omics integration/denoising method, not a sampler.
  * The authors additionally select the reported update/epoch by downstream SVM
    AUC on the phenotype label (omics1.py `prediction`), i.e. model selection
    uses the label.

We keep the architecture, the WGAN-with-weight-clipping optimisation, the
signed+degree-normalised adjacency, the anchor term and the K-update alternating
scheme. Two deviations, both to make the comparison fair rather than to weaken
the baseline, are recorded in `INFO`:
  (1) trained on the training split only (as for every other method here);
  (2) checkpoint selection by generator objective rather than by downstream
      label AUC, because no other method in the comparison may look at labels
      for model selection. Selecting by label AUC would give omicsGAN an
      advantage no other method has.
"""
import numpy as np
import torch
from torch import nn
from torch import linalg as LA

INFO = {
    "paper": "Ahmed et al., Bioinformatics 2022, 38(1):179-186",
    "code": "github.com/compbiolabucf/omicsGAN",
    "nature": "sample-paired transformation of observed profiles, not sampling from noise",
    "deviations": ["train-split only", "checkpoint by generator loss, not label AUC"],
}


class Discriminator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 768), nn.BatchNorm1d(768), nn.ReLU(),
            nn.Linear(768, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, n_input),
        )

    def forward(self, x):
        return self.model(x)


def _norm_adj(adj_bin):
    """Signed + degree-normalised adjacency, exactly as in omics1.py.

    adj_bin: (n_mirna, n_gene) binary miRNA-target indicator.
    Their code sets 1 -> -1 (repression) and 0 -> +1, then divides by
    sqrt(outer(colsum|adj|, rowsum|adj|)).
    """
    a = np.where(adj_bin > 0, -1.0, 1.0).astype(np.float32)   # (n_mi, n_gene)
    s0 = np.abs(a).sum(0)          # per-gene
    s1 = np.abs(a).sum(1)          # per-miRNA
    C = np.sqrt(np.outer(s0, s1))  # (n_gene, n_mi)
    a = a / (C.T + 1e-12)
    return a.astype(np.float32)


def _train_one_view(X_obs, other, adj, device, epochs, lr_D, lr_G,
                    anchor, critic_ite=5, weight_clip=0.01, seed=0, verbose=False):
    """Train one omicsGAN generator view.

    X_obs : (N, D)  observed matrix of the view being updated
    other : (N, D2) current matrix of the other omics
    adj   : (D2, D) normalised signed adjacency mapping other -> this view
    """
    torch.manual_seed(seed)
    N, D = X_obs.shape
    Xt = torch.from_numpy(X_obs.astype(np.float32)).to(device)
    latent = torch.from_numpy((other.astype(np.float32) @ adj)).to(device)

    G = Generator(D).to(device)
    Dis = Discriminator(D).to(device)
    optD = torch.optim.RMSprop(Dis.parameters(), lr=lr_D)
    optG = torch.optim.RMSprop(G.parameters(), lr=lr_G)

    best_loss, best_out = None, None
    for ep in range(epochs):
        for _ in range(critic_ite):
            fake = G(latent)
            Dis.zero_grad()
            lossD = torch.mean(Dis(fake)) - torch.mean(Dis(Xt))
            lossD.backward(retain_graph=True)
            optD.step()
            for p in Dis.parameters():
                p.data.clamp_(-weight_clip, weight_clip)
        G.zero_grad()
        fake = G(latent)
        lossG = -torch.mean(Dis(fake)) + anchor * LA.norm(Xt - fake, 2)
        lossG.backward()
        optG.step()

        if (ep + 1) % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                G.eval()
                out = G(latent)
                cur = (-torch.mean(Dis(out)) + anchor * LA.norm(Xt - out, 2)).item()
                G.train()
            if best_loss is None or cur < best_loss:
                best_loss, best_out = cur, out.cpu().numpy()
            if verbose:
                print(f'    ep{ep+1} lossG={cur:.3f}')
    return best_out


class OmicsGAN:
    """omicsGAN wrapper with the interface used by our comparison harness."""

    def __init__(self, adj_bin, K=3, epochs=600, device='cuda', seed=0,
                 lr_D=5e-6, lr_G=5e-5, anchor_mrna=0.01, anchor_mirna=0.001,
                 verbose=True):
        self.adj_bin = adj_bin              # (n_mirna, n_gene) binary
        self.K = K
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.lr_D, self.lr_G = lr_D, lr_G
        self.anchor_mrna, self.anchor_mirna = anchor_mrna, anchor_mirna
        self.verbose = verbose
        self.out_m = None
        self.out_mi = None

    def fit_transform(self, Xm, Xmi):
        """Run the K-update alternating scheme. Returns (synthetic mRNA, synthetic miRNA),
        each with the same number of rows as the input (this is inherent to the method)."""
        adj = _norm_adj(self.adj_bin)          # (n_mi, n_gene) -> used as other@adj
        adj_T = _norm_adj(self.adj_bin.T)      # (n_gene, n_mi)
        cur_m, cur_mi = Xm.copy(), Xmi.copy()
        self.per_update = []
        for k in range(1, self.K + 1):
            if self.verbose:
                print(f'  [omicsGAN] update {k}/{self.K}')
            new_m = _train_one_view(cur_m, cur_mi, adj, self.device, self.epochs,
                                    self.lr_D, self.lr_G, self.anchor_mrna,
                                    seed=self.seed + k, verbose=False)
            new_mi = _train_one_view(cur_mi, cur_m, adj_T, self.device, self.epochs,
                                     self.lr_D, self.lr_G, self.anchor_mirna,
                                     seed=self.seed + 100 + k, verbose=False)
            cur_m, cur_mi = new_m, new_mi
            self.per_update.append((cur_m.copy(), cur_mi.copy()))
        self.out_m, self.out_mi = cur_m, cur_mi
        return cur_m, cur_mi
