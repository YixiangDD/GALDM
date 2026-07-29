# -*- coding: utf-8 -*-
"""A native joint multi-view VAE baseline.

Canonical shared-latent conditional multi-view VAE: one MLP encoder per modality, fused into a
shared posterior, class-conditioned, two decoders, joint ELBO. Paired samples are generated
from N(0,I) plus a class label.

This is a method designed for paired generation from the outset, unlike GA-LDM (which adds
pathway structure and latent diffusion) and unlike the generic tabular baselines. Same
evaluation protocol as everything else (compute_all_metrics plus XM-Corr), 4 cohorts, 5 seeds."""
import os, sys, json, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_config
from data_pipeline.prepare import prepare_cancer
from evaluation.metrics import compute_all_metrics
from experiments.crossmodal_eval import load_mirtarbase_edges, xm_corr
from models.postprocess import postprocess
from configs import paths as _P

RAW = _P.RAW_ROOT; PROC = _P.PROCESSED_DIR
RES = _P.RESULTS_DIR


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class MultiViewCVAE(nn.Module):
    """shared-latent conditional multi-view VAE (mRNA + miRNA)."""
    def __init__(self, d_m, d_mi, latent=128, hid=512, n_cls=2):
        super().__init__()
        self.latent = latent; self.n_cls = n_cls
        self.enc_m = nn.Sequential(nn.Linear(d_m, hid), nn.ReLU(), nn.Linear(hid, 256), nn.ReLU())
        self.enc_mi = nn.Sequential(nn.Linear(d_mi, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.to_mu = nn.Linear(256 + 128 + n_cls, latent)
        self.to_lv = nn.Linear(256 + 128 + n_cls, latent)
        self.dec_m = nn.Sequential(nn.Linear(latent + n_cls, hid), nn.ReLU(), nn.Linear(hid, d_m))
        self.dec_mi = nn.Sequential(nn.Linear(latent + n_cls, 256), nn.ReLU(), nn.Linear(256, d_mi))

    def encode(self, xm, xmi, c):
        h = torch.cat([self.enc_m(xm), self.enc_mi(xmi), c], 1)
        return self.to_mu(h), self.to_lv(h)

    def decode(self, z, c):
        zc = torch.cat([z, c], 1)
        return self.dec_m(zc), self.dec_mi(zc)

    def forward(self, xm, xmi, c):
        mu, lv = self.encode(xm, xmi, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        rm, rmi = self.decode(z, c)
        return rm, rmi, mu, lv


def train_mvae(Xm, Xmi, y, d_m, d_mi, dev, epochs=800, beta=0.5):
    n_cls = int(y.max()) + 1
    model = MultiViewCVAE(d_m, d_mi, n_cls=n_cls).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xm = torch.tensor(Xm, dtype=torch.float32, device=dev)
    xmi = torch.tensor(Xmi, dtype=torch.float32, device=dev)
    c = F.one_hot(torch.tensor(y, dtype=torch.long, device=dev), n_cls).float()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        rm, rmi, mu, lv = model(xm, xmi, c)
        rec = F.mse_loss(rm, xm) + F.mse_loss(rmi, xmi)
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        loss = rec + beta * kl
        loss.backward(); opt.step()
    return model, n_cls


def generate_mvae(model, n_cls, n, y_ref, dev):
    """Samples from N(0,I) with class labels drawn at the training class proportions."""
    model.eval()
    rng = np.random.RandomState(0)
    # assign labels at the real class proportions
    props = np.bincount(y_ref, minlength=n_cls) / len(y_ref)
    labels = rng.choice(n_cls, size=n, p=props)
    c = F.one_hot(torch.tensor(labels, dtype=torch.long, device=dev), n_cls).float()
    z = torch.randn(n, model.latent, device=dev)
    with torch.no_grad():
        gm, gmi = model.decode(z, c)
    return gm.cpu().numpy(), gmi.cpu().numpy(), labels


def run(cancers=("CESC", "COAD", "HNSC", "KIRC"), seeds=(42, 123, 456, 789, 1024)):
    cfg = get_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mkeys = ["MMD", "KS", "miRNA_Corr", "PCE", "DE", "TSTR", "Bio_FID", "XM_Corr"]
    out = {}
    for cancer in cancers:
        out[cancer] = {k: [] for k in mkeys}
        for seed in seeds:
            set_seed(seed)
            ds, bp = prepare_cancer(RAW, PROC, cancer, seed, verbose=False)
            norm = ds["norm"]
            Xm, Xmi, y = ds["train"]["mrna"], ds["train"]["mirna"], ds["train"]["labels"]
            real_m = Xm * norm["mrna_sd"] + norm["mrna_mu"]
            real_mi = Xmi * norm["mirna_sd"] + norm["mirna_mu"]
            test_m = ds["test"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
            test_y = ds["test"]["labels"]
            n = ds["train"]["n_samples"]
            edges = load_mirtarbase_edges([str(g) for g in ds["gene_ids"]],
                                          [str(m) for m in ds["mirna_ids"]], bp.ens2sym)
            set_seed(seed)
            model, n_cls = train_mvae(Xm, Xmi, y, Xm.shape[1], Xmi.shape[1], dev)
            g_std_m, g_std_mi, gy = generate_mvae(model, n_cls, n, y, dev)
            # de-standardise back to the original space
            gm = g_std_m * norm["mrna_sd"] + norm["mrna_mu"]
            gmi = g_std_mi * norm["mirna_sd"] + norm["mirna_mu"]
            # same post-processing as GA-LDM and the baselines: expression quantile calibration
            gm, gmi, gy = postprocess(gm, gmi, real_m, real_mi, do_filter=False,
                                      calib_strength=cfg.eval.calib_strength,
                                      gen_labels=gy, real_labels=y)
            met = compute_all_metrics(real_m, real_mi, y, gm, gmi, gy, test_m, test_y,
                                      bp.pathway_mask, bp.mirna_family_mask)
            met["XM_Corr"] = xm_corr(real_mi, real_m, gmi, gm, edges)[0]
            for k in mkeys:
                out[cancer][k].append(float(met[k]))
            print("[%s s%d] MVAE BioFID=%.1f PCE=%.3f KS=%.3f miRNA=%.3f XM=%.3f"
                  % (cancer, seed, met["Bio_FID"], met["PCE"], met["KS"], met["miRNA_Corr"], met["XM_Corr"]), flush=True)
        with open(os.path.join(RES, "t2_mvae_baseline.json"), "w") as f:
            json.dump(out, f, indent=2)

    print("\n===== T2-2 multi-view VAE baseline (mean over seeds) =====")
    for cancer in cancers:
        r = out[cancer]
        print("%-5s BioFID=%.1f PCE=%.3f miRNA=%.3f XM=%.3f"
              % (cancer, np.mean(r["Bio_FID"]), np.mean(r["PCE"]), np.mean(r["miRNA_Corr"]), np.mean(r["XM_Corr"])))
    return out


if __name__ == "__main__":
    t0 = time.time()
    run()
    print("\n[done] %.0fs" % (time.time() - t0))
