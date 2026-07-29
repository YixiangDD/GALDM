"""Figures: t-SNE of the latent distribution, per-feature KDE, within-pathway correlation heatmaps."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_tsne(real_m, gen_m, out_path, title="t-SNE", max_n=400):
    from sklearn.manifold import TSNE
    if len(real_m) > max_n:
        real_m = real_m[np.random.choice(len(real_m), max_n, replace=False)]
    if len(gen_m) > max_n:
        gen_m = gen_m[np.random.choice(len(gen_m), max_n, replace=False)]
    X = np.vstack([real_m, gen_m])
    lab = np.array([0] * len(real_m) + [1] * len(gen_m))
    emb = TSNE(n_components=2, perplexity=min(30, len(X) // 4),
               init="pca", random_state=42).fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(emb[lab == 0, 0], emb[lab == 0, 1], s=12, alpha=0.6, label="Real", c="#2c7fb8")
    plt.scatter(emb[lab == 1, 0], emb[lab == 1, 1], s=12, alpha=0.6, label="Generated", c="#de2d26")
    plt.legend(); plt.title(title); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()


def plot_kde(real_m, gen_m, out_path, n_feat=6, title="Feature KDE"):
    var = real_m.var(0)
    top = np.argsort(-var)[:n_feat]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, j in zip(axes.ravel(), top):
        ax.hist(real_m[:, j], bins=30, density=True, alpha=0.5, label="Real", color="#2c7fb8")
        ax.hist(gen_m[:, j], bins=30, density=True, alpha=0.5, label="Gen", color="#de2d26")
        ax.set_title(f"gene {j}"); ax.legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()


def plot_tsne_3d(real_m, gen_m, out_path, title="3D t-SNE", max_n=400):
    """Three-dimensional t-SNE scatter, real vs generated."""
    from sklearn.manifold import TSNE
    if len(real_m) > max_n:
        real_m = real_m[np.random.choice(len(real_m), max_n, replace=False)]
    if len(gen_m) > max_n:
        gen_m = gen_m[np.random.choice(len(gen_m), max_n, replace=False)]
    X = np.vstack([real_m, gen_m])
    lab = np.array([0] * len(real_m) + [1] * len(gen_m))
    emb = TSNE(n_components=3, perplexity=min(30, max(5, len(X) // 4)),
               init="pca", random_state=42).fit_transform(X)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(emb[lab == 0, 0], emb[lab == 0, 1], emb[lab == 0, 2],
               s=14, alpha=0.6, label="Real", c="#2c7fb8")
    ax.scatter(emb[lab == 1, 0], emb[lab == 1, 1], emb[lab == 1, 2],
               s=14, alpha=0.6, label="Generated", c="#de2d26")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_zlabel("t-SNE 3")
    ax.legend(); ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()


def plot_ablation_bars(variant_metrics, out_path, title="Ablation"):
    """Ablation bar chart comparing variants on the key metrics.
    variant_metrics: {variant_name: {metric: mean_value}}"""
    variants = list(variant_metrics.keys())
    metrics = ["MMD", "miRNA_Corr", "PCE", "TSTR", "Bio_FID"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        vals = [variant_metrics[v].get(m, 0) for v in variants]
        ax.bar(range(len(variants)), vals, color="#41b6c4")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax.set_title(m)
    fig.suptitle(title); fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()


def plot_corr_heatmap(real_m, gen_m, pathway_mask, out_path, pw_idx=0, title="Pathway corr"):
    idx = np.where(pathway_mask[pw_idx] > 0)[0][:40]
    def corr(X):
        Xc = X[:, idx] - X[:, idx].mean(0, keepdims=True)
        std = Xc.std(0) + 1e-8
        Xn = Xc / std
        return (Xn.T @ Xn) / Xn.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (C, t) in zip(axes, [(corr(real_m), "Real"), (corr(gen_m), "Generated")]):
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title); fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()
