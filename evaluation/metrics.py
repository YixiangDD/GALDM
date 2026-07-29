"""The seven evaluation metrics, plus paired significance testing.

Distributional fidelity: MMD (lower better; Gaussian kernel, median bandwidth),
                         KS (higher better; fraction of features with two-sample p>0.05)
Biological consistency: miRNA Corr (higher; Frobenius similarity of within-family
                        correlation matrices), PCE (lower; Frobenius distance of
                        within-pathway co-expression), DE (higher; Spearman correlation of
                        differential-expression rankings between classes)
Downstream utility: TSTR (higher; train on synthetic, test on real, XGBoost)
Overall: Bio-FID (lower; Frechet distance after PCA with shrinkage covariance)
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


# ---------- distributional fidelity ----------
def mmd_rbf(X, Y, gamma=None, max_n=500):
    """Squared MMD with a Gaussian kernel, bandwidth from the median heuristic."""
    if len(X) > max_n:
        X = X[np.random.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[np.random.choice(len(Y), max_n, replace=False)]
    XX = cdist(X, X, "sqeuclidean")
    YY = cdist(Y, Y, "sqeuclidean")
    XY = cdist(X, Y, "sqeuclidean")
    if gamma is None:
        med = np.median(np.concatenate([XX.ravel(), YY.ravel(), XY.ravel()]))
        gamma = 1.0 / (med + 1e-8)
    k = lambda d: np.exp(-gamma * d)
    return float(k(XX).mean() + k(YY).mean() - 2 * k(XY).mean())


def ks_pass_rate(real, gen, alpha=0.05, max_feat=2000):
    """Per-feature two-sample KS test; returns the fraction of features with p>alpha."""
    n_feat = min(real.shape[1], max_feat)
    idx = np.random.choice(real.shape[1], n_feat, replace=False) if real.shape[1] > max_feat else range(real.shape[1])
    passed = 0
    for j in idx:
        try:
            _, p = stats.ks_2samp(real[:, j], gen[:, j])
            if p > alpha:
                passed += 1
        except Exception:
            pass
    return passed / len(list(idx))


# ---------- biological consistency ----------
def _corr_matrix(X):
    Xc = X - X.mean(0, keepdims=True)
    std = Xc.std(0) + 1e-8
    Xn = Xc / std
    return (Xn.T @ Xn) / Xn.shape[0]


def mirna_corr(real_mi, gen_mi, family_mask):
    """How well within-family correlation structure survives: normalised Frobenius
    similarity of the within-family correlation matrices. Higher is better."""
    sims = []
    for f in range(family_mask.shape[0]):
        idx = np.where(family_mask[f] > 0)[0]
        if len(idx) < 2:
            continue
        Cr = _corr_matrix(real_mi[:, idx])
        Cg = _corr_matrix(gen_mi[:, idx])
        fro = np.linalg.norm(Cr - Cg)
        denom = np.linalg.norm(Cr) + np.linalg.norm(Cg) + 1e-8
        sims.append(1.0 - fro / denom)
    return float(np.mean(sims)) if sims else 0.0


def pce(real_m, gen_m, pathway_mask):
    """Pathway consistency error: mean Frobenius distance between within-pathway
    co-expression matrices. Lower is better."""
    dists = []
    for p in range(pathway_mask.shape[0]):
        idx = np.where(pathway_mask[p] > 0)[0]
        if len(idx) < 2:
            continue
        if len(idx) > 200:
            idx = idx[:200]
        Cr = _corr_matrix(real_m[:, idx])
        Cg = _corr_matrix(gen_m[:, idx])
        dists.append(np.linalg.norm(Cr - Cg) / len(idx))
    return float(np.mean(dists)) if dists else 0.0


def de_consistency(real_m, real_y, gen_m, gen_y):
    """Differential-expression agreement: Spearman correlation of the between-class t-test
    p-value rankings. Higher is better."""
    def de_rank(X, y):
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            return None
        a, b = X[y == 0], X[y == 1]
        if len(a) < 2 or len(b) < 2:
            return None
        t, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
        return np.nan_to_num(np.abs(t))
    r = de_rank(real_m, real_y)
    g = de_rank(gen_m, gen_y)
    if r is None or g is None:
        return 0.0
    rho, _ = stats.spearmanr(r, g)
    return float(np.nan_to_num(rho))


# ---------- downstream utility ----------
def tstr(gen_m, gen_y, real_test_m, real_test_y):
    """Train on synthetic, test on real (XGBoost)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier as XGBClassifier
    gen_y = np.asarray(gen_y)
    if len(np.unique(gen_y)) < 2:
        return 0.0
    try:
        clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                            subsample=0.8, eval_metric="logloss", verbosity=0)
        clf.fit(gen_m, gen_y)
        pred = clf.predict(real_test_m)
        return float((pred == real_test_y).mean())
    except Exception:
        return 0.0


# ---------- overall ----------
def bio_fid(real_m, gen_m, n_comp=20):
    """Bio-FID: Frechet distance between the two distributions after PCA projection and
    Ledoit-Wolf shrinkage of the covariance. Lower is better.

    Shrinkage stabilises the covariance estimate in this small-sample high-dimensional
    regime, and a small n_comp (default 20) keeps the absolute values in a sane range. Note
    that the estimator is biased at small n, so the value is meaningful mainly as a relative
    ranking across methods, and should be read against the in-sample floor measured by
    experiments/biofid_reference.py rather than against zero.
    """
    from sklearn.covariance import LedoitWolf
    n_comp = min(n_comp, real_m.shape[0] - 1, real_m.shape[1] - 1, gen_m.shape[0] - 1)
    n_comp = max(2, n_comp)
    pca = PCA(n_components=n_comp)
    pca.fit(real_m)
    fr = pca.transform(real_m)
    fg = pca.transform(gen_m)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cov_r = LedoitWolf().fit(fr).covariance_
    cov_g = LedoitWolf().fit(fg).covariance_
    diff = mu_r - mu_g
    from scipy.linalg import sqrtm
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(cov_r + cov_g - 2 * covmean)
    return float(np.real(fid))


def compute_all_metrics(real_m, real_mi, real_y, gen_m, gen_mi, gen_y,
                        test_m, test_y, pathway_mask, family_mask):
    """Computes all seven metrics. PCE uses the first 15 rows of pathway_mask (no background)."""
    pw15 = pathway_mask[:15] if pathway_mask.shape[0] > 15 else pathway_mask
    return {
        "MMD": mmd_rbf(real_m, gen_m),
        "KS": ks_pass_rate(real_m, gen_m),
        "miRNA_Corr": mirna_corr(real_mi, gen_mi, family_mask),
        "PCE": pce(real_m, gen_m, pw15),
        "DE": de_consistency(real_m, real_y, gen_m, gen_y),
        "TSTR": tstr(gen_m, gen_y, test_m, test_y),
        "Bio_FID": bio_fid(real_m, gen_m),
    }


# ---------- regulatory direction consistency (RDC) ----------
def _node_activity(gen_m, gen_mi, pathway_mask, family_mask, n_families, n_pathways):
    """GRN node activity. Nodes 0..M-1 are miRNA families, M..M+K-1 are mRNA pathways.
    Pathway activity is the mean over member genes, family activity the mean over member
    miRNAs. Returns (N, M+K)."""
    M, K = n_families, n_pathways
    N = gen_m.shape[0]
    act = np.zeros((N, M + K), dtype=np.float64)
    # families (nodes 0..M-1)
    for f in range(M):
        idx = np.where(family_mask[f] > 0)[0]
        if len(idx) > 0:
            act[:, f] = gen_mi[:, idx].mean(axis=1)
    # pathways (nodes M..M+K-1); the first K rows of pathway_mask are pathways, no background
    for p in range(K):
        idx = np.where(pathway_mask[p] > 0)[0]
        if len(idx) > 0:
            act[:, M + p] = gen_m[:, idx].mean(axis=1)
    return act


def _pair_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.std() * b.std())
    return float((a * b).mean() / d) if d > 1e-8 else 0.0


def regulatory_direction_consistency(gen_m, gen_mi, grn, pathway_mask, family_mask,
                                     n_families=8, n_pathways=15):
    """Regulatory direction consistency (RDC), the metric PCA-CTG targets by design.

    For every GRN edge, correlate the two endpoints in the node-activity space of the
    generated data: activating edges are expected to correlate positively, suppressive edges
    negatively.
    Returns:
      RDC_acc:    fraction of edges whose sign matches expectation (higher better)
      RDC_margin: mean signed agreement strength, +corr for activating and -corr for
                  suppressive edges (higher better)
    This is measured in gene/pathway activity space, so it is independent of the
    latent-space hinge loss PCA-CTG optimises.
    """
    act = _node_activity(gen_m, gen_mi, pathway_mask, family_mask, n_families, n_pathways)
    correct, total, margins = 0, 0, []
    for (s, t) in grn.activating_edges:
        if s < act.shape[1] and t < act.shape[1]:
            c = _pair_corr(act[:, s], act[:, t])
            correct += (c > 0); total += 1; margins.append(c)      # activating: +c
    for (s, t) in grn.suppressive_edges:
        if s < act.shape[1] and t < act.shape[1]:
            c = _pair_corr(act[:, s], act[:, t])
            correct += (c < 0); total += 1; margins.append(-c)     # suppressive: -c
    if total == 0:
        return {"RDC_acc": float("nan"), "RDC_margin": float("nan")}
    return {"RDC_acc": correct / total, "RDC_margin": float(np.mean(margins))}


# ---------- significance ----------
def paired_wilcoxon(scores_a, scores_b):
    """Paired Wilcoxon signed-rank test across seeds, GA-LDM vs one baseline; returns p."""
    try:
        if len(scores_a) < 2:
            return float("nan")
        _, p = stats.wilcoxon(scores_a, scores_b)
        return float(p)
    except Exception:
        return float("nan")
