"""Maps joint-latent dimensions and GA-DiT tokens onto GRN segments and topological layers.

The joint latent is [mRNA latent | miRNA-common] and the GRN has G = K pathways + M families
segments. Because the latent is globally compressed rather than sliced per segment, the
dimensions are divided evenly across segments: the mRNA half across the K pathways, the
miRNA-common half across the M families. That gives GDVD its per-segment SNR schedule and
PCA-CTG its dimension-to-segment correspondence. GA-DiT tokens take the majority layer of
the segments they cover, which is what the causal attention mask keys on.
"""
import numpy as np


def build_dim_segment_map(mrna_latent=256, mirna_common=256, n_pathways=15,
                          n_families=8, mrna_blocks=None):
    """Returns seg_of_dim (joint_dim,): the segment index of each latent dimension.

    Matches the structured PA-VAE / miRNA-VAE latent layout exactly:
    - the mRNA latent is blocked by pathway (mrna_blocks=K+1 contiguous blocks, background
      included); block i (<K) -> GRN pathway node M+i, the background block -> segment M+K,
      which is standalone and carries no GRN edges;
    - miRNA-common is mixed by a linear map, so it is split evenly across the M families and
      mapped onto GRN family nodes 0..M-1.

    GRN node numbering: 0..M-1 = miRNA families, M..M+K-1 = pathways. Segment M+K is the
    background, for which the trainer must append one causal_depth / node_layers entry.
    """
    M, K = n_families, n_pathways
    n_mrna_blocks = mrna_blocks if mrna_blocks is not None else (K + 1)  # background included
    seg = np.zeros(mrna_latent + mirna_common, dtype=np.int64)
    block_m = mrna_latent // n_mrna_blocks
    for d in range(mrna_latent):
        b = min(d // block_m, n_mrna_blocks - 1)
        seg[d] = (M + b) if b < K else (M + K)   # pathway node, or the background segment
    per_mi = max(1, mirna_common // M)
    for d in range(mirna_common):
        f = min(d // per_mi, M - 1)
        seg[mrna_latent + d] = f
    return seg


def build_token_layer_map(seg_of_dim, node_layers, n_tokens=16):
    """Each token spans a contiguous range of latent dimensions; its layer is the
    majority topological layer of the segments it covers."""
    D = len(seg_of_dim)
    per = D // n_tokens
    token_layer = np.zeros(n_tokens, dtype=np.int64)
    for t in range(n_tokens):
        s, e = t * per, min((t + 1) * per, D)
        segs = seg_of_dim[s:e]
        layers = [node_layers[g] for g in segs]
        # majority layer
        token_layer[t] = int(np.bincount(layers).argmax()) if len(layers) else 0
    return token_layer


def build_token_pathway_map(seg_of_dim, n_tokens=16, n_pathways=15, n_families=8):
    """Representative pathway ID per token, for GA-DiT's learned pathway embedding."""
    D = len(seg_of_dim)
    per = D // n_tokens
    token_pw = np.zeros(n_tokens, dtype=np.int64)
    for t in range(n_tokens):
        s, e = t * per, min((t + 1) * per, D)
        segs = seg_of_dim[s:e]
        token_pw[t] = int(np.bincount(segs).argmax()) if len(segs) else 0
    return token_pw
