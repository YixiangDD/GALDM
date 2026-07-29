"""Builds the multi-omics GRN DAG from real TRRUST and miRTarBase records.

- TRRUST v2: TF -> target gene, annotated Activation / Repression / Unknown, aggregated into
  activating and suppressive edges between pathways.
- miRTarBase: miRNA -> target mRNA, always a suppressive edge.

Outputs, consumed by GDVD and PCA-CTG:
1. nodes = 15 KEGG pathways + 8 miRNA families (G = K + M segments, matching the joint latent);
2. directed edges between pathways/families (TRRUST cross-pathway regulation aggregated,
   plus miRNA family -> pathway suppression);
3. the causal depth of each node (longest incoming path in the DAG), used as the GDVD initial
   SNR offset prior;
4. the activating and suppressive edge sets, used by the PCA-CTG hinge loss;
5. a 3-layer node assignment (miRNA families / upstream TF pathways / downstream effector
   pathways), used as the PCA-CTG topological attention mask.
"""
import os
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .kegg_pathways import get_kegg_pathway_genes, KEGG_15_PATHWAYS

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_TRRUST = os.path.join(_CACHE_DIR, "trrust_human.tsv")
_MIRTARBASE = os.path.join(_CACHE_DIR, "mirtarbase_se_wr.xlsx")


@dataclass
class MultiOmicsGRN:
    n_mirna_families: int
    n_pathways: int                       # number of KEGG pathways, background excluded
    node_names: List[str]                 # length G = M + K
    node_layers: List[int]                # topological layer per node (0 = miRNA family, 1.. = mRNA pathway)
    causal_depth: List[int]               # causal depth per node (longest incoming path)
    activating_edges: List[Tuple[int, int]]   # (src_node, tgt_node), activating
    suppressive_edges: List[Tuple[int, int]]  # (src_node, tgt_node), suppressive (miRNA edges included)
    edge_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.n_mirna_families + self.n_pathways


def _load_trrust() -> pd.DataFrame:
    rows = []
    with open(_TRRUST, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))  # TF, target, mode
    return pd.DataFrame(rows, columns=["tf", "target", "mode"])


def _load_mirtarbase() -> pd.DataFrame:
    df = pd.read_excel(_MIRTARBASE)
    df = df[df["Species (miRNA)"] == "Homo sapiens"]
    return df[["miRNA", "Target Gene"]].rename(
        columns={"miRNA": "mirna", "Target Gene": "target"})


def _gene_to_pathways(kegg: Dict[str, List[str]], hsa_ids: List[str]) -> Dict[str, List[int]]:
    """Gene symbol -> list of KEGG pathway indices it belongs to."""
    g2p = defaultdict(list)
    for i, hsa in enumerate(hsa_ids):
        for g in kegg[hsa]:
            g2p[g].append(i)
    return g2p


def _mirna_root(mirna_name: str) -> str:
    """miRTarBase hsa-miR-21-5p -> normalised root mir-21, matching mirna_families."""
    import re
    s = mirna_name.lower()
    s = re.sub(r"^hsa[-_]", "", s)
    s = re.sub(r"[-_](5p|3p)$", "", s)
    m = re.match(r"(let[-_]?\d+|mir[-_]?\d+)", s)
    if m:
        root = m.group(1)
        if root.startswith("let"):
            return "let-7"
        return root.replace("mir", "mir-").replace("--", "-")
    return s


def build_multiomics_grn(
    pathway_info: Dict,           # from build_pathway_mask, carries hsa_ids and pathway_names
    family_info: Dict,            # from assign_mirna_families, carries family_roots
    selected_gene_syms: set,      # symbols of the selected mRNAs, used to keep only usable edges
    verbose: bool = True,
) -> MultiOmicsGRN:
    """Builds the multi-omics GRN.

    Node numbering: 0..M-1 = miRNA families, M..M+K-1 = KEGG pathways (background excluded).
    """
    hsa_ids = [h for h in pathway_info["hsa_ids"] if h != "background"]
    K = len(hsa_ids)
    M = family_info["n_families"]
    family_roots = family_info["family_roots"]   # List[List[str]], the roots in each family

    kegg = get_kegg_pathway_genes(verbose=False)
    g2p = _gene_to_pathways(kegg, hsa_ids)

    # ---- 1) TRRUST: pathway -> pathway edges, aggregating over the pathways the TF and
    #         the target each belong to ----
    trr = _load_trrust()
    act_counter = defaultdict(int)
    rep_counter = defaultdict(int)
    for _, r in trr.iterrows():
        tf, tgt, mode = r["tf"], r["target"], r["mode"]
        src_paths = g2p.get(tf, [])
        tgt_paths = g2p.get(tgt, [])
        for sp in src_paths:
            for tp in tgt_paths:
                if sp == tp:
                    continue
                key = (M + sp, M + tp)  # pathway node indices are offset by M
                if mode == "Activation":
                    act_counter[key] += 1
                elif mode == "Repression":
                    rep_counter[key] += 1
                else:
                    act_counter[key] += 0.5  # Unknown counts as a weak lean towards activation

    # ---- 2) miRTarBase: miRNA family -> pathway suppressive edges ----
    mtb = _load_mirtarbase()
    # family root -> family node id
    root2fam = {}
    for fam_id, roots in enumerate(family_roots):
        for rt in roots:
            root2fam[rt] = fam_id
    mirna_sup = defaultdict(int)
    for _, r in mtb.iterrows():
        root = _mirna_root(str(r["mirna"]))
        fam = root2fam.get(root)
        if fam is None:
            continue
        tgt_paths = g2p.get(r["target"], [])
        for tp in tgt_paths:
            mirna_sup[(fam, M + tp)] += 1  # miRNA family node `fam` suppresses pathway M+tp

    # ---- 3) net pathway-pathway edges: merge both directions, keep the net one ----
    # Pathway regulation is heavily bidirectional (feedback loops), so each unordered pair is
    # oriented by net flow first, then cycles are removed to leave a DAG.
    # net_pp[(a,b)] is the net support for a->b; activation counts positive, repression
    # negative, and the magnitude decides the orientation.
    MIN_EDGE = 3  # a pathway edge needs at least 3 net TF-target records, for denoising
    pair_dir = {}   # ordered (a,b): net activation/suppression strength
    pair_sign = {}  # (a,b): +1 activating, -1 suppressive
    seen_pairs = set()
    pp_keys = set(k for k in list(act_counter) + list(rep_counter) if k[0] >= M and k[1] >= M)
    for (a, b) in pp_keys:
        if (a, b) in seen_pairs or (b, a) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        # net activation/suppression for a->b and for b->a separately
        ab = act_counter.get((a, b), 0) - rep_counter.get((a, b), 0)
        ba = act_counter.get((b, a), 0) - rep_counter.get((b, a), 0)
        tot_ab = act_counter.get((a, b), 0) + rep_counter.get((a, b), 0)
        tot_ba = act_counter.get((b, a), 0) + rep_counter.get((b, a), 0)
        # orient towards whichever side has more total support
        if tot_ab >= tot_ba:
            src, tgt, net = a, b, ab
        else:
            src, tgt, net = b, a, ba
        strength = max(tot_ab, tot_ba)
        if strength >= MIN_EDGE:
            pair_dir[(src, tgt)] = strength
            pair_sign[(src, tgt)] = 1 if net >= 0 else -1

    # ---- 4) greedy feedback-arc removal: order pathway nodes by net strength and drop the
    #         backward edges ----
    # A simplified Eades heuristic: sort nodes by (out-strength - in-strength) descending to
    # get a linear order, then keep forward edges only.
    out_str = defaultdict(float); in_str = defaultdict(float)
    for (s, t), w in pair_dir.items():
        out_str[s] += w; in_str[t] += w
    path_nodes = list(range(M, M + K))
    order = sorted(path_nodes, key=lambda n: (out_str[n] - in_str[n]), reverse=True)
    rank = {n: i for i, n in enumerate(order)}

    activating: List[Tuple[int, int]] = []
    suppressive: List[Tuple[int, int]] = []
    eweight: Dict[Tuple[int, int], float] = {}
    for (s, t), w in pair_dir.items():
        if rank[s] < rank[t]:               # forward edges only, so the result is acyclic
            (activating if pair_sign[(s, t)] > 0 else suppressive).append((s, t))
            eweight[(s, t)] = w

    # miRNA family -> pathway suppression; miRNAs are always upstream, so no cycle is possible
    for key, c in mirna_sup.items():
        if c >= 1:
            suppressive.append(key)
            eweight[key] = float(c)

    # ---- 5) causal depth = longest incoming path in the DAG; the 3 layers follow from it ----
    adj = defaultdict(list)
    indeg2 = defaultdict(int)
    nodes = list(range(M + K))
    for (s, t) in activating + suppressive:
        adj[s].append(t)
        indeg2[t] += 1
    depth = [0] * (M + K)
    q = deque([n for n in nodes if indeg2[n] == 0])
    tmp_indeg = dict(indeg2)
    while q:
        u = q.popleft()
        for v in adj[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            tmp_indeg[v] -= 1
            if tmp_indeg[v] == 0:
                q.append(v)

    # ---- 6) map nodes onto the L=3 topological layers; miRNA families are fixed at layer 0 ----
    # pathways are split by depth tercile into upstream (1), middle (2) and downstream (3).
    node_layers = [0] * (M + K)
    path_depths = [depth[M + p] for p in range(K)]
    if max(path_depths) > 0:
        d1 = np.percentile(path_depths, 33)
        d2 = np.percentile(path_depths, 67)
        for p in range(K):
            dp = depth[M + p]
            node_layers[M + p] = 1 if dp <= d1 else (2 if dp <= d2 else 3)
    else:
        for p in range(K):
            node_layers[M + p] = 2


    node_names = [f"miRNA_fam_{i}" for i in range(M)] + \
                 [KEGG_15_PATHWAYS[h] for h in hsa_ids]

    grn = MultiOmicsGRN(
        n_mirna_families=M, n_pathways=K,
        node_names=node_names, node_layers=node_layers, causal_depth=depth,
        activating_edges=activating, suppressive_edges=suppressive,
        edge_weights=eweight,
    )
    if verbose:
        print(f"  [grn] nodes {M} miRNA families + {K} pathways, {len(activating)} activating, "
              f"{len(suppressive)} suppressive, causal depth [{min(depth)},{max(depth)}]")
        print(f"  [grn] layers: L0(miRNA)={node_layers[:M].count(0)}, "
              f"L1 upstream={node_layers.count(1)}, L2 middle={node_layers.count(2)}, "
              f"L3 downstream={node_layers.count(3)}")
    return grn
