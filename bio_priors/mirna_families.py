"""miRNA family grouping.

Groups the 200 retained miRNAs into M=8 families using miRBase family annotation. Names
look like hsa-mir-375 or hsa-mir-9-2. Normalisation strips the species prefix (hsa-), the
arm marker (-5p/-3p) and copy suffixes (-1/-2/-3 and letters a/b/c) to get a family root
(mir-9, let-7, mir-200); roots are then packed into 8 balanced buckets by size, with small
families merged into the nearest bucket.
"""
import re
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def _family_root(mirna_id: str) -> str:
    """hsa-mir-196a-2 -> mir-196 ; hsa-let-7a -> let-7 ; hsa-mir-9-2 -> mir-9."""
    s = mirna_id.lower()
    s = re.sub(r"^hsa[-_]", "", s)
    s = re.sub(r"[-_](5p|3p)$", "", s)
    # let-7 series
    m = re.match(r"(let[-_]?\d+)", s)
    if m:
        return "let-7" if m.group(1).startswith("let") else m.group(1)
    # mir-NNN[letter][-copy]
    m = re.match(r"(mir[-_]?\d+)", s)
    if m:
        return re.sub(r"[-_]?", "-", m.group(1), count=0).replace("mir", "mir-").replace("--", "-")
    return s


def assign_mirna_families(mirna_ids: List[str], n_families: int = 8,
                          verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """Returns the miRNA family mask in {0,1}^(M, n_mirna) plus family_info.

    Each miRNA belongs to exactly one family (unlike genes, which may reuse pathways).
    """
    n = len(mirna_ids)
    roots = [_family_root(m) for m in mirna_ids]

    # group by family root, recording member indices per root
    root2idx = defaultdict(list)
    for j, r in enumerate(roots):
        root2idx[r].append(j)

    # greedy balanced bin-packing of roots into n_families buckets, largest first
    roots_sorted = sorted(root2idx.items(), key=lambda kv: -len(kv[1]))
    buckets: List[List[int]] = [[] for _ in range(n_families)]
    bucket_roots: List[List[str]] = [[] for _ in range(n_families)]
    for root, idxs in roots_sorted:
    # drop into the currently smallest bucket
        b = min(range(n_families), key=lambda k: len(buckets[k]))
        buckets[b].extend(idxs)
        bucket_roots[b].append(root)

    mask = np.zeros((n_families, n), dtype=np.float32)
    for b, idxs in enumerate(buckets):
        for j in idxs:
            mask[b, j] = 1.0

    family_info = {
        "n_families": n_families,
        "members_per_family": [len(b) for b in buckets],
        "family_roots": [sorted(set(br)) for br in bucket_roots],
        "n_unique_roots": len(root2idx),
    }
    if verbose:
        print(f"  [mirna_fam] {n} miRNAs -> {n_families} families, "
              f"members per family {family_info['members_per_family']}, "
              f"{len(root2idx)} distinct family roots")
    return mask, family_info
