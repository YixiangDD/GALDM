"""Ensembl gene ID -> HGNC gene symbol mapping.

The expression matrices use versioned Ensembl IDs (e.g. ENSG00000141510.16), while KEGG
pathways and the TRRUST GRN use symbols (e.g. TP53). This module resolves the mapping in
batches through MyGene.info and caches it to local json so repeat runs stay offline. The
mapping is real rather than a hash-based pseudo-grouping, which is what makes the
pathway-aware and GRN-prior components biologically interpretable at all.
"""
import os
import json
import time
import ssl
import urllib.request
import urllib.parse
from typing import List, Dict

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "ensembl2symbol.json")
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def _strip_version(ensembl_id: str) -> str:
    """ENSG00000141510.16 -> ENSG00000141510."""
    return ensembl_id.split(".")[0]


def _load_cache() -> Dict[str, str]:
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: Dict[str, str]):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _query_batch(ensembl_ids: List[str], retries: int = 3) -> Dict[str, str]:
    """Batched MyGene.info POST query, ensembl.gene -> symbol."""
    url = "https://mygene.info/v3/query"
    data = urllib.parse.urlencode({
        "q": ",".join(ensembl_ids),
        "scopes": "ensembl.gene",
        "fields": "symbol",
        "species": "human",
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            resp = urllib.request.urlopen(req, timeout=30, context=_CTX)
            hits = json.loads(resp.read())
            out = {}
            for h in hits:
                qid = h.get("query")
                sym = h.get("symbol")
                if qid and sym and not h.get("notfound"):
                    out[qid] = sym
            return out
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [gene_mapping] batch query failed after {retries} tries: {e}")
                return {}
            time.sleep(2 * (attempt + 1))
    return {}


def map_ensembl_to_symbol(ensembl_ids: List[str], batch_size: int = 500,
                          verbose: bool = True) -> Dict[str, str]:
    """Returns {versioned Ensembl ID -> gene symbol}. Unmapped IDs are absent.

    Uses the local cache (keyed by version-stripped stable ID) to limit network calls.
    """
    cache = _load_cache()
    stable_ids = {eid: _strip_version(eid) for eid in ensembl_ids}
    need = sorted({s for s in stable_ids.values() if s not in cache})

    if need:
        if verbose:
            print(f"  [gene_mapping] querying {len(need)} new genes online (cache hit "
                  f"{len(set(stable_ids.values())) - len(need)})")
        for i in range(0, len(need), batch_size):
            chunk = need[i:i + batch_size]
            result = _query_batch(chunk)
            for sid in chunk:
                cache[sid] = result.get(sid, "")  # empty string = looked up, no symbol
            if verbose:
                print(f"    progress {min(i + batch_size, len(need))}/{len(need)}")
            time.sleep(0.5)
        _save_cache(cache)

    # assemble the result, keeping only IDs that resolved to a symbol
    mapping = {}
    for eid, sid in stable_ids.items():
        sym = cache.get(sid, "")
        if sym:
            mapping[eid] = sym
    if verbose:
        print(f"  [gene_mapping] mapped {len(mapping)}/{len(ensembl_ids)} genes")
    return mapping
