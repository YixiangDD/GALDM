"""Fetches and caches real KEGG pathway membership through the KEGG REST API.

Fifteen classic cancer-related KEGG pathways are used (Methods 2.2). The full gene-symbol
membership of each is pulled from https://rest.kegg.jp/get/hsaXXXXX and cached to local
json, so pathway groupings are real rather than hash-based or hand-curated stubs.
"""
import os
import re
import json
import ssl
import time
import urllib.request
from typing import Dict, List

# The 15 cancer-related KEGG pathways used in the paper (PI3K-Akt, MAPK, p53, Wnt, TGF-beta, ...)
KEGG_15_PATHWAYS: Dict[str, str] = {
    "hsa05200": "Pathways in cancer",
    "hsa04151": "PI3K-Akt signaling",
    "hsa04010": "MAPK signaling",
    "hsa04110": "Cell cycle",
    "hsa04210": "Apoptosis",
    "hsa04310": "Wnt signaling",
    "hsa04350": "TGF-beta signaling",
    "hsa04150": "mTOR signaling",
    "hsa04630": "JAK-STAT signaling",
    "hsa04115": "p53 signaling",
    "hsa04370": "VEGF signaling",
    "hsa04066": "HIF-1 signaling",
    "hsa04068": "FoxO signaling",
    "hsa04012": "ErbB signaling",
    "hsa04330": "Notch signaling",
}

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "kegg_pathway_genes.json")
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def _parse_gene_section(text: str) -> List[str]:
    """Parses gene symbols out of the GENE section of a KEGG flat file.
    Lines look like: 'GENE  10000  AKT3; AKT serine/threonine kinase 3 [KO:...]'
    or continuations:  '      10018  BCL2L11; BCL2 like 11 [KO:...]'
    """
    symbols = []
    in_gene = False
    for line in text.split("\n"):
        if line.startswith("GENE"):
            in_gene = True
            content = line[len("GENE"):]
        elif in_gene and line.startswith(" "):
            content = line
        else:
            if in_gene:
                break
            continue
        # content: '   10000  AKT3; AKT serine/threonine kinase 3 ...'
        m = re.match(r"\s*\d+\s+([A-Za-z0-9\-]+);", content)
        if m:
            symbols.append(m.group(1))
    return symbols


def _fetch_pathway(hsa_id: str, retries: int = 3) -> List[str]:
    url = f"https://rest.kegg.jp/get/{hsa_id}"
    for attempt in range(retries):
        try:
            txt = urllib.request.urlopen(url, timeout=30, context=_CTX).read().decode()
            return _parse_gene_section(txt)
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [kegg] {hsa_id} fetch failed: {e}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def get_kegg_pathway_genes(verbose: bool = True) -> Dict[str, List[str]]:
    """Returns {hsa_id: [gene_symbols]}, backed by the local cache."""
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if all(k in cached and cached[k] for k in KEGG_15_PATHWAYS):
            return cached

    result = {}
    for hsa_id, name in KEGG_15_PATHWAYS.items():
        genes = _fetch_pathway(hsa_id)
        result[hsa_id] = genes
        if verbose:
            print(f"  [kegg] {hsa_id} ({name}): {len(genes)} genes")
        time.sleep(0.4)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result
