"""Fetch the third-party biological prior files into bio_priors/cache/.

Two of the four prior files are not redistributed in this repository because
their licences do not permit it. This script downloads them and verifies that
`bio_priors/grn_builder.py` can read them.

    python scripts/fetch_priors.py

  bio_priors/cache/
    kegg_pathway_genes.json   fetched here     (KEGG REST, 15 pathways)
    trrust_human.tsv          fetched here     (TRRUST v2, CC BY-SA 4.0)
    mirtarbase_se_wr.xlsx     manual download  (miRTarBase 9.0, academic use)
    ensembl2symbol.json       built on first run from mygene.info

None of these are redistributed with the repository. KEGG in particular forbids
redistribution of its data, so the pathway membership is fetched from the KEGG REST API at
setup time rather than shipped. Note that KEGG content changes over time, so a fetch made
today may differ slightly from the snapshot behind the published numbers; the cached
dataset in data_pipeline/processed/ is what reproduces those exactly.

If a download fails (firewall, moved URL), the printed URL can be fetched by
hand and dropped into bio_priors/cache/ under the exact filename shown.
"""
import os
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import paths as _P  # noqa: E402

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
MIRTARBASE_PAGE = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/php/download.php"


def _get(url: str, dest: str) -> bool:
    print(f"  GET {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=120, context=_CTX).read()
    except Exception as e:  # noqa: BLE001
        print(f"  !! failed: {type(e).__name__}: {e}")
        return False
    if len(data) < 1024:
        print(f"  !! suspiciously small response ({len(data)} B), not saving")
        return False
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  -> {dest}  ({len(data) / 1e6:.1f} MB)")
    return True


def main() -> int:
    os.makedirs(_P.CACHE_DIR, exist_ok=True)
    missing = []

    trrust = os.path.join(_P.CACHE_DIR, "trrust_human.tsv")
    if os.path.exists(trrust):
        print(f"[ok]   trrust_human.tsv already present")
    else:
        print("[get]  TRRUST v2 human TF-target network")
        if not _get(TRRUST_URL, trrust):
            missing.append(("trrust_human.tsv", TRRUST_URL))

    mtb = os.path.join(_P.CACHE_DIR, "mirtarbase_se_wr.xlsx")
    if os.path.exists(mtb):
        print(f"[ok]   mirtarbase_se_wr.xlsx already present")
    else:
        # miRTarBase requires accepting its terms on the download page, so the
        # file cannot be pulled non-interactively.
        print("[man]  miRTarBase 9.0 needs a manual download:")
        print(f"       1. open {MIRTARBASE_PAGE}")
        print("       2. pick species=Homo sapiens, support type=Functional MTI,")
        print("          validation=strong evidence (SE) + weak reporter (WR)")
        print(f"       3. save the .xlsx as {mtb}")
        missing.append(("mirtarbase_se_wr.xlsx", MIRTARBASE_PAGE))

    kegg = os.path.join(_P.CACHE_DIR, "kegg_pathway_genes.json")
    if os.path.exists(kegg):
        print("[ok]   kegg_pathway_genes.json already present")
    else:
        print("[get]  KEGG pathway membership (15 pathways, one REST call each)")
        try:
            from bio_priors.kegg_pathways import get_kegg_pathway_genes
            d = get_kegg_pathway_genes(verbose=True)
            n = sum(len(v) for v in d.values())
            print(f"       -> {len(d)} pathways, {n} gene symbols, cached to {kegg}")
        except Exception as e:  # noqa: BLE001
            print(f"  !! failed: {type(e).__name__}: {e}")
            missing.append(("kegg_pathway_genes.json", "https://rest.kegg.jp/get/hsaXXXXX"))

    print("\nensembl2symbol.json is built automatically on the first preprocessing "
          "run (bio_priors/gene_mapping.py queries mygene.info and caches results).")

    if missing:
        print("\nStill missing:")
        for name, url in missing:
            print(f"  {name}  <-  {url}")
        return 1

    # smoke-test that the files parse
    try:
        from bio_priors.grn_builder import _load_trrust, _load_mirtarbase
        print(f"\nparsed TRRUST rows      : {len(_load_trrust())}")
        print(f"parsed miRTarBase rows  : {len(_load_mirtarbase())}")
    except Exception as e:  # noqa: BLE001
        print(f"\n!! files present but failed to parse: {type(e).__name__}: {e}")
        return 1
    print("\nAll priors ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
