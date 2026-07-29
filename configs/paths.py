"""Repo-relative path resolution.

Every script in this repository imports its directories from here instead of
hard-coding absolute paths, so the code runs unchanged on any machine.

Defaults (all relative to the repository root):
    RAW_ROOT      ./data                          raw TCGA files (not shipped)
    PROCESSED_DIR ./data_pipeline/processed       preprocessing cache (.npz/.pkl)
    RESULTS_DIR   ./results                       metric JSON / xlsx
    FIGURES_DIR   ./results/figures
    CACHE_DIR     ./bio_priors/cache              KEGG / Ensembl / GRN caches
    CKPT_DIR      ./checkpoints

Override any of them with environment variables of the same name, e.g.

    export GALDM_RAW_ROOT=/mnt/tcga
    export GALDM_RESULTS_DIR=/scratch/runs
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default: str) -> str:
    return os.environ.get("GALDM_" + name, default)


RAW_ROOT = _env("RAW_ROOT", os.path.join(REPO_ROOT, "data"))
PROCESSED_DIR = _env("PROCESSED_DIR", os.path.join(REPO_ROOT, "data_pipeline", "processed"))
RESULTS_DIR = _env("RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
FIGURES_DIR = _env("FIGURES_DIR", os.path.join(RESULTS_DIR, "figures"))
CACHE_DIR = _env("CACHE_DIR", os.path.join(REPO_ROOT, "bio_priors", "cache"))
CKPT_DIR = _env("CKPT_DIR", os.path.join(REPO_ROOT, "checkpoints"))


def ensure_dirs() -> None:
    """Create the writable output directories if they do not exist."""
    for d in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)


__all__ = [
    "REPO_ROOT", "RAW_ROOT", "PROCESSED_DIR", "RESULTS_DIR",
    "FIGURES_DIR", "CACHE_DIR", "CKPT_DIR", "ensure_dirs",
]
