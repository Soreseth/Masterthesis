"""
download.py — Download all datasets and models needed for the MIA pipeline.

Downloads:
  1. Parameterlab Pile datasets (20 subsets) from HuggingFace
  2. Pythia target models (pythia-2.8b + pythia-6.9b final checkpoints)
  3. Pythia reference models (70m, 160m, 410m, 1b × 8 checkpoints each = 32 models)
  4. DuoLearn reference model (pythia-1b, included in ref models above)

File structure matches what defense.py and precompute_mia_scores.py expect:
  {HF_DIR}/models/EleutherAI__pythia-2.8b/
  {HF_DIR}/models/EleutherAI__pythia-70m_step1000/
  {HF_DIR}/datasets/parameterlab/scaling_mia_the_pile_00_Pile-CC/data/

Usage:
    python download.py                    # download everything
    python download.py --models-only      # download only models
    python download.py --datasets-only    # download only datasets
    python download.py --target-only      # download only target models (2.8b, 6.9b)
    python download.py --ref-only         # download only reference models
    python download.py --hf-dir /path     # custom HuggingFace directory
"""

import os
import argparse

# Disable xet download backend (causes segfaults in some huggingface_hub versions)
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download


# ════════════════════════════════════════════════════════════════════
# Configuration — must match defense.py and precompute_mia_scores.py
# ════════════════════════════════════════════════════════════════════

DEFAULT_HF_DIR = os.path.expanduser("~/hf_home")

# Target models for MIA scoring
TARGET_MODELS = [
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
]

# Reference models: 4 sizes × 8 checkpoints each (= 32 models)
# Used by precompute_mia_scores.py for RefLossDiff, TokenLevelInfoRMIA, WBC
REF_MODEL_SIZES = ["70m", "160m", "410m", "1b"]
REF_STEPS = [1, 512, 1000, 3000, 5000, 10000, 100000, None]  # None = final checkpoint

# Parameterlab Pile datasets (20 subsets)
# https://huggingface.co/collections/parameterlab/scaling-mia-data-and-results
PILE_DATASETS = [
    "Pile-CC", "arxiv", "DM_Mathematics", "Enron_Emails", "EuroParl",
    "FreeLaw", "Github", "Gutenberg_PG-19", "HackerNews", "NIH_ExPorter",
    "OpenSubtitles", "OpenWebText2", "PhilPapers", "PubMed_Abstracts",
    "PubMed_Central", "StackExchange", "Ubuntu_IRC", "USPTO_Backgrounds",
    "wiki", "YoutubeSubtitles",
]


# Download functions

def download_model(repo_id: str, hf_dir: str, revision: str = "main"):
    """Download a single HuggingFace model.

    Args:
        repo_id: e.g. 'EleutherAI/pythia-2.8b'
        hf_dir: Base HF directory
        revision: Git revision (branch, tag, or commit)
    """
    # Convert repo_id to directory name matching existing structure
    # e.g. "EleutherAI/pythia-70m" with revision "step1000"
    #   → "EleutherAI__pythia-70m_step1000"
    dir_name = repo_id.replace("/", "__")
    if revision != "main":
        dir_name += f"_{revision}"

    local_dir = f"{hf_dir}/models/{dir_name}"

    if os.path.exists(local_dir) and any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(local_dir)
    ):
        print(f"  [skip] {dir_name} (already exists)")
        return

    print(f"  [downloading] {dir_name} ...")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    if revision != "main":
        for wrong_file in ["model.safetensors", "pytorch_model.bin"]:
            wrong_path = os.path.join(local_dir, wrong_file)
            sharded = any(
                f.startswith("model-") and f.endswith(".safetensors")
                for f in os.listdir(local_dir)
            )
            if os.path.exists(wrong_path) and sharded:
                os.remove(wrong_path)
                print(f"    [fix] Removed {wrong_file} (wrong main-branch weights)")

    print(f"  [done] {dir_name}")


def download_target_models(hf_dir: str):
    """Download target Pythia models (final checkpoints)."""
    print("\n=== Target Models ===")
    for repo_id in TARGET_MODELS:
        download_model(repo_id, hf_dir)


def download_ref_models(hf_dir: str):
    """Download all 32 reference models (4 sizes × 8 checkpoints)."""
    print("\n=== Reference Models ===")
    for size in REF_MODEL_SIZES:
        repo_id = f"EleutherAI/pythia-{size}"
        for step in REF_STEPS:
            if step is None:
                download_model(repo_id, hf_dir, revision="main")
            else:
                download_model(repo_id, hf_dir, revision=f"step{step}")


def download_dataset(dataset_name: str, hf_dir: str):
    """Download a single parameterlab Pile dataset.

    Args:
        dataset_name: e.g. 'Pile-CC', 'arxiv'
        hf_dir: Base HF directory
    """
    repo_id = f"parameterlab/scaling_mia_the_pile_00_{dataset_name}"
    local_dir = f"{hf_dir}/datasets/parameterlab/scaling_mia_the_pile_00_{dataset_name}"

    if os.path.exists(f"{local_dir}/data"):
        parquets = [f for f in os.listdir(f"{local_dir}/data") if f.endswith(".parquet")]
        if parquets:
            print(f"  [skip] {dataset_name} ({len(parquets)} parquet files)")
            return

    print(f"  [downloading] {dataset_name} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"  [done] {dataset_name}")


def download_datasets(hf_dir: str):
    """Download all 20 parameterlab Pile datasets."""
    print("\n=== Parameterlab Pile Datasets ===")
    for name in PILE_DATASETS:
        download_dataset(name, hf_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Download all datasets and models for the MIA pipeline"
    )
    parser.add_argument("--hf-dir", type=str, default=DEFAULT_HF_DIR,
                        help=f"HuggingFace home directory (default: {DEFAULT_HF_DIR})")
    parser.add_argument("--models-only", action="store_true",
                        help="Download only models (target + reference)")
    parser.add_argument("--datasets-only", action="store_true",
                        help="Download only datasets")
    parser.add_argument("--target-only", action="store_true",
                        help="Download only target models")
    parser.add_argument("--ref-only", action="store_true",
                        help="Download only reference models")
    args = parser.parse_args()

    hf_dir = os.path.abspath(args.hf_dir)
    os.makedirs(f"{hf_dir}/models", exist_ok=True)
    os.makedirs(f"{hf_dir}/datasets/parameterlab", exist_ok=True)

    print(f"HF directory: {hf_dir}")

    if args.target_only:
        download_target_models(hf_dir)
    elif args.ref_only:
        download_ref_models(hf_dir)
    elif args.models_only:
        download_target_models(hf_dir)
        download_ref_models(hf_dir)
    elif args.datasets_only:
        download_datasets(hf_dir)
    else:
        download_target_models(hf_dir)
        download_ref_models(hf_dir)
        download_datasets(hf_dir)

    print("\nExpected structure:")
    print(f"  {hf_dir}/models/EleutherAI__pythia-2.8b/")
    print(f"  {hf_dir}/models/EleutherAI__pythia-70m_step1000/")
    print(f"  {hf_dir}/datasets/parameterlab/scaling_mia_the_pile_00_Pile-CC/data/")


if __name__ == "__main__":
    main()
