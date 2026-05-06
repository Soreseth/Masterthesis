#!/usr/bin/env python3
"""
Download Pythia models, Pile MIA-evaluation datasets (parameterlab), and
optionally MIMIR splits, then create the
``$MIA_ROOT/<subset>/chunks/ctx<X>.jsonl`` skeleton expected by
``src.attacks.precompute_mia_scores``.

Usage:
    # Download everything (target + reference Pythia models, all 8 Pile subsets,
    # chunked at all 4 ctx values).
    python -m src.utils.download

    # Just one subset and one ctx.
    python -m src.utils.download --datasets arxiv --ctx 1024

    # Reference Pythia checkpoints only (used by reference_scores.py).
    python -m src.utils.download --models pythia-70m pythia-160m pythia-410m pythia-1b

    # MIMIR n-gram-overlap splits (used in `paper/tables/mimir/...`).
    python -m src.utils.download --mimir
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DATASETS = ["arxiv", "FreeLaw", "Github", "HackerNews", "OpenWebText2",
                    "Pile-CC", "USPTO_Backgrounds", "wiki"]
DEFAULT_CTXS = [43, 512, 1024, 2048]

# Target models attacked in the thesis.
TARGET_MODELS = ["EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b"]

# Reference checkpoints used by InfoRMIA + WBC.
REFERENCE_MODELS = [
    f"EleutherAI/pythia-{size}{suffix}"
    for size in ["70m", "160m", "410m", "1b"]
    for suffix in ["", "-step1", "-final"]   # plus intermediate steps; expand as needed
]

PILE_DATASETS_HF = {
    ds: f"iliemihai/scaling_mia_the_pile_00_{ds}" for ds in DEFAULT_DATASETS
}

MIN_CHARS = 100


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.cwd() / "hf_cache"))


def _mia_root() -> Path:
    return Path(os.environ.get("MIA_ROOT", Path.cwd() / "mia_scores"))


def download_model(name: str) -> None:
    """Pull a HF model + tokenizer into HF_HOME (idempotent)."""
    print(f"  -> {name}")
    AutoTokenizer.from_pretrained(name, cache_dir=str(_hf_home()))
    AutoModelForCausalLM.from_pretrained(name, cache_dir=str(_hf_home()))


def chunk_text(text: str, tokenizer, max_length: int) -> list[str]:
    """Slice a document into ctx-sized chunks; ctx=43 uses NLTK sentences."""
    if max_length == 43:
        return sent_tokenize(text)
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return [
        tokenizer.decode(tokens[i:i + max_length], skip_special_tokens=True)
        for i in range(0, len(tokens), max_length)
    ]


def write_chunks_for_subset(subset: str, ctxs: list[int], tokenizer) -> None:
    """Materialise chunks/ctx<X>.jsonl files for one Pile subset."""
    hf_name = PILE_DATASETS_HF[subset]
    print(f"\n[{subset}]")
    ds = load_dataset(hf_name, cache_dir=str(_hf_home()))
    members = ds["validation"].filter(lambda x: len(x["text"]) > MIN_CHARS)
    non_members = ds["test"].filter(lambda x: len(x["text"]) > MIN_CHARS)
    print(f"  loaded: {len(members)} members, {len(non_members)} non-members")

    out_dir = _mia_root() / subset / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ctx in ctxs:
        out = out_dir / f"ctx{ctx}.jsonl"
        if out.exists():
            print(f"  skip {out.name} (exists)")
            continue
        with open(out, "w") as f:
            for label, source in [(1, members), (0, non_members)]:
                for i, ex in enumerate(source):
                    chunks = chunk_text(ex["text"], tokenizer, ctx)
                    f.write(json.dumps({
                        "id": f"{'mem' if label else 'non'}_{i}",
                        "label": label,
                        "chunks": chunks,
                    }) + "\n")
        print(f"  wrote {out.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--ctxs", type=int, nargs="+", default=DEFAULT_CTXS,
                   help="Chunk sizes to materialise (default: 43 512 1024 2048)")
    p.add_argument("--models", nargs="+", default=None,
                   help="HF model IDs to download (default: target + reference set)")
    p.add_argument("--skip-models", action="store_true",
                   help="Don't download any models -- datasets/chunks only")
    p.add_argument("--skip-data", action="store_true",
                   help="Don't materialise the chunks/ files -- models only")
    args = p.parse_args()

    print(f"HF_HOME = {_hf_home()}")
    print(f"MIA_ROOT = {_mia_root()}")

    if not args.skip_models:
        models = args.models or (TARGET_MODELS + REFERENCE_MODELS)
        print(f"\nDownloading {len(models)} models...")
        for m in models:
            try:
                download_model(m)
            except Exception as e:
                print(f"  ! {m}: {e}", file=sys.stderr)

    if not args.skip_data:
        print(f"\nMaterialising chunks for {len(args.datasets)} subsets...")
        # Use the smallest target model's tokenizer (Pythia tokenizers are shared).
        tok = AutoTokenizer.from_pretrained(TARGET_MODELS[0], cache_dir=str(_hf_home()))
        for subset in args.datasets:
            write_chunks_for_subset(subset, args.ctxs, tok)

    print("\nDone.")


if __name__ == "__main__":
    main()
