#!/usr/bin/env python3
"""
Compute per-paragraph MIA scores against an undefended Pythia target OR against
a defended (DuoLearn / DP-LoRA) variant. Writes one shard per (start, end)
range so multiple ranges can run in parallel.

Pipeline (per text):
    Pass 1 -- target-only attacks (NoisyNeighbour, CAMIA, Min-K, ...) and cache
             the per-token logits/labels for Pass 2.
    Pass 2 -- for every reference checkpoint in REF_CONFIGS, run RefLossDiff,
             TokenLevelInfoRMIA (multi-temperature × multi-aggregation), and
             WBC (per-window).

Output layout (keyed off `MIA_ROOT`):
    Undefended:
        $MIA_ROOT/<subset>/undefended/<pythia_model>/ctx<CTX>/
            mia_{members,nonmembers}_<S>_<E>.jsonl
    Defended:
        $MIA_ROOT/<subset>/defended/<defense_name>/ctx<CTX>/
            mia_{members,nonmembers}_<S>_<E>.jsonl

Usage:
    # undefended Pythia-2.8B on arxiv at ctx=1024, members 0..1000
    python -m src.attacks.precompute_mia_scores \\
        --pythia_model pythia-2.8b --dataset arxiv \\
        --max_length 1024 --miaset member --range 0 1000 \\
        --member_shot_indices 0,1,2,3,4,5,6 \\
        --nonmember_shot_indices 7,8,9,10,11,12,13

    # defended (DuoLearn alpha=0.2) on the same cell
    python -m src.attacks.precompute_mia_scores \\
        --defended duolearn_a0.2 \\
        --pythia_model pythia-2.8b --dataset arxiv \\
        --max_length 1024 --miaset member --range 0 1000 \\
        --member_shot_indices 0,1,2,3,4,5,6 \\
        --nonmember_shot_indices 7,8,9,10,11,12,13
"""
import argparse
import gc
import json
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import spacy
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.attacks.reference_scores import (RefLossDiff, TokenLevelInfoRMIA, WBC)
from src.attacks.target_scores import (CAMIA, DCPDD, NoisyNeighbour,
                                       RelativeLikelihood, TagTab, inference)
from src.utils.preprocess import (TensorEncoder, create_chunks,
                                  safe_pre_encode_shots)

# -----------------------------------------------------------------------------
# Paths driven by environment variables -- edit your shell rc, not this file.
# -----------------------------------------------------------------------------
MIA_ROOT = Path(os.environ.get("MIA_ROOT", "./mia_scores"))
DEFENDED_MODELS_ROOT = Path(os.environ.get("DEFENDED_MODELS_ROOT", "./defended_models"))
HF_HOME = Path(os.environ.get("HF_HOME", "./hf_cache"))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_MAX_LENGTH = 2048
MIN_CHARS = 100
DC_PDD_FREQ_DICT_PATH = Path(os.environ.get(
    "DCPDD_FREQ_DICT",
    HF_HOME / "GPTNeoXTokenizerFast_realnewslike_freq_dist.pkl",
))

# Reference checkpoints used by RefLossDiff / TokenLevelInfoRMIA / WBC.
# Each tuple: (model_size, step_suffix, hf_dir_name).
# Drop / extend rows here to change the reference set; nothing else needs to
# move.  By default the slurm passes --ref_steps step1 final to keep ~8 of
# these and skip the intermediate checkpoints.
REF_CONFIGS = [
    (size, step, f"EleutherAI__pythia-{size}{('_' + step) if step != 'final' else ''}")
    for size in ("70m", "160m", "410m", "1b")
    for step in ("step1", "step512", "step1000", "step3000",
                 "step5000", "step10000", "step100000", "final")
]

# WBC window sizes tracked individually (others can be added).
WBC_WINDOW_SIZES = [1, 2, 4, 10, 20, 40]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def _nn_batch_size(max_length: int) -> int:
    """NoisyNeighbour batch size -- must divide max_neighbours (=10)."""
    base = {43: 10, 512: 5, 1024: 2, 2048: 1}.get(max_length, 1)
    while 10 % base != 0:
        base -= 1
        if base < 1:
            return 1
    return base


def _load_local_model(local_dir: Path, device, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir), local_files_only=True, return_dict=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_target_model(pythia_model: str, defended: str | None, device):
    """Load the target model.  If `defended` is set, attach the LoRA adapter
    (or full-FT checkpoint) at $DEFENDED_MODELS_ROOT/<defended>/<dataset>/final.

    Returns ``(model, tokenizer)``.
    """
    base_dir = HF_HOME / "models" / f"EleutherAI__{pythia_model}"
    if defended is None:
        return _load_local_model(base_dir, device)

    # Defended branch: load base, then merge adapter if present.
    base_model, tokenizer = _load_local_model(base_dir, device)
    adapter_dir = DEFENDED_MODELS_ROOT / defended
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Defended adapter dir not found: {adapter_dir}\n"
            f"Train it first via `python -m src.defended.<duolearn|dpsgd> --dataset <ds> ...`")

    # The defense scripts save under DEFENDED_MODELS_ROOT/<name>/<dataset>/final
    # -- the user passes the dataset along with --defended via the slurm.
    # Keep it simple here: assume `defended_models/<defended_name>/...` is a
    # PEFT-loadable adapter or a full-FT checkpoint dir.
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(device)
    except ImportError:
        # No PEFT -> fall back to plain checkpoint load (full FT).
        model = AutoModelForCausalLM.from_pretrained(
            str(adapter_dir), local_files_only=True, torch_dtype=torch.float16,
        ).to(device)
    model.eval()
    return model, tokenizer


def load_ref_model(dir_name: str, device):
    """Load a reference checkpoint by directory name (under HF_HOME/models)."""
    path = HF_HOME / "models" / dir_name
    if not path.exists():
        print(f"  WARNING: {path} not found -- skipping reference {dir_name}")
        return None, None
    return _load_local_model(path, device)


# -----------------------------------------------------------------------------
# Output / input path resolvers
# -----------------------------------------------------------------------------
def output_dir(subset: str, pythia_model: str, defended: str | None,
               max_length: int) -> Path:
    if defended is None:
        return MIA_ROOT / subset / "undefended" / pythia_model / f"ctx{max_length}"
    return MIA_ROOT / subset / "defended" / defended / f"ctx{max_length}"


def raw_dataset_dir(subset: str) -> Path:
    """HF dataset directory written by `src.utils.download` containing the
    `members/` and `non_members/` raw-text splits."""
    return MIA_ROOT / subset / "raw"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run(pythia_model: str, defended: str | None, max_length: int,
        miaset: str, dataset: str, dataset_range: list[int],
        member_shot_indices: list[int], nonmember_shot_indices: list[int]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== {dataset} | {pythia_model}{' / ' + defended if defended else ''} | "
          f"ctx={max_length} | {miaset} {dataset_range}")
    model, tokenizer = load_target_model(pythia_model, defended, device)

    raw_dir = raw_dataset_dir(dataset)
    if not (raw_dir / "members").exists() or not (raw_dir / "non_members").exists():
        print(f"ERROR: raw dataset missing under {raw_dir} -- run "
              f"`python -m src.utils.download --datasets {dataset}` first.")
        return

    members = load_from_disk(str(raw_dir / "members"))
    non_members = load_from_disk(str(raw_dir / "non_members"))

    out_dir = output_dir(dataset, pythia_model, defended, max_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    if miaset == "member":
        path = out_dir / f"mia_members_{dataset_range[0]}_{dataset_range[1]}.jsonl"
        label_val = 1
        data_source = members
        idx_range = range(dataset_range[0], dataset_range[1])
        excluded = set(member_shot_indices)
        desc = "Members"
    else:
        end = min(dataset_range[1], len(non_members["text"]))
        path = out_dir / f"mia_nonmembers_{dataset_range[0]}_{end}.jsonl"
        label_val = 0
        data_source = non_members
        idx_range = range(dataset_range[0], end)
        excluded = set(nonmember_shot_indices)
        desc = "Non-members"

    # -- Attacks ------------------------------------------------------------
    nn_attack = NoisyNeighbour(model=model, device=device,
                               batch_size=_nn_batch_size(max_length))
    rel_attacks = RelativeLikelihood(base_model=model, base_tokenizer=tokenizer,
                                     device=device)
    with open(DC_PDD_FREQ_DICT_PATH, "rb") as f:
        freq_dict = np.array(pickle.load(f), dtype=np.float32)
    dcpdd = DCPDD(freq_dict, device=device, a=0.01, apply_smoothing=True)
    tag_tab = TagTab(target_model=model, target_tokenizer=tokenizer,
                     top_k=10, nlp=spacy.load("en_core_web_sm"), device=device,
                     entropy_map=None,
                     min_sentence_len=3 if max_length == 43 else 7,
                     max_sentence_len=40)
    camia_attack = CAMIA(target_model=model, target_tokenizer=tokenizer,
                         device=device, max_len=max_length, calibration_signal={})

    # -- Few-shot prefixes (member + non-member) ----------------------------
    sample_size = min(100, len(members), len(non_members["text"]))
    sample_texts = (list(members["text"][:sample_size]) +
                    list(non_members["text"][:sample_size]))
    target_reserve = int(np.percentile(
        [len(tokenizer.encode(t, add_special_tokens=True)) for t in sample_texts], 95))

    members_shots = members.select(member_shot_indices)["text"]
    global_member_prefix = safe_pre_encode_shots(
        text_list=members_shots, tokenizer=tokenizer,
        max_shot_len=min(max_length - 1, 1023), reserve_for_target=target_reserve)
    non_members_shots = [non_members["text"][i] for i in nonmember_shot_indices]
    global_non_member_prefix = safe_pre_encode_shots(
        text_list=non_members_shots, tokenizer=tokenizer,
        max_shot_len=min(max_length - 1, 1023), reserve_for_target=target_reserve)

    # -- Mini-batch loop: Pass 1 -> Pass 2 -> save ----------------------------
    MINI_BATCH_DOCS = 50
    valid_indices = [i for i in idx_range if i not in excluded]

    total_chunks = total_docs = 0
    for mb_start in range(0, len(valid_indices), MINI_BATCH_DOCS):
        mb_indices = valid_indices[mb_start:mb_start + MINI_BATCH_DOCS]
        mb_num = mb_start // MINI_BATCH_DOCS + 1
        mb_total = (len(valid_indices) + MINI_BATCH_DOCS - 1) // MINI_BATCH_DOCS

        chunk_texts, chunk_results, chunk_doc_ids, target_cache = [], [], [], []
        for idx in tqdm(mb_indices, desc=f"{desc} batch {mb_num}/{mb_total}"):
            text = data_source["text"][idx]
            for chunk in create_chunks(text, tokenizer, max_length):
                if len(chunk) <= 25:
                    continue
                res, raw_data = inference(
                    text=chunk, model=model, tokenizer=tokenizer,
                    negative_prefix=global_non_member_prefix,
                    member_prefix=global_member_prefix,
                    non_member_prefix=global_non_member_prefix,
                    device=device, rel_attacks=rel_attacks, dcpdd=dcpdd,
                    noisy_attack=nn_attack, tagtab_attack=tag_tab,
                    camia_attack=camia_attack)
                if res is None:
                    continue
                chunk_texts.append(chunk)
                chunk_results.append(res)
                chunk_doc_ids.append(idx)
                target_cache.append({
                    "loss": raw_data["loss"].item(),
                    "logits": raw_data["logits"].cpu(),
                    "labels": raw_data["input_ids"].cpu(),
                    "per_token_losses": torch.nn.functional.cross_entropy(
                        raw_data["logits"].squeeze(0),
                        raw_data["input_ids"].squeeze(0),
                        reduction="none",
                    ).cpu().numpy(),
                })
        if not chunk_texts:
            continue
        print(f"  Batch {mb_num}: Pass 1 done -- {len(chunk_texts)} chunks "
              f"from {len(set(chunk_doc_ids))} docs")

        # -- Pass 2: reference-model attacks ---------------------------------
        for size, step, dir_name in tqdm(REF_CONFIGS, desc=f"Ref models (b{mb_num})"):
            ref_m, ref_t = load_ref_model(dir_name, device)
            if ref_m is None:
                continue
            label = f"{size}_{step}"

            rld = RefLossDiff(target_model=model, target_tokenizer=tokenizer,
                              ref_model=ref_m, ref_tokenizer=ref_t, device=device)
            tl_rmia = TokenLevelInfoRMIA(
                target_model=model, target_tokenizer=tokenizer,
                reference_models=[ref_m], reference_tokenizers=[ref_t],
                temperature=2.0, aggregation="mean", device=device)
            wbc = WBC(target_model=model, target_tokenizer=tokenizer,
                      ref_model=ref_m, ref_tokenizer=ref_t, device=device,
                      window_sizes=WBC_WINDOW_SIZES)

            for i, chunk in enumerate(chunk_texts):
                res, tc = chunk_results[i], target_cache[i]
                tc_logits = tc["logits"].to(device)
                tc_labels = tc["labels"].to(device)

                try:
                    s = rld.predict(chunk, target_loss=tc["loss"])
                    res[f"ref_loss_diff_{label}"] = s if not np.isnan(s) else 0.0
                except Exception:
                    res[f"ref_loss_diff_{label}"] = 0.0

                try:
                    res.update(tl_rmia.predict_multi(
                        chunk,
                        temperatures=[0.5, 1.0, 2.0, 5.0],
                        aggregations=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
                        ref_labels=[label],
                        target_logits=tc_logits, target_labels=tc_labels))
                except Exception as e:
                    print(f"TokenLevelInfoRMIA {label} failed: {e}")

                try:
                    res.update(wbc.predict_per_window(
                        chunk, label=label, target_losses=tc["per_token_losses"]))
                except Exception as e:
                    print(f"WBC {label} failed: {e}")

                del tc_logits, tc_labels

            del rld, tl_rmia, wbc, ref_m, ref_t
            torch.cuda.empty_cache()

        print(f"  Batch {mb_num}: Pass 2 done -- {len(REF_CONFIGS)} ref configs")

        # -- Append to shard file --------------------------------------------
        collection = {}
        with open(path, "a") as f:
            for i, res in enumerate(chunk_results):
                doc_id = f"Document_{chunk_doc_ids[i]}"
                f.write(json.dumps({"pred": res, "label": label_val},
                                   cls=TensorEncoder) + "\n")
                collection.setdefault(doc_id, []).append(res)

        # If you also want a doc-level rollup, write under .../document/<...> here.
        total_chunks += len(chunk_texts)
        total_docs += len(set(chunk_doc_ids))
        del chunk_texts, chunk_results, chunk_doc_ids, target_cache, collection
        gc.collect()

    del nn_attack
    torch.cuda.empty_cache()
    print(f"\nDone: {total_chunks} chunks from {total_docs} documents -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pythia_model", default="pythia-2.8b",
                        choices=["pythia-2.8b", "pythia-6.9b"],
                        help="Base target model (also the model the LoRA "
                             "adapter is attached to when --defended is set)")
    parser.add_argument("--defended", default=None,
                        help="Defense name (e.g. duolearn_a0.2 or dplora_eps1.0). "
                             "Looked up under $DEFENDED_MODELS_ROOT/<name>/. "
                             "Omit for undefended runs.")
    parser.add_argument("--miaset", default="member",
                        choices=["member", "nonmember"])
    parser.add_argument("--max_length", type=int, default=512,
                        choices=[43, 512, 1024, 2048])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--range", nargs=2, type=int, default=[0, 0],
                        help="[start, end] doc-index range to score")
    parser.add_argument("--member_shot_indices", required=True,
                        help="Comma-separated member doc indices used as few-shot prefix")
    parser.add_argument("--nonmember_shot_indices", required=True,
                        help="Comma-separated non-member doc indices used as few-shot prefix")
    parser.add_argument("--ref_steps", nargs="+", default=None,
                        help="Restrict reference checkpoints to these step suffixes "
                             "(e.g. --ref_steps step1 final).  Default: all 32 configs.")
    args = parser.parse_args()

    if args.ref_steps:
        kept = [c for c in REF_CONFIGS if c[1] in args.ref_steps]
        REF_CONFIGS[:] = kept
        print(f"[ref_steps={args.ref_steps}] kept {len(REF_CONFIGS)} reference configs")

    try:
        run(
            pythia_model=args.pythia_model,
            defended=args.defended,
            max_length=args.max_length,
            miaset=args.miaset,
            dataset=args.dataset,
            dataset_range=args.range,
            member_shot_indices=[int(x) for x in args.member_shot_indices.split(",")],
            nonmember_shot_indices=[int(x) for x in args.nonmember_shot_indices.split(",")],
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
