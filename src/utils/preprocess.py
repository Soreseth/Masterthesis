"""Preprocessing utilities used by `src.attacks.precompute_mia_scores`.

What lives here:
    - `TensorEncoder`           -- JSONEncoder that handles torch tensors / numpy
    - `safe_pre_encode_shots`   -- encode few-shot prefixes with target-aware truncation
    - `create_chunks`           -- chunk a doc at ctx ∈ {43, 512, 1024, 2048}
    - `mia_dataset`             -- load member/non-member splits from a Pile parquet dir
    - `compute_batch_scores`    -- batched conditional-likelihood scoring (used by topPref)
    - `topPref`                 -- search for an optimal prefix on a validation set
    - `build_freq_dist`         -- token-frequency dictionary used by DCPDD
    - `collect_calibration_signals` -- non-member calibration signals used by CAMIA

Removed (now lives in `src.utils.merge_jsonl`):
    - `merge_scores`, `run_post_merge_cleaning`, `flatten_and_clean_chunk`,
      `fix_chunk_keys`, `fix_jsonl_file`
"""
import json
import os
import pickle
import random
from collections import defaultdict

import datasets
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Project-wide imports (CAMIA + raw_values for calibration signals)
from src.attacks.target_scores import CAMIA, raw_values

MIN_CHARS = 100

# Reproducibility for the helpers that draw random samples.
seed = 42
random.seed(seed)
rng = np.random.default_rng(seed=seed)


class TensorEncoder(json.JSONEncoder):
    """JSON encoder that survives torch tensors and numpy scalars/arrays."""

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.number):
            return obj.item()
        return super().default(obj)


# -----------------------------------------------------------------------------
# Conditional-likelihood + prefix search (used by ReCaLL-style attacks)
# -----------------------------------------------------------------------------
def compute_batch_scores(tokenizer, model, device, target_texts,
                         prefix_text=None, batch_size=4):
    """Batched conditional log-likelihood scores for `[prefix, target]` pairs.

    Strict per-side truncation (max_length=512 each) so that long prefixes
    can't crowd out the target.  Returns negative mean per-token loss.
    """
    scores = []

    if prefix_text:
        prefix_tokens = tokenizer(
            prefix_text, return_tensors="pt", truncation=True,
            max_length=512, add_special_tokens=False,
        )
        prefix_ids = prefix_tokens.input_ids.to(device)
    else:
        prefix_ids = None

    for i in range(0, len(target_texts), batch_size):
        batch_targets = target_texts[i:i + batch_size]
        target_tokens = tokenizer(
            batch_targets, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
            add_special_tokens=False,
        ).to(device)

        target_ids = target_tokens.input_ids
        target_mask = target_tokens.attention_mask

        if prefix_ids is not None:
            current_batch_size = target_ids.shape[0]
            batch_prefix_ids = prefix_ids.expand(current_batch_size, -1)
            batch_prefix_mask = torch.ones_like(batch_prefix_ids)
            input_ids = torch.cat((batch_prefix_ids, target_ids), dim=1)
            attention_mask = torch.cat((batch_prefix_mask, target_mask), dim=1)
        else:
            input_ids = target_ids
            attention_mask = target_mask

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            shift_mask = attention_mask[..., 1:].contiguous()
            loss = loss * shift_mask
            seq_lens = shift_mask.sum(dim=1)
            batch_scores = -(loss.sum(dim=1) / torch.clamp(seq_lens, min=1))
            scores.extend(batch_scores.cpu().numpy())

    return np.array(scores)


def topPref(candidate_prefixes, validation_texts, validation_labels,
            tokenizer, model, device, num_prefix: int, batch_size=8):
    """Rank `candidate_prefixes` by ReCaLL-style validation AUROC.

    Returns the top-`num_prefix` (auroc, prefix) pairs sorted by AUROC desc.
    """
    print(f"Searching for optimal prefix from {len(candidate_prefixes)} candidates...")
    unconditional_scores = compute_batch_scores(
        tokenizer, model, device, validation_texts,
        prefix_text=None, batch_size=batch_size,
    )

    ranked = []
    for prefix in tqdm(candidate_prefixes, desc="Testing prefixes"):
        conditional_scores = compute_batch_scores(
            tokenizer, model, device, validation_texts,
            prefix_text=prefix, batch_size=batch_size,
        )
        recall_scores = conditional_scores / (unconditional_scores + 1e-10)
        try:
            auc = roc_auc_score(validation_labels, recall_scores)
        except ValueError:
            auc = 0.5
        ranked.append((auc, prefix))

    ranked.sort(key=lambda x: x[0], reverse=True)
    if ranked:
        best_score, best_p = ranked[0]
        print(f"WINNER PREFIX: '{best_p[:50]}...'  (AUROC={best_score:.4f})")
    return ranked[:num_prefix]


# -----------------------------------------------------------------------------
# Shot encoding + chunking
# -----------------------------------------------------------------------------
def safe_pre_encode_shots(text_list, tokenizer, max_shot_len: int,
                          reserve_for_target: int = 400, context_window: int = 2048):
    """Encode each shot to a tensor; truncate proportionally if their total
    length plus `reserve_for_target` would exceed `context_window`.

    Each shot is given a minimum of 50 tokens to remain useful.
    """
    if not text_list:
        return []

    encoded_shots = [
        tokenizer.encode(text, add_special_tokens=False, return_tensors="pt",
                         truncation=True, max_length=max_shot_len)
        for text in text_list
    ]
    total_prefix_length = sum(s.shape[1] for s in encoded_shots)
    available_for_prefix = context_window - reserve_for_target

    if total_prefix_length > available_for_prefix:
        ratio = available_for_prefix / total_prefix_length
        target_per_shot = max(int(available_for_prefix / len(text_list)), 50)
        encoded_shots = [
            tokenizer.encode(text, add_special_tokens=False, return_tensors="pt",
                             truncation=True, max_length=target_per_shot)
            for text in text_list
        ]
        new_total = sum(s.shape[1] for s in encoded_shots)
        print(f"  WARN Shots truncated: {total_prefix_length} -> {new_total} "
              f"tokens (ratio: {ratio:.2f})")

    return encoded_shots


def create_chunks(text, tokenizer, max_length):
    """Slice a doc into chunks. ctx=43 -> NLTK sentences; else token windows."""
    if max_length != 43:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return [
            tokenizer.decode(tokens[i:i + max_length], skip_special_tokens=True)
            for i in range(0, len(tokens), max_length)
        ]
    return sent_tokenize(text)


# -----------------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------------
def mia_dataset(dataset_path: str):
    """Load MIA member/non-member splits from a Pile parameterlab parquet dir.

    `train` -> members, `validation+test` -> non-members. Filters docs shorter
    than `MIN_CHARS` and balances counts by clipping members to len(non_members).
    """
    data = load_dataset("parquet", data_files={
        "train":      f"{dataset_path}/data/train-*.parquet",
        "validation": f"{dataset_path}/data/validation-*.parquet",
        "test":       f"{dataset_path}/data/test-*.parquet",
    })
    for split in ("train", "validation", "test"):
        data[split] = data[split].filter(lambda x: len(x["text"]) > MIN_CHARS)

    non_members = concatenate_datasets([data["validation"], data["test"]])
    doc_lengths = [len(x["text"]) for x in non_members]
    if not doc_lengths:
        min_len, max_len = 0, 0
    else:
        min_len, max_len = min(doc_lengths), max(doc_lengths)

    non_members = datasets.Dataset.from_list(list(non_members))
    members = (
        data["train"]
        .filter(lambda x: min_len <= len(x["text"]) <= max_len)
        .shuffle(seed=seed)
        .select(range(len(non_members)))
    )
    return members, non_members


# -----------------------------------------------------------------------------
# Frequency dictionary (DC-PDD) and CAMIA calibration signals
# -----------------------------------------------------------------------------
def build_freq_dist(save_path: str, dataset_path: str, base_tokenizer):
    """Walk allenai/c4-en (streamed) to build a token-frequency dict and
    pickle it to ``<save_path>/<TokenizerClass>_<corpus>_freq_dist.pkl``.

    Output is consumed by ``target_scores.DCPDD`` for non-member token-prior
    calibration.
    """
    data = load_dataset(
        "json",
        data_files={
            "train":      f"{dataset_path}/c4-train*.json.gz",
            "validation": f"{dataset_path}/c4-validation*.json.gz",
        },
        streaming=True,
    )
    freq_dist = [0] * len(base_tokenizer)

    for split_name in ("train", "validation"):
        for sample in data[split_name].iter(batch_size=100_000):
            outputs = base_tokenizer(sample["text"], max_length=2048,
                                     truncation=True)
            for input_ids in outputs["input_ids"]:
                for token_id in input_ids:
                    if token_id < len(freq_dist):
                        freq_dist[token_id] += 1
        print(f"Frequency Distribution: finished {split_name}")

    out = (f"{save_path}/{type(base_tokenizer).__name__}"
           f"_{os.path.basename(os.path.normpath(dataset_path))}_freq_dist.pkl")
    with open(out, "wb") as f:
        pickle.dump(freq_dist, f)
    print(f"Saved frequency distribution -> {out}")


def collect_calibration_signals(non_member_path: str, save_path: str,
                                token_level: int, model, tokenizer, device):
    """Pre-compute CAMIA's per-feature signals over a held-out non-member set.

    Output JSON is loaded by ``CAMIA(calibration_signal=...)`` at attack time
    so the per-feature thresholds reflect the target distribution rather than
    the test sample.
    """
    non_member_dataset = load_from_disk(dataset_path=non_member_path)
    num_signals = min(int(len(non_member_dataset["text"]) * 0.2), 1000)

    out_file = f"{save_path}/calibration_signals.json"
    if os.path.exists(out_file):
        print(f"File already exists at {out_file}")
        return num_signals

    calibration_data = defaultdict(list)
    non_member_dataset = non_member_dataset["text"][:num_signals]

    camia = CAMIA(target_model=model, target_tokenizer=tokenizer,
                  device=device, max_len=token_level, calibration_signal={})

    for text in tqdm(non_member_dataset, desc="Collecting calibration signals"):
        res = raw_values(text, model, tokenizer, device)
        raw_signals = camia.predict(
            input_ids=res["input_ids"].squeeze(0),
            token_log_probs=res["token_log_probs"],
            loss=res["loss"].item(),
            raw_signals=True,
        )
        for k, v in raw_signals.items():
            calibration_data[k].append(v)

    os.makedirs(save_path, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(calibration_data, f, cls=TensorEncoder)
    print(f"Saved {len(non_member_dataset)} calibration signals -> {out_file}")
    return num_signals
