#!/usr/bin/env python3
"""
Precompute per-paragraph MIA scores using Puerto's original method.

Uses nn.Linear + Adam (Puerto's original pipeline) with Puerto's 10 features.
For each seed:
1. Split data using split_documents_puerto (train=1000, known=1000)
2. Train nn.Linear with Adam (100 epochs, outlier removal)
3. Score every paragraph in eval members + eval non-members
4. Save scores + metadata for downstream aggregation


Example Usage:
    python precompute_paragraph_scores_puerto.py --dataset arxiv --seed 670487
    python precompute_paragraph_scores_puerto.py --dataset arxiv  # all 5 seeds
"""
import sys, os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.aggregate import load_jsonl_documents, split_documents_puerto

ALL_SEEDS = [670487, 116739, 26225, 777572, 288389]
CTX = 1024
N_TRAIN = int(os.environ.get("N_TRAIN", 1000))
N_KNOWN = 1000
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 0.01

_MIA_ROOT = os.environ.get("MIA_ROOT", "./mia_scores")
MIA_DIR = os.environ.get("MIA_DIR", os.path.join(_MIA_ROOT, "pythia-2.8b"))
OUT_DIR = os.environ.get("ALL_CHUNK_DIR", os.path.join(_MIA_ROOT, "results", "all_chunk"))


def extract_puerto_features(paragraphs):
    """Extract Puerto's 10 features from paragraph dicts."""
    X, y = [], []
    for para in paragraphs:
        pred = para["pred"]
        ppl = pred.get("ppl", 0)
        lowercase_ppl = pred.get("lowercase_ppl", 1)
        zlib = pred.get("zlib", 1)

        feats = [
            ppl,
            ppl / lowercase_ppl if lowercase_ppl != 0 else 0,
            ppl / zlib if zlib != 0 else 0,
            pred.get("min_k_5", 0),
            pred.get("min_k_10", 0),
            pred.get("min_k_20", 0),
            pred.get("min_k_30", 0),
            pred.get("min_k_40", 0),
            pred.get("min_k_50", 0),
            pred.get("min_k_60", 0),
        ]
        X.append(feats)
        y.append(para["label"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def train_linear_model(X_train, y_train):
    """Train nn.Linear with Adam (Puerto's original method, see https://github.com/parameterlab/mia-scaling/tree/main) with outlier removal."""
    # Remove top/bottom 2.5%
    sort_idx = np.argsort(X_train[:, 0])
    n = len(sort_idx)
    keep = sort_idx[int(0.025 * n):int(0.975 * n)]
    X_train = X_train[keep]
    y_train = y_train[keep]

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(X_train.shape[1], 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(NUM_EPOCHS):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, device

def get_scores(model, X, device):
    """Get raw model scores."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_t).squeeze(-1).cpu().numpy().tolist()

def run_seed(dataset, seed, members, non_members):
    print(f"\n  --- Seed {seed} ---")

    # Set seeds
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = {
        "n_train_docs": N_TRAIN,
        "n_known_docs": N_KNOWN,
        # Fix the training pool at max(N_TRAIN, 1000) so smaller train
        # sizes get subsampled FROM the same 1000-doc pool used at the
        # largest train size. Without this, every (N_TRAIN) PCS file
        # uses a different doc slice and a different eval boundary.
        "n_train_docs_sweep": [max(N_TRAIN, 1000)],
        "n_known_docs_sweep": [N_KNOWN],
    }

    splits = split_documents_puerto(members, non_members, config, seed)

    # Extract training features
    A_para = splits["A_members_para"] + splits["A_non_members_para"]
    y_A = np.array([1]*len(splits["A_members_para"]) + [0]*len(splits["A_non_members_para"]))

    X_A, _ = extract_puerto_features(A_para)

    # Train nn.Linear (Puerto's method)
    model, device = train_linear_model(X_A, y_A)
    print(f"    Trained nn.Linear (Puerto)")

    # Score known non-members
    known_para = splits["known_non_members_para"]
    X_k, _ = extract_puerto_features(known_para)
    known_scores = get_scores(model, X_k, device)
    print(f"    Known scores: {len(known_para)} paragraphs")

    # Score all eval documents
    def score_docs(docs, label):
        results = []
        for doc_idx, doc in enumerate(docs):
            if len(doc) == 0:
                continue
            X_d, _ = extract_puerto_features(doc)
            doc_scores = get_scores(model, X_d, device)
            results.append({
                "doc_id": doc_idx,
                "label": label,
                "n_paragraphs": len(doc),
                "scores": {"PuertoNNLinear": doc_scores},
            })
        return results

    eval_members = score_docs(splits["B_members_docs"], 1)
    eval_non_members = score_docs(splits["B_non_members_docs"], 0)
    print(f"    Scored {len(eval_members)} member docs, {len(eval_non_members)} non-member docs")

    return {
        "seed": seed,
        "config": {
            "n_train": N_TRAIN, "n_known": N_KNOWN, "ctx": CTX,
            "n_train_paragraphs": len(A_para),
            "n_known_paragraphs": len(known_para),
        },
        "models": ["PuertoNNLinear"],
        "model_params": {"PuertoNNLinear": {"method": "nn.Linear+Adam", "epochs": NUM_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE}},
        "known_scores": {"PuertoNNLinear": known_scores},
        "eval_members": eval_members,
        "eval_non_members": eval_non_members,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Single seed, or all 5 if omitted")
    parser.add_argument("--ctx", type=int, default=1024)
    args = parser.parse_args()

    global CTX
    CTX = args.ctx

    print(f"{'='*60}")
    print(f"  Precompute Puerto nn.Linear scores: {args.dataset} ctx={CTX}")
    print(f"{'='*60}")

    # Load data
    mem_file = f"{MIA_DIR}/{args.dataset}/document_{CTX}/members_{CTX}.jsonl"
    non_file = f"{MIA_DIR}/{args.dataset}/document_{CTX}/nonmembers_{CTX}.jsonl"
    members = load_jsonl_documents(mem_file, label=1)
    non_members = load_jsonl_documents(non_file, label=0)
    print(f"  Loaded {len(members)} mem, {len(non_members)} non-mem docs")

    # Run seeds
    seeds = [args.seed] if args.seed else ALL_SEEDS
    out_dir = f"{OUT_DIR}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    for seed in seeds:
        result = run_seed(args.dataset, seed, members, non_members)

        out_path = f"{out_dir}/precomputed_scores_puerto_ac_ctx{CTX}_train{N_TRAIN}_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"    Saved to {out_path}")
        print(f"    File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
