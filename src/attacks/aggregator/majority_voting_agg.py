#!/usr/bin/env python3
"""
Majority voting aggregation from per-feature PCS files.

Pipeline:
  1. For each feature, train an individual logistic regression model in pytorch.
  The prediction logits are the raw scores: per-doc vote = 1 iff the mean
  logit across the doc's paragraphs > 0.
  2. Paragraph level: vote SUM across features -> AUROC.
  3. Document  level: per-doc vote sum -> MWU vs known docs' vote sums.
  4. Collection level: per-collection vote sum -> Student's t-test vs known.

Example usage:
    python -m src.attacks.aggregator.majority_voting_agg \\
        --dataset arxiv --ctx 2048 --train 1000
"""
import argparse
import json
import os
import random

import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

from src.utils.vectorized_stats import mannwhitneyu_u_only

ALL_SEEDS = [670487, 116739, 26225, 777572, 288389]
COLLECTION_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                    150, 200, 300, 400, 500]
N_COLLECTIONS = 1000
LOGIT_RANGE_MIN = -50.0
LOGIT_RANGE_MAX = 50.0

_MIA_ROOT = os.environ.get("MIA_ROOT", "./mia_scores")
MV_DIR = os.environ.get("MV_DIR", os.path.join(_MIA_ROOT, "results", "majority_voting"))
OUT_DIR = os.environ.get(
    "RUN_STATS_OUT_DIR", os.path.join(_MIA_ROOT, "results", "pythia-2.8b"))


def load_pcs(dataset, ctx, train, seed):
    path = (f"{MV_DIR}/{dataset}/precomputed_scores_perfeature_"
            f"ctx{ctx}_train{train}_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _assert_logit_scored(pcs):
    """
    Fail loudly if the per-feature scores don't look like trained logits.
    """
    feat = pcs["models"][0]
    sample_doc = pcs["eval_members"][0]
    sample = np.asarray(sample_doc["scores"][feat], dtype=float)
    if not np.all(np.isfinite(sample)):
        raise ValueError(f"Per-feature scores for {feat!r} contain non-finite values.")
    smin, smax = float(sample.min()), float(sample.max())
    if smin < LOGIT_RANGE_MIN or smax > LOGIT_RANGE_MAX:
        raise ValueError(
            f"Per-feature scores for {feat!r} range {smin:.2f}..{smax:.2f} -- "
            f"that doesn't look like trained logits.  This script assumes the "
            f"upstream PCS was written by precompute_paragraph_scores_perfeature_*.py "
            f"(DiagonalLinear + BCEWithLogits).  Either re-run that or restore "
            f"the prior Youden's J threshold-finding step.")
    if smin >= 0.0 and smax <= 1.0:
        raise ValueError(
            f"Per-feature scores for {feat!r} are in [0, 1] -- looks like "
            f"probabilities, not logits.  Rerun the perfeature precompute or "
            f"transform via logit(p) = log(p / (1 - p)) before voting.")


def vote_sum_per_doc(doc_scores, feature_names):
    """Per-doc majority-vote sum: count features whose mean trained logit > 0
    across all paragraphs of the doc."""
    total = 0
    for feat in feature_names:
        if float(np.mean(doc_scores[feat])) > 0:
            total += 1
    return total


def run_seed(pcs, seed, collection_sizes=None):
    if collection_sizes is None:
        collection_sizes = COLLECTION_SIZES

    feature_names = pcs["models"]
    n_features = len(feature_names)

    print(f"    Voting at the natural-0 logit boundary across "
          f"{n_features} features (no Youden's J).")

    # All eval docs are usable -- no holdout for threshold finding.
    mem_votes = np.array(
        [vote_sum_per_doc(d["scores"], feature_names) for d in pcs["eval_members"]],
        dtype=float)
    non_votes = np.array(
        [vote_sum_per_doc(d["scores"], feature_names) for d in pcs["eval_non_members"]],
        dtype=float)

    # Vote sums for the known (held-out non-member) population.
    n_known = len(pcs["known_scores"][feature_names[0]])
    known_votes = np.array(
        [sum(int(pcs["known_scores"][f][k] > 0) for f in feature_names)
         for k in range(n_known)],
        dtype=float)

    print(f"    Eval: {len(mem_votes)} mem + {len(non_votes)} non-mem  "
          f"(prev. carve-out of 200/class no longer needed)")
    print(f"    Vote-sum means (out of {n_features}): "
          f"mem={mem_votes.mean():.1f}  non={non_votes.mean():.1f}  "
          f"known={known_votes.mean():.1f}")

    # ---------- paragraph-level AUROC --------------------------------------
    all_scores = np.concatenate([mem_votes, non_votes])
    all_labels = np.concatenate(
        [np.ones(len(mem_votes)), np.zeros(len(non_votes))])
    if len(set(all_labels.tolist())) < 2:
        para_auroc = 0.5
    else:
        para_auroc = roc_auc_score(all_labels, all_scores)
    print(f"    para = {para_auroc:.3f}")

    # ---------- document-level AUROC (MWU vs known votes) ------------------
    doc_stats = mannwhitneyu_u_only(
        np.concatenate([mem_votes, non_votes]).reshape(-1, 1),
        known_votes,
    )
    doc_labels = np.concatenate(
        [np.ones(len(mem_votes)), np.zeros(len(non_votes))])
    valid = np.isfinite(doc_stats)
    if valid.sum() == 0 or len(set(doc_labels[valid].tolist())) < 2:
        doc_auroc = 0.5
    else:
        doc_auroc = roc_auc_score(doc_labels[valid], doc_stats[valid])
    print(f"    doc  = {doc_auroc:.3f}")

    # ---------- collection-level AUROC (t-test vs known votes) ------------
    coll_results = {}
    mem_idx = list(range(len(mem_votes)))
    non_idx = list(range(len(non_votes)))

    def _sample_collections(indices, n_colls, cs, base_seed):
        r = random.Random(base_seed)
        out = []
        for _ in range(n_colls):
            if len(indices) >= cs:
                out.append(r.sample(indices, cs))
            else:
                out.append([r.choice(indices) for _ in range(cs)])
        return out

    for cs in collection_sizes:
        mem_colls = _sample_collections(mem_idx, N_COLLECTIONS, cs, seed)
        non_colls = _sample_collections(non_idx, N_COLLECTIONS, cs, seed + 1)

        coll_stats, coll_labels = [], []
        for label, colls, votes_arr in (
            (1, mem_colls, mem_votes),
            (0, non_colls, non_votes),
        ):
            for idx_list in colls:
                votes = votes_arr[idx_list]
                if len(votes) < 2:
                    continue
                try:
                    stat, _ = ttest_ind(votes, known_votes,
                                        equal_var=True, alternative="greater")
                    if np.isfinite(stat):
                        coll_stats.append(stat)
                        coll_labels.append(label)
                except Exception:
                    pass

        if len(set(coll_labels)) >= 2:
            coll_auroc = roc_auc_score(coll_labels, coll_stats)
        else:
            coll_auroc = 0.5
        coll_results[cs] = coll_auroc
        if cs in (50, 100, 200, 500):
            print(f"    coll@{cs} = {coll_auroc:.3f}")

    return {
        "paragraph": {"auroc": para_auroc},
        "document":  {"auroc": doc_auroc},
        "collections": {cs: {"auroc": v} for cs, v in coll_results.items()},
        "n_features": n_features,
        "n_eval_mem": len(mem_votes),
        "n_eval_non": len(non_votes),
        "vote_stats": {
            "mem_mean":   float(mem_votes.mean()),
            "non_mean":   float(non_votes.mean()),
            "known_mean": float(known_votes.mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--train", type=int, nargs="+", default=[1000])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--collection_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--n_collections", type=int, default=1000)
    parser.add_argument("--skip_logit_check", action="store_true",
                        help="Skip the assertion that per-feature scores look "
                             "like trained logits.  Use only if you know the "
                             "upstream perfeature precompute outputs in a "
                             "different format and you've handled it.")
    args = parser.parse_args()

    global N_COLLECTIONS
    N_COLLECTIONS = args.n_collections

    seeds = [args.seed] if args.seed else ALL_SEEDS
    collection_sizes = args.collection_sizes or COLLECTION_SIZES

    print(f"{'=' * 60}")
    print(f"  Majority Voting (natural-0 boundary): {args.dataset} ctx={args.ctx}")
    print(f"  Per-feature trained logit > 0 = vote for member.")
    print(f"  Seeds: {seeds}")
    print(f"{'=' * 60}")

    all_results = []

    for train in args.train:
        for seed in seeds:
            print(f"\n  --- train={train}, seed={seed} ---")
            pcs = load_pcs(args.dataset, args.ctx, train, seed)
            if pcs is None:
                print(f"    SKIP: no PCS file")
                continue

            if not args.skip_logit_check:
                _assert_logit_scored(pcs)

            result = run_seed(pcs, seed, collection_sizes)
            result["seed"] = seed
            result["n_train"] = train
            all_results.append(result)

    # ---------- summary ----------------------------------------------------
    if all_results:
        print(f"\n  {'=' * 50}")
        print(f"  Summary (mean +/- std over {len(seeds)} seeds)")
        for train in args.train:
            tr = [r for r in all_results if r["n_train"] == train]
            if not tr:
                continue
            paras = [r["paragraph"]["auroc"] for r in tr]
            docs = [r["document"]["auroc"] for r in tr]
            line = (f"  train={train}: "
                    f"para={np.mean(paras):.3f}+/-{np.std(paras):.3f}  "
                    f"doc={np.mean(docs):.3f}+/-{np.std(docs):.3f}")
            for cs in (50, 100, 200, 500):
                if cs in collection_sizes:
                    vals = [r["collections"][cs]["auroc"] for r in tr]
                    line += f"  c@{cs}={np.mean(vals):.3f}"
            print(line)

    # ---------- save -------------------------------------------------------
    out_dir = f"{OUT_DIR}/{args.dataset}/brunner_munzel"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/agg_majority_voting_natural0_mwu_ttest_ctx{args.ctx}.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "ctx": args.ctx,
            "method": "majority_voting_natural_0_logit_boundary",
            "n_collections": N_COLLECTIONS,
            "collection_sizes": collection_sizes,
            "train_sizes": args.train,
            "seeds": seeds,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
