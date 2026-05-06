#!/usr/bin/env python3
"""
Majority voting aggregation from per-feature PCS files.

1. Use first 200 eval members + 200 eval non-members to find Youden's J threshold per feature
2. Convert remaining eval + known scores to binary using thresholds
3. Paragraph level: sum of votes across features -> AUROC
4. Document level: sum of votes per doc -> compare against known sums using MWU
5. Collection level: sum of votes per collection -> compare against known sums using t-test

Usage:
    python aggregate_majority_voting.py --dataset arxiv --ctx 2048 --train 1000
"""
import sys, os, json, argparse, random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ttest_ind
from collections import defaultdict

from src.utils.vectorized_stats import mannwhitneyu_u_only

ALL_SEEDS = [670487, 116739, 26225, 777572, 288389]
COLLECTION_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500]
N_COLLECTIONS = 1000
N_THRESHOLD_SAMPLES = 200  # per class for Youden's J

_MIA_ROOT = os.environ.get("MIA_ROOT", "./mia_scores")
MV_DIR = os.environ.get("MV_DIR", os.path.join(_MIA_ROOT, "results", "majority_voting"))
OUT_DIR = os.environ.get(
    "RUN_STATS_OUT_DIR", os.path.join(_MIA_ROOT, "results", "pythia-2.8b"))


def load_pcs(dataset, ctx, train, seed):
    path = f"{MV_DIR}/{dataset}/precomputed_scores_perfeature_fc_ctx{ctx}_train{train}_seed{seed}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def find_youdens_j_thresholds(pcs, n_samples=N_THRESHOLD_SAMPLES):
    """Find Youden's J optimal threshold per feature using first n_samples eval docs per class.

    Returns:
        thresholds: dict feature_name -> threshold
        directions: dict feature_name -> 'greater' or 'less' (whether score > threshold means member)
        remaining_mem_indices: indices of eval_members NOT used for threshold finding
        remaining_non_indices: indices of eval_non_members NOT used for threshold finding
    """
    feature_names = pcs["models"]
    n_mem = min(n_samples, len(pcs["eval_members"]))
    n_non = min(n_samples, len(pcs["eval_non_members"]))

    thresholds = {}
    directions = {}

    for feat in feature_names:
        # Get scores for threshold-finding subset
        mem_scores = [pcs["eval_members"][i]["scores"][feat][0] for i in range(n_mem)]
        non_scores = [pcs["eval_non_members"][i]["scores"][feat][0] for i in range(n_non)]

        all_scores = np.array(mem_scores + non_scores)
        all_labels = np.array([1] * n_mem + [0] * n_non)

        # Try both directions
        try:
            fpr, tpr, thresh = roc_curve(all_labels, all_scores)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_thresh = thresh[best_idx]
            best_auroc = roc_auc_score(all_labels, all_scores)

            if best_auroc >= 0.5:
                thresholds[feat] = best_thresh
                directions[feat] = "greater"  # score > threshold -> member
            else:
                # Flip direction
                fpr2, tpr2, thresh2 = roc_curve(all_labels, -all_scores)
                j_scores2 = tpr2 - fpr2
                best_idx2 = np.argmax(j_scores2)
                thresholds[feat] = -thresh2[best_idx2]
                directions[feat] = "less"  # score < threshold -> member
        except:
            thresholds[feat] = 0.0
            directions[feat] = "greater"

    remaining_mem = list(range(n_mem, len(pcs["eval_members"])))
    remaining_non = list(range(n_non, len(pcs["eval_non_members"])))

    return thresholds, directions, remaining_mem, remaining_non


def to_binary(score, threshold, direction):
    """Convert score to binary vote."""
    if direction == "greater":
        return 1 if score > threshold else 0
    else:
        return 1 if score < threshold else 0


def compute_vote_sum(doc_scores, feature_names, thresholds, directions):
    """Compute sum of binary votes across features for a document's paragraphs."""
    # For first-chunk: 1 paragraph per doc
    total_votes = 0
    for feat in feature_names:
        score = doc_scores[feat][0]  # first (only) paragraph
        total_votes += to_binary(score, thresholds[feat], directions[feat])
    return total_votes


def run_seed(pcs, seed, collection_sizes=None):
    if collection_sizes is None:
        collection_sizes = COLLECTION_SIZES

    feature_names = pcs["models"]
    n_features = len(feature_names)

    # Find thresholds using first 200 eval docs per class
    thresholds, directions, rem_mem_idx, rem_non_idx = find_youdens_j_thresholds(pcs)
    n_greater = sum(1 for d in directions.values() if d == "greater")
    print(f"    Thresholds: {n_features} features, {n_greater} greater / {n_features - n_greater} less")
    print(f"    Eval remaining: {len(rem_mem_idx)} mem + {len(rem_non_idx)} non-mem")

    # Compute vote sums for remaining eval docs
    mem_votes = []
    for i in rem_mem_idx:
        doc = pcs["eval_members"][i]
        votes = compute_vote_sum(doc["scores"], feature_names, thresholds, directions)
        mem_votes.append(votes)

    non_votes = []
    for i in rem_non_idx:
        doc = pcs["eval_non_members"][i]
        votes = compute_vote_sum(doc["scores"], feature_names, thresholds, directions)
        non_votes.append(votes)

    # Compute vote sums for known docs
    known_votes = []
    n_known = len(pcs["known_scores"][feature_names[0]])
    for ki in range(n_known):
        votes = 0
        for feat in feature_names:
            score = pcs["known_scores"][feat][ki]
            votes += to_binary(score, thresholds[feat], directions[feat])
        known_votes.append(votes)

    mem_votes = np.array(mem_votes, dtype=float)
    non_votes = np.array(non_votes, dtype=float)
    known_votes = np.array(known_votes, dtype=float)

    # Paragraph-level AUROC (vote sum as score)
    all_scores = np.concatenate([mem_votes, non_votes])
    all_labels = np.concatenate([np.ones(len(mem_votes)), np.zeros(len(non_votes))])
    para_auroc = roc_auc_score(all_labels, all_scores) if len(set(all_labels.tolist())) >= 2 else 0.5
    print(f"    para={para_auroc:.3f} (vote sums: mem_mean={mem_votes.mean():.1f}/{n_features}, "
          f"non_mean={non_votes.mean():.1f}/{n_features}, known_mean={known_votes.mean():.1f}/{n_features})")

    # Document-level AUROC (MWU: vote sum vs known vote sums)
    doc_stats = mannwhitneyu_u_only(
        np.concatenate([mem_votes, non_votes]).reshape(-1, 1),
        known_votes
    )
    doc_labels = np.concatenate([np.ones(len(mem_votes)), np.zeros(len(non_votes))])
    valid = np.isfinite(doc_stats)
    doc_auroc = roc_auc_score(doc_labels[valid], doc_stats[valid]) if valid.sum() > 0 and len(set(doc_labels[valid].tolist())) >= 2 else 0.5
    print(f"    doc={doc_auroc:.3f}")

    # Collection-level AUROC (t-test: pooled vote sums vs known vote sums)
    coll_results = {}
    for cs in collection_sizes:
        mem_list = list(range(len(mem_votes)))
        non_list = list(range(len(non_votes)))

        rng = random.Random(seed)
        def sample_colls(indices, n, cs, s):
            r = random.Random(s)
            colls = []
            for _ in range(n):
                if len(indices) >= cs:
                    colls.append(r.sample(indices, cs))
                else:
                    colls.append([r.choice(indices) for _ in range(cs)])
            return colls

        mem_colls = sample_colls(mem_list, N_COLLECTIONS, cs, seed)
        non_colls = sample_colls(non_list, N_COLLECTIONS, cs, seed + 1)

        coll_stats, coll_labels = [], []
        for label, colls, votes_arr in [(1, mem_colls, mem_votes), (0, non_colls, non_votes)]:
            for idx_list in colls:
                coll_vote_sums = votes_arr[idx_list]
                if len(coll_vote_sums) < 2:
                    continue
                try:
                    stat, _ = ttest_ind(coll_vote_sums, known_votes,
                                       equal_var=True, alternative='greater')
                    if np.isfinite(stat):
                        coll_stats.append(stat)
                        coll_labels.append(label)
                except:
                    pass

        if len(set(coll_labels)) >= 2:
            coll_auroc = roc_auc_score(coll_labels, coll_stats)
        else:
            coll_auroc = 0.5

        coll_results[cs] = coll_auroc
        if cs in [50, 100, 200, 500]:
            print(f"    coll@{cs}={coll_auroc:.3f}")

    return {
        "paragraph": {"auroc": para_auroc},
        "document": {"auroc": doc_auroc},
        "collections": {cs: {"auroc": v} for cs, v in coll_results.items()},
        "n_features": n_features,
        "n_threshold_samples": N_THRESHOLD_SAMPLES,
        "n_eval_mem": len(rem_mem_idx),
        "n_eval_non": len(rem_non_idx),
        "vote_stats": {
            "mem_mean": float(mem_votes.mean()),
            "non_mean": float(non_votes.mean()),
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
    args = parser.parse_args()

    global N_COLLECTIONS
    N_COLLECTIONS = args.n_collections

    seeds = [args.seed] if args.seed else ALL_SEEDS
    collection_sizes = args.collection_sizes or COLLECTION_SIZES

    print(f"{'='*60}")
    print(f"  Majority Voting: {args.dataset} ctx={args.ctx}")
    print(f"  Youden's J on first {N_THRESHOLD_SAMPLES} eval docs/class")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    all_results = []

    for train in args.train:
        for seed in seeds:
            print(f"\n  --- train={train}, seed={seed} ---")
            pcs = load_pcs(args.dataset, args.ctx, train, seed)
            if pcs is None:
                print(f"    SKIP: no PCS file")
                continue

            result = run_seed(pcs, seed, collection_sizes)
            result["seed"] = seed
            result["n_train"] = train
            all_results.append(result)

    # Summary
    if all_results:
        print(f"\n  {'='*50}")
        print(f"  Summary (mean ± std over {len(seeds)} seeds)")
        for train in args.train:
            tr = [r for r in all_results if r["n_train"] == train]
            if not tr:
                continue
            paras = [r["paragraph"]["auroc"] for r in tr]
            docs = [r["document"]["auroc"] for r in tr]
            print(f"  train={train}: para={np.mean(paras):.3f}±{np.std(paras):.3f}  "
                  f"doc={np.mean(docs):.3f}±{np.std(docs):.3f}", end="")
            for cs in [50, 100, 200, 500]:
                if cs in collection_sizes:
                    vals = [r["collections"][cs]["auroc"] for r in tr]
                    print(f"  c@{cs}={np.mean(vals):.3f}", end="")
            print()

    # Save
    out_dir = f"{OUT_DIR}/{args.dataset}/brunner_munzel"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/agg_majority_voting_mwu_ttest_ctx{args.ctx}.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "ctx": args.ctx,
            "method": "majority_voting_youdens_j",
            "n_threshold_samples": N_THRESHOLD_SAMPLES,
            "n_collections": N_COLLECTIONS,
            "collection_sizes": collection_sizes,
            "train_sizes": args.train,
            "seeds": seeds,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
