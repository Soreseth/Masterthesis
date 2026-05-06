#!/usr/bin/env python3
"""
Aggregate precomputed paragraph scores (PCS) into document- and collection-level AUROCs.

Loads precomputed per-paragraph MIA scores and applies statistical tests
without retraining any models. Supports the bundled-classifier and
per-classifier PCS variants for Pythia-2.8b, Pythia-6.9b, and MIMIR.

Statistical tests:
  - Document level: Mann-Whitney U (mwu) or Brunner-Munzel (bm)
  - Collection level: Student's t-test (ttest) or Brunner-Munzel (bm)

Uses vectorized implementations from vectorized_stats.py for speed.
Collection sampling uses random.Random(seed) to match Puerto et al.

Usage:
    python -m src.evaluation.run_stats --dataset arxiv --ctx 1024 --pcs_type extended_2.8b
    python -m src.evaluation.run_stats --dataset arxiv --ctx 1024 --pcs_type puerto_2.8b --train 1000
    python -m src.evaluation.run_stats --dataset arxiv --ctx 1024 --pcs_type mlp_2.8b --seed 670487
"""

import sys, os, json, argparse, random
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.vectorized_stats import mannwhitneyu_u_only, brunnermunzel_w_only

ALL_SEEDS = [670487, 116739, 26225, 777572, 288389]
COLLECTION_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500]
N_COLLECTIONS = 100

OUT_DIR = os.environ.get(
    "RUN_STATS_OUT_DIR",
    os.path.join(os.environ.get("MIA_ROOT", "./mia_scores"), "results", "pythia-2.8b"),
)

# On-disk filename prefixes per variant.
PCS_PREFIXES = {
    "extended_2.8b":   "precomputed_scores_ctx",
    "puerto_2.8b":     "precomputed_scores_puerto_ctx",
    "extended_6.9b":   "precomputed_scores_6.9b_ctx",
    "puerto_6.9b":     "precomputed_scores_puerto_6.9b_ctx",
    "extended_mimir":  "precomputed_scores_ctx",
    # Per-classifier PCS variants (one classifier per file). Produced by
    # precompute_one_classifier.py for sklearn estimators and
    # precompute_one_mlp.py for MLP.
    "lr_2.8b":   "precomputed_scores_lr_ctx",
    "svc_2.8b":  "precomputed_scores_svc_ctx",
    "rf_2.8b":   "precomputed_scores_rf_ctx",
    "xgb_2.8b":  "precomputed_scores_xgb_ctx",
    "mlp_2.8b":  "precomputed_scores_mlp_ctx",
    "lr_6.9b":   "precomputed_scores_lr_6.9b_ctx",
    "svc_6.9b":  "precomputed_scores_svc_6.9b_ctx",
    "rf_6.9b":   "precomputed_scores_rf_6.9b_ctx",
    "xgb_6.9b":  "precomputed_scores_xgb_6.9b_ctx",
    "mlp_6.9b":  "precomputed_scores_mlp_6.9b_ctx",
}

PCS_DIR = os.environ.get(
    "PCS_DIR",
    os.path.join(os.environ.get("MIA_ROOT", "./mia_scores"), "results", "pcs"),
)
# Variants whose files live under <dataset>/per_classifier/
# (one classifier per file).
PER_CLASSIFIER_TYPES = {
    "lr_2.8b",  "svc_2.8b",  "rf_2.8b",  "xgb_2.8b", "mlp_2.8b",
    "lr_6.9b",  "svc_6.9b",  "rf_6.9b",  "xgb_6.9b", "mlp_6.9b",
}


def load_pcs(dataset, ctx, train, seed, pcs_type):
    """Load a precomputed scores file."""
    prefix = PCS_PREFIXES[pcs_type]
    if pcs_type in PER_CLASSIFIER_TYPES:
        path = f"{PCS_DIR}/{dataset}/per_classifier/{prefix}{ctx}_train{train}_seed{seed}.json"
    else:
        path = f"{PCS_DIR}/{dataset}/{prefix}{ctx}_train{train}_seed{seed}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_tpr_at_fpr(labels, scores, fpr_thresholds=[0.05, 0.01, 0.001, 0.0001]):
    """Compute TPR at specific FPR thresholds."""
    fpr, tpr, _ = roc_curve(labels, scores)
    results = {}
    for t in fpr_thresholds:
        results[f"tpr@{t}fpr"] = float(np.interp(t, fpr, tpr))
    return results


def evaluate_paragraph(pcs, model_name):
    """Paragraph-level AUROC from precomputed scores."""
    all_scores = []
    all_labels = []
    for doc in pcs["eval_members"]:
        all_scores.extend(doc["scores"][model_name])
        all_labels.extend([1] * doc["n_paragraphs"])
    for doc in pcs["eval_non_members"]:
        all_scores.extend(doc["scores"][model_name])
        all_labels.extend([0] * doc["n_paragraphs"])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) < 2:
        return {"auroc": 0.5, "n_samples": len(all_labels), "level": "paragraph"}

    auroc = roc_auc_score(all_labels, all_scores)
    result = {"auroc": auroc, "n_samples": len(all_labels), "level": "paragraph"}
    result.update(compute_tpr_at_fpr(all_labels, all_scores))
    return result


def evaluate_document(pcs, model_name, stat_test="mwu"):
    """Document-level AUROC using vectorized stats."""
    known_scores = np.array(pcs["known_scores"][model_name])

    # Group docs by paragraph count for batched processing
    doc_data = []  # (scores_array, label)
    for label, docs_key in [(1, "eval_members"), (0, "eval_non_members")]:
        for doc in pcs[docs_key]:
            scores = doc["scores"][model_name]
            if len(scores) < 1:
                continue
            doc_data.append((np.array(scores), label))

    if not doc_data:
        return {"auroc": 0.5, "n_samples": 0, "level": "document"}

    # Group by n_paragraphs for vectorized batch processing
    from collections import defaultdict
    by_size = defaultdict(list)
    for scores, label in doc_data:
        by_size[len(scores)].append((scores, label))

    doc_stats = []
    doc_labels = []

    for n_para, group in by_size.items():
        samples = np.array([g[0] for g in group])
        labels = [g[1] for g in group]

        if stat_test == "bm":
            stats = -brunnermunzel_w_only(samples, known_scores)
        else:  # mwu
            stats = mannwhitneyu_u_only(samples, known_scores)

        for i, stat in enumerate(stats):
            if np.isfinite(stat):
                doc_stats.append(stat)
                doc_labels.append(labels[i])

    doc_stats = np.array(doc_stats)
    doc_labels = np.array(doc_labels)
    doc_stats = np.nan_to_num(doc_stats, nan=0.0, posinf=1e6, neginf=-1e6)

    if len(doc_labels) < 2 or len(np.unique(doc_labels)) < 2:
        return {"auroc": 0.5, "n_samples": len(doc_labels), "level": "document"}

    auroc = roc_auc_score(doc_labels, doc_stats)
    result = {"auroc": auroc, "n_samples": len(doc_labels), "level": "document"}
    result.update(compute_tpr_at_fpr(doc_labels, doc_stats))
    return result


def evaluate_collection(pcs, model_name, coll_size, seed,
                        n_collections=N_COLLECTIONS, stat_test="ttest"):
    """Collection-level AUROC using t-test or BM."""
    known_scores = np.array(pcs["known_scores"][model_name])

    # Build doc score lists
    mem_docs_scores = [doc["scores"][model_name] for doc in pcs["eval_members"]]
    non_docs_scores = [doc["scores"][model_name] for doc in pcs["eval_non_members"]]

    # Sample collections using random.Random (matching Puerto)
    def sample_collections(docs_scores, n_colls, cs, s):
        rng = random.Random(s)
        indices = list(range(len(docs_scores)))
        collections = []
        for _ in range(n_colls):
            if len(indices) >= cs:
                sampled = rng.sample(indices, cs)
            else:
                sampled = [rng.choice(indices) for _ in range(cs)]
            collections.append(sampled)
        return collections

    mem_colls = sample_collections(mem_docs_scores, n_collections, coll_size, seed)
    non_colls = sample_collections(non_docs_scores, n_collections, coll_size, seed + 1)

    coll_stats = []
    coll_labels = []

    for label, colls, docs_scores in [(1, mem_colls, mem_docs_scores),
                                       (0, non_colls, non_docs_scores)]:
        for idx_list in colls:
            # Pool all paragraphs from the sampled documents
            coll_scores = []
            for idx in idx_list:
                coll_scores.extend(docs_scores[idx])
            coll_scores = np.array(coll_scores)

            if len(coll_scores) < 2:
                continue

            try:
                if stat_test == "bm":
                    stat = -brunnermunzel_w_only(coll_scores.reshape(1, -1), known_scores)[0]
                else:  # ttest
                    stat, _ = ttest_ind(coll_scores, known_scores,
                                       equal_var=True, alternative='greater')

                if np.isfinite(stat):
                    coll_stats.append(stat)
                    coll_labels.append(label)
            except Exception:
                pass

    coll_stats = np.array(coll_stats)
    coll_labels = np.array(coll_labels)
    coll_stats = np.nan_to_num(coll_stats, nan=0.0, posinf=1e6, neginf=-1e6)

    if len(coll_labels) < 2 or len(np.unique(coll_labels)) < 2:
        return {"auroc": 0.5, "n_samples": len(coll_labels),
                "level": "collection", "collection_size": coll_size}

    auroc = roc_auc_score(coll_labels, coll_stats)
    result = {"auroc": auroc, "n_samples": len(coll_labels),
              "level": "collection", "collection_size": coll_size}
    result.update(compute_tpr_at_fpr(coll_labels, coll_stats))
    return result


def subsample_known(pcs, known_size, seed):
    """Subsample known scores by document index, matching aggregate.py's
    np.random.RandomState(seed).permutation approach.
    Scores are flattened, subsample first N scores
    (approximation; exact doc-level subsampling would need doc boundaries).
    """
    total_known = len(list(pcs["known_scores"].values())[0])
    if known_size >= total_known:
        return pcs  # no subsampling needed

    rng = np.random.RandomState(seed)
    known_idx = rng.permutation(total_known)[:known_size]
    known_idx = sorted(known_idx)

    # Create a shallow copy with subsampled known scores
    pcs_sub = dict(pcs)
    pcs_sub["known_scores"] = {}
    for model_name, scores in pcs["known_scores"].items():
        pcs_sub["known_scores"][model_name] = [scores[i] for i in known_idx]

    return pcs_sub


def run_evaluation(dataset, ctx, train, seed, pcs_type,
                   doc_test="mwu", coll_test="ttest",
                   collection_sizes=None, known_sizes=None):
    """Run full evaluation for one PCS file."""
    if collection_sizes is None:
        collection_sizes = COLLECTION_SIZES

    pcs = load_pcs(dataset, ctx, train, seed, pcs_type)
    if pcs is None:
        print(f"    SKIP: no PCS file for {dataset} ctx={ctx} train={train} seed={seed} ({pcs_type})")
        return None

    # If no known sweep, run once with full known set
    if known_sizes is None:
        known_sizes = [None]  # None means use all

    models = pcs["models"]
    results = {}

    for model_name in models:
        print(f"    Model: {model_name}")

        # Paragraph level (independent of known size)
        para = evaluate_paragraph(pcs, model_name)
        print(f"      para={para['auroc']:.3f}")

        known_sweep_results = {}

        for known_size in known_sizes:
            if known_size is not None:
                pcs_k = subsample_known(pcs, known_size, seed)
                k_label = known_size
                n_known_actual = len(pcs_k["known_scores"][model_name])
                print(f"      known={known_size} ({n_known_actual} scores)")
            else:
                pcs_k = pcs
                k_label = "all"
                n_known_actual = len(pcs_k["known_scores"][model_name])

            # Document level
            doc = evaluate_document(pcs_k, model_name, stat_test=doc_test)
            print(f"        doc={doc['auroc']:.3f}")

            # Collection level - sweep sizes
            coll_results = {}
            for cs in collection_sizes:
                coll = evaluate_collection(pcs_k, model_name, cs, seed,
                                           n_collections=N_COLLECTIONS,
                                           stat_test=coll_test)
                coll_results[cs] = coll
                if cs in [50, 100, 200, 500]:
                    print(f"        coll@{cs}={coll['auroc']:.3f}")

            known_sweep_results[k_label] = {
                "n_known": n_known_actual,
                "document": doc,
                "collections": coll_results,
            }

        results[model_name] = {
            "paragraph": para,
            "known_sweep": known_sweep_results,
        }

        # For backward compatibility, also store top-level doc/collections from first known_size
        first_key = list(known_sweep_results.keys())[0]
        results[model_name]["document"] = known_sweep_results[first_key]["document"]
        results[model_name]["collections"] = known_sweep_results[first_key]["collections"]

    return {
        "dataset": dataset,
        "ctx": ctx,
        "n_train": train,
        "seed": seed,
        "pcs_type": pcs_type,
        "doc_test": doc_test,
        "coll_test": coll_test,
        "models": models,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate from precomputed paragraph scores")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--train", type=int, nargs="+",
                        default=[1000, 500, 200, 100, 50, 10])
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed, or all 5 if omitted")
    parser.add_argument("--pcs_type", type=str, default="extended_2.8b",
                        choices=list(PCS_PREFIXES.keys()))
    parser.add_argument("--doc_test", type=str, default="mwu",
                        choices=["mwu", "bm"],
                        help="Document-level test: mwu (Mann-Whitney U) or bm (Brunner-Munzel)")
    parser.add_argument("--coll_test", type=str, default="ttest",
                        choices=["ttest", "bm"],
                        help="Collection-level test: ttest (Student's t) or bm (Brunner-Munzel)")
    parser.add_argument("--collection_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--n_collections", type=int, default=100)
    parser.add_argument("--known_sizes", type=int, nargs="+", default=None,
                        help="Known partition sizes to sweep. If omitted, use full known set.")
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Optional subdirectory under <OUT_DIR>/<dataset>/brunner_munzel/. "
                             "Use this to isolate per-(train, seed) runs so parallel "
                             "tasks do not race on the same agg JSON path.")
    args = parser.parse_args()

    global N_COLLECTIONS
    N_COLLECTIONS = args.n_collections

    seeds = [args.seed] if args.seed else ALL_SEEDS
    collection_sizes = args.collection_sizes or COLLECTION_SIZES

    print(f"{'='*60}")
    print(f"  Aggregate from PCS: {args.dataset} ctx={args.ctx}")
    print(f"  PCS type: {args.pcs_type}")
    print(f"  Tests: doc={args.doc_test}, coll={args.coll_test}")
    print(f"  Train sizes: {args.train}")
    print(f"  Known sizes: {args.known_sizes or 'all'}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    all_results = []

    for train in args.train:
        for seed in seeds:
            print(f"\n  --- train={train}, seed={seed} ---")
            result = run_evaluation(
                args.dataset, args.ctx, train, seed, args.pcs_type,
                doc_test=args.doc_test, coll_test=args.coll_test,
                collection_sizes=collection_sizes,
                known_sizes=args.known_sizes,
            )
            if result is not None:
                all_results.append(result)

    # Save
    out_dir = f"{OUT_DIR}/{args.dataset}/brunner_munzel"
    if args.output_subdir:
        out_dir = os.path.join(out_dir, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    test_suffix = f"{args.doc_test}_{args.coll_test}"
    out_path = f"{out_dir}/agg_{args.pcs_type}_{test_suffix}_ctx{args.ctx}.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "ctx": args.ctx,
            "pcs_type": args.pcs_type,
            "doc_test": args.doc_test,
            "coll_test": args.coll_test,
            "n_collections": N_COLLECTIONS,
            "collection_sizes": collection_sizes,
            "train_sizes": args.train,
            "seeds": seeds,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  Summary (mean ± std over {len(seeds)} seeds)")
    if all_results:
        models = all_results[0]["models"]
        for model_name in models:
            print(f"\n  Model: {model_name}")
            for train in args.train:
                train_results = [r for r in all_results if r["n_train"] == train]
                if not train_results:
                    continue
                paras = [r["results"][model_name]["paragraph"]["auroc"] for r in train_results]
                docs = [r["results"][model_name]["document"]["auroc"] for r in train_results]
                print(f"    train={train}: para={np.mean(paras):.3f}±{np.std(paras):.3f}  "
                      f"doc={np.mean(docs):.3f}±{np.std(docs):.3f}", end="")
                for cs in [50, 100, 200, 500]:
                    if cs in collection_sizes:
                        vals = [r["results"][model_name]["collections"][str(cs) if isinstance(list(r["results"][model_name]["collections"].keys())[0], str) else cs]["auroc"]
                                for r in train_results
                                if cs in r["results"][model_name]["collections"]]
                        if vals:
                            print(f"  c@{cs}={np.mean(vals):.3f}", end="")
                print()


if __name__ == "__main__":
    main()
