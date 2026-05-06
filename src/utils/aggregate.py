"""
aggregate.py -- MIA Classifier Screening & Evaluation (HPC version)
===================================================================
Implements Puerto et al. (NAACL 2025) methodology with THREE aggregation levels:
  - PARAGRAPH: Direct classification on individual paragraphs
  - DOCUMENT: Aggregate paragraph scores per document using Mann-Whitney U test
  - COLLECTION: Aggregate paragraph scores across k documents using t-test

Usage:
  python aggregate.py --model pythia-2.8b --dataset Pile-CC --feature_mode reduced
  python aggregate.py --model pythia-2.8b --dataset arxiv --aggregation_levels paragraph document collection
"""

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------

import json
import os
import gc
import argparse
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, brunnermunzel
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    GridSearchCV,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="MIA Classifier Screening & Evaluation")
    parser.add_argument("--model", type=str, default="pythia-2.8b",
                        help="Model name (pythia-2.8b, pythia-6.9b)")
    parser.add_argument("--dataset", type=str, default="Pile-CC",
                        help="Dataset name (Pile-CC, arxiv, FreeLaw, Github, HackerNews, OpenWebText2, USPTO_Backgrounds, wiki)")
    parser.add_argument("--base_dir", type=str,
                        default=os.environ.get("MIA_ROOT", "./mia_scores"),
                        help="Base directory for MIA scores (env: MIA_ROOT)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/{model}/{dataset})")
    parser.add_argument("--context_sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--aggregation_levels", type=str, nargs="+",
                        default=["paragraph", "document", "collection"],
                        choices=["paragraph", "document", "collection"])
    parser.add_argument("--feature_mode", type=str, default="reduced",
                        choices=["puerto", "reduced", "grouplda", "all"])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[670487, 116739, 26225, 777572, 288389])
    parser.add_argument("--max_docs_per_class", type=int, default=None,
                        help="Max documents per class (None = use all)")
    parser.add_argument("--n_train_docs", type=int, nargs="+", default=[1000, 500, 200, 100, 50, 10],
                        help="Training docs per class sweep (largest first for CV). Default: [1000, 500, 200, 100, 50, 10]")
    parser.add_argument("--n_known_docs", type=int, nargs="+", default=[1000, 750, 500, 300, 200, 100, 50, 25],
                        help="Known docs per class sweep. Default: [1000, 750, 500, 300, 200, 100, 50, 25]")
    parser.add_argument("--tune_hyperparameters", action="store_true")
    parser.add_argument("--n_cv_folds", type=int, default=5)
    parser.add_argument("--run_known_partition_sweep", action="store_true", default=True)
    parser.add_argument("--run_token_count_analysis", action="store_true", default=True)
    parser.add_argument("--run_collection_size_sweep", action="store_true", default=True)
    parser.add_argument("--docs_per_collection", type=int, default=50)
    parser.add_argument("--num_collections", type=int, default=100)
    parser.add_argument("--sweep_collection_sizes", type=int, nargs="+",
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500],
                        help="Collection sizes for collection size sweep")
    parser.add_argument("--n_pca_components", type=int, default=1)
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Subdirectory under output_dir for results (e.g. 'brunner_munzel')")
    parser.add_argument("--pretuned_params_file", type=str, default=None,
                        help="JSON file with pretuned CV params (skip CV tuning)")
    parser.add_argument("--cv_only", action="store_true",
                        help="Run CV tuning only, save best params to JSON, then exit. "
                             "Uses first seed and largest train size.")
    parser.add_argument("--stat_test", type=str, default="mwu_ttest",
                        choices=["mwu_ttest", "brunnermunzel"],
                        help="Statistical test for aggregation. "
                             "mwu_ttest: Mann-Whitney U (document) + t-test (collection). "
                             "brunnermunzel: Brunner-Munzel W statistic (both levels).")
    return parser.parse_args()


def build_config(args):
    """Build config dict from argparse args."""
    input_dir = os.path.join(args.base_dir, args.model, args.dataset)
    output_dir = args.output_dir or os.path.join("results", args.model, args.dataset)
    if args.output_subdir:
        output_dir = os.path.join(output_dir, args.output_subdir)
    else:
        # Auto-create subdirectory based on statistical test
        stat_subdir = {"mwu_ttest": "mwu_ttest", "brunnermunzel": "brunner_munzel"}
        output_dir = os.path.join(output_dir, stat_subdir.get(args.stat_test, args.stat_test))
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "model_name": args.model,
        "dataset_name": args.dataset,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "context_sizes": args.context_sizes,
        "seeds": args.seeds,

        # Puerto et al. split sizes (list for sweep, largest first)
        "n_train_docs_sweep": sorted(args.n_train_docs, reverse=True),
        "n_train_docs": max(args.n_train_docs), 
        "n_known_docs_sweep": sorted(args.n_known_docs, reverse=True),
        "n_known_docs": max(args.n_known_docs),

        # CV-based hyperparameter tuning
        "tune_hyperparameters": args.tune_hyperparameters,
        "n_cv_folds": args.n_cv_folds,
        "models_to_tune": ["PuertoLinearMap", "LogisticRegression", "SVC", "RandomForest", "XGBoost"],

        # Puerto et al. features (L=10)
        "puerto_features": [
            "ppl", "ppl_lowercase_ratio", "ppl_zlib_ratio",
            "min_k_5", "min_k_10", "min_k_20", "min_k_30",
            "min_k_40", "min_k_50", "min_k_60",
        ],
        "derived_features": {
            "ppl_lowercase_ratio": ("ppl", "lowercase_ppl", "divide"),
            "ppl_zlib_ratio": ("ppl", "zlib", "divide"),
        },

        # Feature mode
        "feature_mode": args.feature_mode,
        "n_pca_components": args.n_pca_components,
        "grouplda_n_components": 1,
        "grouplda_n_lda_train": None,  # Use all A partition paragraphs

        # Aggregation levels
        "aggregation_levels": args.aggregation_levels,

        # Collection-level settings
        "docs_per_collection": args.docs_per_collection,
        "num_collections": args.num_collections,
        "sweep_collection_sizes": args.sweep_collection_sizes,

        # Advanced analyses
        "run_known_partition_sweep": args.run_known_partition_sweep,
        "run_token_count_analysis": args.run_token_count_analysis,
        "run_collection_size_sweep": args.run_collection_size_sweep,

        # Max docs
        "max_docs_per_class": args.max_docs_per_class,

        # Pretuned params file
        "pretuned_params_file": args.pretuned_params_file,

        # CV-only mode
        "cv_only": args.cv_only,

        # Statistical test
        "stat_test": args.stat_test,

    }
    return config


# --------------------------------------------------------------
# Cell 3: Data Loading
# --------------------------------------------------------------

def load_jsonl_documents(path, label, max_docs=None):
    """
    Load JSONL file and return list of documents.
    Each document is a list of paragraph feature dicts.
    """
    documents = []
    with open(path) as f:
        for line in f:
            if max_docs and len(documents) >= max_docs:
                break
            obj = json.loads(line)
            preds = obj.get("pred", obj.get("preds", []))
            
            if isinstance(preds, list) and len(preds) > 0:
                if isinstance(preds[0], dict):
                    doc_paragraphs = []
                    for para in preds:
                        doc_paragraphs.append({"pred": para, "label": label})
                    documents.append(doc_paragraphs)
            elif isinstance(preds, dict):
                documents.append([{"pred": preds, "label": label}])
    
    return documents


def split_documents_puerto(members, non_members, config, seed):
    """
    Apply Puerto et al.'s document-level split with fixed eval set.

    The eval set (B partition) is ALWAYS the same regardless of n_train_docs
    or n_known_docs. This is achieved by:
      1. Shuffle all docs with the seed
      2. Reserve a FIXED pool for training (max n_train docs per class)
      3. Reserve a FIXED pool for known + extended known (non-members only)
      4. Everything else = B partition (eval), constant across all sweep sizes

    For smaller training/known sizes, we subsample WITHIN the reserved pools.

    Split structure:
      Members:     [ A_pool (max_train) | discarded (max_known_total) | B (eval) ]
      Non-members: [ A_pool (max_train) | Known_pool (max_known_total) | B (eval) ]

    Config keys used:
      - n_train_docs: actual training docs to use (subsampled from A_pool)
      - n_known_docs: actual known docs to use (subsampled from Known_pool)
      - n_train_docs_sweep[0]: max training pool size (= largest train size)
      - n_known_docs_sweep[0]: max known pool size
    """
    n_train = config["n_train_docs"]  # actual training size (may be < max)
    n_known = config["n_known_docs"]  # actual known size

    # Fixed pool sizes (determines eval set boundary, never changes)
    train_sizes = config.get("n_train_docs_sweep", [n_train])
    known_sizes = config.get("n_known_docs_sweep", [n_known])

    max_train = max(train_sizes)  # training pool
    max_known_total = max(known_sizes)  # known pool

    # Shuffle with seed
    rng = np.random.RandomState(seed)
    mem_idx = rng.permutation(len(members))
    non_idx = rng.permutation(len(non_members))

    members_shuffled = [members[i] for i in mem_idx]
    non_members_shuffled = [non_members[i] for i in non_idx]

    # Fixed boundary: eval set starts at the same index regardless of n_train
    eval_start = max_train + max_known_total

    # -- Training pool (fixed size = max_train) --
    train_pool_mem = members_shuffled[:max_train]
    train_pool_non = non_members_shuffled[:max_train]

    # -- Known pool (fixed size = max_known_total, non-members only) --
    known_pool = non_members_shuffled[max_train:eval_start]

    # -- Eval set (everything after, FIXED across all sweep sizes) --
    B_members_docs = members_shuffled[eval_start:]
    B_non_members_docs = non_members_shuffled[eval_start:]

    # Subsample training docs from pool (for smaller n_train)
    if n_train < max_train:
        train_sub_idx = rng.permutation(max_train)[:n_train]
        A_members_docs = [train_pool_mem[i] for i in sorted(train_sub_idx)]
        A_non_members_docs = [train_pool_non[i] for i in sorted(train_sub_idx)]
    else:
        A_members_docs = train_pool_mem
        A_non_members_docs = train_pool_non

    # Subsample known docs from pool
    known_non_members_docs = known_pool[:n_known]
    all_known_for_sweep_docs = known_pool

    # Flatten paragraphs for A partition
    A_members_para = []
    A_members_doc_ids = []
    for doc_id, doc in enumerate(A_members_docs):
        for para in doc:
            A_members_para.append(para)
            A_members_doc_ids.append(f"mem_{doc_id}")

    A_non_members_para = []
    A_non_members_doc_ids = []
    for doc_id, doc in enumerate(A_non_members_docs):
        for para in doc:
            A_non_members_para.append(para)
            A_non_members_doc_ids.append(f"non_{doc_id}")

    # Flatten known paragraphs
    known_non_members_para = [para for doc in known_non_members_docs for para in doc]

    print(f"\n  Split [seed={seed}]: pool={max_train} train + {max_known_total} known, actual={n_train} train + {n_known} known")
    print(f"    A (train): {len(A_members_para)} mem + {len(A_non_members_para)} non-mem paragraphs ({n_train} docs/class)")
    print(f"    Known: {len(known_non_members_docs)} docs ({len(known_non_members_para)} para)")
    print(f"    B (eval): {len(B_members_docs)} mem + {len(B_non_members_docs)} non-mem docs (FIXED)")

    return {
        "A_members_para": A_members_para,
        "A_non_members_para": A_non_members_para,
        "A_members_doc_ids": A_members_doc_ids,
        "A_non_members_doc_ids": A_non_members_doc_ids,
        "known_non_members_para": known_non_members_para,
        "known_non_members_docs": known_non_members_docs,
        "all_known_for_sweep_docs": all_known_for_sweep_docs,
        "B_members_docs": B_members_docs,
        "B_non_members_docs": B_non_members_docs,
    }


def load_data_for_context_size(context_size, config):
    """Load raw documents for a given context size."""
    input_dir = config["input_dir"]
    max_docs = config.get("max_docs_per_class", None)

    member_file = os.path.join(input_dir, f"document_{context_size}", f"members_{context_size}.jsonl")
    nonmember_file = os.path.join(input_dir, f"document_{context_size}", f"nonmembers_{context_size}.jsonl")

    print(f"  Looking for:")
    print(f"    Members: {member_file}")
    print(f"    Non-members: {nonmember_file}")

    for f in [member_file, nonmember_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Not found: {f}")

    print(f"  Loading members: {os.path.basename(member_file)}" + (f" (max {max_docs} docs)" if max_docs else ""))
    members = load_jsonl_documents(member_file, label=1, max_docs=max_docs) # Member Label = 1
    print(f"    {len(members)} documents loaded")

    print(f"  Loading non-members: {os.path.basename(nonmember_file)}" + (f" (max {max_docs} docs)" if max_docs else ""))
    non_members = load_jsonl_documents(nonmember_file, label=0, max_docs=max_docs) # Non-member Label = 0
    print(f"    {len(non_members)} documents loaded")

    n_para_mem = sum(len(doc) for doc in members)
    n_para_non = sum(len(doc) for doc in non_members)
    print(f"    Total paragraphs: {n_para_mem} member + {n_para_non} non-member")

    return members, non_members


# --------------------------------------------------------------
# Cell 4: Feature Extraction
# --------------------------------------------------------------

def compute_derived_features(pred, derived_config):
    """Compute derived features from existing ones."""
    for derived_name, (feat1, feat2, op) in derived_config.items():
        val1 = pred.get(feat1, np.nan)
        val2 = pred.get(feat2, np.nan)
        
        if op == "divide":
            if val2 != 0 and not np.isnan(val2):
                pred[derived_name] = val1 / val2
            else:
                pred[derived_name] = 0.0
        elif op == "subtract":
            pred[derived_name] = val1 - val2
        elif op == "multiply":
            pred[derived_name] = val1 * val2
    return pred


def extract_features(paragraphs, feature_names=None, derived_config=None):
    """
    Extract features from paragraph dicts.

    Args:
        paragraphs: List of {"pred": {...}, "label": int}
        feature_names: List of feature names to extract. If None, extract all.
        derived_config: Dict of derived features to compute (e.g. ppl_zlib_ratio)

    Returns:
        X, y, feature_names
    """
    if not paragraphs:
        return np.array([]), np.array([]), []

    if feature_names is None:
        feature_names = sorted(paragraphs[0]["pred"].keys())

    X_list = []
    y_list = []

    for para in paragraphs:
        pred = para["pred"] if not derived_config else para["pred"].copy()
        if derived_config:
            pred = compute_derived_features(pred, derived_config)

        features = [pred.get(feat_name, 0.0) for feat_name in feature_names]
        X_list.append(features)
        y_list.append(para["label"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)

    return X, y, feature_names




# --------------------------------------------------------------
# Cell 4b: Feature Reduction (Reference Filtering + GroupPCA for CAMIA)
# --------------------------------------------------------------

# Reference checkpoints to keep (step1 = near-random, final = fully trained)
KEEP_REF_STEPS = ["step1", "final"]

def filter_reference_features(feature_keys):
    """
    Filter reference-model features to keep only step1 and final.
    
    Reference features follow patterns like:
      - ref_loss_diff_70m_step1
      - wbc_70m_final_w1
      - tl_informia_70m_step1_t0.5_mk5
    
    Returns: List of feature keys to keep
    """
    import re
    ref_prefixes = ['ref_loss_', 'wbc_', 'tl_informia_']
    
    # Patterns to match step boundaries (step1 but not step10, step100, etc.)
    # Match: _step1_ or _step1$ (end of string) or _step1_something
    keep_patterns = [
        re.compile(r'_step1(?:_|$)'),      # _step1_ or _step1 at end
        re.compile(r'_final(?:_|$)'),      # _final_ or _final at end
    ]
    
    # Features to exclude entirely in reduced mode (superseded by better alternatives)
    exclude_prefixes = ['ref_loss_diff_', 'mod_renyi_']
    exclude_exact = {'modified_entropies_mean'}

    filtered = []
    removed_count = 0
    excluded_count = 0

    for key in feature_keys:
        # Exclude ref_loss_diff, mod_renyi, modified_entropies_mean
        if any(key.startswith(p) for p in exclude_prefixes) or key in exclude_exact:
            excluded_count += 1
            continue

        is_ref_feature = any(key.startswith(p) for p in ref_prefixes)

        if is_ref_feature:
            keep = any(pattern.search(key) for pattern in keep_patterns)
            if keep:
                filtered.append(key)
            else:
                removed_count += 1
        else:
            filtered.append(key)

    if removed_count > 0:
        print(f"  Reference filtering: removed {removed_count} features (kept step1 + final only)")
    if excluded_count > 0:
        print(f"  Excluded {excluded_count} features (ref_loss_diff, mod_renyi, modified_entropies_mean)")
    
    return filtered


def _parse_camia_groups(feature_keys):
    """
    Parse CAMIA features into groups for GroupPCA.
    
    Handles patterns like:
      - camia_rep2_cut_200, camia_rep2_cut_300  -> group: camia_rep2_cut
      - camia_rep2_cal_T, camia_rep2_cal_200    -> group: camia_rep2_cal
    """
    camia_groups = defaultdict(list)
    other_features = []
    
    for key in feature_keys:
        if key.startswith('camia_'):
            # Pattern: camia_rep{N}_{method}_{param}
            # Group by: camia_rep{N}_{method} (everything except last underscore segment)
            parts = key.rsplit('_', 1)  # Split off the last segment (param)
            if len(parts) == 2:
                group_name = parts[0]  # e.g., "camia_rep2_cut"
                camia_groups[group_name].append(key)
            else:
                other_features.append(key)
        else:
            other_features.append(key)
    
    return dict(camia_groups), other_features


def apply_feature_reduction(X_train, y_train, X_evals, feature_keys, 
                            n_pca_components=2, seed=42):
    """
    Apply feature reduction:
      1. Filter reference features to step1 + final only
      2. Apply GroupPCA to CAMIA features (n_components per group)
      3. Keep other features as singletons
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (unused, kept for API consistency)
        X_evals: List of evaluation feature matrices to transform
        feature_keys: List of feature names matching X columns
        n_pca_components: PCA components per CAMIA group (default=2)
        seed: Random seed
    
    Returns:
        X_train_reduced, y_train, X_evals_reduced, new_feature_keys
    """
    from sklearn.decomposition import PCA
    
    # Step 0: Align X columns to feature_keys (pad missing features with 0)
    # Some documents may lack TagTab features, resulting in fewer columns
    # Pad missing features to match feature_keys (some docs lack TagTab features)
    n_expected = len(feature_keys)
    if X_train.shape[1] < n_expected:
        X_train = np.pad(X_train, ((0, 0), (0, n_expected - X_train.shape[1])), constant_values=0.0)
        print(f"  Aligned train features: padded {n_expected - X_train.shape[1]} missing columns")
    X_evals = [
        np.pad(X, ((0, 0), (0, n_expected - X.shape[1])), constant_values=0.0)
        if X.shape[1] < n_expected else X
        for X in X_evals
    ]

    # Step 1: Filter reference features
    filtered_keys = filter_reference_features(feature_keys)

    # Build index mapping for filtered features
    key_to_orig_idx = {k: i for i, k in enumerate(feature_keys)}
    filtered_indices = [key_to_orig_idx[k] for k in filtered_keys]

    # Subset to filtered features
    X_train_filtered = X_train[:, filtered_indices]
    X_evals_filtered = [X[:, filtered_indices] for X in X_evals]
    
    print(f"  After reference filtering: {len(feature_keys)} -> {len(filtered_keys)} features")
    
    # Step 2: Parse CAMIA groups
    camia_groups, other_features = _parse_camia_groups(filtered_keys)
    
    print(f"\n  CAMIA GroupPCA: {len(camia_groups)} groups")
    for gname, members in sorted(camia_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {gname}: {len(members)} features -> {n_pca_components} components")
    
    # Build column index mapping for filtered features
    filtered_key_to_idx = {k: i for i, k in enumerate(filtered_keys)}
    
    # Step 3: Fit PCA for each CAMIA group
    pca_transformers = []  # (group_name, pca_model, feature_indices)
    
    for gname, members in sorted(camia_groups.items()):
        indices = [filtered_key_to_idx[m] for m in members]
        
        if len(indices) < n_pca_components:
            # Too few features for PCA, keep as singletons
            other_features.extend(members)
            continue
        
        X_group = X_train_filtered[:, indices]
        
        # Check for constant features
        valid_cols = np.std(X_group, axis=0) > 1e-10
        if valid_cols.sum() < n_pca_components:
            other_features.extend(members)
            continue
        
        # Filter to valid columns
        valid_indices = [indices[i] for i in range(len(indices)) if valid_cols[i]]
        X_group = X_train_filtered[:, valid_indices]
        
        # Fit PCA
        n_comp = min(n_pca_components, len(valid_indices))
        pca = PCA(n_components=n_comp, random_state=seed)
        try:
            pca.fit(X_group)
            pca_transformers.append((gname, pca, valid_indices))
        except Exception as e:
            print(f"    Warning: PCA failed for {gname}: {e}")
            other_features.extend(members)
    
    # Get indices for other (non-CAMIA) features
    other_indices = [filtered_key_to_idx[k] for k in other_features if k in filtered_key_to_idx]
    
    # Step 4: Transform all data
    def transform_data(X):
        parts = []
        new_keys = []
        
        # Add PCA components for each CAMIA group
        for gname, pca, indices in pca_transformers:
            X_group = X[:, indices]
            X_pca = pca.transform(X_group)
            parts.append(X_pca)
            for i in range(X_pca.shape[1]):
                new_keys.append(f"pca_{gname}_{i}")
        
        # Add other features (singletons)
        if other_indices:
            parts.append(X[:, other_indices])
            for idx in other_indices:
                new_keys.append(filtered_keys[idx])
        
        if parts:
            X_reduced = np.hstack(parts)
        else:
            X_reduced = X
            new_keys = filtered_keys
        
        return X_reduced, new_keys
    
    X_train_reduced, new_keys = transform_data(X_train_filtered)
    X_evals_reduced = [transform_data(X_e)[0] for X_e in X_evals_filtered]
    
    print(f"\n  Final reduction: {len(feature_keys)} -> {len(new_keys)} features")
    print(f"    - CAMIA: {sum(len(m) for m in camia_groups.values())} -> {len(pca_transformers) * n_pca_components} (GroupPCA)")
    print(f"    - Other: {len(other_indices)} singletons")

    # Build a transform_fn that takes raw features (all_feature_keys columns)
    # and produces reduced features
    _filtered_indices = filtered_indices  # capture for closure
    def make_transform_fn():
        def transform_fn(X_raw):
            X_filt = X_raw[:, _filtered_indices]
            return transform_data(X_filt)[0]
        return transform_fn

    return X_train_reduced, y_train, X_evals_reduced, new_keys, make_transform_fn()


# --------------------------------------------------------------
# Cell 4c: GroupLDA Feature Reduction (Alternative)
# --------------------------------------------------------------

def _parse_feature_groups_for_lda(feature_keys):
    """
    Parse feature names to identify groups by prefix for LDA.
    
    Groups are identified by common prefixes like:
    - renyi_05_*, renyi_2_*, renyi_inf_*
    - min_k_*, min_k_plus_*
    - ref_loss_*, wbc_*, tl_informia_*
    - recall_*, conrecall_*
    """
    groups = defaultdict(list)
    singletons = []
    
    group_patterns = [
        ('ref_loss_', 'ref_loss'),
        ('wbc_', 'wbc'),
        ('tl_informia_', 'tl_informia'),
        ('renyi_05_', 'renyi_05'),
        ('renyi_2_', 'renyi_2'),
        ('renyi_inf_', 'renyi_inf'),
        ('min_k_plus_', 'min_k_plus'),
        ('min_k_', 'min_k'),
        ('conrecall_', 'conrecall'),
        ('recall_', 'recall'),
        ('entropies_', 'entropies'),
        ('modified_entropies_', 'modified_entropies'),
        ('gap_prob_', 'gap_prob'),
        ('camia_', 'camia'),
        ('acmia_', 'acmia'),
        ('noisy_', 'noisy'),
        ('tag_tab_', 'tag_tab'),
        ('dc_pdd_', 'dc_pdd'),
    ]
    
    for key in feature_keys:
        matched = False
        for prefix, group_name in group_patterns:
            if key.startswith(prefix):
                groups[group_name].append(key)
                matched = True
                break
        if not matched:
            singletons.append(key)
    
    return dict(groups), singletons


def apply_group_lda(X_train, y_train, X_evals, feature_keys, 
                    n_components_per_group=1, n_lda_train=None, seed=42):
    """
    Apply LDA within each feature group to reduce dimensionality.
    
    For each group of related features (e.g., all renyi_05_* features),
    fit LDA to find the most discriminative linear combination.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels
        X_evals: List of evaluation feature matrices to transform
        feature_keys: List of feature names matching X columns
        n_components_per_group: LDA components per group (1 for binary)
        n_lda_train: Max samples for LDA fitting (None = use all)
        seed: Random seed
    
    Returns:
        X_train_reduced, y_train, X_evals_reduced, new_feature_keys
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    rng = np.random.RandomState(seed)
    groups, singletons = _parse_feature_groups_for_lda(feature_keys)
    
    print(f"\n  GroupLDA: {len(groups)} groups, {len(singletons)} singletons")
    for gname, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"    {gname}: {len(members)} features")
    
    # Subsample for LDA fitting if requested
    if n_lda_train and len(X_train) > n_lda_train:
        lda_idx = rng.permutation(len(X_train))[:n_lda_train]
        X_lda_fit = X_train[lda_idx]
        y_lda_fit = y_train[lda_idx]
        print(f"  Using {n_lda_train} samples for LDA fitting")
    else:
        X_lda_fit = X_train
        y_lda_fit = y_train
    
    # Build column index mapping
    key_to_idx = {k: i for i, k in enumerate(feature_keys)}
    
    # Collect LDA transformers and singleton indices
    lda_transformers = []  # (group_name, lda_model, feature_indices)
    singleton_indices = [key_to_idx[s] for s in singletons if s in key_to_idx]
    
    for gname, members in sorted(groups.items()):
        indices = [key_to_idx[m] for m in members if m in key_to_idx]
        if len(indices) < 2:
            singleton_indices.extend(indices)
            continue
        
        X_group = X_lda_fit[:, indices]
        
        # Check for constant features
        valid_cols = np.std(X_group, axis=0) > 1e-10
        if valid_cols.sum() < 2:
            singleton_indices.extend(indices)
            continue
        
        indices = [indices[i] for i in range(len(indices)) if valid_cols[i]]
        X_group = X_lda_fit[:, indices]
        
        # Fit LDA
        n_comp = min(n_components_per_group, len(np.unique(y_lda_fit)) - 1, len(indices))
        if n_comp < 1:
            singleton_indices.extend(indices)
            continue
            
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        try:
            lda.fit(X_group, y_lda_fit)
            lda_transformers.append((gname, lda, indices))
        except Exception as e:
            print(f"    Warning: LDA failed for {gname}: {e}")
            singleton_indices.extend(indices)
    
    # Transform all data
    def transform_data(X):
        parts = []
        new_keys = []
        
        # Add LDA components
        for gname, lda, indices in lda_transformers:
            X_group = X[:, indices]
            X_lda = lda.transform(X_group)
            parts.append(X_lda)
            for i in range(X_lda.shape[1]):
                new_keys.append(f"lda_{gname}_{i}")
        
        # Add singletons
        if singleton_indices:
            parts.append(X[:, singleton_indices])
            for idx in singleton_indices:
                new_keys.append(feature_keys[idx])
        
        if parts:
            X_reduced = np.hstack(parts)
        else:
            X_reduced = X
            new_keys = feature_keys
        
        return X_reduced, new_keys
    
    X_train_reduced, new_keys = transform_data(X_train)
    X_evals_reduced = [transform_data(X_e)[0] for X_e in X_evals]
    
    print(f"  Reduced: {len(feature_keys)} -> {len(new_keys)} features")
    
    return X_train_reduced, y_train, X_evals_reduced, new_keys


# --------------------------------------------------------------
# Cell 5: Models & Hyperparameter Tuning
# --------------------------------------------------------------

def get_hyperparameter_grids():
    """
    Conservative grids optimized for transfer across training sizes (10-1000 docs).
    
    Design principles:
    - Bias toward regularization (transfers better to small data)
    - Incorporate Yogesh's key additions: class_weight, subsample, higher n_estimators
    - Keep tractable (~400 total combinations)
    """
    return {
        "PuertoLinearMap": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
        },  # 8 combinations (same grid, but tuned on Puerto 10 features)

        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
        },  # 8 combinations
        
        "SVC": [
            {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear"],
            },
            {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf"],
                "gamma": ["scale", "auto", 0.01],
            },
            {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["poly"],
                "gamma": ["scale", "auto"],
                "degree": [2, 3],
            },
        ],  # 3 + 9 + 12 = 24 combinations
        
        "RandomForest": {
            "n_estimators": [500, 1000, 2000],       
            "max_depth": [5, 10, 20],                
            "min_samples_leaf": [5, 10],             
            "min_samples_split": [5, 10],           
            "max_features": ["sqrt", 0.5, 0.7],      
            "class_weight": ["balanced"],     
            "ccp_alpha": [0.0, 0.01, 0.1],      
        },  # 3 × 3 × 2 × 2 × 3 × 1 × 3 = 324 combinations
        
        "XGBoost": {
            "n_estimators": [500, 1000],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.001, 0.01, 0.1],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.3, 0.5, 0.7],
            "reg_lambda": [0.1, 1.0, 10.0],          
        },  # 2 × 3 × 3 × 2 × 3 × 3 = 324 combinations
    }

def tune_model_cv(model_name, X_train, y_train, groups=None, n_folds=5, seed=42):
    """
    Tune hyperparameters for a single model using cross-validation.
    Note: CV tuning always uses sklearn (CPU) since GridSearchCV needs sklearn-compatible estimators.
    """

    grids = get_hyperparameter_grids()

    if model_name not in grids:
        return {}, None, None

    param_grid = grids[model_name]

    if model_name in ("LogisticRegression", "PuertoLinearMap"):
        base_model = LogisticRegression(max_iter=5000, random_state=seed)

    elif model_name == "SVC":
        base_model = SVC(probability=False, random_state=seed)

    elif model_name == "RandomForest":
        base_model = RandomForestClassifier(
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed
        )

    elif model_name == "XGBoost":
        base_model = XGBClassifier(
            subsample=0.8, reg_lambda=1.0, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=seed
        )

    else:
        return {}, None, None
    
    # Use StratifiedGroupKFold if groups provided
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(
        base_model, param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=2 if model_name in ["XGBoost", "SVC"] else -1, 
        refit=False,
        verbose=0,
    )
    
    if groups is not None:
        grid_search.fit(X_train, y_train, groups=groups)
    else:
        grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    cv_results = cv_results.sort_values("rank_test_score")
    
    return best_params, best_score, cv_results

    
def create_tuned_model(model_name, best_params, seed=42):
    """Create a model instance with tuned hyperparameters."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    if model_name == "PuertoLinearMap":
        params = {"max_iter": 5000, "C": 1.0, "random_state": seed}
        params.update(best_params)
        return LogisticRegression(**params)

    elif model_name == "LogisticRegression":
        params = {"max_iter": 5000, "C": 1.0, "random_state": seed}
        params.update(best_params)
        return LogisticRegression(**params)

    elif model_name == "SVC":
        params = {"probability": False, "random_state": seed}
        params.update(best_params)
        return SVC(**params)

    elif model_name == "RandomForest":
        params = {"max_features": "sqrt", "n_jobs": -1, "random_state": seed}
        params.update(best_params)
        return RandomForestClassifier(**params)

    elif model_name == "XGBoost":
        params = {
            "subsample": 0.8, "reg_lambda": 1.0,
            "eval_metric": "logloss", "tree_method": "hist",
            "n_jobs": -1, "random_state": seed
        }
        params.update(best_params)
        return XGBClassifier(**params)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# --------------------------------------------------------------




# --------------------------------------------------------------
# Cell 6: Aggregation Functions
# --------------------------------------------------------------

def get_paragraph_scores(model, scaler, paragraphs, feature_names, derived_config, feature_mode, all_feature_keys=None, transform_fn=None):
    """
    Get paragraph-level scores from trained model.
    
    Returns: np.array of shape (n_paragraphs,)
    """
    if feature_mode == "puerto":
        X, _, _ = extract_features(paragraphs, feature_names, derived_config)
    else:
        X, _, eval_keys = extract_features(paragraphs)

        # Align features to match what transform_fn or scaler expects
        # transform_fn expects raw feature count (all_feature_keys), scaler expects reduced count
        if transform_fn is not None and all_feature_keys is not None:
            # Align to all_feature_keys (raw features) BEFORE transform_fn
            if eval_keys != all_feature_keys:
                eval_key_to_idx = {k: i for i, k in enumerate(eval_keys)}
                X_aligned = np.zeros((X.shape[0], len(all_feature_keys)), dtype=np.float32)
                for i, k in enumerate(all_feature_keys):
                    if k in eval_key_to_idx:
                        X_aligned[:, i] = X[:, eval_key_to_idx[k]]
                X = X_aligned
            X = transform_fn(X)
        elif transform_fn is not None:
            # No all_feature_keys -- try to align by column count
            if X.shape[1] < scaler.n_features_in_:
                X = np.pad(X, ((0, 0), (0, scaler.n_features_in_ - X.shape[1])), constant_values=0.0)
            X = transform_fn(X)
        else:
            # No transform_fn -- align directly to scaler
            expected_n = scaler.n_features_in_
            if X.shape[1] != expected_n:
                if all_feature_keys is not None and len(all_feature_keys) == expected_n:
                    eval_key_to_idx = {k: i for i, k in enumerate(eval_keys)}
                    X_aligned = np.zeros((X.shape[0], expected_n), dtype=np.float32)
                    for i, k in enumerate(all_feature_keys):
                        if k in eval_key_to_idx:
                            X_aligned[:, i] = X[:, eval_key_to_idx[k]]
                    X = X_aligned
                elif X.shape[1] < expected_n:
                    X = np.pad(X, ((0, 0), (0, expected_n - X.shape[1])), constant_values=0.0)
                else:
                    X = X[:, :expected_n]

    X_scaled = scaler.transform(X)

    # Use raw scores (logits) instead of probabilities -- this is critical for
    # the statistical tests (Mann-Whitney U, t-test) at document/collection level.
    # predict_proba compresses scores into [0,1], destroying the signal that
    # statistical tests rely on.
    try:
        is_xgb = isinstance(model, XGBClassifier)
    except ImportError:
        is_xgb = False

    if is_xgb:
        scores = model.predict(X_scaled, output_margin=True)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_scaled)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_scaled)[:, 1]
    else:
        scores = model.predict(X_scaled).astype(float)

    return scores


def evaluate_paragraph_level(model, scaler, B_members_docs, B_non_members_docs, 
                             feature_names, derived_config, feature_mode, 
                             all_feature_keys=None, transform_fn=None):
    """
    Paragraph-level evaluation: Direct AUROC on individual paragraphs.
    """
    # Flatten all paragraphs
    B_members_para = [para for doc in B_members_docs for para in doc]
    B_non_members_para = [para for doc in B_non_members_docs for para in doc]
    all_para = B_members_para + B_non_members_para
    
    scores = get_paragraph_scores(
        model, scaler, all_para, feature_names, derived_config, 
        feature_mode, all_feature_keys, transform_fn
    )
    
    y_true = np.array([1] * len(B_members_para) + [0] * len(B_non_members_para))
    
    auroc = roc_auc_score(y_true, scores)

    # TPR at low FPR thresholds
    fpr, tpr, _ = roc_curve(y_true, scores)
    tpr_at_5pct = np.interp(0.05, fpr, tpr)
    tpr_at_1pct = np.interp(0.01, fpr, tpr)
    tpr_at_0_1pct = np.interp(0.001, fpr, tpr)
    tpr_at_0_01pct = np.interp(0.0001, fpr, tpr)

    return {
        "auroc": auroc,
        "tpr_at_5pct_fpr": tpr_at_5pct,
        "tpr_at_1pct_fpr": tpr_at_1pct,
        "tpr_at_0.1pct_fpr": tpr_at_0_1pct,
        "tpr_at_0.01pct_fpr": tpr_at_0_01pct,
        "n_samples": len(y_true),
        "level": "paragraph",
        "_y_true": y_true,
        "_y_score": scores,
    }




def _doc_stat(doc_scores, known_scores, stat_test="mwu_ttest"):
    """Compute document-level statistic. Returns (statistic, pvalue)."""
    if stat_test == "brunnermunzel":
        result = brunnermunzel(doc_scores, known_scores, alternative='greater', distribution='t')
        return (-result.statistic, result.pvalue)
    else:
        return mannwhitneyu(doc_scores, known_scores, alternative='greater')


def _coll_stat(coll_scores, known_scores, stat_test="mwu_ttest"):
    """Compute collection-level statistic. Returns (statistic, pvalue)."""
    if stat_test == "brunnermunzel":
        result = brunnermunzel(coll_scores, known_scores, alternative='greater', distribution='t')
        return (-result.statistic, result.pvalue)
    else:
        return ttest_ind(coll_scores, known_scores, equal_var=True, alternative='greater')


def evaluate_document_level(model, scaler, B_members_docs, B_non_members_docs,
                            known_non_members_para, feature_names, derived_config,
                            feature_mode, all_feature_keys=None, transform_fn=None,
                            stat_test="mwu_ttest"):
    """
    Document-level evaluation using statistical test.

    For each document:
      1. Get paragraph scores
      2. Compare against known non-member paragraph scores
      3. Use test statistic as document score

    stat_test: 'mwu_ttest' (Mann-Whitney U) or 'brunnermunzel' (BM W-stat)
    """
    known_scores = get_paragraph_scores(
        model, scaler, known_non_members_para, feature_names, derived_config,
        feature_mode, all_feature_keys, transform_fn
    )

    doc_scores = []
    doc_labels = []

    for doc in B_members_docs:
        if len(doc) < 2:
            continue

        doc_para_scores = get_paragraph_scores(
            model, scaler, doc, feature_names, derived_config,
            feature_mode, all_feature_keys, transform_fn
        )

        try:
            statistic, _ = _doc_stat(doc_para_scores, known_scores, stat_test)
            if np.isfinite(statistic):
                doc_scores.append(statistic)
                doc_labels.append(1)
        except Exception:
            pass

    for doc in B_non_members_docs:
        if len(doc) < 2:
            continue

        doc_para_scores = get_paragraph_scores(
            model, scaler, doc, feature_names, derived_config,
            feature_mode, all_feature_keys, transform_fn
        )

        try:
            statistic, _ = _doc_stat(doc_para_scores, known_scores, stat_test)
            if np.isfinite(statistic):
                doc_scores.append(statistic)
                doc_labels.append(0)
        except Exception:
            pass

    doc_scores = np.array(doc_scores)
    doc_scores = np.nan_to_num(doc_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    doc_labels = np.array(doc_labels)

    if len(doc_labels) < 2 or len(np.unique(doc_labels)) < 2:
        return {
            "auroc": 0.5, "tpr_at_5pct_fpr": 0.0, "tpr_at_1pct_fpr": 0.0,
            "tpr_at_0.1pct_fpr": 0.0, "tpr_at_0.01pct_fpr": 0.0,
            "n_samples": len(doc_labels), "level": "document",
            "_y_true": doc_labels, "_y_score": doc_scores,
        }

    auroc = roc_auc_score(doc_labels, doc_scores)

    fpr, tpr, _ = roc_curve(doc_labels, doc_scores)
    tpr_at_5pct = np.interp(0.05, fpr, tpr)
    tpr_at_1pct = np.interp(0.01, fpr, tpr)
    tpr_at_0_1pct = np.interp(0.001, fpr, tpr)
    tpr_at_0_01pct = np.interp(0.0001, fpr, tpr)
    
    return {
        "auroc": auroc,
        "tpr_at_5pct_fpr": tpr_at_5pct,
        "tpr_at_1pct_fpr": tpr_at_1pct,
        "tpr_at_0.1pct_fpr": tpr_at_0_1pct,
        "tpr_at_0.01pct_fpr": tpr_at_0_01pct,
        "n_samples": len(doc_labels),
        "level": "document",
        "_y_true": doc_labels,
        "_y_score": doc_scores,
    }


def sample_collections(docs, n_collections, docs_per_collection, seed):
    """
    Sample collections of documents.
    Each collection is a list of docs_per_collection documents.
    Uses random.sample to match Puerto et al.'s original implementation.
    """
    import random
    rng = random.Random(seed)
    collections = []
    indices = list(range(len(docs)))

    for _ in range(n_collections):
        if len(docs) >= docs_per_collection:
            sampled = rng.sample(indices, docs_per_collection)
        else:
            sampled = [rng.choice(indices) for _ in range(docs_per_collection)]
        collection = [docs[i] for i in sampled]
        collections.append(collection)

    return collections


def evaluate_collection_level(model, scaler, B_members_docs, B_non_members_docs,
                              known_non_members_para, feature_names, derived_config,
                              feature_mode, config, seed,
                              all_feature_keys=None, transform_fn=None,
                              stat_test="mwu_ttest"):
    """
    Collection-level evaluation using statistical test.

    stat_test: 'mwu_ttest' (Student's t-test) or 'brunnermunzel' (BM W-stat)
    """
    n_collections = config.get("num_collections", 100)
    docs_per_collection = config.get("docs_per_collection", 50)

    known_scores = get_paragraph_scores(
        model, scaler, known_non_members_para, feature_names, derived_config,
        feature_mode, all_feature_keys, transform_fn
    )

    member_collections = sample_collections(B_members_docs, n_collections, docs_per_collection, seed)
    nonmember_collections = sample_collections(B_non_members_docs, n_collections, docs_per_collection, seed + 1)

    collection_scores = []
    collection_labels = []

    for collection in member_collections:
        collection_para = [para for doc in collection for para in doc]

        if len(collection_para) < 1:
            continue

        collection_para_scores = get_paragraph_scores(
            model, scaler, collection_para, feature_names, derived_config,
            feature_mode, all_feature_keys, transform_fn
        )

        try:
            statistic, _ = _coll_stat(collection_para_scores, known_scores, stat_test)
            if np.isfinite(statistic):
                collection_scores.append(statistic)
                collection_labels.append(1)
        except Exception:
            pass

    for collection in nonmember_collections:
        collection_para = [para for doc in collection for para in doc]

        if len(collection_para) < 1:
            continue

        collection_para_scores = get_paragraph_scores(
            model, scaler, collection_para, feature_names, derived_config,
            feature_mode, all_feature_keys, transform_fn
        )

        try:
            statistic, _ = _coll_stat(collection_para_scores, known_scores, stat_test)
            if np.isfinite(statistic):
                collection_scores.append(statistic)
                collection_labels.append(0)
        except Exception:
            pass

    collection_scores = np.array(collection_scores)
    collection_scores = np.nan_to_num(collection_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    collection_labels = np.array(collection_labels)

    if len(collection_labels) < 2 or len(np.unique(collection_labels)) < 2:
        return {
            "auroc": 0.5, "tpr_at_5pct_fpr": 0.0, "tpr_at_1pct_fpr": 0.0,
            "tpr_at_0.1pct_fpr": 0.0, "tpr_at_0.01pct_fpr": 0.0,
            "n_samples": len(collection_labels), "level": "collection",
            "docs_per_collection": docs_per_collection,
            "_y_true": collection_labels, "_y_score": collection_scores,
        }

    auroc = roc_auc_score(collection_labels, collection_scores)

    fpr, tpr, _ = roc_curve(collection_labels, collection_scores)
    tpr_at_5pct = np.interp(0.05, fpr, tpr)
    tpr_at_1pct = np.interp(0.01, fpr, tpr)
    tpr_at_0_1pct = np.interp(0.001, fpr, tpr)
    tpr_at_0_01pct = np.interp(0.0001, fpr, tpr)
    
    return {
        "auroc": auroc,
        "tpr_at_5pct_fpr": tpr_at_5pct,
        "tpr_at_1pct_fpr": tpr_at_1pct,
        "tpr_at_0.1pct_fpr": tpr_at_0_1pct,
        "tpr_at_0.01pct_fpr": tpr_at_0_01pct,
        "n_samples": len(collection_labels),
        "level": "collection",
        "docs_per_collection": docs_per_collection,
        "_y_true": collection_labels,
        "_y_score": collection_scores,
    }


# --------------------------------------------------------------
# Cell 6b: Advanced Analysis Functions
# --------------------------------------------------------------

def run_known_partition_sweep(model, scaler, B_members_docs, B_non_members_docs,
                              all_known_non_members_docs, feature_names, derived_config,
                              feature_mode, config, seed,
                              all_feature_keys=None, transform_fn=None,
                              known_sizes=None, collection_sizes=None,
                              stat_test="mwu_ttest"):
    """
    Sweep over known partition sizes (number of documents) and collection sizes.
    
    Replicates Puerto et al. Figure showing Collection-MIA AUROC
    vs Known Partition Size with different collection sizes.
    
    Args:
        all_known_non_members_docs: List of known non-member DOCUMENTS (not paragraphs)
        known_sizes: List of known partition sizes (# documents) to test
        collection_sizes: List of docs per collection
    
    Returns:
        DataFrame with columns: known_size, collection_size, auroc, tpr_at_1pct_fpr, ...
    """
    if known_sizes is None:
        known_sizes = [10, 25, 50, 100, 200, 300, 500, 750, 1000]
    if collection_sizes is None:
        collection_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500]
    
    n_collections = config.get("num_collections", 100)
    results = []
    
    total_known_docs = len(all_known_non_members_docs)
    print(f"\n  Known partition sweep: {len(known_sizes)} sizes × {len(collection_sizes)} collection sizes")
    print(f"    Total known documents available: {total_known_docs}")
    
    for known_size in known_sizes:
        # Subsample known documents
        if known_size > total_known_docs:
            print(f"    Skipping known_size={known_size} (only {total_known_docs} docs available)")
            continue
        
        rng = np.random.RandomState(seed)
        known_idx = rng.permutation(total_known_docs)[:known_size]
        known_docs_subset = [all_known_non_members_docs[i] for i in known_idx]
        
        # Flatten to paragraphs for scoring
        known_para_subset = [para for doc in known_docs_subset for para in doc]
        
        if len(known_para_subset) < 10:
            print(f"    Skipping known_size={known_size} (only {len(known_para_subset)} paragraphs)")
            continue
        
        # Get scores for this known subset
        known_scores = get_paragraph_scores(
            model, scaler, known_para_subset, feature_names, derived_config,
            feature_mode, all_feature_keys, transform_fn
        )
        
        for coll_size in collection_sizes:
            # Sample collections
            member_collections = sample_collections(B_members_docs, n_collections, coll_size, seed)
            nonmember_collections = sample_collections(B_non_members_docs, n_collections, coll_size, seed + 1)
            
            collection_scores = []
            collection_labels = []
            
            # Process member collections
            for collection in member_collections:
                collection_para = [para for doc in collection for para in doc]
                if len(collection_para) < 1:
                    continue
                
                coll_para_scores = get_paragraph_scores(
                    model, scaler, collection_para, feature_names, derived_config,
                    feature_mode, all_feature_keys, transform_fn
                )
                
                try:
                    statistic, _ = _coll_stat(coll_para_scores, known_scores, stat_test)
                    if np.isfinite(statistic):
                        collection_scores.append(statistic)
                        collection_labels.append(1)
                except:
                    pass

            # Process non-member collections
            for collection in nonmember_collections:
                collection_para = [para for doc in collection for para in doc]
                if len(collection_para) < 1:
                    continue

                coll_para_scores = get_paragraph_scores(
                    model, scaler, collection_para, feature_names, derived_config,
                    feature_mode, all_feature_keys, transform_fn
                )

                try:
                    statistic, _ = _coll_stat(coll_para_scores, known_scores, stat_test)
                    if np.isfinite(statistic):
                        collection_scores.append(statistic)
                        collection_labels.append(0)
                except:
                    pass
            
            if len(collection_scores) < 10:
                continue
            
            coll_scores = np.array(collection_scores)
            coll_scores = np.nan_to_num(coll_scores, nan=0.0, posinf=1e6, neginf=-1e6)
            coll_labels = np.array(collection_labels)
            
            auroc = roc_auc_score(coll_labels, coll_scores)
            fpr, tpr, _ = roc_curve(coll_labels, coll_scores)
            tpr_5 = np.interp(0.05, fpr, tpr)
            tpr_1 = np.interp(0.01, fpr, tpr)
            tpr_01 = np.interp(0.001, fpr, tpr)
            tpr_001 = np.interp(0.0001, fpr, tpr)

            results.append({
                "known_size": known_size,
                "collection_size": coll_size,
                "auroc": auroc,
                "tpr_at_5pct_fpr": tpr_5,
                "tpr_at_1pct_fpr": tpr_1,
                "tpr_at_0.1pct_fpr": tpr_01,
                "tpr_at_0.01pct_fpr": tpr_001,
                "n_samples": len(coll_labels),
            })
    
    return pd.DataFrame(results)


def run_token_count_analysis(model, scaler, B_members_docs, B_non_members_docs,
                             known_non_members_para, feature_names, derived_config,
                             feature_mode, ctx_size, all_feature_keys=None, transform_fn=None,
                             token_bins=None, stat_test="mwu_ttest"):
    """
    Analyze AUROC by document token count (number of paragraphs × context_size).
    
    Groups documents by their token count and computes AUROC for each bin.
    
    Args:
        ctx_size: Context size (512, 1024, 2048) to compute actual token count
    
    Returns:
        DataFrame with columns: token_bin, n_docs, auroc, tpr_at_1pct_fpr, ...
    """
    if token_bins is None:
        # Bins in actual tokens (paragraphs × ctx_size)
        token_bins = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), 
                      (4000, 5000), (5000, 6000), (6000, 8000), (8000, float('inf'))]
    
    # Get known scores
    known_scores = get_paragraph_scores(
        model, scaler, known_non_members_para, feature_names, derived_config,
        feature_mode, all_feature_keys, transform_fn
    )
    
    # Compute document scores with token counts
    doc_data = []
    
    for doc in B_members_docs:
        if len(doc) < 2:
            continue
        
        # Actual token count = n_paragraphs × ctx_size
        n_tokens = len(doc) * ctx_size
        doc_para_scores = get_paragraph_scores(
            model, scaler, doc, feature_names, derived_config,
            feature_mode, all_feature_keys, transform_fn
        )
        
        try:
            statistic, _ = _doc_stat(doc_para_scores, known_scores, stat_test)
            if np.isfinite(statistic):
                doc_data.append({"n_tokens": n_tokens, "n_paragraphs": len(doc), "score": statistic, "label": 1})
        except:
            pass

    for doc in B_non_members_docs:
        if len(doc) < 2:
            continue

        n_tokens = len(doc) * ctx_size
        doc_para_scores = get_paragraph_scores(
            model, scaler, doc, feature_names, derived_config,
            feature_mode, all_feature_keys, transform_fn
        )

        try:
            statistic, _ = _doc_stat(doc_para_scores, known_scores, stat_test)
            if np.isfinite(statistic):
                doc_data.append({"n_tokens": n_tokens, "n_paragraphs": len(doc), "score": statistic, "label": 0})
        except:
            pass
    
    df = pd.DataFrame(doc_data)
    results = []

    if df.empty:
        return pd.DataFrame(results)

    for low, high in token_bins:
        bin_df = df[(df["n_tokens"] >= low) & (df["n_tokens"] < high)]
        if len(bin_df) < 10:
            continue
        
        try:
            auroc = roc_auc_score(bin_df["label"], bin_df["score"])
            fpr, tpr, _ = roc_curve(bin_df["label"], bin_df["score"])
            tpr_5 = np.interp(0.05, fpr, tpr)
            tpr_1 = np.interp(0.01, fpr, tpr)
            tpr_01 = np.interp(0.001, fpr, tpr)

            results.append({
                "token_bin_low": low,
                "token_bin_high": high,
                "token_bin_mid": (low + min(high, df["n_tokens"].max())) / 2,
                "n_docs": len(bin_df),
                "auroc": auroc,
                "tpr_at_5pct_fpr": tpr_5,
                "tpr_at_1pct_fpr": tpr_1,
                "tpr_at_0.1pct_fpr": tpr_01,
            })
        except:
            pass
    
    return pd.DataFrame(results)


def run_collection_size_sweep(model, scaler, B_members_docs, B_non_members_docs,
                              known_non_members_para, feature_names, derived_config,
                              feature_mode, config, seed,
                              all_feature_keys=None, transform_fn=None,
                              collection_sizes=None, stat_test="mwu_ttest"):
    """
    Sweep over collection sizes to show compounding effect.
    
    Also returns paragraph-level AUROC as baseline.
    
    Returns:
        DataFrame with columns: collection_size, auroc, auroc_std, ...
        (collection_size=0 represents paragraph-level baseline)
    """
    if collection_sizes is None:
        collection_sizes = [1, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500]
    
    n_collections = config.get("num_collections", 100)
    
    # Get known scores
    known_scores = get_paragraph_scores(
        model, scaler, known_non_members_para, feature_names, derived_config,
        feature_mode, all_feature_keys, transform_fn
    )
    
    # First compute paragraph-level baseline
    B_members_para = [para for doc in B_members_docs for para in doc]
    B_non_members_para = [para for doc in B_non_members_docs for para in doc]
    all_para = B_members_para + B_non_members_para
    
    para_scores = get_paragraph_scores(
        model, scaler, all_para, feature_names, derived_config,
        feature_mode, all_feature_keys, transform_fn
    )
    para_labels = np.array([1] * len(B_members_para) + [0] * len(B_non_members_para))
    
    para_auroc = roc_auc_score(para_labels, para_scores)
    fpr, tpr, _ = roc_curve(para_labels, para_scores)
    para_tpr_5 = np.interp(0.05, fpr, tpr)
    para_tpr_1 = np.interp(0.01, fpr, tpr)
    para_tpr_01 = np.interp(0.001, fpr, tpr)

    results = [{
        "collection_size": 0,  # 0 = paragraph level
        "auroc": para_auroc,
        "tpr_at_5pct_fpr": para_tpr_5,
        "tpr_at_1pct_fpr": para_tpr_1,
        "tpr_at_0.1pct_fpr": para_tpr_01,
        "level": "paragraph",
    }]
    
    # Sweep collection sizes
    for coll_size in collection_sizes:
        member_collections = sample_collections(B_members_docs, n_collections, coll_size, seed)
        nonmember_collections = sample_collections(B_non_members_docs, n_collections, coll_size, seed + 1)
        
        collection_scores = []
        collection_labels = []
        
        for collection in member_collections:
            collection_para = [para for doc in collection for para in doc]
            if len(collection_para) < 1:
                continue
            
            coll_para_scores = get_paragraph_scores(
                model, scaler, collection_para, feature_names, derived_config,
                feature_mode, all_feature_keys, transform_fn
            )
            
            try:
                statistic, _ = _coll_stat(coll_para_scores, known_scores, stat_test)
                if np.isfinite(statistic):
                    collection_scores.append(statistic)
                    collection_labels.append(1)
            except:
                pass

        for collection in nonmember_collections:
            collection_para = [para for doc in collection for para in doc]
            if len(collection_para) < 1:
                continue

            coll_para_scores = get_paragraph_scores(
                model, scaler, collection_para, feature_names, derived_config,
                feature_mode, all_feature_keys, transform_fn
            )

            try:
                statistic, _ = _coll_stat(coll_para_scores, known_scores, stat_test)
                if np.isfinite(statistic):
                    collection_scores.append(statistic)
                    collection_labels.append(0)
            except:
                pass
        
        if len(collection_scores) < 10:
            continue
        
        coll_scores = np.array(collection_scores)
        coll_scores = np.nan_to_num(coll_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        coll_labels = np.array(collection_labels)
        
        auroc = roc_auc_score(coll_labels, coll_scores)
        fpr, tpr, _ = roc_curve(coll_labels, coll_scores)
        tpr_5 = np.interp(0.05, fpr, tpr)
        tpr_1 = np.interp(0.01, fpr, tpr)
        tpr_01 = np.interp(0.001, fpr, tpr)

        results.append({
            "collection_size": coll_size,
            "auroc": auroc,
            "tpr_at_5pct_fpr": tpr_5,
            "tpr_at_1pct_fpr": tpr_1,
            "tpr_at_0.1pct_fpr": tpr_01,
            "level": "collection",
        })
    
    return pd.DataFrame(results)


    # Plotting removed for HPC -- results saved to CSV/JSON instead


# --------------------------------------------------------------
# Cell 7: Main Experiment Loop
# --------------------------------------------------------------

def run_single_seed_experiment(members, non_members, seed, ctx_size, config, aggregation_levels,
                               pretuned_params=None):
    """
    Run a single experiment: train once, evaluate at all aggregation levels.

    Args:
        members: List of member documents
        non_members: List of non-member documents
        seed: Random seed for splitting
        ctx_size: Context size (512, 1024, 2048)
        config: Configuration dict
        aggregation_levels: List of levels to evaluate: ["paragraph", "document", "collection"]
        pretuned_params: Dict of {model_name: best_params} from prior CV tuning.
                         If provided, skips CV and uses these params directly.

    Returns:
        (seed_results, best_params_all) -- results list and tuned params (for reuse)
    """
    feature_mode = config.get("feature_mode", "puerto")
    
    # Split data
    splits = split_documents_puerto(members, non_members, config, seed)
    
    # Get training paragraphs and their document IDs (for StratifiedGroupKFold in CV)
    A_para = splits["A_members_para"] + splits["A_non_members_para"]
    A_doc_ids = splits["A_members_doc_ids"] + splits["A_non_members_doc_ids"]
    A_doc_ids = np.array(A_doc_ids)  # Convert to numpy array for sklearn
    
    # Extract features based on mode
    if feature_mode == "puerto":
        feature_names = config["puerto_features"]
        derived_config = config.get("derived_features", None)
        X_A, y_A, _ = extract_features(A_para, feature_names, derived_config)
        all_feature_keys = None
        transform_fn = None
        
    elif feature_mode in ("all", "reduced", "grouplda"):
        X_A, y_A, all_feature_keys = extract_features(A_para)
        derived_config = None
        
        if feature_mode == "reduced":
            # Apply reference filtering + GroupPCA for CAMIA
            B_para = [para for doc in splits["B_members_docs"] for para in doc]
            B_para += [para for doc in splits["B_non_members_docs"] for para in doc]
            X_B, _, _ = extract_features(B_para)
            
            n_pca = config.get("n_pca_components", 1)
            X_A, y_A, [X_B_reduced], reduced_keys, reduction_transform_fn = apply_feature_reduction(
                X_A, y_A, [X_B], all_feature_keys,
                n_pca_components=n_pca, seed=seed
            )
            
            # Use the transform function returned by apply_feature_reduction
            transform_fn = reduction_transform_fn
            feature_names = reduced_keys
            del X_B, B_para
            
        elif feature_mode == "grouplda":
            # Apply GroupLDA to all feature groups
            B_para = [para for doc in splits["B_members_docs"] for para in doc]
            B_para += [para for doc in splits["B_non_members_docs"] for para in doc]
            X_B, _, _ = extract_features(B_para)
            
            n_comp = config.get("grouplda_n_components", 1)
            n_lda_train = config.get("grouplda_n_lda_train", 2000)
            X_A, y_A, [X_B_reduced], reduced_keys = apply_group_lda(
                X_A, y_A, [X_B], all_feature_keys,
                n_components_per_group=n_comp, n_lda_train=n_lda_train, seed=seed
            )
            
            # Create transform function for later use
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            
            # Re-extract and re-fit for transform function
            X_train_full, y_train_full, _ = extract_features(A_para)
            groups, singletons = _parse_feature_groups_for_lda(all_feature_keys)
            key_to_idx = {k: i for i, k in enumerate(all_feature_keys)}
            
            # Subsample for LDA fitting
            rng = np.random.RandomState(seed)
            if n_lda_train and len(X_train_full) > n_lda_train:
                lda_idx = rng.permutation(len(X_train_full))[:n_lda_train]
                X_lda_fit = X_train_full[lda_idx]
                y_lda_fit = y_train_full[lda_idx]
            else:
                X_lda_fit = X_train_full
                y_lda_fit = y_train_full
            
            lda_models = {}
            singleton_indices = [key_to_idx[s] for s in singletons if s in key_to_idx]
            
            for gname, members_feat in groups.items():
                indices = [key_to_idx[m] for m in members_feat if m in key_to_idx]
                if len(indices) >= 2:
                    X_group = X_lda_fit[:, indices]
                    valid_cols = np.std(X_group, axis=0) > 1e-10
                    if valid_cols.sum() >= 2:
                        valid_indices = [indices[i] for i in range(len(indices)) if valid_cols[i]]
                        X_group = X_lda_fit[:, valid_indices]
                        lda = LinearDiscriminantAnalysis(n_components=n_comp)
                        try:
                            lda.fit(X_group, y_lda_fit)
                            lda_models[gname] = (lda, valid_indices)
                        except:
                            singleton_indices.extend(valid_indices)
                    else:
                        singleton_indices.extend(indices)
                else:
                    singleton_indices.extend(indices)
            
            def transform_fn(X_raw):
                parts = []
                for gname, (lda, indices) in lda_models.items():
                    X_grp = X_raw[:, indices]
                    parts.append(lda.transform(X_grp))
                if singleton_indices:
                    parts.append(X_raw[:, singleton_indices])
                return np.hstack(parts) if parts else X_raw
            
            feature_names = reduced_keys
            del X_B, B_para, X_train_full
            
        else:
            # "all" mode: just filter reference features, no reduction
            filtered_keys = filter_reference_features(all_feature_keys)
            key_to_orig_idx = {k: i for i, k in enumerate(all_feature_keys)}
            filtered_indices = [key_to_orig_idx[k] for k in filtered_keys]
            X_A = X_A[:, filtered_indices]
            feature_names = filtered_keys
            
            def transform_fn(X_raw):
                return X_raw[:, filtered_indices]
    
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    
    print(f"    Features: {X_A.shape[1]} (mode={feature_mode})")

    # -- Phase 1: Hyperparameter Tuning via CV ------------------
    tune_hyperparameters = config.get("tune_hyperparameters", False)
    n_cv_folds = config.get("n_cv_folds", 5)
    models_to_tune = config.get("models_to_tune", ["LogisticRegression", "RandomForest", "XGBoost"])

    # Scale data once for tuning
    scaler_for_tuning = StandardScaler()
    X_A_scaled = scaler_for_tuning.fit_transform(X_A)

    # Reuse pretuned params if provided (skip CV)
    best_params_all = {}
    tuning_results = []

    if pretuned_params is not None:
        best_params_all = pretuned_params
        print(f"    Using pretuned params from prior CV ({len(pretuned_params)} models)")

    elif tune_hyperparameters:
        # Use full training set for CV tuning (only runs once on largest train size)
        X_cv = X_A_scaled
        y_cv = y_A
        doc_ids_cv = A_doc_ids
        print(f"    CV Tuning ({n_cv_folds}-fold on all {len(X_A_scaled)} paragraphs)...")

        # Prepare Puerto features for PuertoLinearMap CV
        puerto_feat_names = config.get("puerto_features", [])
        puerto_derived = config.get("derived_features", None)
        A_para_for_cv = splits["A_members_para"] + splits["A_non_members_para"]
        X_A_puerto_cv, _, _ = extract_features(A_para_for_cv, puerto_feat_names, puerto_derived)
        scaler_puerto_cv = StandardScaler()
        X_puerto_cv_scaled = scaler_puerto_cv.fit_transform(X_A_puerto_cv)

        for model_name in models_to_tune:
            print(f"      Tuning {model_name}...", end=" ")

            # PuertoLinearMap tunes on Puerto 10 features, others on reduced features
            if model_name == "PuertoLinearMap":
                cv_X, cv_y = X_puerto_cv_scaled, y_A
            else:
                cv_X, cv_y = X_cv, y_A

            best_params, best_score, _ = tune_model_cv(
                model_name, cv_X, cv_y,
                groups=doc_ids_cv,
                n_folds=n_cv_folds, seed=seed
            )

            if best_params:
                best_params_all[model_name] = best_params
                print(f"CV-AUROC={best_score:.4f}, params={best_params}")

                tuning_results.append({
                    "model": model_name,
                    "seed": seed,
                    "context_size": ctx_size,
                    "best_cv_auroc": best_score,
                    "best_params": str(best_params),
                })
            else:
                print("(not tunable)")
    
    # -- Phase 2: Train all models on Full A Partition --
    print(f"    Training on full A partition...")
    seed_results = []
    model_names = ["PuertoLinearMap", "LogisticRegression", "SVC", "RandomForest", "XGBoost"]

    # Extract puerto features for PuertoLinearMap (with outlier trimming)
    puerto_feat_names = config.get("puerto_features", [])
    puerto_derived = config.get("derived_features", None)
    A_para_all = splits["A_members_para"] + splits["A_non_members_para"]
    X_A_puerto, y_A_puerto, _ = extract_features(A_para_all, puerto_feat_names, puerto_derived)

    for name in model_names:
        # -- Train each model --
        if name == "PuertoLinearMap":
            # LogReg tuned via CV on Puerto 10 features
            puerto_params = best_params_all.get("PuertoLinearMap", {})
            model = create_tuned_model("PuertoLinearMap", puerto_params, seed=seed)
            params_used = puerto_params
            scaler = StandardScaler()
            model.fit(scaler.fit_transform(X_A_puerto), y_A_puerto)
            # Puerto-specific eval settings
            cur_feature_mode = "puerto"
            cur_feature_names = puerto_feat_names
            cur_derived_config = puerto_derived
            cur_all_feature_keys = None
            cur_transform_fn = None

        else:
            # Standard sklearn models on full feature set
            if name in best_params_all:
                model = create_tuned_model(name, best_params_all[name], seed=seed)
                params_used = best_params_all[name]
            else:
                model = create_tuned_model(name, {}, seed=seed)
                params_used = {}
            scaler = StandardScaler()
            model.fit(scaler.fit_transform(X_A), y_A)
            cur_feature_mode = feature_mode
            cur_feature_names = feature_names
            cur_derived_config = derived_config
            cur_all_feature_keys = all_feature_keys
            cur_transform_fn = transform_fn

        # -- Save model interpretation (feature importances / coefficients) --
        # Only for the largest training size and first seed to avoid redundant files
        n_train = config["n_train_docs"]
        max_train_size = max(config.get("n_train_docs_sweep", [n_train]))
        if n_train == max_train_size and seed == config.get("seeds", [seed])[0]:
            output_dir = config.get("output_dir", "results")
            interp_data = []

            if name in ("LogisticRegression", "PuertoLinearMap"):
                # LogReg coefficients -- directly interpretable
                coefs = model.coef_[0]
                feat_names = cur_feature_names if cur_feature_mode == "puerto" else (feature_names or [f"f{i}" for i in range(len(coefs))])
                for fname, coef in zip(feat_names, coefs):
                    interp_data.append({"feature": fname, "importance": coef, "method": "coefficient"})

            elif name in ("RandomForest", "XGBoost"):
                importances = model.feature_importances_
                feat_names = feature_names or [f"f{i}" for i in range(len(importances))]
                for fname, imp in zip(feat_names, importances):
                    interp_data.append({"feature": fname, "importance": imp, "method": "feature_importance"})

            if interp_data:
                interp_df = pd.DataFrame(interp_data)
                interp_df = interp_df.sort_values("importance", key=abs, ascending=False)
                interp_path = os.path.join(output_dir, f"feature_importance_{name}_ctx{ctx_size}.csv")
                interp_df.to_csv(interp_path, index=False)

        # -- Phase 3: Evaluate on B Partition at ALL aggregation levels --
        for aggregation_level in aggregation_levels:
            if aggregation_level == "paragraph":
                result = evaluate_paragraph_level(
                    model, scaler,
                    splits["B_members_docs"], splits["B_non_members_docs"],
                    cur_feature_names, cur_derived_config, cur_feature_mode,
                    cur_all_feature_keys, cur_transform_fn
                )
            elif aggregation_level == "document":
                result = evaluate_document_level(
                    model, scaler,
                    splits["B_members_docs"], splits["B_non_members_docs"],
                    splits["known_non_members_para"],
                    cur_feature_names, cur_derived_config, cur_feature_mode,
                    cur_all_feature_keys, cur_transform_fn,
                    stat_test=config.get("stat_test", "mwu_ttest"),
                )
            elif aggregation_level == "collection":
                result = evaluate_collection_level(
                    model, scaler,
                    splits["B_members_docs"], splits["B_non_members_docs"],
                    splits["known_non_members_para"],
                    cur_feature_names, cur_derived_config, cur_feature_mode,
                    config, seed,
                    cur_all_feature_keys, cur_transform_fn,
                    stat_test=config.get("stat_test", "mwu_ttest"),
                )
            else:
                raise ValueError(f"Unknown aggregation_level: {aggregation_level}")

            # Save paragraph score distributions (first 1000 per class) for visualization
            # Shows how score distributions change across training sizes
            if aggregation_level == "paragraph":
                y_true_raw = result.get("_y_true")
                y_score_raw = result.get("_y_score")
                if y_true_raw is not None and y_score_raw is not None:
                    output_dir = config.get("output_dir", "results")
                    n_train = config["n_train_docs"]
                    member_mask = y_true_raw == 1
                    nonmember_mask = y_true_raw == 0
                    member_scores = y_score_raw[member_mask][:1000]
                    nonmember_scores = y_score_raw[nonmember_mask][:1000]
                    dist_df = pd.DataFrame({
                        "score": np.concatenate([member_scores, nonmember_scores]),
                        "label": ["member"] * len(member_scores) + ["nonmember"] * len(nonmember_scores),
                        "classifier": name,
                        "n_train_docs": n_train,
                        "seed": seed,
                        "context_size": ctx_size,
                    })
                    dist_path = os.path.join(
                        output_dir,
                        f"score_dist_{name}_ctx{ctx_size}_train{n_train}_seed{seed}.csv"
                    )
                    dist_df.to_csv(dist_path, index=False)

            # Extract raw predictions before building result entry
            y_true = result.pop("_y_true", None)
            y_score = result.pop("_y_score", None)

            # Save predictions CSV
            if y_true is not None and y_score is not None:
                output_dir = config.get("output_dir", "results")
                n_train = config["n_train_docs"]
                pred_df = pd.DataFrame({
                    "y_true": y_true,
                    "y_score": y_score,
                    "classifier": name,
                    "context_size": ctx_size,
                    "n_train_docs": n_train,
                    "aggregation_level": aggregation_level,
                    "seed": seed,
                    "param_type": "tuned" if name in best_params_all else "default",
                })
                pred_path = os.path.join(
                    output_dir,
                    f"predictions_{name}_{aggregation_level}_ctx{ctx_size}_train{n_train}_seed{seed}.csv"
                )
                pred_df.to_csv(pred_path, index=False)

            # Include tuning info in results
            result_entry = {
                "model": name,
                "seed": seed,
                "context_size": ctx_size,
                "n_train_docs": config["n_train_docs"],
                "aggregation_level": aggregation_level,
                "n_features": X_A.shape[1],
                "tuned": name in best_params_all,
                "params": str(params_used) if params_used else "default",
                **result,
            }

            # Add CV score if available
            if name in best_params_all:
                for tr in tuning_results:
                    if tr["model"] == name:
                        result_entry["cv_auroc"] = tr["best_cv_auroc"]
                        break

            seed_results.append(result_entry)
            print(f"      {name}/{aggregation_level}: AUROC={result['auroc']:.4f}")
    
    gc.collect()
    return seed_results, best_params_all


def aggregate_across_seeds(all_seed_results, ctx_size):
    """Aggregate results across seeds to get mean ± std."""
    df = pd.DataFrame(all_seed_results)
    
    agg_results = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        
        result = {
            "model": model,
            "context_size": ctx_size,
            "n_seeds": len(model_df),
            "auroc_mean": model_df["auroc"].mean(),
            "auroc_std": model_df["auroc"].std(),
            "tpr_at_1pct_fpr_mean": model_df["tpr_at_1pct_fpr"].mean(),
            "tpr_at_1pct_fpr_std": model_df["tpr_at_1pct_fpr"].std(),
            "level": model_df["level"].iloc[0],
        }
        
        # Add all TPR@FPR metrics if present
        for tpr_key in ["tpr_at_5pct_fpr", "tpr_at_0.1pct_fpr", "tpr_at_0.01pct_fpr"]:
            if tpr_key in model_df.columns:
                result[f"{tpr_key}_mean"] = model_df[tpr_key].mean()
                result[f"{tpr_key}_std"] = model_df[tpr_key].std()
        
        agg_results.append(result)
    
    return pd.DataFrame(agg_results).sort_values("auroc_mean", ascending=False)


def print_model_comparison(cv_summary, context_size, aggregation_level):
    """Print model comparison table."""
    df = cv_summary.sort_values("auroc_mean", ascending=False)
    print(f"\n  Model Comparison -- {aggregation_level.upper()}-level (ctx={context_size})")
    print(f"  {'Model':<25} {'AUROC':>14} {'TPR@1%FPR':>14}")
    print(f"  {'-'*55}")
    for _, row in df.iterrows():
        auroc_str = f"{row['auroc_mean']:.4f} +/- {row['auroc_std']:.4f}"
        tpr_str = f"{row['tpr_at_1pct_fpr_mean']:.4f} +/- {row['tpr_at_1pct_fpr_std']:.4f}"
        print(f"  {row['model']:<25} {auroc_str:>14} {tpr_str:>14}")




# --------------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------------

def main():
    args = parse_args()
    CONFIG = build_config(args)

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = CONFIG["output_dir"]

    print(f"\n{'='*60}")
    print(f"  CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Model             : {CONFIG['model_name']}")
    print(f"  Dataset           : {CONFIG['dataset_name']}")
    print(f"  Input Dir         : {CONFIG['input_dir']}")
    print(f"  Output Dir        : {OUTPUT_DIR}")
    print(f"  Aggregation Levels: {CONFIG['aggregation_levels']}")
    print(f"  Feature Mode      : {CONFIG['feature_mode']}")
    print(f"  Context Sizes     : {CONFIG['context_sizes']}")
    print(f"  Seeds             : {CONFIG['seeds']}")
    print(f"  HP Tuning         : {CONFIG['tune_hyperparameters']}")
    print(f"  Max Docs/Class    : {CONFIG.get('max_docs_per_class', 'all')}")
    print(f"{'='*60}\n")

    # Save config
    config_path = os.path.join(OUTPUT_DIR, f"config_{TIMESTAMP}.json")
    with open(config_path, "w") as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
                   for k, v in CONFIG.items()}, f, indent=2)

    all_seed_results = []
    all_aggregated_results = []

    for ctx_size in CONFIG["context_sizes"]:
        print("\n" + "=" * 60)
        print(f"  CONTEXT SIZE: {ctx_size} tokens")
        print("=" * 60)

        # Load Documents (once per context size)
        print(f"\nLoading Documents (ctx={ctx_size})...")
        members, non_members = load_data_for_context_size(ctx_size, CONFIG)
        print(f"  Feature mode: {CONFIG['feature_mode']}")
        print(f"  Aggregation levels: {CONFIG['aggregation_levels']}")



        # -- CV-only mode: tune hyperparameters, save to JSON, exit --
        if CONFIG.get("cv_only", False):
            print(f"\n  === CV-ONLY MODE ===")
            cv_seed = CONFIG["seeds"][0]
            cv_n_train = max(CONFIG.get("n_train_docs_sweep", [CONFIG["n_train_docs"]]))
            cv_config = {**CONFIG, "n_train_docs": cv_n_train, "tune_hyperparameters": True}

            print(f"  Seed: {cv_seed}, Train docs: {cv_n_train}")
            _, cv_params = run_single_seed_experiment(
                members, non_members, cv_seed, ctx_size, cv_config,
                aggregation_levels=["paragraph"],  # minimal eval, just need CV params
                pretuned_params=None
            )

            if cv_params:
                cv_out_path = os.path.join(OUTPUT_DIR, f"cv_params_ctx{ctx_size}.json")
                with open(cv_out_path, "w") as _f:
                    json.dump(cv_params, _f, indent=2)
                print(f"\n  Saved CV params to {cv_out_path}")
                print(f"  Models tuned: {list(cv_params.keys())}")
                for name, params in cv_params.items():
                    print(f"    {name}: {params}")
            else:
                print("  WARNING: No CV params returned!")

            print(f"\n  === CV-ONLY COMPLETE ===")
            continue  # skip to next context size (or exit)

        # Sweep over training sizes (CV only at the largest)
        # Models are trained once per (seed, train_size) and evaluated at all aggregation levels
        train_sizes = CONFIG.get("n_train_docs_sweep", [CONFIG["n_train_docs"]])
        # Load pretuned params from file if provided (skip CV entirely)
        tuned_params = None
        if CONFIG.get("pretuned_params_file"):
            with open(CONFIG["pretuned_params_file"]) as f:
                tuned_params = json.load(f)
            print(f"  Loaded pretuned params from {CONFIG['pretuned_params_file']} ({len(tuned_params)} models)")

        for n_train in train_sizes:
            print(f"\n  === n_train_docs={n_train} ===")
            sweep_config = {**CONFIG, "n_train_docs": n_train}

            # Only enable CV tuning for the first (largest) training size
            if n_train == train_sizes[0]:
                sweep_config["tune_hyperparameters"] = CONFIG.get("tune_hyperparameters", False)
            else:
                sweep_config["tune_hyperparameters"] = False

            agg_levels = CONFIG["aggregation_levels"]
            print(f"  Running {len(CONFIG['seeds'])} seeds, {len(agg_levels)} levels (train={n_train})...")

            train_size_results = []

            for seed_idx, seed in enumerate(CONFIG["seeds"]):
                print(f"\n    -- Seed {seed_idx + 1}/{len(CONFIG['seeds'])}: {seed} --")

                # Train once, evaluate at all aggregation levels
                seed_results, returned_params = run_single_seed_experiment(
                    members, non_members, seed, ctx_size, sweep_config, agg_levels,
                    pretuned_params=tuned_params
                )

                # Lock CV params after first seed of first (largest) training size
                if tuned_params is None and returned_params:
                    tuned_params = returned_params
                    print(f"    (CV params locked from n_train={n_train}: {list(tuned_params.keys())})")

                train_size_results.extend(seed_results)

            all_seed_results.extend(train_size_results)

            # Aggregate and save per aggregation level
            collection_aggregated = None  # Track collection-level for sweep best-model selection
            document_aggregated = None   # Track document-level for token count analysis
            for agg_level in agg_levels:
                level_results = [r for r in train_size_results if r["aggregation_level"] == agg_level]
                if not level_results:
                    continue

                aggregated = aggregate_across_seeds(level_results, ctx_size)
                aggregated["aggregation_level"] = agg_level
                aggregated["n_train_docs"] = n_train
                all_aggregated_results.append(aggregated)

                if agg_level == "collection":
                    collection_aggregated = aggregated
                if agg_level == "document":
                    document_aggregated = aggregated

                print(f"\n  Results ({agg_level}-level, train={n_train}, mean +/- std over {len(CONFIG['seeds'])} seeds):")
                for _, row in aggregated.iterrows():
                    tpr_01_str = ""
                    if "tpr_at_0.1pct_fpr_mean" in row:
                        tpr_01_str = f"  TPR@0.1%FPR={row['tpr_at_0.1pct_fpr_mean']:.4f}"
                    print(f"    {row['model']:25s}: AUROC={row['auroc_mean']:.4f}+/-{row['auroc_std']:.4f}  "
                          f"TPR@1%FPR={row['tpr_at_1pct_fpr_mean']:.4f}{tpr_01_str}")

                # Save level-specific results
                pd.DataFrame(level_results).to_csv(os.path.join(
                    OUTPUT_DIR, f"seed_results_{agg_level}_ctx{ctx_size}_train{n_train}_{TIMESTAMP}.csv"), index=False)
                aggregated.to_csv(os.path.join(
                    OUTPUT_DIR, f"aggregated_{agg_level}_ctx{ctx_size}_train{n_train}_{TIMESTAMP}.csv"), index=False)

                print_model_comparison(aggregated, ctx_size, agg_level)

        # Advanced Analyses (optional, run across all seeds for mean +/- std)
        if "collection" in agg_levels and CONFIG.get("run_known_partition_sweep") and collection_aggregated is not None:
            print(f"\n  Running known partition sweep ({len(CONFIG['seeds'])} seeds)...")
            best_model_name = collection_aggregated.iloc[0]["model"]
            best_model_params = tuned_params.get(best_model_name, {}) if tuned_params else {}
            feature_mode = CONFIG.get("feature_mode", "puerto")

            all_sweep_dfs = []
            for seed in CONFIG["seeds"]:
                splits = split_documents_puerto(members, non_members, CONFIG, seed)
                model = create_tuned_model(best_model_name, best_model_params, seed=seed)

                A_para = splits["A_members_para"] + splits["A_non_members_para"]
                sweep_feature_names = None
                sweep_derived_config = None
                sweep_all_feature_keys = None
                sweep_transform_fn = None

                if feature_mode == "puerto":
                    sweep_feature_names = CONFIG["puerto_features"]
                    sweep_derived_config = CONFIG.get("derived_features")
                    X_A, y_A, _ = extract_features(A_para, sweep_feature_names, sweep_derived_config)
                else:
                    X_A, y_A, sweep_all_feature_keys = extract_features(A_para)
                    if feature_mode == "reduced":
                        B_para = [p for doc in splits["B_members_docs"] for p in doc]
                        B_para += [p for doc in splits["B_non_members_docs"] for p in doc]
                        X_B_tmp, _, _ = extract_features(B_para)
                        n_pca = CONFIG.get("n_pca_components", 1)
                        X_A, y_A, [X_B_tmp], reduced_keys, sweep_transform_fn = apply_feature_reduction(
                            X_A, y_A, [X_B_tmp], sweep_all_feature_keys,
                            n_pca_components=n_pca, seed=seed)
                        del X_B_tmp
                        sweep_feature_names = reduced_keys

                scaler = StandardScaler()
                model.fit(scaler.fit_transform(X_A), y_A)

                sweep_df = run_known_partition_sweep(
                    model, scaler, splits["B_members_docs"], splits["B_non_members_docs"],
                    splits["all_known_for_sweep_docs"],
                    sweep_feature_names, sweep_derived_config,
                    feature_mode, CONFIG, seed,
                    all_feature_keys=sweep_all_feature_keys, transform_fn=sweep_transform_fn,
                    known_sizes=CONFIG.get("n_known_docs_sweep"),
                    collection_sizes=CONFIG.get("sweep_collection_sizes"),
                    stat_test=CONFIG.get("stat_test", "mwu_ttest"),
                )
                sweep_df["seed"] = seed
                all_sweep_dfs.append(sweep_df)
                print(f"    seed={seed} done")

            combined_sweep = pd.concat(all_sweep_dfs, ignore_index=True)
            sweep_path = os.path.join(OUTPUT_DIR, f"known_sweep_ctx{ctx_size}_{TIMESTAMP}.csv")
            combined_sweep.to_csv(sweep_path, index=False)
            print(f"  Saved: {sweep_path}")

        if "collection" in agg_levels and CONFIG.get("run_collection_size_sweep") and collection_aggregated is not None:
            print(f"\n  Running collection size sweep ({len(CONFIG['seeds'])} seeds)...")
            best_model_name = collection_aggregated.iloc[0]["model"]
            best_model_params = tuned_params.get(best_model_name, {}) if tuned_params else {}
            feature_mode = CONFIG.get("feature_mode", "puerto")

            all_coll_dfs = []
            for seed in CONFIG["seeds"]:
                splits = split_documents_puerto(members, non_members, CONFIG, seed)
                model = create_tuned_model(best_model_name, best_model_params, seed=seed)

                A_para = splits["A_members_para"] + splits["A_non_members_para"]
                coll_feature_names = None
                coll_derived_config = None
                coll_all_feature_keys = None
                coll_transform_fn = None

                if feature_mode == "puerto":
                    coll_feature_names = CONFIG["puerto_features"]
                    coll_derived_config = CONFIG.get("derived_features")
                    X_A, y_A, _ = extract_features(A_para, coll_feature_names, coll_derived_config)
                else:
                    X_A, y_A, coll_all_feature_keys = extract_features(A_para)
                    if feature_mode == "reduced":
                        B_para = [p for doc in splits["B_members_docs"] for p in doc]
                        B_para += [p for doc in splits["B_non_members_docs"] for p in doc]
                        X_B_tmp, _, _ = extract_features(B_para)
                        n_pca = CONFIG.get("n_pca_components", 1)
                        X_A, y_A, [X_B_tmp], _, coll_transform_fn = apply_feature_reduction(
                            X_A, y_A, [X_B_tmp], coll_all_feature_keys,
                            n_pca_components=n_pca, seed=seed)
                        del X_B_tmp

                scaler = StandardScaler()
                model.fit(scaler.fit_transform(X_A), y_A)

                coll_df = run_collection_size_sweep(
                    model, scaler, splits["B_members_docs"], splits["B_non_members_docs"],
                    splits["known_non_members_para"],
                    coll_feature_names, coll_derived_config,
                    feature_mode, CONFIG, seed,
                    all_feature_keys=coll_all_feature_keys, transform_fn=coll_transform_fn,
                    stat_test=CONFIG.get("stat_test", "mwu_ttest"),
                )
                coll_df["seed"] = seed
                all_coll_dfs.append(coll_df)
                print(f"    seed={seed} done")

            combined_coll = pd.concat(all_coll_dfs, ignore_index=True)
            coll_path = os.path.join(OUTPUT_DIR, f"collection_sweep_ctx{ctx_size}_{TIMESTAMP}.csv")
            combined_coll.to_csv(coll_path, index=False)
            print(f"  Saved: {coll_path}")

        if "document" in agg_levels and CONFIG.get("run_token_count_analysis") and document_aggregated is not None:
            print(f"\n  Running token count analysis ({len(CONFIG['seeds'])} seeds)...")
            best_model_name = document_aggregated.iloc[0]["model"]
            best_model_params = tuned_params.get(best_model_name, {}) if tuned_params else {}
            feature_mode = CONFIG.get("feature_mode", "puerto")

            all_token_dfs = []
            for seed in CONFIG["seeds"]:
                splits = split_documents_puerto(members, non_members, CONFIG, seed)
                model = create_tuned_model(best_model_name, best_model_params, seed=seed)

                A_para = splits["A_members_para"] + splits["A_non_members_para"]
                tok_feature_names = None
                tok_derived_config = None
                tok_all_feature_keys = None
                tok_transform_fn = None

                if feature_mode == "puerto":
                    tok_feature_names = CONFIG["puerto_features"]
                    tok_derived_config = CONFIG.get("derived_features")
                    X_A, y_A, _ = extract_features(A_para, tok_feature_names, tok_derived_config)
                else:
                    X_A, y_A, tok_all_feature_keys = extract_features(A_para)
                    if feature_mode == "reduced":
                        B_para = [p for doc in splits["B_members_docs"] for p in doc]
                        B_para += [p for doc in splits["B_non_members_docs"] for p in doc]
                        X_B_tmp, _, _ = extract_features(B_para)
                        n_pca = CONFIG.get("n_pca_components", 1)
                        X_A, y_A, [X_B_tmp], _, tok_transform_fn = apply_feature_reduction(
                            X_A, y_A, [X_B_tmp], tok_all_feature_keys,
                            n_pca_components=n_pca, seed=seed)
                        del X_B_tmp

                scaler = StandardScaler()
                model.fit(scaler.fit_transform(X_A), y_A)

                token_df = run_token_count_analysis(
                    model, scaler, splits["B_members_docs"], splits["B_non_members_docs"],
                    splits["known_non_members_para"],
                    tok_feature_names, tok_derived_config,
                    feature_mode, ctx_size,
                    all_feature_keys=tok_all_feature_keys, transform_fn=tok_transform_fn,
                    stat_test=CONFIG.get("stat_test", "mwu_ttest"),
                )
                token_df["seed"] = seed
                all_token_dfs.append(token_df)
                print(f"    seed={seed} done")

            combined_token = pd.concat(all_token_dfs, ignore_index=True)
            token_path = os.path.join(OUTPUT_DIR, f"token_count_analysis_ctx{ctx_size}_{TIMESTAMP}.csv")
            combined_token.to_csv(token_path, index=False)
            print(f"  Saved: {token_path}")

        # Cleanup per context size
        del members, non_members
        gc.collect()

    # -- Combined Results --------------------------------------
    if all_seed_results:
        all_df = pd.DataFrame(all_seed_results)
        all_df.to_csv(os.path.join(OUTPUT_DIR, f"all_seed_results_{TIMESTAMP}.csv"), index=False)

    if all_aggregated_results:
        combined_df = pd.concat(all_aggregated_results, ignore_index=True)
        combined_df.to_csv(os.path.join(OUTPUT_DIR, f"all_aggregated_{TIMESTAMP}.csv"), index=False)

        # Print summary table
        print(f"\n{'='*80}")
        print(f"  COMBINED RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Level':4s} | {'Model':20s} | {'Ctx':5s} | {'AUROC':>18s} | {'TPR@1%':>8s}")
        print("-" * 80)

        for agg_level in CONFIG["aggregation_levels"]:
            level_df = combined_df[combined_df["aggregation_level"] == agg_level]
            for ctx in CONFIG["context_sizes"]:
                ctx_df = level_df[level_df["context_size"] == ctx].sort_values("auroc_mean", ascending=False)
                for _, row in ctx_df.iterrows():
                    print(f"{agg_level[:4]:4s} | {row['model']:20s} | {int(ctx):5d} | "
                          f"{row['auroc_mean']:.4f} +/- {row['auroc_std']:.4f} | "
                          f"{row['tpr_at_1pct_fpr_mean']:.4f}")

    # Save JSON summary
    summary = {
        "timestamp": TIMESTAMP,
        "model": CONFIG["model_name"],
        "dataset": CONFIG["dataset_name"],
        "feature_mode": CONFIG["feature_mode"],
        "aggregation_levels": CONFIG["aggregation_levels"],
        "context_sizes": CONFIG["context_sizes"],
        "n_seeds": len(CONFIG["seeds"]),
    }
    with open(os.path.join(OUTPUT_DIR, f"summary_{TIMESTAMP}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
