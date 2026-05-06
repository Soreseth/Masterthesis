#!/usr/bin/env python3
"""
Precompute per-paragraph MIA scores using ALL chunks per document with extended features.

Example Usage:
    python precompute_paragraph_scores_all_chunk.py --dataset arxiv --seed 670487 --ctx 2048
"""
import sys, os, json, re, argparse, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.aggregate import (load_jsonl_documents, split_documents_puerto,
                                  extract_features, apply_feature_reduction)

ALL_SEEDS = [670487, 116739, 26225, 777572, 288389]
CTX = 2048
N_TRAIN = int(os.environ.get("N_TRAIN", 1000))
N_KNOWN = 1000

PUERTO_FEATURES = [
    "ppl", "ppl_lowercase_ratio", "ppl_zlib_ratio",
    "min_k_5", "min_k_10", "min_k_20", "min_k_30",
    "min_k_40", "min_k_50", "min_k_60",
]
DERIVED_FEATURES = {
    "ppl_lowercase_ratio": ("ppl", "lowercase_ppl", "divide"),
    "ppl_zlib_ratio": ("ppl", "zlib", "divide"),
}
PLM_PARAMS = {"C": 0.1, "penalty": "l1", "solver": "saga"}

# Sklearn-default fallbacks used when no <model>_<dataset>_<ctx>.json exists
# yet for this (model, dataset, ctx). See the 6.9b sister script for the
# rationale -- without these, only PuertoLinearMap would be trained.
DEFAULT_PARAMS = {
    "LogisticRegression": {"C": 1.0,  "penalty": "l2", "solver": "lbfgs"},
    "SVC":                {"C": 1.0,  "kernel": "rbf", "gamma": "scale"},
    "RandomForest":       {"n_estimators": 200, "max_depth": None,
                            "min_samples_leaf": 1, "n_jobs": -1},
    "XGBoost":            {"n_estimators": 200, "max_depth": 6,
                            "learning_rate": 0.1, "subsample": 1.0,
                            "colsample_bytree": 1.0},
}

_MIA_ROOT = os.environ.get("MIA_ROOT", "./mia_scores")
MIA_DIR = os.environ.get("MIA_DIR", os.path.join(_MIA_ROOT, "pythia-2.8b"))
OUT_DIR = os.environ.get("ALL_CHUNK_DIR", os.path.join(_MIA_ROOT, "results", "all_chunk"))
CV_DIR = os.environ.get(
    "CV_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "configs", "cv_params"),
)


def load_cv_params(dataset, pythia_model="pythia-2.8b"):
    """Load tuned CV hyperparameters for the current (model, dataset, CTX).

    Reads from `configs/cv_params/<pythia_model>_<dataset>_<ctx>.json`
    (the committed thesis-table values). Falls back to ctx=512 of the same
    (model, dataset) when the requested ctx isn't tuned yet -- useful at
    ctx=43, which shares the per-paragraph feature schema with ctx=512.
    Returns {} if neither exists, which makes the per-classifier defaults
    in DEFAULT_PARAMS kick in for every estimator.
    """
    candidates = [CTX]
    if CTX != 512:
        candidates.append(512)
    for ctx_to_try in candidates:
        path = f"{CV_DIR}/{pythia_model}_{dataset}_{ctx_to_try}.json"
        if os.path.exists(path):
            with open(path) as f:
                params = json.load(f)
            if ctx_to_try != CTX:
                print(f"    [INFO] {pythia_model}_{dataset}_{CTX}.json missing -- "
                      f"falling back to {pythia_model}_{dataset}_{ctx_to_try}.json")
            return params
    return {}


def create_model(model_name, params, seed):
    if model_name in ("LogisticRegression", "PuertoLinearMap"):
        return LogisticRegression(max_iter=5000, random_state=seed, **params)
    elif model_name == "SVC":
        return SVC(probability=False, random_state=seed, **params)
    elif model_name == "RandomForest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    elif model_name == "XGBoost":
        return XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="logloss",
                             verbosity=0, **params)


def get_scores(model, model_name, X_scaled):
    if hasattr(model, 'decision_function'):
        return model.decision_function(X_scaled).tolist()
    return model.predict_proba(X_scaled)[:, 1].tolist()


def run_seed(dataset, seed, members, non_members, cv_params):
    print(f"\n  --- Seed {seed} ---")

    config = {
        "n_train_docs": N_TRAIN,
        "n_known_docs": N_KNOWN,
        "n_train_docs_sweep": [max(N_TRAIN, 1000)],
        "n_known_docs_sweep": [N_KNOWN],
    }

    splits = split_documents_puerto(members, non_members, config, seed)

    A_para = splits["A_members_para"] + splits["A_non_members_para"]
    y_A = np.array([1]*len(splits["A_members_para"]) + [0]*len(splits["A_non_members_para"]))

    X_A_raw, _, all_feature_keys = extract_features(A_para)

    B_para = [p for doc in splits["B_members_docs"] for p in doc]
    B_para += [p for doc in splits["B_non_members_docs"] for p in doc]
    known_para = splits["known_non_members_para"]
    X_B_raw, _, _ = extract_features(B_para)
    X_K_raw, _, _ = extract_features(known_para)

    n_pca = 1
    X_A_reduced, y_A, [X_B_reduced, X_K_reduced], reduced_keys_pca, transform_fn = apply_feature_reduction(
        X_A_raw, y_A, [X_B_raw, X_K_raw], all_feature_keys,
        n_pca_components=n_pca, seed=seed
    )
    del X_B_raw, X_K_raw, B_para

    scaler_ext = StandardScaler()
    X_A_ext_scaled = scaler_ext.fit_transform(X_A_reduced)

    X_A_prt, _, _ = extract_features(A_para, PUERTO_FEATURES, DERIVED_FEATURES)
    scaler_prt = StandardScaler()
    X_A_prt_scaled = scaler_prt.fit_transform(X_A_prt)

    models_to_train = {}
    models_to_train["PuertoLinearMap"] = {"params": PLM_PARAMS, "is_puerto": True}
    for m_name in ["LogisticRegression", "SVC", "RandomForest", "XGBoost"]:
        if m_name in cv_params:
            models_to_train[m_name] = {"params": cv_params[m_name], "is_puerto": False}
        elif m_name in DEFAULT_PARAMS:
            print(f"    [INFO] {m_name}: cv_params missing, using sklearn defaults")
            models_to_train[m_name] = {"params": DEFAULT_PARAMS[m_name], "is_puerto": False}

    trained = {}
    training_results = {}
    for m_name, m_info in models_to_train.items():
        model = create_model(m_name, m_info["params"], seed)
        X_train = X_A_prt_scaled if m_info["is_puerto"] else X_A_ext_scaled
        t0 = time.perf_counter()
        model.fit(X_train, y_A)
        fit_seconds = time.perf_counter() - t0
        train_scores = get_scores(model, m_name, X_train)
        train_auroc = float(roc_auc_score(y_A, train_scores))
        trained[m_name] = {
            "model": model,
            "scaler": scaler_prt if m_info["is_puerto"] else scaler_ext,
            "is_puerto": m_info["is_puerto"],
            "transform_fn": None if m_info["is_puerto"] else transform_fn,
        }
        training_results[m_name] = {
            "train_auroc": train_auroc,
            "fit_seconds": float(fit_seconds),
            "n_train_paragraphs": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_pos": int((y_A == 1).sum()),
            "n_neg": int((y_A == 0).sum()),
            "feature_set": "puerto" if m_info["is_puerto"] else "extended",
            "params": m_info["params"],
        }
        print(f"    Trained {m_name}  (train AUROC = {train_auroc:.4f}, "
              f"fit = {fit_seconds:.1f}s)")

    known_scores = {}
    for m_name, m_info in trained.items():
        if m_info["is_puerto"]:
            X_k, _, _ = extract_features(known_para, PUERTO_FEATURES, DERIVED_FEATURES)
        else:
            X_k = X_K_reduced
        X_k_scaled = m_info["scaler"].transform(X_k)
        known_scores[m_name] = get_scores(m_info["model"], m_name, X_k_scaled)
    print(f"    Known scores: {len(known_para)} paragraphs")

    def score_docs(docs, label):
        results = []
        for doc_idx, doc in enumerate(docs):
            if len(doc) == 0:
                continue
            doc_scores = {}
            for m_name, m_info in trained.items():
                if m_info["is_puerto"]:
                    X_d, _, _ = extract_features(doc, PUERTO_FEATURES, DERIVED_FEATURES)
                else:
                    X_d_raw, _, _ = extract_features(doc)
                    n_expected = len(all_feature_keys)
                    if X_d_raw.shape[1] < n_expected:
                        X_d_raw = np.pad(X_d_raw, ((0, 0), (0, n_expected - X_d_raw.shape[1])), constant_values=0.0)
                    X_d = m_info["transform_fn"](X_d_raw)
                X_d_scaled = m_info["scaler"].transform(X_d)
                doc_scores[m_name] = get_scores(m_info["model"], m_name, X_d_scaled)
            results.append({
                "doc_id": doc_idx,
                "label": label,
                "n_paragraphs": len(doc),
                "scores": doc_scores,
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
        "models": list(trained.keys()),
        "model_params": {m: models_to_train[m]["params"] for m in trained},
        "training_results": training_results,
        "known_scores": known_scores,
        "eval_members": eval_members,
        "eval_non_members": eval_non_members,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pythia_model", default="pythia-2.8b",
                        choices=["pythia-2.8b", "pythia-6.9b"],
                        help="Picks which configs/cv_params/<model>_<dataset>_<ctx>.json "
                             "to load tuned hyperparameters from.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ctx", type=int, default=2048)
    args = parser.parse_args()

    global CTX
    CTX = args.ctx

    print(f"{'='*60}")
    print(f"  PCS All-Chunk Extended: {args.dataset} ctx={CTX}")
    print(f"{'='*60}")

    mem_file = f"{MIA_DIR}/{args.dataset}/document_{CTX}/members_{CTX}.jsonl"
    non_file = f"{MIA_DIR}/{args.dataset}/document_{CTX}/nonmembers_{CTX}.jsonl"
    members = load_jsonl_documents(mem_file, label=1)
    non_members = load_jsonl_documents(non_file, label=0)
    print(f"  Loaded {len(members)} mem, {len(non_members)} non-mem docs (all chunks)")

    cv_params = load_cv_params(args.dataset, args.pythia_model)
    print(f"  CV params for: {list(cv_params.keys())}")

    seeds = [args.seed] if args.seed else ALL_SEEDS
    out_dir = f"{OUT_DIR}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    for seed in seeds:
        result = run_seed(args.dataset, seed, members, non_members, cv_params)

        out_path = f"{out_dir}/precomputed_scores_ctx{CTX}_train{N_TRAIN}_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"    Saved to {out_path}")
        print(f"    File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
