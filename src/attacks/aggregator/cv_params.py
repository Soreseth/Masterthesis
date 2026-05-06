#!/usr/bin/env python3
"""
Find the most optimal hyperparameter for the aggregator on the training dataset with GridSearchCV per (dataset, ctx) on the
classifiers shortlisted by ``motivation_lazypredict.py``.

Example Usage:
    python -m src.attacks.aggregator.cv_params \\
        --dataset arxiv --ctx 1024 --pythia_model pythia-2.8b
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.attacks.aggregator.motivation_lazypredict import load_features

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def _build_estimator(name: str, params: dict, seed: int = 42):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=5000, random_state=seed, **params)
    if name == "SVC":
        return SVC(probability=False, random_state=seed, **params)
    if name == "RandomForest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="logloss",
                             verbosity=0, **params)
    raise ValueError(f"Unknown classifier {name!r}. "
                     f"Add it to _build_estimator (and configs/cv_grids.yaml).")


def grid_search(name: str, X, y, grid: dict, n_splits: int = 5, seed: int = 42):
    """
    Run GridSearchCV on `name` with `grid` on (X, y). Returns
    (best_params, cv_results) where cv_results is a list of one dict per
    candidate configuration (params + per-fold scores + mean/std/rank).
    """
    base = _build_estimator(name, {}, seed=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    print(f"  GridSearchCV: {name}  ({len(list(_param_grid_size(grid)))} configs)")
    gs = GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=False)
    gs.fit(X, y)
    print(f"    best ROC AUC = {gs.best_score_:.4f}  with {gs.best_params_}")

    cvr = gs.cv_results_
    n = len(cvr["params"])
    split_keys = [f"split{i}_test_score" for i in range(n_splits)]
    cv_results = [
        {
            "params": cvr["params"][i],
            "mean_test_score": float(cvr["mean_test_score"][i]),
            "std_test_score":  float(cvr["std_test_score"][i]),
            "rank_test_score": int(cvr["rank_test_score"][i]),
            "split_test_scores": [float(cvr[k][i]) for k in split_keys],
        }
        for i in range(n)
    ]
    return gs.best_params_, cv_results


def _param_grid_size(grid: dict):
    n = 1
    for v in grid.values():
        n *= max(1, len(v))
    return [n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pythia_model", default="pythia-2.8b")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--shortlist", default=str(CONFIGS / "lazypredict_shortlist.json"))
    ap.add_argument("--grids", default=str(CONFIGS / "cv_grids.yaml"))
    ap.add_argument("--out_dir", default=str(CONFIGS / "cv_params"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading features for {args.dataset} / {args.pythia_model} / ctx{args.ctx}")
    X, y, _ = load_features(args.dataset, args.pythia_model, args.ctx)
    X = StandardScaler().fit_transform(X)
    print(f"  X.shape = {X.shape}")

    with open(args.shortlist) as f:
        shortlist = json.load(f)
    with open(args.grids) as f:
        grids = yaml.safe_load(f)

    print(f"\nClassifiers to tune ({len(shortlist)}): {shortlist}")
    best_params_per_clf = {}
    cv_results_per_clf = {}
    for clf_name in shortlist:
        if clf_name not in grids:
            print(f"  [skip] no grid in {args.grids} for {clf_name}")
            continue
        try:
            best_params, cv_results = grid_search(
                clf_name, X, y, grids[clf_name], seed=args.seed)
            best_params_per_clf[clf_name] = best_params
            cv_results_per_clf[clf_name] = cv_results
        except Exception as e:
            print(f"  ! {clf_name} failed: {e}")

    base = f"{args.pythia_model}_{args.dataset}_{args.ctx}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Existing schema: {clf_name: best_params}. extended_aggregator's loader
    # depends on this shape, so don't change it.
    best_path = out_dir / f"{base}.json"
    best_path.write_text(json.dumps(best_params_per_clf, indent=2, default=str))
    print(f"\n-> {best_path}")

    # Sidecar with full per-config CV scores (the diagram's "Validation Results").
    cvr_path = out_dir / f"{base}.cv_results.json"
    cvr_path.write_text(json.dumps({
        "dataset": args.dataset,
        "pythia_model": args.pythia_model,
        "ctx": args.ctx,
        "seed": args.seed,
        "scoring": "roc_auc",
        "cv_results": cv_results_per_clf,
    }, indent=2, default=str))
    print(f"-> {cvr_path}")


if __name__ == "__main__":
    main()
