#!/usr/bin/env python3
"""
Stage 2 of the aggregator pipeline: GridSearchCV per (dataset, ctx) on the
classifiers shortlisted by ``motivation_lazypredict.py``.

Inputs:
    configs/lazypredict_shortlist.json     (which classifiers to tune)
    configs/cv_grids.yaml                  (per-classifier search grids)
    $MIA_ROOT/<subset>/undefended/<model>/ctx<CTX>/{members,nonmembers}.jsonl

Outputs:
    configs/cv_params/<pythia_model>_<dataset>_<ctx>.json
        e.g. {"LogisticRegression": {"C": 0.01, "penalty": "l1", "solver": "saga"}, ...}
    The committed JSONs in this directory are the ones that back the thesis
    tables; new runs overwrite the matching file in place.

Usage:
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
    """Run GridSearchCV on `name` with `grid` on (X, y). Returns best params."""
    base = _build_estimator(name, {}, seed=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    print(f"  GridSearchCV: {name}  ({len(list(_param_grid_size(grid)))} configs)")
    gs = GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=False)
    gs.fit(X, y)
    print(f"    best ROC AUC = {gs.best_score_:.4f}  with {gs.best_params_}")
    return gs.best_params_


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
    out = {}
    for clf_name in shortlist:
        if clf_name not in grids:
            print(f"  [skip] no grid in {args.grids} for {clf_name}")
            continue
        try:
            out[clf_name] = grid_search(clf_name, X, y, grids[clf_name], seed=args.seed)
        except Exception as e:
            print(f"  ! {clf_name} failed: {e}")

    out_path = Path(args.out_dir) / f"{args.pythia_model}_{args.dataset}_{args.ctx}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
