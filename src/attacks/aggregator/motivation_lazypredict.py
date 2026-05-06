#!/usr/bin/env python3
"""
Stage 1 of the aggregator pipeline: run lazypredict on the 644-feature MIA
matrix to identify which sklearn-compatible classifiers separate members from
non-members at all. The shortlist is consumed by ``cv_params.py`` to focus
GridSearchCV only on classifiers worth tuning.

Outputs:
    configs/lazypredict_shortlist.json
        ["LogisticRegression", "SVC", "RandomForest", "XGBoost", "MLP", ...]
    paper/tables/lazypredict_ranking_<dataset>_ctx<CTX>.tex
        ranking table for the thesis appendix

Inputs:
    Merged MIA score JSONL files at:
        $MIA_ROOT/<subset>/undefended/pythia-2.8b/ctx<CTX>/members.jsonl
        $MIA_ROOT/<subset>/undefended/pythia-2.8b/ctx<CTX>/nonmembers.jsonl

Usage:
    python -m src.attacks.aggregator.motivation_lazypredict \\
        --dataset arxiv --ctx 1024 --pythia_model pythia-2.8b
"""
import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MIA_ROOT = Path(os.environ.get("MIA_ROOT", "./mia_scores"))
CONFIGS = Path(__file__).resolve().parents[3] / "configs"
PAPER = Path(__file__).resolve().parents[3] / "paper"


def _load_jsonl(path: Path):
    """Read a (possibly gzipped) JSONL of {pred, label} chunks."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def load_features(subset: str, pythia_model: str, ctx: int):
    """Returns (X, y, feature_names). Each row = one chunk's flat feature vec."""
    base = MIA_ROOT / subset / "undefended" / pythia_model / f"ctx{ctx}"
    rows, labels = [], []
    for side, label in (("members", 1), ("nonmembers", 0)):
        path = base / f"{side}.jsonl"
        if not path.exists():
            path = base / f"{side}.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(f"merged MIA file missing: {path}")
        for obj in _load_jsonl(path):
            preds = obj.get("pred", obj.get("preds", []))
            if isinstance(preds, dict):
                preds = [preds]
            for chunk in preds:
                if not isinstance(chunk, dict):
                    continue
                rows.append(chunk)
                labels.append(label)

    df = pd.DataFrame(rows).fillna(0.0)
    feature_names = list(df.columns)
    X = df.values.astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, feature_names


def run_lazypredict(X, y, top_k: int = 10):
    """Train ~30 classifiers via lazypredict and return the top-k by AUROC."""
    from lazypredict.Supervised import LazyClassifier
    from sklearn.model_selection import train_test_split

    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, preds = clf.fit(X_tr, X_te, y_tr, y_te)
    return models.sort_values("ROC AUC", ascending=False).head(top_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pythia_model", default="pythia-2.8b")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_shortlist", default=str(CONFIGS / "lazypredict_shortlist.json"))
    args = ap.parse_args()

    print(f"Loading features for {args.dataset} / {args.pythia_model} / ctx{args.ctx}")
    X, y, _ = load_features(args.dataset, args.pythia_model, args.ctx)
    print(f"  X.shape = {X.shape}  (N={len(y)}, members={(y==1).sum()})")

    print(f"\nRunning lazypredict (top {args.top_k}) ...")
    ranking = run_lazypredict(X, y, top_k=args.top_k)
    print(ranking[["ROC AUC", "Accuracy", "F1 Score"]].to_string())

    # Write the shortlist (one per line) for cv_params.py to consume.
    shortlist = ranking.index.tolist()
    Path(args.out_shortlist).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_shortlist).write_text(json.dumps(shortlist, indent=2))
    print(f"\n-> shortlist saved to {args.out_shortlist}")

    # Latex table for the thesis appendix.
    tex_dir = PAPER / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / f"lazypredict_ranking_{args.dataset}_ctx{args.ctx}.tex"
    ranking.to_latex(tex_path, float_format="%.3f")
    print(f"-> LaTeX ranking saved to {tex_path}")


if __name__ == "__main__":
    main()
