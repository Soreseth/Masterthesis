#!/usr/bin/env python3
"""
GroupLDA aggregation variant.

Treats each chunk's 644-dim feature vector as observations grouped by document
and applies Linear Discriminant Analysis to project onto a 1-D
member/non-member axis. The per-chunk projection is then aggregated to the
document level via mean / median / max.

Useful as an ablation against:
    - extended_aggregator.py (5 separate sklearn classifiers per cell)
    - majority_voting_agg.py (per-feature Youden-J thresholds + voting)

Inputs:
    Same as extended_aggregator.py -- merged MIA jsonls under
    $MIA_ROOT/<subset>/{undefended/<model> | defended/<name>}/ctx<CTX>/

Output (PCS file):
    $MIA_ROOT/<subset>/aggregation/<model>/ctx<CTX>/GroupLDA/
        train<N>_known<K>_seed<S>_<aggregation_method>.json

Usage:
    python -m src.attacks.aggregator.group_lda \\
        --dataset arxiv --ctx 1024 --pythia_model pythia-2.8b \\
        --n_train 1000 --n_known 1000 --seed 670487
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

MIA_ROOT = Path(os.environ.get("MIA_ROOT", "./mia_scores"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pythia_model", default="pythia-2.8b")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_known", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=670487)
    ap.add_argument("--defended", default=None)
    ap.add_argument("--aggregation_method", default="brunner_munzel",
                    choices=["mwu_ttest", "brunner_munzel"])
    args = ap.parse_args()

    raise NotImplementedError(
        "GroupLDA is a thesis ablation. Port the pipeline from "
        "diagnose_glda.py + aggregate_pile_noglda.slurm in the legacy Scripts/ "
        "tree, plug in the same split_documents_puerto / extract_features path "
        "as extended_aggregator.py, then drop in:\n"
        "    lda = LinearDiscriminantAnalysis(n_components=1)\n"
        "    lda.fit(X_A, y_A)\n"
        "    chunk_scores = lda.transform(X_B).ravel()\n"
        "Write results to $MIA_ROOT/<subset>/aggregation/<model>/ctx<X>/GroupLDA/...")


if __name__ == "__main__":
    main()
