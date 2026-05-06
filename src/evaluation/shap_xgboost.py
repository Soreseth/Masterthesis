#!/usr/bin/env python3
"""Aggregate per-seed SHAP files for one or more datasets.

Reads:  {OUT_DIR}/{dataset}/per_seed/seed*_train{N_TRAIN}_ctx{CTX}.json
Writes per dataset:
  {OUT_DIR}/{dataset}/shap_mean_abs_xgboost_train{N_TRAIN}_ctx{CTX}.json
  {OUT_DIR}/{dataset}/shap_mean_abs_xgboost_train{N_TRAIN}_ctx{CTX}.csv
  {OUT_DIR}/{dataset}/shap_top{TOP_N}_xgboost_train{N_TRAIN}_ctx{CTX}.png
  {OUT_DIR}/{dataset}/shap_bottom{BOTTOM_N}_xgboost_train{N_TRAIN}_ctx{CTX}.png
"""
import os, json, csv, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.environ.get(
    "SHAP_OUT_DIR",
    os.path.join(os.environ.get("MIA_ROOT", "./mia_scores"),
                 "results", "interpretability", "shap_43"),
)
ALL_SEEDS = [670487, 116739, 26225, 777572, 288389]
N_TRAIN = 1000


def plot_bar(names, values, out_path, xlim=None):
    """Save a horizontal bar plot of feature importances on a log x-axis.

    Args:
        names: Iterable of feature names (top-to-bottom in the plot).
        values: Iterable of mean(|SHAP|) values aligned with `names`.
        out_path: PNG file path to write.
        xlim: Optional (xmin, xmax) for the log x-axis. If None, matplotlib
            picks the bounds.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(names))))
    ax.barh(range(len(names)), values, color="#1f77b4")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel("mean(|SHAP value|) [log scale]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}", flush=True)


def common_log_xlim(values, pad=1.15):
    """Pick (xmin, xmax) for a log axis covering the strictly-positive entries.

    Args:
        values: 1-D numpy array of feature importances (may contain zeros).
        pad: Multiplicative padding factor applied to the min (divide) and
            max (multiply) so bars don't touch the axis edges.

    Returns:
        Tuple (xmin, xmax), or None if `values` has no positive entries.
    """
    nz = values[values > 0]
    if nz.size == 0:
        return None
    return (nz.min() / pad, nz.max() * pad)


def aggregate_dataset(dataset, ctx, top_n, bottom_n, expected_seeds):
    """Average the per-seed SHAP files for one dataset and emit JSON/CSV/PNG.

    Reads ``{OUT_DIR}/{dataset}/per_seed/seed*_train{N_TRAIN}_ctx{ctx}.json``,
    averages the per-feature mean(|SHAP|) across seeds, and writes the
    seed-averaged JSON, a sorted CSV, and bar plots for the top-N and
    bottom-N features.

    Args:
        dataset: Pile subset name, e.g. ``"arxiv"``.
        ctx: Context length (43, 512, 1024, 2048).
        top_n: Number of highest-importance features to plot.
        bottom_n: Number of lowest-importance (still positive) features to plot.
        expected_seeds: Iterable of seed ids that should be present; missing
            seeds trigger a WARNING but are not fatal.
    """
    in_dir = f"{OUT_DIR}/{dataset}/per_seed"
    pattern = f"{in_dir}/seed*_train{N_TRAIN}_ctx{ctx}.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[{dataset}] no per-seed files matching {pattern}", flush=True)
        return
    print(f"[{dataset}] aggregating {len(files)} per-seed files", flush=True)

    per_seed = []
    seeds_found = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        per_seed.append(dict(zip(data["features"], data["mean_abs_shap"])))
        seeds_found.append(data["seed"])

    if expected_seeds:
        missing = sorted(set(expected_seeds) - set(seeds_found))
        if missing:
            print(f"[{dataset}] WARNING missing seeds: {missing}", flush=True)

    all_keys = sorted(set().union(*[d.keys() for d in per_seed]))
    matrix = np.array([[d.get(k, np.nan) for k in all_keys] for d in per_seed])
    n_seeds_per_feat = np.sum(~np.isnan(matrix), axis=0)
    avg = np.nanmean(matrix, axis=0)

    out_dir = f"{OUT_DIR}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    json_path = f"{out_dir}/shap_mean_abs_xgboost_train{N_TRAIN}_ctx{ctx}.json"
    with open(json_path, "w") as f:
        json.dump({
            "seeds": seeds_found,
            "n_train_docs": N_TRAIN,
            "ctx": ctx,
            "features": all_keys,
            "mean_abs_shap_seed_avg": avg.tolist(),
            "n_seeds_per_feature": n_seeds_per_feat.tolist(),
            "per_seed_mean_abs": [{k: float(v) for k, v in d.items()} for d in per_seed],
        }, f, indent=2)
    print(f"  JSON -> {json_path}", flush=True)

    csv_path = f"{out_dir}/shap_mean_abs_xgboost_train{N_TRAIN}_ctx{ctx}.csv"
    sorted_idx = np.argsort(avg)[::-1]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "mean_abs_shap_seed_avg", "n_seeds"])
        for i in sorted_idx:
            writer.writerow([all_keys[i], f"{avg[i]:.8g}", int(n_seeds_per_feat[i])])
    print(f"  CSV  -> {csv_path}", flush=True)

    xlim = common_log_xlim(avg)

    order_desc = np.argsort(avg)[::-1]
    top_idx = order_desc[:top_n]
    plot_bar(
        [all_keys[i] for i in top_idx], avg[top_idx],
        f"{out_dir}/shap_top{top_n}_xgboost_train{N_TRAIN}_ctx{ctx}.png",
        xlim=xlim,
    )

    nz_sorted_desc = [i for i in order_desc if avg[i] > 0]
    bottom_idx = nz_sorted_desc[-bottom_n:]
    if bottom_idx:
        plot_bar(
            [all_keys[i] for i in bottom_idx], avg[bottom_idx],
            f"{out_dir}/shap_bottom{bottom_n}_xgboost_train{N_TRAIN}_ctx{ctx}.png",
            xlim=xlim,
        )
    else:
        print(f"  [{dataset}] no non-zero features for bottom plot -- skipping",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--ctx", type=int, default=43)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--bottom-n", type=int, default=30)
    args = parser.parse_args()

    for ds in args.datasets:
        aggregate_dataset(ds, args.ctx, args.top_n, args.bottom_n, ALL_SEEDS)


if __name__ == "__main__":
    main()
