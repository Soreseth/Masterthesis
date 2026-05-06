#!/usr/bin/env python3
"""
Merge sharded MIA score files into one members.jsonl / nonmembers.jsonl per
(model_or_defense, dataset, ctx). Drops intermediate-step reference features
to reduce file size and harmonise schema across splits.

Background:
    Old splits (doc range 0-1999) carry ALL 32 reference checkpoints per chunk:
        step1, step512, step1000, step3000, step5000, step10000, step100000, final
        x {70m, 160m, 410m, 1b}
    Newer splits (doc range 2000-5999) used --ref_steps step1 final, so they
    already have only 8 reference configs.  This merger filters every chunk's
    `pred` dict to drop the 6 dropped-step keys, so downstream code can treat
    every doc uniformly with the reduced ~644-feature schema.

Input  (per shard):
    $MIA_ROOT/<subset>/<kind>/<key>/ctx<CTX>/mia_{members,nonmembers}_<S>_<E>.jsonl
        kind ∈ {undefended, defended}
        key  is the model name or defense name (e.g. "pythia-2.8b", "duolearn_a0.2")

Output (single file per side):
    $MIA_ROOT/<subset>/<kind>/<key>/ctx<CTX>/{members,nonmembers}.jsonl[.gz]

Usage:
    python -m src.utils.merge_jsonl
    python -m src.utils.merge_jsonl --datasets arxiv FreeLaw --ctx 1024
    python -m src.utils.merge_jsonl --kind defended --key duolearn_a0.2 --gzip
"""
import argparse
import glob
import gzip
import json
import os
import re

DEFAULT_DATASETS = ["arxiv", "FreeLaw", "Github", "HackerNews", "OpenWebText2",
                    "Pile-CC", "USPTO_Backgrounds", "wiki"]
DEFAULT_CTX = 2048

# Reference-checkpoint suffixes NOT in the reduced set -- features whose name
# contains any of these as a `_stepX_` or `_stepX$` segment get dropped.
DROP_STEPS = ["step512", "step1000", "step3000", "step5000",
              "step10000", "step100000"]
DROP_PATTERNS = [re.compile(rf"_{s}(?:_|$)") for s in DROP_STEPS]


def _round_floats(obj, ndigits):
    """Recursively round every float in a nested structure."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    return obj


def _filter_pred(pred_dict, ndigits):
    """Drop dropped-step keys and round remaining floats."""
    if not isinstance(pred_dict, dict):
        return pred_dict
    out = {}
    for k, v in pred_dict.items():
        if any(p.search(k) for p in DROP_PATTERNS):
            continue
        out[k] = _round_floats(v, ndigits)
    return out


def _filter_obj(obj, ndigits):
    """Apply _filter_pred to every chunk in a doc's preds list (or single dict)."""
    preds = obj.get("pred", obj.get("preds", []))
    if isinstance(preds, list) and preds and isinstance(preds[0], dict):
        obj["pred"] = [_filter_pred(p, ndigits) for p in preds]
    elif isinstance(preds, dict):
        obj["pred"] = _filter_pred(preds, ndigits)
    return obj


def _open_out(path, use_gzip):
    if use_gzip:
        return gzip.open(path + ".gz", "wt", compresslevel=6)
    return open(path, "w")


def _shard_start_index(fname, prefix):
    """Extract the integer start of a shard filename for sort order."""
    base = os.path.basename(fname).replace(prefix, "").replace(".jsonl", "")
    return int(base.split("_")[0])


def merge_one_split(doc_dir, mtype, prefix, outname, ndigits, use_gzip):
    """Merge all shards of one side (members or nonmembers) into one file."""
    outpath = os.path.join(doc_dir, outname)
    actual_outpath = outpath + (".gz" if use_gzip else "")

    # Always re-merge; remove plain + gz variants so we never read stale content.
    for p in (outpath, outpath + ".gz"):
        if os.path.exists(p):
            os.remove(p)
            print(f"  {mtype}: removed stale {os.path.basename(p)}")

    shards = glob.glob(os.path.join(doc_dir, f"{prefix}*.jsonl"))
    shards = [f for f in shards if os.path.basename(f) != outname]
    if not shards:
        print(f"  {mtype}: no shards, skip")
        return

    shards.sort(key=lambda f: _shard_start_index(f, prefix))

    seen_ids = set()
    n_total, n_dup = 0, 0
    per_file_stats = []

    with _open_out(outpath, use_gzip) as out:
        for sf in shards:
            n_docs_file = 0
            n_keys_dropped = 0
            with open(sf) as f:
                for line in f:
                    obj = json.loads(line)
                    doc_id = obj.get("id", f"doc_{n_total}")
                    if doc_id in seen_ids:
                        n_dup += 1
                        continue
                    seen_ids.add(doc_id)

                    if isinstance(obj.get("pred"), list) and obj["pred"]:
                        first = obj["pred"][0]
                        before = len(first) if isinstance(first, dict) else 0
                        obj = _filter_obj(obj, ndigits)
                        first = obj["pred"][0]
                        after = len(first) if isinstance(first, dict) else 0
                        n_keys_dropped = before - after
                    else:
                        obj = _filter_obj(obj, ndigits)

                    out.write(json.dumps(obj, separators=(",", ":")) + "\n")
                    n_docs_file += 1
                    n_total += 1
            per_file_stats.append((os.path.basename(sf), n_docs_file, n_keys_dropped))

    try:
        size_mb = os.path.getsize(actual_outpath) / 1e6
    except OSError:
        size_mb = -1.0

    print(f"  {mtype}: merged {len(shards)} shards -> {n_total} docs, "
          f"{size_mb:.1f} MB  ({n_dup} dup skipped)")
    if per_file_stats:
        oldest, newest = per_file_stats[0], per_file_stats[-1]
        print(f"    oldest shard ({oldest[0]}): {oldest[2]} keys dropped per chunk")
        print(f"    newest shard ({newest[0]}): {newest[2]} keys dropped per chunk")


def merge_dataset(mia_root, dataset, kind, key, ctx, ndigits, use_gzip):
    """Merge both sides for one (subset, kind, key, ctx) cell."""
    doc_dir = f"{mia_root}/{dataset}/{kind}/{key}/ctx{ctx}"
    if not os.path.isdir(doc_dir):
        print(f"  {dataset}: no {doc_dir}, SKIP")
        return
    print(f"\n=== {dataset} / {kind} / {key} / ctx{ctx} ===")
    merge_one_split(doc_dir, "members",    "mia_members_",    "members.jsonl",    ndigits, use_gzip)
    merge_one_split(doc_dir, "nonmembers", "mia_nonmembers_", "nonmembers.jsonl", ndigits, use_gzip)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mia_root", default=os.environ.get("MIA_ROOT", "./mia_scores"))
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--kind", default="undefended", choices=["undefended", "defended"])
    p.add_argument("--key", default="pythia-2.8b",
                   help="Model name (undefended) or defense name (defended)")
    p.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    p.add_argument("--precision", type=int, default=6,
                   help="Decimal places for floats (default 6; 0 = no rounding)")
    p.add_argument("--gzip", action="store_true",
                   help="Write .jsonl.gz instead of .jsonl (~5-10x smaller)")
    args = p.parse_args()

    print(f"Merging shards under {args.mia_root}")
    print(f"  kind={args.kind}  key={args.key}  ctx={args.ctx}")
    print(f"  dropping {len(DROP_STEPS)} intermediate checkpoints: {DROP_STEPS}")
    print(f"  float precision: {args.precision} decimals  gzip: {args.gzip}")

    for ds in args.datasets:
        merge_dataset(args.mia_root, ds, args.kind, args.key, args.ctx,
                      args.precision, args.gzip)
    print("\nDone!")


if __name__ == "__main__":
    main()
