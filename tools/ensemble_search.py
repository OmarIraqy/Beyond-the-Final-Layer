#!/usr/bin/env python3
"""Ensemble combination search — find the best model combination by feature concatenation.

Reads pre-extracted features from ensemble_extract.py and evaluates all possible
combinations of models. For each combination, features are concatenated and
L2-normalized, then evaluated with per-class mAP and test-weighted-mAP.

Usage:
    python tools/ensemble_search.py
    python tools/ensemble_search.py --max-models 4 --top-k 10
    python tools/ensemble_search.py --greedy
    python tools/ensemble_search.py --include vit_large_dinov3,swin_large_in22k,convnextv2_large_in22k
"""

import os
import sys
import json
import argparse
import time
import multiprocessing as mp
from itertools import combinations
from functools import partial

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.evaluator import eval_func, TEST_CLASS_WEIGHTS
from src.data.dataset import IDX_TO_CLASS

# Module-level globals for worker processes (inherited via fork on Linux).
# Set by _init_worker(); avoids pickling large arrays across processes.
_WORKER_MODELS = None
_WORKER_SHARED_META = None


def load_features(features_dir, split="val", include=None, exclude=None):
    """Load manifest and all feature arrays.

    Returns:
        models: dict  {exp_name: {"qf": np.array, "gf": np.array, "meta": dict}}
        shared_meta: dict  {"pids", "camids", "class_labels", "num_query"}
    """
    manifest_path = os.path.join(features_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run ensemble_extract.py first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load shared metadata
    meta_path = os.path.join(features_dir, f"{split}_meta.npz")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Shared metadata not found: {meta_path}")
    shared = np.load(meta_path)
    num_query = int(shared["num_query"])
    shared_meta = {
        "q_pids": shared["pids"][:num_query],
        "g_pids": shared["pids"][num_query:],
        "q_camids": shared["camids"][:num_query],
        "g_camids": shared["camids"][num_query:],
        "q_class_labels": shared["class_labels"][:num_query],
        "g_class_labels": shared["class_labels"][num_query:],
        "num_query": num_query,
    }

    models = {}
    for entry in manifest:
        name = entry["experiment"]
        if include is not None and name not in include:
            continue
        if exclude is not None and name in exclude:
            continue
        qf_path = os.path.join(features_dir, name, f"{split}_qf.npy")
        gf_path = os.path.join(features_dir, name, f"{split}_gf.npy")
        if not os.path.isfile(qf_path) or not os.path.isfile(gf_path):
            print(f"  Warning: features not found for {name}, skipping")
            continue
        models[name] = {
            "qf": np.load(qf_path),
            "gf": np.load(gf_path),
            "meta": entry,
        }
    return models, shared_meta


def _init_worker(models, shared_meta):
    """Initializer for pool workers — stash refs to shared data."""
    global _WORKER_MODELS, _WORKER_SHARED_META
    _WORKER_MODELS = models
    _WORKER_SHARED_META = shared_meta


def _eval_combo_worker(combo):
    """Worker target: evaluate one combination using module-level globals."""
    return evaluate_ensemble(list(combo), _WORKER_MODELS, _WORKER_SHARED_META)


def evaluate_ensemble(model_names, models, shared_meta):
    """Evaluate a combination of models by concatenating features.

    Returns dict with metrics.
    """
    # Concatenate and L2-normalize
    qf_parts = [models[n]["qf"] for n in model_names]
    gf_parts = [models[n]["gf"] for n in model_names]
    qf = np.hstack(qf_parts)
    gf = np.hstack(gf_parts)
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)

    # Cosine distance
    sim = qf @ gf.T
    dist = 1.0 - sim

    q_pids = shared_meta["q_pids"]
    g_pids = shared_meta["g_pids"]
    q_camids = shared_meta["q_camids"]
    g_camids = shared_meta["g_camids"]
    q_class_labels = shared_meta["q_class_labels"]
    g_class_labels = shared_meta["g_class_labels"]

    # Overall metrics
    cmc, mAP = eval_func(dist, q_pids, g_pids, q_camids, g_camids)

    metrics = {
        "mAP": float(mAP),
        "R1": float(cmc[0]) if len(cmc) > 0 else 0.0,
        "R5": float(cmc[4]) if len(cmc) > 4 else 0.0,
        "R10": float(cmc[9]) if len(cmc) > 9 else 0.0,
    }

    # Per-class mAP
    per_class = {}
    for cls_idx, cls_name in IDX_TO_CLASS.items():
        q_mask = q_class_labels == cls_idx
        g_mask = g_class_labels == cls_idx
        if q_mask.sum() == 0 or g_mask.sum() == 0:
            continue
        cls_dist = dist[np.ix_(q_mask, g_mask)]
        cls_cmc, cls_mAP = eval_func(
            cls_dist,
            q_pids[q_mask], g_pids[g_mask],
            q_camids[q_mask], g_camids[g_mask],
        )
        per_class[cls_name] = float(cls_mAP)

    metrics["per_class_mAP"] = per_class
    metrics["test_weighted_mAP"] = sum(
        TEST_CLASS_WEIGHTS.get(cls, 0.0) * per_class.get(cls, 0.0)
        for cls in TEST_CLASS_WEIGHTS
    )
    return metrics


def greedy_search(models, shared_meta, num_workers=1):
    """Greedy forward selection: start with best single, add models one at a time."""
    model_names = list(models.keys())
    print(f"\n--- Greedy Search ({len(model_names)} models, {num_workers} workers) ---")

    pool = mp.Pool(num_workers, initializer=_init_worker, initargs=(models, shared_meta))

    # Evaluate all singles
    single_combos = [(name,) for name in model_names]
    single_results = {}
    for combo, m in zip(single_combos, pool.imap(_eval_combo_worker, single_combos)):
        single_results[combo[0]] = m["test_weighted_mAP"]

    # Sort by score
    ranked = sorted(single_results.items(), key=lambda x: -x[1])
    print("\nIndividual model scores:")
    for name, score in ranked:
        print(f"  {name:45s}  {score:.4f}")

    # Greedy forward selection
    selected = [ranked[0][0]]
    best_score = ranked[0][1]
    remaining = [n for n, _ in ranked[1:]]
    history = [(list(selected), best_score)]

    print(f"\nStarting with: {selected[0]} ({best_score:.4f})")

    while remaining:
        # Build trial combos for all remaining candidates
        trial_combos = [tuple(selected + [c]) for c in remaining]
        trial_scores = pool.map(_eval_combo_worker, trial_combos)

        best_next = None
        best_next_score = best_score
        for candidate, m in zip(remaining, trial_scores):
            if m["test_weighted_mAP"] > best_next_score:
                best_next_score = m["test_weighted_mAP"]
                best_next = candidate

        if best_next is None:
            print("No improvement found, stopping.")
            break

        selected.append(best_next)
        remaining.remove(best_next)
        best_score = best_next_score
        history.append((list(selected), best_score))
        print(f"  + {best_next:45s}  -> {best_score:.4f}  (n={len(selected)})")

    pool.close()
    pool.join()
    return history


def _combo_generator(model_names, min_models, max_models):
    """Yield all combinations as tuples."""
    for k in range(min_models, max_models + 1):
        yield from combinations(model_names, k)


def exhaustive_search(models, shared_meta, min_models=2, max_models=None, num_workers=1):
    """Try all combinations of models within size bounds."""
    model_names = sorted(models.keys())
    n = len(model_names)

    if max_models is None:
        max_models = n

    # Count total combinations
    from math import comb
    total = sum(comb(n, k) for k in range(min_models, max_models + 1))
    print(f"\n--- Exhaustive Search ---")
    print(f"Models: {n}, combo sizes: {min_models}..{max_models}, total combos: {total:,}, workers: {num_workers}")

    results = []
    best_so_far = 0.0

    # Use a chunk size that balances IPC overhead vs. responsiveness
    chunksize = max(1, min(256, total // (num_workers * 4)))

    pool = mp.Pool(num_workers, initializer=_init_worker, initargs=(models, shared_meta))
    combo_iter = _combo_generator(model_names, min_models, max_models)

    pbar = tqdm(total=total, desc="Searching")
    for combo, m in zip(
        _combo_generator(model_names, min_models, max_models),
        pool.imap(_eval_combo_worker, combo_iter, chunksize=chunksize),
    ):
        score = m["test_weighted_mAP"]
        results.append({
            "models": list(combo),
            "num_models": len(combo),
            "combined_feature_dim": sum(models[name]["qf"].shape[1] for name in combo),
            **m,
        })
        if score > best_so_far:
            best_so_far = score
            pbar.set_postfix({"best": f"{best_so_far:.4f}", "combo": ",".join(combo)[:60]})
        pbar.update(1)
    pbar.close()

    pool.close()
    pool.join()

    results.sort(key=lambda r: -r["test_weighted_mAP"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Ensemble Combination Search")
    parser.add_argument("--features-dir", type=str, default="outputs/ensemble_features")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <features-dir>/ensemble_results.json)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--min-models", type=int, default=2)
    parser.add_argument("--max-models", type=int, default=None,
                        help="Max models per combination (default: no limit)")
    parser.add_argument("--include", type=str, default=None,
                        help="Comma-separated experiment names to include (only these)")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Comma-separated experiment names to exclude")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy forward selection instead of exhaustive search")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Print top K results to console")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    args = parser.parse_args()

    if args.workers is None:
        args.workers = mp.cpu_count()

    if args.include and args.exclude:
        parser.error("--include and --exclude are mutually exclusive")

    if args.output is None:
        args.output = os.path.join(args.features_dir, "ensemble_results.json")

    include = set(args.include.split(",")) if args.include else None
    exclude = set(args.exclude.split(",")) if args.exclude else None

    print(f"Loading features from {args.features_dir}...")
    models, shared_meta = load_features(args.features_dir, args.split, include, exclude)
    print(f"Loaded {len(models)} models:")
    for name, data in sorted(models.items()):
        meta = data["meta"]
        print(f"  {name:45s}  dim={meta['feature_dim']:5d}  best_wmAP={meta['best_test_weighted_mAP']:.4f}")

    if len(models) < 1:
        print("No models loaded. Nothing to do.")
        return

    t0 = time.time()

    # Also evaluate individual models for the JSON (parallel)
    print(f"\nEvaluating individual models ({args.workers} workers)...")
    sorted_names = sorted(models.keys())
    single_combos = [(name,) for name in sorted_names]

    pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(models, shared_meta))
    single_metrics = list(tqdm(
        pool.imap(_eval_combo_worker, single_combos),
        total=len(single_combos), desc="Singles",
    ))
    pool.close()
    pool.join()

    individual = []
    for name, m in zip(sorted_names, single_metrics):
        individual.append({
            "experiment": name,
            "backbone": models[name]["meta"]["backbone"],
            "feature_dim": models[name]["meta"]["feature_dim"],
            "best_epoch": models[name]["meta"]["best_epoch"],
            "val_test_weighted_mAP": models[name]["meta"]["best_test_weighted_mAP"],
            "evaluated_test_weighted_mAP": m["test_weighted_mAP"],
            **m,
        })
    individual.sort(key=lambda x: -x["test_weighted_mAP"])

    print("\nIndividual model results:")
    for entry in individual:
        print(f"  {entry['experiment']:45s}  weighted_mAP={entry['test_weighted_mAP']:.4f}  "
              f"mAP={entry['mAP']:.4f}  R1={entry['R1']:.4f}")

    # Run search
    if args.greedy:
        history = greedy_search(models, shared_meta, num_workers=args.workers)
        # Convert greedy history to results format
        results = []
        for selected, score in history:
            m = evaluate_ensemble(selected, models, shared_meta)
            results.append({
                "models": selected,
                "num_models": len(selected),
                "combined_feature_dim": sum(models[n]["qf"].shape[1] for n in selected),
                **m,
            })
        results.sort(key=lambda r: -r["test_weighted_mAP"])
    else:
        results = exhaustive_search(
            models, shared_meta,
            min_models=args.min_models,
            max_models=args.max_models,
            num_workers=args.workers,
        )

    elapsed = time.time() - t0

    # Print top-K
    print(f"\n{'='*80}")
    print(f"Top {min(args.top_k, len(results))} ensembles (searched {len(results)} combos in {elapsed:.1f}s):")
    print(f"{'='*80}")
    for i, r in enumerate(results[:args.top_k]):
        model_str = " + ".join(r["models"])
        print(f"\n  #{i+1}  test_weighted_mAP={r['test_weighted_mAP']:.4f}  "
              f"mAP={r['mAP']:.4f}  R1={r['R1']:.4f}  dim={r['combined_feature_dim']}")
        print(f"      Models ({r['num_models']}): {model_str}")
        pc = r.get("per_class_mAP", {})
        if pc:
            cls_str = "  ".join(f"{c}={v:.4f}" for c, v in sorted(pc.items()))
            print(f"      Per-class: {cls_str}")

    # Save JSON
    output_data = {
        "search_config": {
            "features_dir": args.features_dir,
            "split": args.split,
            "min_models": args.min_models,
            "max_models": args.max_models,
            "greedy": args.greedy,
            "num_experiments": len(models),
            "total_combinations_evaluated": len(results),
            "elapsed_seconds": round(elapsed, 1),
        },
        "individual_models": individual,
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
