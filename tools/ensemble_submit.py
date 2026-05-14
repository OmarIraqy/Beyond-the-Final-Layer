#!/usr/bin/env python3
"""Generate competition submissions for the top-N ensemble configurations.

Reads the ensemble search results JSON and pre-extracted TEST features,
then produces a submission CSV for each of the best N configurations.

Prerequisites:
    1. Run ensemble_extract.py --split test   (extract test features)
    2. Run ensemble_search.py                 (produce results JSON)

Usage:
    python tools/ensemble_submit.py --results outputs/ensemble_features/ensemble_results.json --top-n 5
    python tools/ensemble_submit.py --results outputs/ensemble_features/ensemble_results.json --top-n 3 --rerank
"""

import os
import sys
import csv
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.postprocess.distance import cosine_similarity
from src.postprocess.rerank import re_ranking


def load_test_features(features_dir, model_names):
    """Load pre-extracted test features for the given models.

    Returns:
        dict {name: {"qf": np.array, "gf": np.array}}
    """
    features = {}
    for name in model_names:
        qf_path = os.path.join(features_dir, name, "test_qf.npy")
        gf_path = os.path.join(features_dir, name, "test_gf.npy")
        if not os.path.isfile(qf_path) or not os.path.isfile(gf_path):
            raise FileNotFoundError(
                f"Test features not found for {name}. "
                f"Run: python tools/ensemble_extract.py --split test"
            )
        features[name] = {
            "qf": np.load(qf_path),
            "gf": np.load(gf_path),
        }
    return features


def build_ensemble_features(model_names, features):
    """Concatenate and L2-normalize features from multiple models."""
    qf = np.hstack([features[n]["qf"] for n in model_names])
    gf = np.hstack([features[n]["gf"] for n in model_names])
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    return qf, gf


def write_submission(indices, output_path, top_k=100):
    """Write submission CSV in competition format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for i, row in enumerate(indices):
            image_name = f"{i + 1:06d}.jpg"
            index_str = " ".join(str(idx + 1) for idx in row[:top_k])
            writer.writerow([image_name, index_str])


def main():
    parser = argparse.ArgumentParser(description="Generate submissions for top ensemble configs")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to ensemble_results.json from ensemble_search.py")
    parser.add_argument("--features-dir", type=str, default="outputs/ensemble_features",
                        help="Directory with pre-extracted features")
    parser.add_argument("--output-dir", type=str, default="outputs/ensemble_submissions2",
                        help="Directory to write submission CSVs")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top configurations to generate submissions for")
    parser.add_argument("--max-models", type=int, default=3,
                        help="Only consider configs with at most this many models")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Top-K gallery matches per query in submission")
    parser.add_argument("--rerank", action="store_true",
                        help="Apply k-reciprocal re-ranking")
    parser.add_argument("--rerank-k1", type=int, default=20)
    parser.add_argument("--rerank-k2", type=int, default=6)
    parser.add_argument("--rerank-lambda", type=float, default=0.3)
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} ensemble results from {args.results}")

    # Filter by max-models if specified
    if args.max_models is not None:
        results = [r for r in results if r["num_models"] <= args.max_models]
        print(f"Filtered to {len(results)} configs with <= {args.max_models} models")

    top_configs = results[:args.top_n]
    print(f"Generating submissions for top {len(top_configs)} configurations\n")

    # Collect all unique model names needed
    all_models = set()
    for cfg in top_configs:
        all_models.update(cfg["models"])

    # Load test features once for all needed models
    print(f"Loading test features for {len(all_models)} models...")
    features = load_test_features(args.features_dir, all_models)
    print("Features loaded.\n")

    os.makedirs(args.output_dir, exist_ok=True)
    summary = []

    for rank, cfg in enumerate(top_configs, 1):
        model_names = cfg["models"]
        n_models = cfg["num_models"]
        score = cfg["test_weighted_mAP"]
        dim = cfg["combined_feature_dim"]

        print(f"--- Config #{rank} (val weighted_mAP={score:.4f}, {n_models} models, dim={dim}) ---")
        print(f"    Models: {' + '.join(model_names)}")

        # Build concatenated features
        qf, gf = build_ensemble_features(model_names, features)
        print(f"    Features: qf={qf.shape}, gf={gf.shape}")

        # Compute distance
        q_g_sim = cosine_similarity(qf, gf)

        if args.rerank:
            print(f"    Re-ranking (k1={args.rerank_k1}, k2={args.rerank_k2}, λ={args.rerank_lambda})...")
            q_q_sim = cosine_similarity(qf, qf)
            g_g_sim = cosine_similarity(gf, gf)
            dist = re_ranking(
                q_g_sim, q_q_sim, g_g_sim,
                k1=args.rerank_k1, k2=args.rerank_k2,
                lambda_value=args.rerank_lambda,
            )
        else:
            dist = 1.0 - q_g_sim

        # Top-K ranking
        indices = np.argsort(dist, axis=1)[:, :args.top_k]

        # Write submission
        tag = f"rank{rank}_{n_models}models"
        if args.rerank:
            tag += "_rerank"
        csv_path = os.path.join(args.output_dir, f"submission_{tag}.csv")
        write_submission(indices, csv_path, args.top_k)
        print(f"    Saved: {csv_path}\n")

        summary.append({
            "rank": rank,
            "models": model_names,
            "num_models": n_models,
            "combined_feature_dim": dim,
            "val_test_weighted_mAP": score,
            "rerank": args.rerank,
            "submission_path": csv_path,
        })

    # Save summary
    summary_path = os.path.join(args.output_dir, "submissions_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
