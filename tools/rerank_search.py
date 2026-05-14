#!/usr/bin/env python3
"""S5: Re-ranking hyperparameter grid search on the val set.

Loads pre-extracted val features (qf.npy / gf.npy) from an experiment output
directory, then sweeps over k1, k2, lambda combinations and reports the best
settings by test-weighted-mAP.  No retraining required.

Usage:
    # Use pre-saved features in an experiment output dir
    python tools/rerank_search.py --features-dir outputs/vit_large_dinov3

    # Specify feature files explicitly
    python tools/rerank_search.py --qf outputs/vit_large_dinov3/qf.npy \
                                  --gf outputs/vit_large_dinov3/gf.npy \
                                  --config outputs/vit_large_dinov3/config.yaml

    # Narrow the search grid
    python tools/rerank_search.py --features-dir outputs/vit_large_dinov3 \
        --k1 10 20 30 --k2 4 6 8 --lam 0.2 0.3 0.4
"""

import os
import sys
import argparse
import itertools
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.data import UrbanReIDDataset
from src.postprocess.rerank import re_ranking
from src.postprocess.distance import cosine_similarity
from src.engine.evaluator import eval_func, TEST_CLASS_WEIGHTS
from src.data.dataset import IDX_TO_CLASS


def _test_weighted_map(dist, q_pids, g_pids, q_camids, g_camids, q_cls, g_cls):
    """Compute test-weighted-mAP from a distance matrix."""
    overall_cmc, overall_mAP = eval_func(dist, q_pids, g_pids, q_camids, g_camids)

    per_class = {}
    for cls_idx, cls_name in IDX_TO_CLASS.items():
        qm = q_cls == cls_idx
        gm = g_cls == cls_idx
        if qm.sum() == 0 or gm.sum() == 0:
            continue
        cls_dist = dist[np.ix_(qm, gm)]
        _, cls_mAP = eval_func(cls_dist, q_pids[qm], g_pids[gm], q_camids[qm], g_camids[gm])
        per_class[cls_name] = float(cls_mAP)

    tw_map = sum(TEST_CLASS_WEIGHTS[cls] * per_class.get(cls, 0.0) for cls in TEST_CLASS_WEIGHTS)
    return float(overall_mAP), tw_map, per_class


def main():
    parser = argparse.ArgumentParser(description="Re-ranking hyperparameter grid search")
    parser.add_argument("--features-dir", type=str, default=None,
                        help="Experiment output dir containing qf.npy, gf.npy, config.yaml")
    parser.add_argument("--qf", type=str, default=None, help="Path to query features .npy")
    parser.add_argument("--gf", type=str, default=None, help="Path to gallery features .npy")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--k1", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--k2", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--lam", type=float, nargs="+", default=[0.2, 0.3, 0.4],
                        dest="lam")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Print top-k configurations")
    args = parser.parse_args()

    # Resolve feature paths
    if args.features_dir:
        qf_path = args.qf or os.path.join(args.features_dir, "qf.npy")
        gf_path = args.gf or os.path.join(args.features_dir, "gf.npy")
        cfg_path = args.config if args.config != "configs/default.yaml" \
            else os.path.join(args.features_dir, "config.yaml")
    else:
        qf_path = args.qf
        gf_path = args.gf
        cfg_path = args.config

    if not qf_path or not os.path.isfile(qf_path):
        raise FileNotFoundError(f"Query features not found: {qf_path}")
    if not gf_path or not os.path.isfile(gf_path):
        raise FileNotFoundError(f"Gallery features not found: {gf_path}")

    print(f"Loading features: {qf_path}, {gf_path}")
    qf = np.load(qf_path).astype(np.float32)
    gf = np.load(gf_path).astype(np.float32)
    print(f"  qf: {qf.shape}  gf: {gf.shape}")

    # Load val metadata
    cfg = load_config(cfg_path)
    dataset = UrbanReIDDataset(cfg)

    val_q = list(dataset.val_query)
    val_g = list(dataset.val_gallery)
    q_pids = np.array([s.pid for s in val_q])
    g_pids = np.array([s.pid for s in val_g])
    q_camids = np.array([s.camid for s in val_q])
    g_camids = np.array([s.camid for s in val_g])
    q_cls = np.array([s.class_label for s in val_q])
    g_cls = np.array([s.class_label for s in val_g])

    if len(qf) != len(q_pids):
        raise ValueError(f"qf length {len(qf)} != val_query length {len(q_pids)}. "
                         "Make sure features were extracted on the val split.")

    # Baseline: cosine distance without re-ranking
    print("\nBaseline (no re-ranking):")
    sim = cosine_similarity(qf, gf)
    dist_base = 1.0 - sim
    mAP_base, tw_base, cls_base = _test_weighted_map(dist_base, q_pids, g_pids, q_camids, g_camids, q_cls, g_cls)
    print(f"  mAP={mAP_base:.4f}  test-weighted-mAP={tw_base:.4f}")
    for cls, v in sorted(cls_base.items()):
        print(f"    {cls}: {v:.4f}")

    # Pre-compute self-similarity matrices (reused across all configurations)
    print("\nPre-computing q-q and g-g similarity...")
    q_q_sim = cosine_similarity(qf, qf)
    g_g_sim = cosine_similarity(gf, gf)
    q_g_sim = sim

    # Grid search
    grid = list(itertools.product(args.k1, args.k2, args.lam))
    print(f"\nSearching {len(grid)} configurations: "
          f"k1={args.k1}  k2={args.k2}  lam={args.lam}")

    results: List[dict] = []
    for k1, k2, lam in grid:
        dist = re_ranking(q_g_sim, q_q_sim, g_g_sim, k1=k1, k2=k2, lambda_value=lam)
        mAP, tw_map, per_cls = _test_weighted_map(dist, q_pids, g_pids, q_camids, g_camids, q_cls, g_cls)
        results.append({
            "k1": k1, "k2": k2, "lam": lam,
            "mAP": mAP, "tw_map": tw_map, "per_cls": per_cls,
        })
        print(f"  k1={k1:3d} k2={k2:2d} lam={lam:.2f}  "
              f"mAP={mAP:.4f}  tw-mAP={tw_map:.4f}")

    results.sort(key=lambda r: r["tw_map"], reverse=True)

    print(f"\n{'='*60}")
    print(f"Top {args.top_k} configurations by test-weighted-mAP:")
    print(f"{'='*60}")
    for i, r in enumerate(results[: args.top_k]):
        delta = r["tw_map"] - tw_base
        sign = "+" if delta >= 0 else ""
        print(f"\n#{i+1}  k1={r['k1']}  k2={r['k2']}  lam={r['lam']:.2f}")
        print(f"     mAP={r['mAP']:.4f}  tw-mAP={r['tw_map']:.4f}  ({sign}{delta:.4f} vs baseline)")
        for cls, v in sorted(r["per_cls"].items()):
            b = cls_base.get(cls, 0.0)
            print(f"     {cls}: {v:.4f} ({'+' if v>=b else ''}{v-b:.4f})")

    best = results[0]
    print(f"\nBest config: k1={best['k1']} k2={best['k2']} lam={best['lam']:.2f}")
    print(f"  → Add to your config:")
    print(f"     test:")
    print(f"       rerank: true")
    print(f"       rerank_k1: {best['k1']}")
    print(f"       rerank_k2: {best['k2']}")
    print(f"       rerank_lambda: {best['lam']}")


if __name__ == "__main__":
    main()
