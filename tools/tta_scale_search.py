#!/usr/bin/env python3
"""Multi-scale TTA grid search on the val set.

Extracts features at each candidate scale, then sweeps over all combinations
(1-scale, 2-scale, 3-scale, ...) and reports the best by test-weighted-mAP.

Usage:
    # Basic search over common scales
    python tools/tta_scale_search.py \\
        --config configs/experiment/vit_large_dinov3.yaml \\
        test.weight=outputs/vit_large_dinov3/best_model.pth

    # Custom scale grid + limit max scales per combo
    python tools/tta_scale_search.py \\
        --config configs/experiment/vit_large_dinov3.yaml \\
        test.weight=outputs/vit_large_dinov3/best_model.pth \\
        --scales 224 256 288 320 352 416 \\
        --max-combo 3

    # With flip TTA disabled
    python tools/tta_scale_search.py \\
        --config configs/experiment/vit_large_dinov3.yaml \\
        test.weight=outputs/vit_large_dinov3/best_model.pth \\
        --no-flip

    # Also test with/without re-ranking
    python tools/tta_scale_search.py \\
        --config configs/experiment/vit_large_dinov3.yaml \\
        test.weight=outputs/vit_large_dinov3/best_model.pth \\
        --test-rerank
"""

import os
import sys
import argparse
import itertools
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.data import UrbanReIDDataset, build_val_loader
from src.data.dataset import IDX_TO_CLASS
from src.models import ReIDModel
from src.engine.evaluator import extract_features, eval_func, TEST_CLASS_WEIGHTS
from src.postprocess.distance import cosine_similarity
from src.postprocess.rerank import re_ranking
from src.postprocess.class_mask import apply_class_mask
from src.utils.device import get_default_device
from src.utils.logger import setup_logger

import torch


def _build_scale_loader(cfg, dataset, scale):
    """Build a val loader with cfg.input.size_test overridden to `scale`."""
    cfg.input.size_test = list(scale)
    loader, num_q = build_val_loader(cfg, dataset)
    return loader, num_q


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


def _combinations_capped(candidates: List, max_k: int):
    """Yield all non-empty combinations of `candidates` up to size `max_k`."""
    for k in range(1, min(max_k, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, k):
            yield list(combo)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scale TTA grid search on val set"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--features-dir", type=str, default=None,
        help="Experiment output dir containing config.yaml (alternative to --config)",
    )
    parser.add_argument(
        "--scales", type=int, nargs="+",
        default=[32, 64, 128, 224, 256, 288, 320, 352, 416, 480],
        help="Candidate scale sizes (square: each value is both H and W). "
             "Default: 32, 64, 128, 224, 256, 288, 320, 352, 416, 480",
    )
    parser.add_argument(
        "--max-combo", type=int, default=None,
        help="Max scales per combination (default: all). Use 1 for single-scale only.",
    )
    parser.add_argument(
        "--flip", action=argparse.BooleanOptionalAction, default=True,
        help="Enable/disable horizontal flip TTA at each scale (default: --flip)",
    )
    parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=False,
        help="Enable/disable re-ranking (default: --no-rerank)",
    )
    parser.add_argument(
        "--class-mask", action="store_true", default=False,
        help="Apply cross-class masking",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Print top K configurations")
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory to cache per-scale features (.npy). Resuming reuses cache.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides (e.g. test.weight=path/to/weights.pth)",
    )
    args = parser.parse_args()

    # Resolve config path
    if args.features_dir:
        cfg_path = os.path.join(args.features_dir, "config.yaml")
        if args.config != "configs/default.yaml":
            cfg_path = args.config
    else:
        cfg_path = args.config

    cfg = load_config(cfg_path, args.overrides if args.overrides else None)

    max_combo = args.max_combo if args.max_combo is not None else len(args.scales)

    # Build candidate scales as [h, w] pairs
    candidate_scales = [[s, s] for s in args.scales]

    print(f"Candidate scales: {candidate_scales}")
    print(f"Max combo size: {max_combo}")
    print(f"Flip TTA: {args.flip}")
    print(f"Re-ranking: {args.rerank}")
    print(f"Class mask: {args.class_mask}")

    # Setup
    device = get_default_device()
    cache_dir = args.cache_dir or cfg.trainer.output_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Dataset
    dataset = UrbanReIDDataset(cfg)

    # Model
    has_class_obj = any(obj.type == "class_ce" for obj in cfg.objectives)
    num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0
    model = ReIDModel(cfg, num_pids=dataset.num_train_pids, num_obj_classes=num_obj_classes)

    if cfg.test.weight:
        ckpt = torch.load(cfg.test.weight, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {cfg.test.weight}")
    else:
        print("WARNING: No test.weight specified — evaluating with random weights")

    model = model.to(device)

    # Val metadata
    val_q = list(dataset.val_query)
    val_g = list(dataset.val_gallery)
    num_val_query = len(val_q)
    q_pids = np.array([s.pid for s in val_q])
    g_pids = np.array([s.pid for s in val_g])
    q_camids = np.array([s.camid for s in val_q])
    g_camids = np.array([s.camid for s in val_g])
    q_cls = np.array([s.class_label for s in val_q])
    g_cls = np.array([s.class_label for s in val_g])

    print(f"Val query: {num_val_query}, Val gallery: {len(val_g)}")

    # Phase 1: Extract features at each candidate scale
    per_scale_features: Dict[str, np.ndarray] = {}
    for scale in candidate_scales:
        key = f"{scale[0]}x{scale[1]}"
        cache_path = os.path.join(cache_dir, f"val_feats_{key}.npy") if cache_dir else None

        if cache_path and os.path.isfile(cache_path):
            print(f"  Loading cached: {cache_path}")
            per_scale_features[key] = np.load(cache_path).astype(np.float32)
        else:
            print(f"  Extracting features at scale {scale} ...")
            loader, nq = _build_scale_loader(cfg, dataset, scale)
            if nq != num_val_query:
                raise RuntimeError(
                    f"Scale {scale}: num_query={nq} but expected {num_val_query}. "
                    "Dataset changed? Check config."
                )
            feats = extract_features(model, loader, flip=args.flip, feat_norm=True)
            per_scale_features[key] = feats
            if cache_path:
                np.save(cache_path, feats.astype(np.float32))
                print(f"  Cached to {cache_path}")

    assert len(per_scale_features) == len(candidate_scales)
    print(f"\nExtracted features at {len(per_scale_features)} scales. "
          f"Feature shape: {list(per_scale_features.values())[0].shape}")

    # Phase 2: Test all combinations
    scale_keys = list(per_scale_features.keys())
    combos = list(_combinations_capped(scale_keys, max_combo))
    print(f"\nSearching {len(combos)} scale combinations ...")

    results: List[dict] = []
    for combo in combos:
        # Average features across scales in the combo
        scale_feats = [per_scale_features[k] for k in combo]
        avg_feats = np.mean(scale_feats, axis=0)
        # Re-normalize
        norms = np.linalg.norm(avg_feats, axis=1, keepdims=True) + 1e-12
        avg_feats = avg_feats / norms

        qf = avg_feats[:num_val_query]
        gf = avg_feats[num_val_query:]

        # Similarity -> distance
        sim = cosine_similarity(qf, gf)
        dist = 1.0 - sim

        # Optional re-ranking
        if args.rerank:
            dist = re_ranking(
                sim,
                cosine_similarity(qf, qf),
                cosine_similarity(gf, gf),
                k1=cfg.test.rerank_k1,
                k2=cfg.test.rerank_k2,
                lambda_value=cfg.test.rerank_lambda,
            )

        # Optional class mask
        if args.class_mask:
            dist = apply_class_mask(dist, q_cls, g_cls)

        mAP, tw_map, per_cls = _test_weighted_map(
            dist, q_pids, g_pids, q_camids, g_camids, q_cls, g_cls,
        )
        results.append({
            "scales": combo,
            "num_scales": len(combo),
            "mAP": mAP,
            "tw_map": tw_map,
            "per_cls": per_cls,
        })

        scales_str = " + ".join(combo)
        print(f"  [{scales_str:30s}] n={len(combo):1d}  mAP={mAP:.4f}  tw-mAP={tw_map:.4f}")

    results.sort(key=lambda r: r["tw_map"], reverse=True)

    # Baseline: the combination with all scales
    baseline = None
    for r in results:
        if r["num_scales"] == len(per_scale_features):
            baseline = r
            break
    # If no "all scales" combo (due to max_combo), use single best scale
    if baseline is None:
        baseline = max(
            [r for r in results if r["num_scales"] == 1],
            key=lambda r: r["tw_map"],
            default=results[0],
        )

    print(f"\n{'='*60}")
    print(f"Top {args.top_k} configurations by test-weighted-mAP:")
    print(f"{'='*60}")

    for i, r in enumerate(results[:args.top_k]):
        delta = r["tw_map"] - baseline["tw_map"]
        sign = "+" if delta >= 0 else ""
        scales_str = " + ".join(r["scales"])
        print(f"\n#{i+1}  scales=[{scales_str}]  n={r['num_scales']}")
        print(f"     mAP={r['mAP']:.4f}  tw-mAP={r['tw_map']:.4f}  ({sign}{delta:.4f} vs baseline)")
        for cls, v in sorted(r["per_cls"].items()):
            b = baseline.get("per_cls", {}).get(cls, 0.0) if baseline else 0.0
            print(f"     {cls}: {v:.4f} ({'+' if v>=b else ''}{v-b:.4f})")

    best = results[0]
    print(f"\nBest config: scales={best['scales']}")
    print(f"  → Add to your experiment YAML:")
    print(f"     test:")
    print(f"       scales:")
    for s in best["scales"]:
        h, w = s.split("x")
        print(f"         - [{h}, {w}]")
    print(f"       flip_test: {str(args.flip).lower()}")
    if args.rerank:
        print(f"       rerank: true")
        print(f"       rerank_k1: {cfg.test.rerank_k1}")
        print(f"       rerank_k2: {cfg.test.rerank_k2}")
        print(f"       rerank_lambda: {cfg.test.rerank_lambda}")
    if args.class_mask:
        print(f"       class_mask: true")


if __name__ == "__main__":
    main()
