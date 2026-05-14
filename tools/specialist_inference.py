#!/usr/bin/env python3
"""Specialist-ensemble inference: each model handles one object class.

Each specialist is loaded from its own config + checkpoint.  Images are routed
to their specialist purely by class label (from the classes CSV files).
Cross-class distances are set to inf so specialists never compete across classes.

Usage (one --specialist pair per class, in class-index order 0→3):

    python tools/specialist_inference.py \\
        --specialist configs/exp_container.yaml   outputs/container/best_model.pth \\
        --specialist configs/exp_crosswalk.yaml   outputs/crosswalk/best_model.pth \\
        --specialist configs/exp_rubbish.yaml     outputs/rubbish/best_model.pth  \\
        --specialist configs/exp_trafficsign.yaml outputs/trafficsign/best_model.pth \\
        --output-dir outputs/specialist_ensemble

Class order (must match --specialist order):
    0 → Container
    1 → Crosswalk
    2 → RubbishBins
    3 → TrafficSign

Optional flags:
    --flip        horizontal flip TTA
    --no-rerank   skip re-ranking (re-ranking is ON by default)
    --no-val      skip validation evaluation
    --no-test     skip test submission generation
"""

import os
import sys
import csv
import argparse
import logging
import json
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.utils.device import get_default_device
from src.utils.seed import set_seed
from src.data.dataset import UrbanReIDDataset, ImageDataset, IDX_TO_CLASS
from src.data.dataloader import build_collate_fn
from src.data.transforms import build_test_transforms
from src.models import ReIDModel
from src.engine.evaluator import extract_features, eval_func
from src.postprocess.distance import cosine_similarity
from src.postprocess.rerank import re_ranking

logger = logging.getLogger("specialist_inference")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "specialist_inference.log")),
        ],
    )


def _load_model(cfg, num_train_pids: int, checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Build ReIDModel from cfg, load checkpoint weights."""
    has_class_obj = any(obj.type == "class_ce" for obj in cfg.objectives)
    num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0
    model = ReIDModel(cfg, num_pids=num_train_pids, num_obj_classes=num_obj_classes)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    logger.info(f"  Loaded checkpoint: {checkpoint}")
    return model.to(device).eval()


def _extract_subset(
    model: torch.nn.Module,
    cfg,
    samples,
    flip: bool = False,
    feat_norm: bool = True,
) -> np.ndarray:
    """Extract features for a list of Sample objects using a specialist's transforms."""
    transform = build_test_transforms(cfg)
    dataset_obj = ImageDataset(samples, transform=transform)
    collate_fn = build_collate_fn(
        cfg,
        force_batch_pad=cfg.input.resize_strategy.lower() == "class_letterbox",
    )
    loader = DataLoader(
        dataset_obj,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return extract_features(model, loader, flip=flip, feat_norm=feat_norm)


def _build_dist_matrix(
    specialists: list,       # list of (model, cfg) in class-index order
    query_samples: list,
    gallery_samples: list,
    flip: bool,
    rerank: bool,
    cfg0,                    # first config used for rerank hyper-params
    split_name: str,
) -> np.ndarray:
    """Assemble a full [num_query, num_gallery] distance matrix using per-class specialists.

    Cross-class pairs remain inf (specialists never score across classes).
    Samples with unknown class label (-1) fall back to all specialists
    (minimum distance across all specialists).
    """
    nq = len(query_samples)
    ng = len(gallery_samples)
    dist = np.full((nq, ng), np.inf, dtype=np.float32)

    q_classes = np.array([s.class_label for s in query_samples])
    g_classes = np.array([s.class_label for s in gallery_samples])

    unknown_q = np.sum(q_classes == -1)
    unknown_g = np.sum(g_classes == -1)
    if unknown_q > 0:
        logger.warning(f"[{split_name}] {unknown_q} query samples have unknown class label (-1)")
    if unknown_g > 0:
        logger.warning(f"[{split_name}] {unknown_g} gallery samples have unknown class label (-1)")

    # Per-class specialist pass
    for cls_idx, (model, cfg) in enumerate(specialists):
        cls_name = IDX_TO_CLASS[cls_idx]
        q_mask = q_classes == cls_idx
        g_mask = g_classes == cls_idx
        q_count = int(q_mask.sum())
        g_count = int(g_mask.sum())

        if q_count == 0 or g_count == 0:
            logger.warning(
                f"[{split_name}] Class {cls_name}: {q_count} queries, {g_count} gallery items — skipping"
            )
            continue

        logger.info(
            f"[{split_name}] Extracting features for class {cls_name} "
            f"(q={q_count}, g={g_count}) …"
        )

        q_subs = [query_samples[i] for i in np.where(q_mask)[0]]
        g_subs = [gallery_samples[i] for i in np.where(g_mask)[0]]

        qf = _extract_subset(model, cfg, q_subs, flip=flip)
        gf = _extract_subset(model, cfg, g_subs, flip=flip)

        if rerank:
            logger.info(f"  Re-ranking class {cls_name} …")
            q_q_sim = cosine_similarity(qf, qf)
            q_g_sim = cosine_similarity(qf, gf)
            g_g_sim = cosine_similarity(gf, gf)
            cls_dist = re_ranking(
                q_g_sim, q_q_sim, g_g_sim,
                k1=cfg0.test.rerank_k1,
                k2=cfg0.test.rerank_k2,
                lambda_value=cfg0.test.rerank_lambda,
            )
        else:
            cls_dist = 1.0 - cosine_similarity(qf, gf)

        q_idx = np.where(q_mask)[0]
        g_idx = np.where(g_mask)[0]
        dist[np.ix_(q_idx, g_idx)] = cls_dist

    # Fallback for unknown-class queries: assign minimum distance across all specialists
    unk_q_idx = np.where(q_classes == -1)[0]
    if len(unk_q_idx) > 0:
        logger.info(f"[{split_name}] Computing fallback distances for {len(unk_q_idx)} unknown-class queries …")
        unk_samples = [query_samples[i] for i in unk_q_idx]
        # Collect gallery features from every specialist (all gallery items)
        fallback_feats_q = []
        fallback_feats_g = []
        for cls_idx, (model, cfg) in enumerate(specialists):
            qf_unk = _extract_subset(model, cfg, unk_samples, flip=flip)
            gf_all = _extract_subset(model, cfg, gallery_samples, flip=flip)
            fallback_feats_q.append(qf_unk)
            fallback_feats_g.append(gf_all)
        # Take the minimum distance across specialists per gallery item
        all_dists = np.stack([
            1.0 - cosine_similarity(fallback_feats_q[i], fallback_feats_g[i])
            for i in range(len(specialists))
        ], axis=0)  # [num_specialists, num_unk_q, ng]
        best_dist = np.min(all_dists, axis=0)
        dist[unk_q_idx, :] = best_dist

    return dist


# ---------------------------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------------------------

def run_val(specialists, dataset, flip, rerank, cfg0, output_dir):
    logger.info("=" * 60)
    logger.info("VALIDATION SET")
    logger.info("=" * 60)

    q_samples = list(dataset.val_query)
    g_samples = list(dataset.val_gallery)
    logger.info(f"Val query: {len(q_samples)}, Val gallery: {len(g_samples)}")

    dist = _build_dist_matrix(
        specialists, q_samples, g_samples,
        flip=flip, rerank=rerank, cfg0=cfg0, split_name="val",
    )

    q_pids = np.array([s.pid for s in q_samples])
    g_pids = np.array([s.pid for s in g_samples])
    q_camids = np.array([s.camid for s in q_samples])
    g_camids = np.array([s.camid for s in g_samples])
    q_classes = np.array([s.class_label for s in q_samples])
    g_classes = np.array([s.class_label for s in g_samples])

    cmc, mAP = eval_func(dist, q_pids, g_pids, q_camids, g_camids)

    TEST_CLASS_WEIGHTS = {
        "TrafficSign": 0.627,
        "Container":   0.180,
        "Crosswalk":   0.098,
        "RubbishBins": 0.095,
    }

    metrics = {
        "mAP": float(mAP),
        "R1":  float(cmc[0]) if len(cmc) > 0 else 0.0,
        "R5":  float(cmc[4]) if len(cmc) > 4 else 0.0,
        "R10": float(cmc[9]) if len(cmc) > 9 else 0.0,
    }

    for cls_idx, cls_name in IDX_TO_CLASS.items():
        q_mask = q_classes == cls_idx
        g_mask = g_classes == cls_idx
        if q_mask.sum() == 0 or g_mask.sum() == 0:
            continue
        cls_dist = dist[np.ix_(np.where(q_mask)[0], np.where(g_mask)[0])]
        cls_cmc, cls_mAP = eval_func(
            cls_dist,
            q_pids[q_mask], g_pids[g_mask],
            q_camids[q_mask], g_camids[g_mask],
        )
        metrics[f"mAP_{cls_name}"] = float(cls_mAP)
        metrics[f"R1_{cls_name}"]  = float(cls_cmc[0]) if len(cls_cmc) > 0 else 0.0

    metrics["test_weighted_mAP"] = sum(
        TEST_CLASS_WEIGHTS[cls] * metrics.get(f"mAP_{cls}", 0.0)
        for cls in TEST_CLASS_WEIGHTS
    )

    logger.info("--- Validation Metrics ---")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved val metrics → {output_dir}/val_metrics.json")

    return metrics


# ---------------------------------------------------------------------------
# Test submission
# ---------------------------------------------------------------------------

def run_test(specialists, dataset, flip, rerank, cfg0, output_dir):
    logger.info("=" * 60)
    logger.info("TEST SET (submission)")
    logger.info("=" * 60)

    q_samples = list(dataset.query)
    g_samples = list(dataset.gallery)
    logger.info(f"Test query: {len(q_samples)}, Test gallery: {len(g_samples)}")

    dist = _build_dist_matrix(
        specialists, q_samples, g_samples,
        flip=flip, rerank=rerank, cfg0=cfg0, split_name="test",
    )

    top_k = cfg0.submission.top_k
    indices = np.argsort(dist, axis=1)[:, :top_k]

    output_path = os.path.join(output_dir, "submission_specialist.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for i, row in enumerate(indices):
            image_name = f"{i + 1:06d}.jpg"
            index_str = " ".join(str(idx + 1) for idx in row)
            writer.writerow([image_name, index_str])

    logger.info(f"Submission written → {output_path}")
    logger.info(f"  Queries: {len(indices)}, Top-K: {top_k}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Specialist ensemble inference (one model per class)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--specialist",
        metavar=("CONFIG", "CHECKPOINT"),
        nargs=2,
        action="append",
        required=True,
        help="Config YAML and checkpoint path for one specialist. "
             "Repeat 4 times, in class-index order: "
             "Container(0), Crosswalk(1), RubbishBins(2), TrafficSign(3).",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/specialist_ensemble",
        help="Directory for metrics JSON and submission CSV.",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="Enable horizontal flip TTA.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        default=False,
        help="Disable re-ranking (re-ranking is enabled by default).",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        default=False,
        help="Skip validation evaluation.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        default=False,
        help="Skip test submission generation.",
    )
    args = parser.parse_args()

    if len(args.specialist) != 4:
        parser.error(
            f"Exactly 4 --specialist pairs required (one per class), got {len(args.specialist)}."
        )

    _setup_logging(args.output_dir)
    rerank = not args.no_rerank
    device = get_default_device()

    logger.info("Specialist inference")
    logger.info(f"  Output dir : {args.output_dir}")
    logger.info(f"  Flip TTA   : {args.flip}")
    logger.info(f"  Re-ranking : {rerank}")

    # ── Load all 4 configs ──────────────────────────────────────────────────
    cfgs = []
    for i, (cfg_path, ckpt_path) in enumerate(args.specialist):
        cls_name = IDX_TO_CLASS[i]
        logger.info(f"Loading config [{i}] {cls_name}: {cfg_path}")
        cfg = load_config(cfg_path)
        cfgs.append(cfg)

    # ── Dataset (all specialists share the same dataset CSVs) ───────────────
    # Use the first config's dataset paths; they all point to the same dataset.
    set_seed(cfgs[0].trainer.seed)
    dataset = UrbanReIDDataset(cfgs[0])
    logger.info(
        f"Dataset loaded: train_pids={dataset.num_train_pids}, "
        f"val_query={len(dataset.val_query)}, val_gallery={len(dataset.val_gallery)}, "
        f"query={len(dataset.query)}, gallery={len(dataset.gallery)}"
    )

    # ── Load all 4 models ───────────────────────────────────────────────────
    specialists = []
    for i, (cfg_path, ckpt_path) in enumerate(args.specialist):
        cls_name = IDX_TO_CLASS[i]
        logger.info(f"Loading specialist model [{i}] {cls_name} …")
        model = _load_model(cfgs[i], dataset.num_train_pids, ckpt_path, device)
        specialists.append((model, cfgs[i]))
        logger.info(f"  Model {i} ({cls_name}) ready on {device}")

    # ── Validation ──────────────────────────────────────────────────────────
    if not args.no_val:
        run_val(specialists, dataset, args.flip, rerank, cfgs[0], args.output_dir)

    # ── Test submission ─────────────────────────────────────────────────────
    if not args.no_test:
        run_test(specialists, dataset, args.flip, rerank, cfgs[0], args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
