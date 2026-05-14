#!/usr/bin/env python3
"""Evaluate a checkpoint against the validation set and print per-class mAP.

Also writes a CSV in submission format (val query imageName -> top-K val gallery
indices, 1-indexed) saved next to the checkpoint as <stem>_val_submission.csv.

Usage:
    python tools/eval_checkpoint.py \\
        --config  configs/experiment/vit_large_dinov3.yaml \\
        --checkpoint outputs/vit_large_dinov3/best_model.pth

    # Disable rerank / enable flip TTA:
    python tools/eval_checkpoint.py \\
        --config  configs/experiment/vit_large_dinov3.yaml \\
        --checkpoint outputs/vit_large_dinov3/best_model.pth \\
        -- test.rerank=false test.flip_test=true
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_default_device
from src.utils.logger import setup_logger
from src.data import UrbanReIDDataset, build_val_loader
from src.data.dataset import IDX_TO_CLASS
from src.models import ReIDModel
from src.engine.evaluator import eval_func, extract_features
from src.postprocess.distance import cosine_similarity
from src.postprocess.rerank import re_ranking
from src.postprocess.class_mask import apply_class_mask


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the val split")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--", dest="overrides", nargs="*", default=[],
                        help="Config key=value overrides")
    # Collect everything after '--' as overrides
    argv = sys.argv[1:]
    sep = argv.index("--") if "--" in argv else len(argv)
    main_argv = argv[:sep]
    overrides = argv[sep + 1:] if sep < len(argv) else []

    args = parser.parse_args(main_argv)

    cfg = load_config(args.config, overrides if overrides else None)
    set_seed(cfg.trainer.seed)
    device = get_default_device()

    logger = setup_logger("urban_reid", cfg.trainer.output_dir)
    logger.info(f"Config     : {args.config}")
    logger.info(f"Checkpoint : {args.checkpoint}")

    # Dataset
    dataset = UrbanReIDDataset(cfg)
    logger.info(
        f"Val query: {len(dataset.val_query)} images  |  "
        f"Val gallery: {len(dataset.val_gallery)} images"
    )

    # Model
    has_class_obj = any(obj.type == "class_ce" for obj in cfg.objectives)
    num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0
    model = ReIDModel(cfg, num_pids=dataset.num_train_pids, num_obj_classes=num_obj_classes)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_best_map = ckpt.get("best_mAP", None)
    logger.info(
        f"Loaded checkpoint  epoch={ckpt_epoch}"
        + (f"  saved_best_mAP={ckpt_best_map:.4f}" if ckpt_best_map is not None else "")
    )

    model = model.to(device)

    # Val loader + metadata
    val_loader, num_val_query = build_val_loader(cfg, dataset)
    val_samples = list(dataset.val_query) + list(dataset.val_gallery)
    q_pids         = np.array([s.pid          for s in val_samples[:num_val_query]])
    g_pids         = np.array([s.pid          for s in val_samples[num_val_query:]])
    q_camids       = np.array([s.camid        for s in val_samples[:num_val_query]])
    g_camids       = np.array([s.camid        for s in val_samples[num_val_query:]])
    q_class_labels = np.array([s.class_label  for s in val_samples[:num_val_query]])
    g_class_labels = np.array([s.class_label  for s in val_samples[num_val_query:]])
    q_image_names  = [os.path.basename(s.img_path) for s in val_samples[:num_val_query]]

    # Extract features once — reused for metrics and CSV
    logger.info("Extracting features...")
    features = extract_features(
        model, val_loader,
        flip=cfg.test.flip_test,
        feat_norm=cfg.test.feat_norm,
    )
    qf = features[:num_val_query]
    gf = features[num_val_query:]
    logger.info(f"Query: {qf.shape}  Gallery: {gf.shape}")

    # Distance matrix
    q_g_sim = cosine_similarity(qf, gf)
    if cfg.test.rerank:
        logger.info("Applying re-ranking...")
        q_q_sim = cosine_similarity(qf, qf)
        g_g_sim = cosine_similarity(gf, gf)
        dist = re_ranking(
            q_g_sim, q_q_sim, g_g_sim,
            k1=cfg.test.rerank_k1, k2=cfg.test.rerank_k2,
            lambda_value=cfg.test.rerank_lambda,
        )
    else:
        dist = 1.0 - q_g_sim

    if cfg.test.class_mask:
        logger.info("Applying class mask...")
        dist = apply_class_mask(dist, q_class_labels.tolist(), g_class_labels.tolist())

    # ── Compute metrics ───────────────────────────────────────────────────────
    cmc, mAP = eval_func(dist, q_pids, g_pids, q_camids, g_camids)
    metrics = {
        "mAP": float(mAP),
        "R1":  float(cmc[0]) if len(cmc) > 0 else 0.0,
        "R5":  float(cmc[4]) if len(cmc) > 4 else 0.0,
        "R10": float(cmc[9]) if len(cmc) > 9 else 0.0,
    }
    for cls_idx, cls_name in IDX_TO_CLASS.items():
        qm = q_class_labels == cls_idx
        gm = g_class_labels == cls_idx
        if qm.sum() == 0 or gm.sum() == 0:
            continue
        cls_cmc, cls_mAP = eval_func(
            dist[np.ix_(qm, gm)], q_pids[qm], g_pids[gm], q_camids[qm], g_camids[gm]
        )
        metrics[f"mAP_{cls_name}"] = float(cls_mAP)
        metrics[f"R1_{cls_name}"]  = float(cls_cmc[0]) if len(cls_cmc) > 0 else 0.0

    # ── Pretty-print ──────────────────────────────────────────────────────────
    col = 22
    print("\n" + "=" * 52)
    print(f"  Checkpoint : {os.path.basename(args.checkpoint)}")
    print("=" * 52)
    print(f"  {'mAP':<{col}}: {metrics['mAP']:.4f}")
    print(f"  {'Rank-1':<{col}}: {metrics['R1']:.4f}")
    print(f"  {'Rank-5':<{col}}: {metrics['R5']:.4f}")
    print(f"  {'Rank-10':<{col}}: {metrics['R10']:.4f}")
    print("-" * 52)
    print("  Per-class mAP:")
    for cls_idx, cls_name in sorted(IDX_TO_CLASS.items()):
        map_key = f"mAP_{cls_name}"
        r1_key  = f"R1_{cls_name}"
        if map_key in metrics:
            print(f"    {cls_name:<{col}}: mAP={metrics[map_key]:.4f}  R1={metrics[r1_key]:.4f}")
        else:
            print(f"    {cls_name:<{col}}: no samples")
    print("=" * 52 + "\n")

    # ── Write val submission CSV ───────────────────────────────────────────────
    top_k = min(cfg.submission.top_k, len(g_pids))
    ranked_indices = np.argsort(dist, axis=1)[:, :top_k]

    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f"{ckpt_stem}_val_submission.csv",
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for img_name, row in zip(q_image_names, ranked_indices):
            writer.writerow([img_name, " ".join(str(idx + 1) for idx in row)])

    logger.info(f"Val submission CSV written to {csv_path}")
    logger.info(f"  Queries: {len(q_image_names)}, Top-K: {top_k}")


if __name__ == "__main__":
    main()
