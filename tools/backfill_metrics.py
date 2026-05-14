#!/usr/bin/env python3
"""Backfill missing per-class mAP and test-weighted-mAP into TensorBoard logs.

For completed experiments whose checkpoints exist but whose TB logs lack
per-class mAP (mAP_Container, mAP_Crosswalk, etc.) and/or test-weighted-mAP,
this script:
  1. Loads each periodic checkpoint (checkpoint_ep{N}.pth)
  2. Runs the evaluator to compute per-class val metrics
  3. Optionally runs train-mAP evaluation for per-class train metrics
  4. Writes the missing scalars into the existing TensorBoard log directory

Experiments that are still running (no final_model.pth) are skipped.

Usage:
    python tools/backfill_metrics.py [--skip-train] [--experiments exp1 exp2 ...]
"""

import os
import sys
import glob
import argparse
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.utils.device import get_default_device
from src.utils.seed import set_seed
from src.data import UrbanReIDDataset, build_val_loader
from src.data.dataloader import build_train_probe_loader
from src.data.dataset import IDX_TO_CLASS
from src.models import ReIDModel
from src.engine.evaluator import (
    eval_func, extract_features, TEST_CLASS_WEIGHTS,
)
from src.postprocess.distance import cosine_similarity

logger = logging.getLogger("backfill")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

OUTPUTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def discover_experiments(outputs_root, filter_names=None):
    """Return list of (name, dir) for completed experiments only."""
    experiments = []
    for name in sorted(os.listdir(outputs_root)):
        exp_dir = os.path.join(outputs_root, name)
        if not os.path.isdir(exp_dir):
            continue
        if filter_names and name not in filter_names:
            continue
        # Skip experiments still running (no final_model.pth)
        if not os.path.exists(os.path.join(exp_dir, "final_model.pth")):
            logger.info(f"SKIP {name} — still running (no final_model.pth)")
            continue
        experiments.append((name, exp_dir))
    return experiments


def find_checkpoints(exp_dir, eval_period=5):
    """Return sorted list of (epoch_0indexed, checkpoint_path).

    checkpoint_ep5.pth  -> epoch 4  (0-indexed)
    checkpoint_ep10.pth -> epoch 9
    ...
    """
    ckpts = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "checkpoint_ep*.pth"))):
        basename = os.path.basename(path)
        # e.g. "checkpoint_ep25.pth" -> 25
        ep_num = int(basename.replace("checkpoint_ep", "").replace(".pth", ""))
        epoch_0 = ep_num - 1  # trainer uses 0-indexed epochs for TB
        ckpts.append((epoch_0, path))
    return sorted(ckpts, key=lambda x: x[0])


def load_model(cfg, dataset, checkpoint_path):
    """Instantiate model and load checkpoint weights."""
    device = get_default_device()
    has_class_obj = any(obj.type == "class_ce" for obj in cfg.objectives)
    num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0
    model = ReIDModel(cfg, num_pids=dataset.num_train_pids,
                      num_obj_classes=num_obj_classes)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Strip DDP "module." prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def compute_val_metrics(model, cfg, val_loader, num_val_query,
                        q_pids, g_pids, q_camids, g_camids,
                        q_class_labels, g_class_labels):
    """Compute full val metrics including per-class mAP and weighted mAP."""
    features = extract_features(
        model, val_loader,
        flip=cfg.test.flip_test,
        feat_norm=cfg.test.feat_norm,
    )
    qf = features[:num_val_query]
    gf = features[num_val_query:]

    sim = cosine_similarity(qf, gf)
    dist = 1.0 - sim

    cmc, mAP = eval_func(dist, q_pids, g_pids, q_camids, g_camids)

    metrics = {
        "mAP": float(mAP),
        "R1": float(cmc[0]) if len(cmc) > 0 else 0.0,
        "R5": float(cmc[4]) if len(cmc) > 4 else 0.0,
        "R10": float(cmc[9]) if len(cmc) > 9 else 0.0,
    }

    # Per-class mAP
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
        metrics[f"mAP_{cls_name}"] = float(cls_mAP)
        metrics[f"R1_{cls_name}"] = float(cls_cmc[0]) if len(cls_cmc) > 0 else 0.0

    # Test-weighted mAP
    metrics["test-weighted-mAP"] = sum(
        TEST_CLASS_WEIGHTS[cls] * metrics.get(f"mAP_{cls}", 0.0)
        for cls in TEST_CLASS_WEIGHTS
    )

    return metrics


def compute_train_metrics(model, cfg, train_probe_loader, train_pids,
                          train_class_labels):
    """Compute per-class train mAP (lightweight version — no chunked eval,
    uses the same logic as eval_train_map but returns only per-class metrics)."""
    from src.engine.evaluator import eval_train_map
    results = eval_train_map(
        model, train_probe_loader, train_pids,
        train_class_labels=train_class_labels,
        flip=cfg.test.flip_test,
        feat_norm=cfg.test.feat_norm,
    )
    return results


def check_existing_tb_tags(tb_dir):
    """Read existing TensorBoard events to find which tags already exist at
    which epochs. Returns dict: {tag: set_of_epochs}."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        existing = {}
        for tag in ea.Tags().get("scalars", []):
            steps = {e.step for e in ea.Scalars(tag)}
            existing[tag] = steps
        return existing
    except Exception as e:
        logger.warning(f"Could not read existing TB events: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Backfill missing TB metrics")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Only process these experiment names. Default: all completed.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip train-mAP backfill (much faster)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be done, no model loading")
    args = parser.parse_args()

    outputs_root = os.path.abspath(OUTPUTS_ROOT)
    experiments = discover_experiments(outputs_root, args.experiments)
    if not experiments:
        logger.info("No completed experiments to process.")
        return

    logger.info(f"Found {len(experiments)} completed experiment(s): "
                f"{[n for n, _ in experiments]}")

    # ── Load dataset once (shared across all experiments) ─────────────────
    # Use any completed experiment's config to load the dataset;
    # dataset root and CSV paths are the same for all.
    sample_cfg = load_config(os.path.join(experiments[0][1], "config.yaml"))
    dataset = UrbanReIDDataset(sample_cfg)
    logger.info(f"Dataset loaded: {len(dataset.train)} train, "
                f"{len(dataset.val_query)} val_query, "
                f"{len(dataset.val_gallery)} val_gallery")

    # Pre-compute val arrays (same for all experiments)
    val_samples = list(dataset.val_query) + list(dataset.val_gallery)
    num_val_query = len(dataset.val_query)
    q_pids = np.array([s.pid for s in val_samples[:num_val_query]])
    g_pids = np.array([s.pid for s in val_samples[num_val_query:]])
    q_camids = np.array([s.camid for s in val_samples[:num_val_query]])
    g_camids = np.array([s.camid for s in val_samples[num_val_query:]])
    q_class_labels = np.array([s.class_label for s in val_samples[:num_val_query]])
    g_class_labels = np.array([s.class_label for s in val_samples[num_val_query:]])

    # Pre-compute train arrays
    train_pids = np.array([s.pid for s in dataset.train])
    train_class_labels = np.array([s.class_label for s in dataset.train])

    # Per-class tags we want to exist
    val_required_tags = set()
    for cls_name in IDX_TO_CLASS.values():
        val_required_tags.add(f"val/mAP_{cls_name}")
        val_required_tags.add(f"val/R1_{cls_name}")
    val_required_tags.add("val/test-weighted-mAP")

    train_required_tags = set()
    for cls_name in IDX_TO_CLASS.values():
        train_required_tags.add(f"train/train_mAP_{cls_name}")
        train_required_tags.add(f"train/train_R1_{cls_name}")
    train_required_tags.add("train/train-test-weighted-mAP")

    for exp_name, exp_dir in experiments:
        logger.info(f"\n{'='*60}\nProcessing: {exp_name}\n{'='*60}")

        cfg = load_config(os.path.join(exp_dir, "config.yaml"))
        eval_period = cfg.trainer.eval_period
        tb_dir = os.path.join(exp_dir, "tb_logs")

        # Check what's already in TB
        existing = check_existing_tb_tags(tb_dir)

        checkpoints = find_checkpoints(exp_dir, eval_period)
        if not checkpoints:
            logger.warning(f"No periodic checkpoints found in {exp_dir}")
            continue

        # Determine which epochs need backfill
        val_epochs_needed = []
        train_epochs_needed = []
        for epoch_0, ckpt_path in checkpoints:
            # Check if val per-class tags are missing for this epoch
            val_missing = any(
                epoch_0 not in existing.get(tag, set())
                for tag in val_required_tags
            )
            if val_missing:
                val_epochs_needed.append((epoch_0, ckpt_path))

            # Check if train per-class tags are missing
            if not args.skip_train:
                train_missing = any(
                    epoch_0 not in existing.get(tag, set())
                    for tag in train_required_tags
                )
                if train_missing:
                    train_epochs_needed.append((epoch_0, ckpt_path))

        if not val_epochs_needed and not train_epochs_needed:
            logger.info(f"  All metrics already present — nothing to do.")
            continue

        logger.info(f"  Val backfill needed for epochs: "
                    f"{[e+1 for e, _ in val_epochs_needed]}")
        if not args.skip_train:
            logger.info(f"  Train backfill needed for epochs: "
                        f"{[e+1 for e, _ in train_epochs_needed]}")

        if args.dry_run:
            continue

        # Build val loader with this experiment's config (image size may differ)
        val_loader, _ = build_val_loader(cfg, dataset)

        # Build train probe loader if needed
        train_probe_loader = None
        if train_epochs_needed and not args.skip_train:
            train_probe_loader = build_train_probe_loader(cfg, dataset)

        # Open a new TB writer that appends to the existing log dir
        writer = SummaryWriter(log_dir=tb_dir)

        # ── Val backfill ──────────────────────────────────────────────────
        for epoch_0, ckpt_path in val_epochs_needed:
            ep_display = epoch_0 + 1
            logger.info(f"  [val] epoch {ep_display}: loading {os.path.basename(ckpt_path)}")
            model = load_model(cfg, dataset, ckpt_path)

            metrics = compute_val_metrics(
                model, cfg, val_loader, num_val_query,
                q_pids, g_pids, q_camids, g_camids,
                q_class_labels, g_class_labels,
            )

            # Write only the per-class and weighted tags (don't duplicate mAP/R1/R5/R10)
            tags_written = []
            for key, val in metrics.items():
                tag = f"val/{key}"
                if tag in val_required_tags:
                    if epoch_0 not in existing.get(tag, set()):
                        writer.add_scalar(tag, val, epoch_0)
                        tags_written.append(f"{tag}={val:.4f}")

            logger.info(f"    mAP={metrics['mAP']:.4f}  "
                        f"test-weighted-mAP={metrics.get('test-weighted-mAP', 0):.4f}  "
                        f"wrote {len(tags_written)} tags")

            del model
            torch.cuda.empty_cache()

        # ── Train backfill ────────────────────────────────────────────────
        if train_probe_loader and not args.skip_train:
            for epoch_0, ckpt_path in train_epochs_needed:
                ep_display = epoch_0 + 1
                logger.info(f"  [train] epoch {ep_display}: loading {os.path.basename(ckpt_path)}")
                model = load_model(cfg, dataset, ckpt_path)

                train_results = compute_train_metrics(
                    model, cfg, train_probe_loader, train_pids, train_class_labels,
                )

                tags_written = []
                for key, val in train_results.items():
                    tag = f"train/{key}"
                    if tag in train_required_tags:
                        if epoch_0 not in existing.get(tag, set()):
                            writer.add_scalar(tag, val, epoch_0)
                            tags_written.append(f"{tag}={val:.4f}")

                logger.info(f"    train_mAP={train_results['train_mAP']:.4f}  "
                            f"train-test-weighted-mAP="
                            f"{train_results.get('train-test-weighted-mAP', 0):.4f}  "
                            f"wrote {len(tags_written)} tags")

                del model
                torch.cuda.empty_cache()

        writer.close()
        logger.info(f"  Done — TB logs updated in {tb_dir}")

    logger.info("\nBackfill complete.")


if __name__ == "__main__":
    main()
