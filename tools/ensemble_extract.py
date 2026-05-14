#!/usr/bin/env python3
"""Ensemble feature extraction — extract val/test features from all experiments.

For each experiment, finds the checkpoint with the best val/test-weighted-mAP
(read from TensorBoard logs), loads the model, extracts features, and saves
them as .npy files for downstream ensemble search.

Usage:
    python tools/ensemble_extract.py
    python tools/ensemble_extract.py --split test
    python tools/ensemble_extract.py --include vit_large_dinov3,swin_large_in22k
    python tools/ensemble_extract.py --exclude specialist_trafficsignal_vit_large
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.config import load_config
from src.utils.device import get_default_device
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data import UrbanReIDDataset, build_val_loader, build_test_loader
from src.models import ReIDModel
from src.engine.evaluator import extract_features


SKIP_DIRS = {"backfill_logs", "ensemble_features"}
TB_TAG = "val/test-weighted-mAP"


def discover_experiments(experiments_dir, include=None, exclude=None):
    """Find experiment directories that have config.yaml."""
    experiments = []
    for name in sorted(os.listdir(experiments_dir)):
        if name in SKIP_DIRS:
            continue
        exp_dir = os.path.join(experiments_dir, name)
        if not os.path.isdir(exp_dir):
            continue
        if not os.path.isfile(os.path.join(exp_dir, "config.yaml")):
            continue
        if include is not None and name not in include:
            continue
        if exclude is not None and name in exclude:
            continue
        experiments.append(name)
    return experiments


def get_best_epoch_from_tb(tb_dir):
    """Read TB logs and return (best_epoch, best_value) for val/test-weighted-mAP.

    Returns None if the tag doesn't exist.
    """
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    if TB_TAG not in scalars:
        return None
    events = ea.Scalars(TB_TAG)
    if not events:
        return None
    best = max(events, key=lambda e: e.value)
    return best.step, best.value


def find_checkpoint(exp_dir, best_epoch):
    """Map a 0-indexed epoch to the nearest checkpoint file.

    Checkpoints are named checkpoint_ep{N}.pth where N = epoch+1.
    Falls back to best_model.pth if exact match not found.
    """
    # Exact match
    exact = os.path.join(exp_dir, f"checkpoint_ep{best_epoch + 1}.pth")
    if os.path.isfile(exact):
        return exact

    # Find nearest available checkpoint
    pattern = os.path.join(exp_dir, "checkpoint_ep*.pth")
    ckpts = glob.glob(pattern)
    if ckpts:
        # Parse epoch numbers
        def parse_ep(p):
            base = os.path.basename(p)
            return int(base.replace("checkpoint_ep", "").replace(".pth", ""))
        ckpts_with_ep = [(parse_ep(p), p) for p in ckpts]
        # Find closest to best_epoch+1
        ckpts_with_ep.sort(key=lambda x: abs(x[0] - (best_epoch + 1)))
        return ckpts_with_ep[0][1]

    # Last resort
    best_path = os.path.join(exp_dir, "best_model.pth")
    if os.path.isfile(best_path):
        return best_path

    return None


def main():
    parser = argparse.ArgumentParser(description="Ensemble Feature Extraction")
    parser.add_argument("--output-dir", type=str, default="outputs/ensemble_features")
    parser.add_argument("--experiments-dir", type=str, default="outputs")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--include", type=str, default=None,
                        help="Comma-separated experiment names to include (only these)")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Comma-separated experiment names to exclude")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if features already exist")
    args = parser.parse_args()

    if args.include and args.exclude:
        parser.error("--include and --exclude are mutually exclusive")

    include = set(args.include.split(",")) if args.include else None
    exclude = set(args.exclude.split(",")) if args.exclude else None

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("ensemble_extract", args.output_dir)
    device = get_default_device()

    # Discover experiments
    experiments = discover_experiments(args.experiments_dir, include, exclude)
    logger.info(f"Found {len(experiments)} experiments: {experiments}")

    # Track shared val metadata (saved once since all experiments use the same split)
    val_meta_saved = False
    manifest = []

    for exp_name in experiments:
        exp_dir = os.path.join(args.experiments_dir, exp_name)
        feat_dir = os.path.join(args.output_dir, exp_name)

        # Check if already extracted
        qf_path = os.path.join(feat_dir, f"{args.split}_qf.npy")
        gf_path = os.path.join(feat_dir, f"{args.split}_gf.npy")
        meta_path = os.path.join(feat_dir, "meta.json")
        if not args.force and os.path.isfile(qf_path) and os.path.isfile(gf_path) and os.path.isfile(meta_path):
            logger.info(f"[{exp_name}] Features already exist, skipping (use --force to re-extract)")
            with open(meta_path) as f:
                manifest.append(json.load(f))
            continue

        # Read TB logs for best epoch
        tb_dir = os.path.join(exp_dir, "tb_logs")
        if not os.path.isdir(tb_dir):
            logger.warning(f"[{exp_name}] No tb_logs/ directory, skipping")
            continue
        result = get_best_epoch_from_tb(tb_dir)
        if result is None:
            logger.warning(f"[{exp_name}] No '{TB_TAG}' tag in TB logs, skipping")
            continue
        best_epoch, best_wmAP = result
        logger.info(f"[{exp_name}] Best val/test-weighted-mAP={best_wmAP:.4f} at epoch {best_epoch}")

        # Find checkpoint
        ckpt_path = find_checkpoint(exp_dir, best_epoch)
        if ckpt_path is None:
            logger.warning(f"[{exp_name}] No checkpoint found near epoch {best_epoch}, skipping")
            continue
        logger.info(f"[{exp_name}] Using checkpoint: {os.path.basename(ckpt_path)}")

        # Load config
        cfg_path = os.path.join(exp_dir, "config.yaml")
        cfg = load_config(cfg_path)
        set_seed(cfg.trainer.seed)

        # Build dataset
        dataset = UrbanReIDDataset(cfg)

        # Build model
        has_class_obj = any(obj.type == "class_ce" for obj in cfg.objectives)
        num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0
        model = ReIDModel(cfg, num_pids=dataset.num_train_pids, num_obj_classes=num_obj_classes)

        # Load weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        logger.info(f"[{exp_name}] Model loaded ({cfg.backbone.name})")

        # Build loader
        if args.split == "val":
            loader, num_query = build_val_loader(cfg, dataset)
            samples = list(dataset.val_query) + list(dataset.val_gallery)
        else:
            loader, num_query = build_test_loader(cfg, dataset)
            samples = list(dataset.query) + list(dataset.gallery)

        # Extract features
        features = extract_features(
            model, loader,
            flip=cfg.test.flip_test,
            feat_norm=cfg.test.feat_norm,
        )
        qf = features[:num_query]
        gf = features[num_query:]

        # Ensure L2 normalization
        qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
        gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)

        logger.info(f"[{exp_name}] Features: qf={qf.shape}, gf={gf.shape}")

        # Save features
        os.makedirs(feat_dir, exist_ok=True)
        np.save(qf_path, qf.astype(np.float32))
        np.save(gf_path, gf.astype(np.float32))

        # Save metadata
        meta = {
            "experiment": exp_name,
            "backbone": cfg.backbone.name,
            "checkpoint": os.path.basename(ckpt_path),
            "best_epoch": int(best_epoch),
            "best_test_weighted_mAP": float(best_wmAP),
            "feature_dim": int(qf.shape[1]),
            "num_query": int(qf.shape[0]),
            "num_gallery": int(gf.shape[0]),
            "flip_test": bool(cfg.test.flip_test),
            "feat_norm": bool(cfg.test.feat_norm),
            "input_size_test": list(cfg.input.size_test),
            "split": args.split,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)

        # Save shared val metadata (pids, camids, class_labels) once
        if not val_meta_saved:
            pids = np.array([s.pid for s in samples])
            camids = np.array([s.camid for s in samples])
            class_labels = np.array([s.class_label for s in samples])
            np.savez(
                os.path.join(args.output_dir, f"{args.split}_meta.npz"),
                pids=pids,
                camids=camids,
                class_labels=class_labels,
                num_query=np.array(num_query),
            )
            val_meta_saved = True
            logger.info(f"Saved shared {args.split} metadata: {len(pids)} samples, {num_query} queries")

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest with {len(manifest)} experiments to {manifest_path}")


if __name__ == "__main__":
    main()
