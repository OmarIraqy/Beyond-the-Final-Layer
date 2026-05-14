#!/usr/bin/env python3
"""Feature extraction tool — extract and save features for any split.

Usage:
    python tools/extract_features.py --config outputs/config.yaml --split val
    python tools/extract_features.py --config outputs/config.yaml --split test
"""

import os
import sys
import argparse
import numpy as np

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_default_device
from src.utils.logger import setup_logger
from src.data import UrbanReIDDataset, build_val_loader, build_test_loader
from src.models import ReIDModel
from src.engine.evaluator import extract_features


def main():
    parser = argparse.ArgumentParser(description="Urban ReID Feature Extraction")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides if args.overrides else None)
    set_seed(cfg.trainer.seed)
    device = get_default_device()

    output_dir = args.output_dir or cfg.trainer.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("urban_reid", output_dir)

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
        logger.info(f"Loaded weights from {cfg.test.weight}")

    model = model.to(device)

    # Build loader
    if args.split == "val":
        loader, num_query = build_val_loader(cfg, dataset)
        prefix = "val"
    else:
        loader, num_query = build_test_loader(cfg, dataset)
        prefix = "test"

    # Extract
    features = extract_features(
        model, loader,
        flip=cfg.test.flip_test,
        feat_norm=cfg.test.feat_norm,
    )
    qf = features[:num_query]
    gf = features[num_query:]

    np.save(os.path.join(output_dir, f"{prefix}_qf.npy"), qf)
    np.save(os.path.join(output_dir, f"{prefix}_gf.npy"), gf)
    logger.info(f"Saved {prefix} features: qf={qf.shape}, gf={gf.shape}")


if __name__ == "__main__":
    main()
