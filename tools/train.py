#!/usr/bin/env python3
"""Training entrypoint.

Usage:
    python tools/train.py --config configs/default.yaml
    python tools/train.py --config configs/experiment/baseline_resnet50.yaml
    python tools/train.py --config configs/default.yaml solver.lr=1e-4 backbone.name=convnext_base

    # Multi-GPU (DDP via torchrun)
    torchrun --nproc_per_node=2 tools/train.py --config configs/default.yaml

Every run saves its full resolved config to the output directory for reproducibility.
"""

import os
import sys
import argparse
import logging
import numpy as np

import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.device import get_default_device
from src.utils.distributed import setup_ddp, cleanup_ddp, is_main_process, get_rank
from src.data import UrbanReIDDataset, build_train_loader, build_val_loader
from src.data.dataloader import build_train_probe_loader
from src.models import ReIDModel
from src.losses import ObjectiveCombiner
from src.engine.trainer import Trainer, build_optimizer, build_scheduler


def main():
    parser = argparse.ArgumentParser(description="Urban ReID Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("overrides", nargs="*",
                        help="Override config keys: key=value")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, args.overrides if args.overrides else None)

    # DDP setup (if launched via torchrun)
    setup_ddp()
    rank = get_rank()
    device = get_default_device(rank)

    # Seed everything for reproducibility
    set_seed(cfg.trainer.seed + rank)

    # Output directory
    os.makedirs(cfg.trainer.output_dir, exist_ok=True)

    # Logger
    logger = setup_logger("urban_reid", cfg.trainer.output_dir, rank=rank)

    if is_main_process():
        logger.info(f"Config:\n{cfg}")
        logger.info(f"Device: {device}")
        # Save config immediately for reproducibility
        config_path = save_config(cfg, cfg.trainer.output_dir)
        logger.info(f"Config saved to {config_path}")

    # Dataset
    logger.info("Loading dataset...")
    dataset = UrbanReIDDataset(cfg)
    logger.info(
        f"Train: {len(dataset.train)} images, {dataset.num_train_pids} identities, "
        f"{dataset.num_train_cams} cameras"
    )
    logger.info(f"Val query: {len(dataset.val_query)}, Val gallery: {len(dataset.val_gallery)}")

    # Determine if we have class-aware objectives
    has_class_obj = any(
        obj.type == "class_ce" for obj in cfg.objectives
    )
    num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0

    # Model
    model = ReIDModel(cfg, num_pids=dataset.num_train_pids, num_obj_classes=num_obj_classes)
    model = model.to(device)

    # Save embed_dim before DDP wrapping (needed for loss construction)
    embed_dim = model.head.in_dim
    num_pids = dataset.num_train_pids

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model: {cfg.backbone.name}, {n_params:.1f}M parameters ({n_trainable:.1f}M trainable)")
        if cfg.backbone.freeze_percent > 0:
            logger.info(f"Backbone freeze_percent: {cfg.backbone.freeze_percent:.0%}")

    # DDP wrapping
    if dist.is_initialized():
        ddp_kwargs = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [rank]
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    # Dataloaders
    train_loader = build_train_loader(cfg, dataset)
    val_loader, num_val_query = build_val_loader(cfg, dataset)

    # Pre-compute val pids/camids to avoid iterating dataloader twice during eval
    val_samples = list(dataset.val_query) + list(dataset.val_gallery)
    val_pids = np.array([s.pid for s in val_samples])
    val_camids = np.array([s.camid for s in val_samples])
    val_class_labels = np.array([s.class_label for s in val_samples])

    # Train probe loader for train mAP (overfitting indicator)
    train_probe_loader = None
    train_pids = None
    train_class_labels = None
    if cfg.trainer.eval_train_map:
        train_probe_loader = build_train_probe_loader(cfg, dataset)
        train_pids = np.array([s.pid for s in dataset.train])
        train_class_labels = np.array([s.class_label for s in dataset.train])
        logger.info(f"Train mAP enabled: {len(dataset.train)} train images will be probed at eval")

    # Losses
    objective_combiner = ObjectiveCombiner(
        cfg.objectives,
        num_classes=num_pids,
        embed_dim=embed_dim,
        num_train_cams=dataset.num_train_cams,
    ).to(device)
    logger.info(f"Objectives: {[obj.name for obj in cfg.objectives]}")

    # Optimizer and scheduler (include loss params e.g. ArcFace weights, center embeddings)
    optimizer = build_optimizer(cfg, model, objective_combiner)
    scheduler = build_scheduler(cfg, optimizer)

    # Trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        objective_combiner=objective_combiner,
        optimizer=optimizer,
        scheduler=scheduler,
        val_loader=val_loader,
        num_val_query=num_val_query,
        val_pids=val_pids,
        val_camids=val_camids,
        val_class_labels=val_class_labels,
        train_probe_loader=train_probe_loader,
        train_pids=train_pids,
        train_class_labels=train_class_labels,
    )

    # Resume from checkpoint if specified
    if cfg.trainer.resume:
        trainer.resume(cfg.trainer.resume)

    # Train
    trainer.train()

    cleanup_ddp()


if __name__ == "__main__":
    main()
