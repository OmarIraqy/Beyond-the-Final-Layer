#!/usr/bin/env python3
"""Multi-block feature inference for ViT- and Swin-family ReID models.

Builds a per-image feature vector by concatenating block-level tokens from
multiple transformer blocks, then either evaluates on the validation set or
generates a competition submission CSV for the test set.

  ViT-family  (DINOv3, DINOv2, EVA, plain ViT, …)
      backbone.blocks  → CLS token ``output[:, 0, :]`` per block.

  Swin-family (swin_*, swinv2_*, …)
      backbone.layers  → flattened per-stage blocks
      No CLS token — uses global-average-pool of spatial tokens
      ``output.mean(dim=1)`` per block, the closest equivalent summary.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
# Validate — specific blocks
python tools/infer_multiblock.py \\
    --config outputs/vit_large_dinov3/config.yaml \\
    --split val \\
    --blocks 18,20,22,23 \\
    test.weight=outputs/vit_large_dinov3/checkpoint_ep80.pth

# Validate — last 4 blocks
python tools/infer_multiblock.py \\
    --config outputs/vit_large_dinov3/config.yaml \\
    --split val \\
    --blocks last:4 \\
    test.weight=outputs/vit_large_dinov3/checkpoint_ep80.pth

# Test submission — all blocks
python tools/infer_multiblock.py \\
    --config outputs/vit_large_dinov3/config.yaml \\
    --split test \\
    --blocks all \\
    test.weight=outputs/vit_large_dinov3/checkpoint_ep80.pth
──────────────────────────────────────────────────────────────────────────────
Output files (saved to --output-dir or cfg.trainer.output_dir)
──────────────────────────────────────────────────────────────────────────────
  multiblock_<blocks_tag>_val_qf.npy
  multiblock_<blocks_tag>_val_gf.npy
  multiblock_<blocks_tag>_val_metrics.json    ← val split
  multiblock_<blocks_tag>_test_qf.npy
  multiblock_<blocks_tag>_test_gf.npy
  multiblock_<blocks_tag>_submission.csv      ← test split
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.utils.device import get_default_device, get_module_device, move_to_device
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data import UrbanReIDDataset, build_val_loader, build_test_loader
from src.data.dataset import IDX_TO_CLASS
from src.models import ReIDModel, unwrap_backbone_module
from src.engine.evaluator import eval_func, TEST_CLASS_WEIGHTS
from src.postprocess.distance import cosine_similarity
from src.postprocess.rerank import re_ranking
from src.postprocess.class_mask import apply_class_mask


# ── Multi-block token extractor ──────────────────────────────────────────────

_CONTAINER = (nn.Sequential, nn.ModuleList)


def _get_all_blocks(backbone: nn.Module):
    """Return (arch, flat_block_list) for ViT or Swin backbones.

    ViT  → arch='vit',  blocks = list(backbone.blocks)
    Swin → arch='swin', blocks = [blk for stage in backbone.layers for blk in stage.blocks]
    """
    backbone = unwrap_backbone_module(backbone)
    if hasattr(backbone, "blocks") and isinstance(backbone.blocks, _CONTAINER):
        return "vit", list(backbone.blocks)
    if hasattr(backbone, "layers") and isinstance(backbone.layers, _CONTAINER):
        flat = []
        for stage in backbone.layers:
            if hasattr(stage, "blocks") and isinstance(stage.blocks, _CONTAINER):
                flat.extend(list(stage.blocks))
        if flat:
            return "swin", flat
    raise ValueError(
        "Unsupported backbone: needs backbone.blocks (ViT-family) "
        "or backbone.layers with .blocks (Swin-family)."
    )


class MultiBlockCLSExtractor:
    """Hook backbone blocks to capture a per-block feature vector.

    ViT-family  (backbone.blocks)
        Extracts the CLS token ``output[:, 0, :]`` — shape ``[B, D]`` per block.

    Swin-family (backbone.layers → flattened blocks)
        Extracts the global-average-pooled spatial tokens ``output.mean(dim=1)``
        — shape ``[B, C]`` per block.  Swin has no CLS token; GAP gives the
        closest equivalent summary of what each block has attended to.

    After ``backbone(images)`` runs, ``get_features()`` returns the
    concatenation across all selected blocks: ``[B, K × D]``.

    Args:
        backbone: timm ViT or Swin backbone.
        block_indices: sorted list of block indices (0-based) to hook.

    Raises:
        ValueError: unsupported backbone, or an index is out of range.
    """

    def __init__(self, backbone: nn.Module, block_indices: list):
        self.arch, all_blocks = _get_all_blocks(backbone)
        self.num_blocks = len(all_blocks)

        for idx in block_indices:
            if not (0 <= idx < self.num_blocks):
                raise ValueError(
                    f"Block index {idx} out of range [0, {self.num_blocks - 1}]"
                )

        self.block_indices = block_indices
        self._captured: dict = {}
        self._hooks = []

        for idx in block_indices:
            self._hooks.append(
                all_blocks[idx].register_forward_hook(self._make_hook(idx))
            )

    def _make_hook(self, idx: int):
        def hook_fn(module, inp, out):
            if self.arch == "vit":
                # ViT block output: [B, N, D] — position 0 is always the CLS token
                self._captured[idx] = out[:, 0, :].detach()
            else:
                # Swin block output can be:
                #   [B, H*W, C] (3D, older timm) — GAP over dim 1
                #   [B, H, W, C] (4D, newer timm) — GAP over dims 1 and 2
                # Using reshape so we always average over all spatial positions.
                if out.dim() == 3:
                    self._captured[idx] = out.mean(dim=1).detach()       # [B, C]
                else:
                    self._captured[idx] = out.mean(dim=(1, 2)).detach()  # [B, C]
        return hook_fn

    def clear(self):
        """Clear stored tensors between batches."""
        self._captured.clear()

    def get_features(self) -> torch.Tensor:
        """Return concatenated block tokens ``[B, K × D]``."""
        parts = [self._captured[idx] for idx in self.block_indices]
        return torch.cat(parts, dim=1)

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_multiblock_features(
    model: ReIDModel,
    extractor: MultiBlockCLSExtractor,
    dataloader,
    flip: bool = False,
    feat_norm: bool = True,
) -> np.ndarray:
    """Extract multi-block features for all samples in a dataloader.

    Runs only the backbone forward pass (not the ReID head) to capture
    intermediate block tokens (CLS for ViT, GAP for Swin). Optional
    horizontal-flip TTA averages features from original + flipped image.

    Args:
        model: ReIDModel — only ``model.backbone`` is called.
        extractor: attached ``MultiBlockCLSExtractor``.
        dataloader: yields dicts with key ``"images"``.
        flip: if True, average original + horizontally flipped features.
        feat_norm: if True, L2-normalise the concatenated feature vector.

    Returns:
        ``[N, D_concat]`` float32 numpy array.
    """
    model.eval()
    device = get_module_device(model)
    all_features = []

    for batch in tqdm(dataloader, desc="Extracting multi-block features", leave=False):
        images = move_to_device(batch["images"], device)
        pixel_mask = move_to_device(batch.get("pixel_mask"), device)
        patch_mask = move_to_device(batch.get("patch_mask"), device)
        camids = move_to_device(batch.get("camids"), device)

        if camids is not None and hasattr(model.backbone, "set_camids"):
            model.backbone.set_camids(camids)

        extractor.clear()
        model.backbone(images, pixel_mask=pixel_mask, patch_mask=patch_mask)
        ff = extractor.get_features()   # [B, D_concat]

        if flip:
            extractor.clear()
            if camids is not None and hasattr(model.backbone, "set_camids"):
                model.backbone.set_camids(camids)
            pixel_mask_flip = torch.flip(pixel_mask, dims=[2]) if pixel_mask is not None else None
            if patch_mask is not None and patch_mask.ndim == 3:
                patch_mask_flip = torch.flip(patch_mask, dims=[2])
            else:
                patch_mask_flip = patch_mask
            model.backbone(
                torch.flip(images, dims=[3]),
                pixel_mask=pixel_mask_flip,
                patch_mask=patch_mask_flip,
            )
            ff = ff + extractor.get_features()

        if feat_norm:
            ff = F.normalize(ff, dim=1)

        all_features.append(ff.cpu())

    return torch.cat(all_features, dim=0).numpy()


# ── --blocks parser ───────────────────────────────────────────────────────────

def parse_blocks(arg: str, num_blocks: int) -> list:
    """Parse the ``--blocks`` CLI argument into a sorted list of indices.

    Accepted formats:
      - ``"all"``          → every block (0 … num_blocks-1)
      - ``"last:N"``       → the last N blocks
      - ``"18,20,22,23"``  → explicit comma-separated indices
    """
    arg = arg.strip().lower()
    if arg == "all":
        return list(range(num_blocks))
    if arg.startswith("last:"):
        n = int(arg.split(":")[1])
        return list(range(max(0, num_blocks - n), num_blocks))
    return sorted(int(x.strip()) for x in arg.split(","))


# ── Val branch ────────────────────────────────────────────────────────────────

def _run_val(cfg, model, extractor, dataset, output_dir, blocks_tag, logger):
    loader, num_query = build_val_loader(cfg, dataset)
    logger.info(
        f"Val split — query: {len(dataset.val_query)}, "
        f"gallery: {len(dataset.val_gallery)}"
    )

    features = extract_multiblock_features(
        model, extractor, loader,
        flip=cfg.test.flip_test,
        feat_norm=cfg.test.feat_norm,
    )
    qf, gf = features[:num_query], features[num_query:]
    logger.info(f"Feature shapes — qf: {qf.shape}, gf: {gf.shape}")

    np.save(os.path.join(output_dir, f"multiblock_{blocks_tag}_val_qf.npy"), qf)
    np.save(os.path.join(output_dir, f"multiblock_{blocks_tag}_val_gf.npy"), gf)

    q_pids       = np.array([s.pid         for s in dataset.val_query])
    g_pids       = np.array([s.pid         for s in dataset.val_gallery])
    q_camids     = np.array([s.camid       for s in dataset.val_query])
    g_camids     = np.array([s.camid       for s in dataset.val_gallery])
    q_cls_labels = np.array([s.class_label for s in dataset.val_query])
    g_cls_labels = np.array([s.class_label for s in dataset.val_gallery])

    sim = cosine_similarity(qf, gf)
    if cfg.test.rerank:
        logger.info("Applying re-ranking…")
        dist = re_ranking(
            sim,
            cosine_similarity(qf, qf),
            cosine_similarity(gf, gf),
            k1=cfg.test.rerank_k1,
            k2=cfg.test.rerank_k2,
            lambda_value=cfg.test.rerank_lambda,
        )
    else:
        dist = 1.0 - sim

    cmc, mAP = eval_func(dist, q_pids, g_pids, q_camids, g_camids)
    metrics = {
        "blocks": extractor.block_indices,
        "mAP":    float(mAP),
        "R1":     float(cmc[0]) if len(cmc) > 0 else 0.0,
        "R5":     float(cmc[4]) if len(cmc) > 4 else 0.0,
        "R10":    float(cmc[9]) if len(cmc) > 9 else 0.0,
    }

    # Per-class mAP
    for cls_idx, cls_name in IDX_TO_CLASS.items():
        q_mask = q_cls_labels == cls_idx
        g_mask = g_cls_labels == cls_idx
        if q_mask.sum() == 0 or g_mask.sum() == 0:
            continue
        cls_cmc, cls_mAP = eval_func(
            dist[np.ix_(q_mask, g_mask)],
            q_pids[q_mask], g_pids[g_mask],
            q_camids[q_mask], g_camids[g_mask],
        )
        metrics[f"mAP_{cls_name}"] = float(cls_mAP)
        metrics[f"R1_{cls_name}"]  = float(cls_cmc[0]) if len(cls_cmc) > 0 else 0.0

    metrics["test-weighted-mAP"] = sum(
        TEST_CLASS_WEIGHTS[cls] * metrics.get(f"mAP_{cls}", 0.0)
        for cls in TEST_CLASS_WEIGHTS
    )

    log_parts = [
        f"mAP={metrics['mAP']:.4f}",
        f"R1={metrics['R1']:.4f}",
        f"R5={metrics['R5']:.4f}",
        f"R10={metrics['R10']:.4f}",
    ]
    for cls_idx, cls_name in sorted(IDX_TO_CLASS.items()):
        key = f"mAP_{cls_name}"
        if key in metrics:
            log_parts.append(f"{key}={metrics[key]:.4f}")
    if "test-weighted-mAP" in metrics:
        log_parts.append(f"test-weighted-mAP={metrics['test-weighted-mAP']:.4f}")
    logger.info("  ".join(log_parts))

    metrics_path = os.path.join(output_dir, f"multiblock_{blocks_tag}_val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


# ── Test branch ───────────────────────────────────────────────────────────────

def _run_test(cfg, model, extractor, dataset, output_dir, blocks_tag, logger):
    loader, num_query = build_test_loader(cfg, dataset)
    logger.info(
        f"Test split — query: {len(dataset.query)}, gallery: {len(dataset.gallery)}"
    )

    features = extract_multiblock_features(
        model, extractor, loader,
        flip=cfg.test.flip_test,
        feat_norm=cfg.test.feat_norm,
    )
    qf, gf = features[:num_query], features[num_query:]
    logger.info(f"Feature shapes — qf: {qf.shape}, gf: {gf.shape}")

    np.save(os.path.join(output_dir, f"multiblock_{blocks_tag}_test_qf.npy"), qf)
    np.save(os.path.join(output_dir, f"multiblock_{blocks_tag}_test_gf.npy"), gf)

    sim = cosine_similarity(qf, gf)
    if cfg.test.rerank:
        logger.info("Applying re-ranking…")
        dist = re_ranking(
            sim,
            cosine_similarity(qf, qf),
            cosine_similarity(gf, gf),
            k1=cfg.test.rerank_k1,
            k2=cfg.test.rerank_k2,
            lambda_value=cfg.test.rerank_lambda,
        )
    else:
        dist = 1.0 - sim

    if cfg.test.class_mask:
        logger.info("Applying class mask…")
        dist = apply_class_mask(
            dist,
            [s.class_label for s in dataset.query],
            [s.class_label for s in dataset.gallery],
        )

    top_k   = cfg.submission.top_k
    indices = np.argsort(dist, axis=1)[:, :top_k]

    submission_path = os.path.join(
        output_dir, f"multiblock_{blocks_tag}_submission.csv"
    )
    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for i, row in enumerate(indices):
            writer.writerow([f"{i + 1:06d}.jpg", " ".join(str(idx + 1) for idx in row)])

    logger.info(f"Submission written to {submission_path}")
    logger.info(f"  Queries: {len(indices)}, Top-K: {top_k}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-block CLS token inference for ViT-family ReID models. "
            "Extracts concatenated CLS tokens from selected transformer blocks "
            "and runs evaluation (val) or generates a submission CSV (test)."
        )
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment config YAML (or a saved output config.yaml).",
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="'val' prints mAP/CMC metrics; 'test' generates a submission CSV.",
    )
    parser.add_argument(
        "--blocks", default="all",
        help=(
            "Which backbone blocks to extract CLS tokens from. "
            "Options: 'all', 'last:N' (last N blocks), "
            "or comma-separated indices e.g. '18,20,22,23'. "
            "Default: 'all'."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: cfg.trainer.output_dir).",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dot-path overrides, e.g. test.weight=outputs/.../best.pth",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides if args.overrides else None)
    set_seed(cfg.trainer.seed)

    output_dir = args.output_dir or cfg.trainer.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("urban_reid", output_dir)
    device = get_default_device()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = UrbanReIDDataset(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    has_class_obj  = any(obj.type == "class_ce" for obj in cfg.objectives)
    num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0
    model = ReIDModel(cfg, num_pids=dataset.num_train_pids, num_obj_classes=num_obj_classes)

    if not cfg.test.weight:
        raise ValueError(
            "test.weight must be specified. "
            "Pass it as an override: test.weight=outputs/.../checkpoint.pth"
        )
    ckpt = torch.load(cfg.test.weight, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    logger.info(f"Loaded weights from {cfg.test.weight}")

    model = model.to(device).eval()

    # ── Block selection ───────────────────────────────────────────────────────
    backbone = model.backbone
    arch, all_blocks = _get_all_blocks(backbone)   # raises ValueError if unsupported
    num_backbone_blocks = len(all_blocks)
    block_indices = parse_blocks(args.blocks, num_backbone_blocks)

    # Human-readable tag for output file names
    arg_lower = args.blocks.strip().lower()
    if arg_lower == "all":
        blocks_tag = "all"
    elif arg_lower.startswith("last:"):
        blocks_tag = arg_lower.replace(":", "")
    else:
        blocks_tag = args.blocks.strip().replace(",", "-").replace(" ", "")

    token_kind = "CLS token" if arch == "vit" else "GAP of spatial tokens"
    logger.info(f"Backbone : {cfg.backbone.name}  ({num_backbone_blocks} blocks total, arch={arch})")
    logger.info(f"Blocks   : {block_indices}  ({len(block_indices)} selected)")
    logger.info(f"Feature  : {token_kind} per block, concatenated & L2-normalised")

    # ── Attach hooks ──────────────────────────────────────────────────────────
    extractor = MultiBlockCLSExtractor(backbone, block_indices)

    try:
        if args.split == "val":
            _run_val(cfg, model, extractor, dataset, output_dir, blocks_tag, logger)
        else:
            _run_test(cfg, model, extractor, dataset, output_dir, blocks_tag, logger)
    finally:
        extractor.remove_hooks()
        logger.info("Hooks removed.")


if __name__ == "__main__":
    main()
