#!/usr/bin/env python3
"""Evaluation entrypoint — run mAP/CMC on validation split.

Usage:
    python tools/evaluate.py --config configs/default.yaml test.weight=outputs/best_model.pth
"""

import os
import sys
import argparse
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_default_device
from src.utils.logger import setup_logger
from src.data import UrbanReIDDataset, build_val_loader
from src.data.dataset import ImageDataset, IDX_TO_CLASS
from src.models import ReIDModel
from src.engine.evaluator import extract_features, extract_features_multiscale, eval_func, TEST_CLASS_WEIGHTS
from src.postprocess.distance import cosine_similarity
from src.postprocess.rerank import re_ranking
from src.postprocess.class_mask import apply_class_mask
from src.postprocess.query_expansion import local_query_expansion
from src.postprocess.semantic_bonus import apply_confidence_bonus
from src.postprocess.query_majority_vote import query_majority_vote


class _SimpleTransform:
    """Wrap a torchvision transform for ImageDataset which passes extra kwargs."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, sample=None, is_train=None):
        return self.transform(img)


def _extract_class_probs(classifier, loader, device: torch.device) -> np.ndarray:
    """Run SemanticClassifier on all images in loader, return (N, C) numpy array."""
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            probs = classifier.predict_probs(imgs)
            all_probs.append(probs.cpu())
    return torch.cat(all_probs, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Urban ReID Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides if args.overrides else None)
    set_seed(cfg.trainer.seed)
    device = get_default_device()

    logger = setup_logger("urban_reid", cfg.trainer.output_dir)

    # Dataset
    dataset = UrbanReIDDataset(cfg)
    logger.info(f"Val query: {len(dataset.val_query)}, Val gallery: {len(dataset.val_gallery)}")

    # Model
    has_class_obj = any(obj.type == "class_ce" for obj in cfg.objectives)
    num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0
    model = ReIDModel(cfg, num_pids=dataset.num_train_pids, num_obj_classes=num_obj_classes)

    # Load weights
    if cfg.test.weight:
        ckpt = torch.load(cfg.test.weight, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Handle DDP state_dict (remove 'module.' prefix)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {cfg.test.weight}")
    else:
        logger.warning("No test.weight specified — evaluating with random weights")

    model = model.to(device)

    # Val loader
    val_loader, num_val_query = build_val_loader(cfg, dataset)
    val_samples = list(dataset.val_query) + list(dataset.val_gallery)
    val_pids = np.array([s.pid for s in val_samples])
    val_camids = np.array([s.camid for s in val_samples])
    val_class_labels = np.array([s.class_label for s in val_samples])

    # Feature extraction
    scales = getattr(cfg.test, "scales", None)
    if scales:
        logger.info(f"Multi-scale TTA with scales: {scales}")
        def _build_val_loader_for_scale(scale):
            cfg.input.size_test = list(scale)
            loader, _ = build_val_loader(cfg, dataset)
            return loader
        features = extract_features_multiscale(
            model, _build_val_loader_for_scale, scales,
            flip=cfg.test.flip_test,
            feat_norm=cfg.test.feat_norm,
        )
    else:
        features = extract_features(
            model, val_loader,
            flip=cfg.test.flip_test,
            feat_norm=cfg.test.feat_norm,
        )

    qf = features[:num_val_query]
    gf = features[num_val_query:]

    # Query Expansion
    lqe_k = getattr(cfg.test, "lqe_k", 0)
    if lqe_k > 0:
        lqe_alpha = getattr(cfg.test, "lqe_alpha", 3.0)
        logger.info(f"Applying Local Query Expansion: k={lqe_k}, alpha={lqe_alpha}")
        qf = local_query_expansion(qf, gf, k=lqe_k, alpha=lqe_alpha, feat_norm=cfg.test.feat_norm)

    # Distance computation
    sim = cosine_similarity(qf, gf)
    dist = 1.0 - sim

    if cfg.test.rerank:
        logger.info("Applying re-ranking...")
        dist = re_ranking(
            sim, cosine_similarity(qf, qf), cosine_similarity(gf, gf),
            k1=cfg.test.rerank_k1, k2=cfg.test.rerank_k2,
            lambda_value=cfg.test.rerank_lambda,
        )

    if cfg.test.class_mask:
        logger.info("Applying class mask...")
        q_cls = val_class_labels[:num_val_query]
        g_cls = val_class_labels[num_val_query:]
        dist = apply_class_mask(dist, q_cls, g_cls)

    # Optional semantic confidence bonus
    if cfg.test.semantic_classifier:
        from Classifier.model import SemanticClassifier

        logger.info(f"Loading semantic classifier from {cfg.test.semantic_classifier}...")
        classifier = SemanticClassifier.from_checkpoint(
            cfg.test.semantic_classifier,
            backbone_name=cfg.test.semantic_backbone,
            img_size=cfg.test.semantic_img_size,
            device=device,
        )
        cls_tf = T.Compose([
            T.Resize((cfg.test.semantic_img_size, cfg.test.semantic_img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        q_cls_ds = ImageDataset(dataset.val_query, transform=_SimpleTransform(cls_tf))
        g_cls_ds = ImageDataset(dataset.val_gallery, transform=_SimpleTransform(cls_tf))
        q_cls_loader = DataLoader(
            q_cls_ds, batch_size=cfg.test.batch_size,
            shuffle=False, num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
        )
        g_cls_loader = DataLoader(
            g_cls_ds, batch_size=cfg.test.batch_size,
            shuffle=False, num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
        )
        logger.info("Extracting semantic class probabilities for query...")
        q_probs = _extract_class_probs(classifier, q_cls_loader, device)
        logger.info("Extracting semantic class probabilities for gallery...")
        g_probs = _extract_class_probs(classifier, g_cls_loader, device)
        logger.info(f"Applying confidence bonus (alpha={cfg.test.semantic_alpha})...")
        dist = apply_confidence_bonus(dist, q_probs, g_probs, alpha=cfg.test.semantic_alpha)
        logger.info("Semantic confidence bonus applied.")
        del classifier

    # Optional Query Majority Voting
    if cfg.test.query_majority_vote_window > 0:
        window = cfg.test.query_majority_vote_window
        logger.info(f"Applying Query Majority Voting (window={window})...")
        dist = query_majority_vote(dist, qf, window=window)
        logger.info("Query Majority Voting applied.")

    cmc, mAP = eval_func(dist, val_pids[:num_val_query], val_pids[num_val_query:],
                         val_camids[:num_val_query], val_camids[num_val_query:])

    metrics = {
        "mAP": float(mAP),
        "R1": float(cmc[0]) if len(cmc) > 0 else 0.0,
        "R5": float(cmc[4]) if len(cmc) > 4 else 0.0,
        "R10": float(cmc[9]) if len(cmc) > 9 else 0.0,
    }

    # Per-class mAP
    for cls_idx, cls_name in IDX_TO_CLASS.items():
        q_mask = val_class_labels[:num_val_query] == cls_idx
        g_mask = val_class_labels[num_val_query:] == cls_idx
        if q_mask.sum() == 0 or g_mask.sum() == 0:
            continue
        cls_dist = dist[np.ix_(q_mask, g_mask)]
        cls_cmc, cls_mAP = eval_func(
            cls_dist,
            val_pids[:num_val_query][q_mask], val_pids[num_val_query:][g_mask],
            val_camids[:num_val_query][q_mask], val_camids[num_val_query:][g_mask],
        )
        metrics[f"mAP_{cls_name}"] = float(cls_mAP)
        metrics[f"R1_{cls_name}"] = float(cls_cmc[0]) if len(cls_cmc) > 0 else 0.0

    if any(k.startswith("mAP_") for k in metrics):
        metrics["test-weighted-mAP"] = sum(
            TEST_CLASS_WEIGHTS[cls] * metrics.get(f"mAP_{cls}", 0.0)
            for cls in TEST_CLASS_WEIGHTS
        )

    logger.info(f"Results: {metrics}")


if __name__ == "__main__":
    main()
