#!/usr/bin/env python3
"""Generate competition submission CSV.

Replicates the exact output format of PAT's update.py:
  - Query images named 000001.jpg to 000928.jpg (sequential, 1-indexed)
  - Gallery indices are 1-indexed
  - Top 100 matches per query
  - CSV columns: imageName,Corresponding Indexes

Usage:
    python tools/submit.py --config configs/default.yaml test.weight=outputs/best_model.pth
    python tools/submit.py --config outputs/config.yaml   # reuse saved config for reproducibility
"""

import os
import sys
import csv
import argparse
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# UrbanReid root — gives access to Classifier/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.device import get_default_device
from src.utils.logger import setup_logger
from src.data import UrbanReIDDataset, build_test_loader
from src.data.dataset import ImageDataset
from src.models import ReIDModel
from src.engine.evaluator import extract_features, extract_features_multiscale
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
    parser = argparse.ArgumentParser(description="Urban ReID Submission Generator")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides if args.overrides else None)
    set_seed(cfg.trainer.seed)
    device = get_default_device()

    logger = setup_logger("urban_reid", cfg.trainer.output_dir)

    # Save config for this submission run
    submission_dir = os.path.dirname(os.path.abspath(cfg.submission.output_path))
    os.makedirs(submission_dir, exist_ok=True)

    # Dataset
    dataset = UrbanReIDDataset(cfg)
    logger.info(f"Query: {len(dataset.query)}, Gallery: {len(dataset.gallery)}")

    # Model
    has_class_obj = any(obj.type == "class_ce" for obj in cfg.objectives)
    num_obj_classes = cfg.dataset.num_obj_classes if has_class_obj else 0
    model = ReIDModel(cfg, num_pids=dataset.num_train_pids, num_obj_classes=num_obj_classes)

    # Load weights
    if not cfg.test.weight:
        raise ValueError("test.weight must be set for submission generation")
    ckpt = torch.load(cfg.test.weight, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    logger.info(f"Loaded weights from {cfg.test.weight}")

    model = model.to(device)

    # Build test loader once to get num_query (shared across scales)
    test_loader, num_query = build_test_loader(cfg, dataset)

    # Multi-scale test-time augmentation
    scales = getattr(cfg.test, "scales", None)
    if scales:
        logger.info(f"Multi-scale TTA with scales: {scales}")
        def _build_test_loader_for_scale(scale):
            cfg.input.size_test = list(scale)
            loader, _ = build_test_loader(cfg, dataset)
            return loader
        features = extract_features_multiscale(
            model, _build_test_loader_for_scale, scales,
            flip=cfg.test.flip_test,
            feat_norm=cfg.test.feat_norm,
        )
    else:
        # Extract features with optional flip TTA
        logger.info("Extracting features...")
        features = extract_features(
            model, test_loader,
            flip=cfg.test.flip_test,
            feat_norm=cfg.test.feat_norm,
        )

    qf = features[:num_query]
    gf = features[num_query:]
    logger.info(f"Query features: {qf.shape}, Gallery features: {gf.shape}")

    # Query Expansion
    lqe_k = getattr(cfg.test, "lqe_k", 0)
    if lqe_k > 0:
        lqe_alpha = getattr(cfg.test, "lqe_alpha", 3.0)
        logger.info(f"Applying Local Query Expansion: k={lqe_k}, alpha={lqe_alpha}")
        qf = local_query_expansion(qf, gf, k=lqe_k, alpha=lqe_alpha, feat_norm=cfg.test.feat_norm)

    # Save features for analysis
    np.save(os.path.join(cfg.trainer.output_dir, "qf.npy"), qf)
    np.save(os.path.join(cfg.trainer.output_dir, "gf.npy"), gf)

    # Compute similarity
    q_g_sim = cosine_similarity(qf, gf)

    # Optional class masking
    if cfg.test.class_mask:
        logger.info("Applying class mask...")
        q_classes = [s.class_label for s in dataset.query]
        g_classes = [s.class_label for s in dataset.gallery]
        dist = apply_class_mask(q_g_sim, q_classes, g_classes)


    # Re-ranking
    if cfg.test.rerank:
        logger.info("Applying re-ranking...")
        logger.info(f"  Re-rank params: k1={cfg.test.rerank_k1}, k2={cfg.test.rerank_k2}, lambda={cfg.test.rerank_lambda}")
        q_q_sim = cosine_similarity(qf, qf)
        g_g_sim = cosine_similarity(gf, gf)
        dist = re_ranking(
            q_g_sim, q_q_sim, g_g_sim,
            k1=cfg.test.rerank_k1, k2=cfg.test.rerank_k2,
            lambda_value=cfg.test.rerank_lambda,
        )
    else:
        dist = 1.0 - q_g_sim  # similarity -> distance

    # Optional semantic confidence bonus
    if cfg.test.semantic_classifier:
        from src.Classifier.model import SemanticClassifier

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
        q_cls_ds = ImageDataset(dataset.query, transform=_SimpleTransform(cls_tf))
        g_cls_ds = ImageDataset(dataset.gallery, transform=_SimpleTransform(cls_tf))
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

    # Top-K ranking
    top_k = cfg.submission.top_k
    indices = np.argsort(dist, axis=1)[:, :top_k]

    # Write submission CSV — EXACT format matching competition
    output_path = cfg.submission.output_path
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for i, row in enumerate(indices):
            image_name = f"{i + 1:06d}.jpg"
            # Gallery indices are 1-indexed (match PAT's update.py)
            index_str = " ".join(str(idx + 1) for idx in row)
            writer.writerow([image_name, index_str])

    logger.info(f"Submission written to {output_path}")
    logger.info(f"  Queries: {len(indices)}, Top-K: {top_k}")

    # Save the config used for this submission alongside the CSV
    save_config(cfg, os.path.dirname(os.path.abspath(output_path)))


if __name__ == "__main__":
    main()
