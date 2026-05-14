"""ReID evaluator — mAP and CMC computation."""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Optional

from ..postprocess.distance import cosine_similarity
from ..postprocess.query_expansion import local_query_expansion
from ..data.dataset import IDX_TO_CLASS
from ..utils.device import get_module_device, move_to_device

logger = logging.getLogger("urban_reid")

# Weights derived from test-set class distribution
TEST_CLASS_WEIGHTS = {
    "TrafficSign":  0.627,
    "Container":    0.180,
    "Crosswalk":    0.098,
    "RubbishBins":  0.095,
}


def eval_func(
    dist_mat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> tuple:
    """Compute CMC and mAP.

    Args:
        dist_mat: [num_query, num_gallery] distance matrix (lower = better)
        q_pids: query person IDs
        g_pids: gallery person IDs
        q_camids: query camera IDs
        g_camids: gallery camera IDs
        max_rank: max rank for CMC

    Returns:
        (cmc, mAP) — cmc is array of shape [max_rank], mAP is float
    """
    num_q, num_g = dist_mat.shape

    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(dist_mat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        # Remove gallery samples with same pid AND same camid
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove
        match = matches[q_idx][keep]

        if not np.any(match):
            continue

        num_valid_q += 1

        cmc = match.cumsum()
        cmc[cmc > 1] = 1  # binary: found at least one match

        all_cmc.append(cmc[:max_rank])

        # AP
        num_rel = match.sum()
        tmp_cmc = match.cumsum()
        tmp_cmc = tmp_cmc * match  # only count at positions where match occurs
        precision = tmp_cmc / (np.arange(len(match)) + 1.0)
        ap = precision.sum() / num_rel
        all_AP.append(ap)

    if num_valid_q == 0:
        return np.zeros(max_rank), 0.0

    all_cmc = np.array(all_cmc, dtype=np.float32)
    cmc = all_cmc.mean(axis=0)
    mAP = np.mean(all_AP)

    return cmc, mAP


def extract_features(
    model: torch.nn.Module,
    dataloader,
    flip: bool = False,
    feat_norm: bool = True,
) -> np.ndarray:
    """Extract features from a dataloader.

    Args:
        model: ReID model with .extract() method
        dataloader: yields dicts with "images" key
        flip: if True, average features from original + horizontally flipped
        feat_norm: if True, L2-normalize features

    Returns:
        features: [N, D] numpy array
    """
    model.eval()
    extract_model = model.module if hasattr(model, "module") else model
    device = get_module_device(extract_model)
    all_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            images = move_to_device(batch["images"], device)
            pixel_mask = move_to_device(batch.get("pixel_mask"), device)
            patch_mask = move_to_device(batch.get("patch_mask"), device)
            camids = move_to_device(batch.get("camids"), device)

            ff = extract_model.extract(images, pixel_mask=pixel_mask, patch_mask=patch_mask, camids=camids)

            if flip:
                images_flip = torch.flip(images, dims=[3])
                pixel_mask_flip = None
                if pixel_mask is not None:
                    pixel_mask_flip = torch.flip(pixel_mask, dims=[2])

                patch_mask_flip = None
                if patch_mask is not None:
                    patch_mask_flip = torch.flip(patch_mask, dims=[2]) if patch_mask.ndim == 3 else patch_mask

                ff_flip = extract_model.extract(
                    images_flip,
                    pixel_mask=pixel_mask_flip,
                    patch_mask=patch_mask_flip,
                    camids=camids,
                )
                ff = ff + ff_flip

            if feat_norm:
                ff = F.normalize(ff, dim=1)

            all_features.append(ff.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    return features


def extract_features_multiscale(
    model: torch.nn.Module,
    dataloader_builder,
    scales,
    flip: bool = False,
    feat_norm: bool = True,
) -> np.ndarray:
    """Extract features with multi-scale test-time augmentation.

    Args:
        model: ReID model with .extract() method
        dataloader_builder: callable(cfg) -> dataloader; used to rebuild at each scale
        scales: list of [H, W] sizes, e.g. [[224,224], [256,256], [288,288]]
        flip: if True, also apply horizontal flip TTA at each scale
        feat_norm: if True, L2-normalize features

    Returns:
        features: [N, D] numpy array (averaged across scales)
    """
    all_scale_features = []
    for scale in scales:
        scale_dataloader = dataloader_builder(scale)
        feats = extract_features(
            model, scale_dataloader,
            flip=flip, feat_norm=feat_norm,
        )
        all_scale_features.append(feats)

    # Average features across scales, then re-normalize
    avg_features = np.mean(all_scale_features, axis=0)
    if feat_norm:
        norms = np.linalg.norm(avg_features, axis=1, keepdims=True) + 1e-12
        avg_features = avg_features / norms
    return avg_features


class ReIDEvaluator:
    """Evaluate ReID model on a val split with mAP and CMC."""

    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader,
        num_query: int,
        q_pids: Optional[np.ndarray] = None,
        g_pids: Optional[np.ndarray] = None,
        q_camids: Optional[np.ndarray] = None,
        g_camids: Optional[np.ndarray] = None,
        q_class_labels: Optional[np.ndarray] = None,
        g_class_labels: Optional[np.ndarray] = None,
        dataloader_builder=None,
    ) -> Dict[str, float]:
        """Run full evaluation pipeline.

        Args:
            model: ReID model
            dataloader: val loader (query + gallery concatenated)
            num_query: number of query samples
            q_pids/g_pids/q_camids/g_camids: if None, extracted from dataloader
            q_class_labels/g_class_labels: object class indices for per-class mAP
            dataloader_builder: optional callable(scale) -> dataloader for multi-scale TTA

        Returns:
            dict with mAP, R1, R5, R10, and per-class mAP
        """
        scales = getattr(self.cfg.test, "scales", None)
        if scales and dataloader_builder is not None:
            features = extract_features_multiscale(
                model, dataloader_builder, scales,
                flip=self.cfg.test.flip_test,
                feat_norm=self.cfg.test.feat_norm,
            )
        else:
            features = extract_features(
                model, dataloader,
                flip=self.cfg.test.flip_test,
                feat_norm=self.cfg.test.feat_norm,
            )

        qf = features[:num_query]
        gf = features[num_query:]

        # Query Expansion
        lqe_k = getattr(self.cfg.test, "lqe_k", 0)
        if lqe_k > 0:
            lqe_alpha = getattr(self.cfg.test, "lqe_alpha", 3.0)
            qf = local_query_expansion(qf, gf, k=lqe_k, alpha=lqe_alpha, feat_norm=self.cfg.test.feat_norm)

        # If pids/camids not passed, collect from dataloader dataset
        if q_pids is None:
            all_pids = []
            all_camids = []
            for batch in dataloader:
                all_pids.extend(batch["pids"].tolist())
                all_camids.extend(batch["camids"].tolist())
            all_pids = np.array(all_pids)
            all_camids = np.array(all_camids)
            q_pids = all_pids[:num_query]
            g_pids = all_pids[num_query:]
            q_camids = all_camids[:num_query]
            g_camids = all_camids[num_query:]

        # Cosine similarity -> distance
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
        if q_class_labels is not None and g_class_labels is not None:
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

            metrics["test-weighted-mAP"] = sum(
                TEST_CLASS_WEIGHTS[cls] * metrics.get(f"mAP_{cls}", 0.0)
                for cls in TEST_CLASS_WEIGHTS
            )

        log_parts = [
            f"mAP={metrics['mAP']:.4f}",
            f"R1={metrics['R1']:.4f} R5={metrics['R5']:.4f} R10={metrics['R10']:.4f}",
        ]
        for cls_idx, cls_name in sorted(IDX_TO_CLASS.items()):
            key = f"mAP_{cls_name}"
            if key in metrics:
                log_parts.append(f"{key}={metrics[key]:.4f}")
        if "test-weighted-mAP" in metrics:
            log_parts.append(f"test-weighted-mAP={metrics['test-weighted-mAP']:.4f}")
        logger.info("Evaluation: " + "  ".join(log_parts))
        return metrics


def eval_train_map(
    model: torch.nn.Module,
    dataloader,
    train_pids: np.ndarray,
    train_class_labels: Optional[np.ndarray] = None,
    flip: bool = False,
    feat_norm: bool = True,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """Compute mAP on the training set as an overfitting indicator.

    Each training image is used as query against all other training images.
    The query image itself is masked out (distance set to inf) so it never
    inflates the result.  Identities with no other instance (singletons) are
    skipped, matching standard ReID evaluation practice.

    Memory cost: O(chunk_size × N × D) for the similarity chunk, not O(N²).

    Args:
        model: ReID model with .extract() method
        dataloader: sequential loader over training set (test transforms,
                    no augmentation); order must match train_pids
        train_pids: [N] pid array aligned with dataloader order
        train_class_labels: [N] class label array for per-class metrics
        flip: horizontal flip TTA
        feat_norm: L2-normalise features
        chunk_size: number of queries processed per matmul chunk

    Returns:
        dict with train_mAP, train_R1, and per-class train metrics
    """
    features = extract_features(model, dataloader, flip=flip, feat_norm=feat_norm)
    # features: [N, D] numpy float32

    n = len(train_pids)
    feat_t = torch.from_numpy(features)  # keep on CPU; matmul is vectorised via BLAS

    all_AP = []
    all_R1 = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        qf_chunk = feat_t[start:end]  # [chunk, D]

        # Cosine similarity: [chunk, N]  (features already L2-normalised)
        sim_chunk = torch.mm(qf_chunk, feat_t.t()).numpy()  # [chunk, N]

        for local_i, q_idx in enumerate(range(start, end)):
            q_pid = train_pids[q_idx]

            sim_row = sim_chunk[local_i].copy()
            sim_row[q_idx] = -np.inf  # mask self: will sort to the end

            # Descending similarity order (ascending distance)
            order = np.argsort(-sim_row)
            # Drop the self entry (last after masking, but explicit is safer)
            order = order[order != q_idx]

            match = (train_pids[order] == q_pid).astype(np.int32)
            if not np.any(match):
                continue  # singleton identity — no positives to rank

            num_rel = match.sum()
            # Precision at each relevant position
            tmp_cmc = match.cumsum() * match
            precision = tmp_cmc / (np.arange(len(match)) + 1.0)
            ap = precision.sum() / num_rel
            all_AP.append(ap)
            all_R1.append(float(match[0]))

    train_mAP = float(np.mean(all_AP)) if all_AP else 0.0
    train_R1 = float(np.mean(all_R1)) if all_R1 else 0.0

    results = {"train_mAP": train_mAP, "train_R1": train_R1}

    # Per-class train mAP
    if train_class_labels is not None:
        all_AP_arr = np.array(all_AP) if all_AP else np.array([])
        all_R1_arr = np.array(all_R1) if all_R1 else np.array([])
        # Build a mask of which queries (by original index) contributed to all_AP
        # We need to re-iterate to track which indices were valid
        valid_indices = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            for q_idx in range(start, end):
                q_pid = train_pids[q_idx]
                # Check if this query has positives (same logic as above)
                pos_count = (train_pids == q_pid).sum() - 1  # exclude self
                if pos_count > 0:
                    valid_indices.append(q_idx)
        valid_indices = np.array(valid_indices)

        for cls_idx, cls_name in IDX_TO_CLASS.items():
            if len(valid_indices) == 0:
                continue
            cls_mask = train_class_labels[valid_indices] == cls_idx
            if cls_mask.sum() == 0:
                continue
            cls_aps = all_AP_arr[cls_mask]
            cls_r1s = all_R1_arr[cls_mask]
            results[f"train_mAP_{cls_name}"] = float(cls_aps.mean())
            results[f"train_R1_{cls_name}"] = float(cls_r1s.mean())

        results["train-test-weighted-mAP"] = sum(
            TEST_CLASS_WEIGHTS[cls] * results.get(f"train_mAP_{cls}", 0.0)
            for cls in TEST_CLASS_WEIGHTS
        )

    log_parts = [f"mAP={train_mAP:.4f}", f"R1={train_R1:.4f}",
                 f"(queries with positives: {len(all_AP)}/{n})"]
    for cls_idx, cls_name in sorted(IDX_TO_CLASS.items()):
        key = f"train_mAP_{cls_name}"
        if key in results:
            log_parts.append(f"{key}={results[key]:.4f}")
        if "train-test-weighted-mAP" in results:
            log_parts.append(f"train-test-weighted-mAP={results['train-test-weighted-mAP']:.4f}")
    logger.info("Train eval: " + "  ".join(log_parts))
    return results
