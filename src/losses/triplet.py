"""Triplet loss with hard positive/negative mining."""

import torch
import torch.nn as nn

from .registry import register_loss


def _hard_mine_triplets(embeddings: torch.Tensor, pids: torch.Tensor, margin: float):
    """Compute batch-hard triplet loss.

    For each anchor, find the hardest positive (max dist) and
    hardest negative (min dist) within the batch.
    """
    dist_mat = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

    B = pids.size(0)
    is_pos = pids.unsqueeze(0).eq(pids.unsqueeze(1))  # [B, B]
    is_neg = ~is_pos

    # Hardest positive: max distance among positives
    dist_ap = dist_mat.clone()
    dist_ap[~is_pos] = 0.0
    dist_ap, _ = dist_ap.max(dim=1)  # [B]

    # Hardest negative: min distance among negatives
    dist_an = dist_mat.clone()
    dist_an[~is_neg] = float("inf")
    dist_an, _ = dist_an.min(dim=1)  # [B]

    loss = torch.clamp(dist_ap - dist_an + margin, min=0.0).mean()
    return loss


def _soft_margin_triplet(embeddings: torch.Tensor, pids: torch.Tensor):
    """Soft-margin triplet loss (no fixed margin, uses softplus)."""
    dist_mat = torch.cdist(embeddings, embeddings, p=2)

    is_pos = pids.unsqueeze(0).eq(pids.unsqueeze(1))
    is_neg = ~is_pos

    dist_ap = dist_mat.clone()
    dist_ap[~is_pos] = 0.0
    dist_ap, _ = dist_ap.max(dim=1)

    dist_an = dist_mat.clone()
    dist_an[~is_neg] = float("inf")
    dist_an, _ = dist_an.min(dim=1)

    loss = torch.nn.functional.softplus(dist_ap - dist_an).mean()
    return loss


@register_loss("triplet")
class TripletLoss(nn.Module):
    """Triplet loss with batch-hard mining.

    Supports 7C class-weighted triplet: compute per-class triplet loss and apply
    per-class weights (e.g. upweight TrafficSign which has 62.7% test weight).
    Configure via objective.class_triplet_weights = [w0, w1, w2, w3].
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.margin = getattr(cfg, "margin", 0.3)
        self.hard_mining = getattr(cfg, "hard_mining", True)
        raw_weights = list(getattr(cfg, "class_triplet_weights", []))
        self.class_triplet_weights = raw_weights if raw_weights else []

    def _triplet(self, embeddings, pids):
        if self.margin > 0 and self.hard_mining:
            return _hard_mine_triplets(embeddings, pids, self.margin)
        return _soft_margin_triplet(embeddings, pids)

    def forward(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        embeddings = model_outputs["embeddings"]
        pids = batch["pids"].to(embeddings.device)

        if not self.class_triplet_weights:
            return self._triplet(embeddings, pids)

        # 7C: per-class weighted triplet
        class_labels = batch.get("class_labels")
        if class_labels is None:
            return self._triplet(embeddings, pids)

        class_labels = class_labels.to(embeddings.device)
        total_loss = torch.tensor(0.0, device=embeddings.device)
        num_classes = len(self.class_triplet_weights)
        n_valid = 0

        for cls_idx in range(num_classes):
            mask = class_labels == cls_idx
            if mask.sum() < 2:
                continue
            cls_emb = embeddings[mask]
            cls_pids = pids[mask]
            if cls_pids.unique().numel() < 2:
                continue
            w = self.class_triplet_weights[cls_idx]
            total_loss = total_loss + w * self._triplet(cls_emb, cls_pids)
            n_valid += 1

        if n_valid == 0:
            return self._triplet(embeddings, pids)

        return total_loss / n_valid
