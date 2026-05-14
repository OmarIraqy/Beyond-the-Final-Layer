"""Embedding heads for ReID: BNNeck, ArcFace, plain linear."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeckHead(nn.Module):
    """Standard BNNeck head for ReID.

    Flow: features -> bottleneck (BN) -> classifier (linear, no bias)
    Training returns both raw embeddings and classifier logits.
    """

    def __init__(self, in_dim: int, num_pids: int, num_obj_classes: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.num_pids = num_pids

        self.bottleneck = nn.BatchNorm1d(in_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        self.classifier = nn.Linear(in_dim, num_pids, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        self.class_classifier = None
        if num_obj_classes > 0:
            self.class_classifier = nn.Linear(in_dim, num_obj_classes, bias=False)
            nn.init.normal_(self.class_classifier.weight, std=0.001)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: [B, D] from backbone

        Returns:
            dict with keys: embeddings, bn_embeddings, id_logits, class_logits
        """
        bn_feat = self.bottleneck(features)
        id_logits = self.classifier(bn_feat)

        out = {
            "embeddings": features,
            "bn_embeddings": bn_feat,
            "id_logits": id_logits,
        }

        if self.class_classifier is not None:
            out["class_logits"] = self.class_classifier(bn_feat)

        return out

    def extract(self, features: torch.Tensor, neck_feat: str = "after") -> torch.Tensor:
        """Inference: return embeddings (before or after BN)."""
        if neck_feat == "after":
            return self.bottleneck(features)
        return features


class ArcFaceHead(nn.Module):
    """ArcFace / CosFace margin-based classification head."""

    def __init__(self, in_dim: int, num_pids: int, num_obj_classes: int = 0,
                 s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.in_dim = in_dim
        self.num_pids = num_pids
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_pids, in_dim))
        nn.init.xavier_uniform_(self.weight)

        self.bottleneck = nn.BatchNorm1d(in_dim)
        self.bottleneck.bias.requires_grad_(False)

        self.class_classifier = None
        if num_obj_classes > 0:
            self.class_classifier = nn.Linear(in_dim, num_obj_classes, bias=False)

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> dict:
        bn_feat = self.bottleneck(features)
        # Cosine similarity
        cosine = F.linear(F.normalize(bn_feat), F.normalize(self.weight))

        if labels is not None and self.training:
            # ArcFace margin
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            id_logits = self.s * torch.cos(theta + one_hot * self.m)
        else:
            id_logits = self.s * cosine

        out = {
            "embeddings": features,
            "bn_embeddings": bn_feat,
            "id_logits": id_logits,
        }
        if self.class_classifier is not None:
            out["class_logits"] = self.class_classifier(bn_feat)
        return out

    def extract(self, features: torch.Tensor, neck_feat: str = "after") -> torch.Tensor:
        if neck_feat == "after":
            return self.bottleneck(features)
        return features


def build_head(cfg, feature_dim: int, num_pids: int, num_obj_classes: int = 0) -> nn.Module:
    """Build embedding head from config."""
    head_type = cfg.head.type.lower()
    dim = cfg.head.embed_dim if cfg.head.embed_dim else feature_dim

    if head_type == "bnneck":
        return BNNeckHead(dim, num_pids, num_obj_classes)
    elif head_type == "arcface":
        s = getattr(cfg.head, "s", 64.0)
        m = getattr(cfg.head, "m", 0.5)
        return ArcFaceHead(dim, num_pids, num_obj_classes, s=s, m=m)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
