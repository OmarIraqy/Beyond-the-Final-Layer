"""ArcFace loss — using pytorch-metric-learning implementation.

This is a classification-based loss with its own learnable weight matrix.
When using this loss, use a BNNeck head (not ArcFaceHead) — the angular
margin is handled entirely within the loss function.
"""

import math
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import ArcFaceLoss as PMLArcFaceLoss

from .registry import register_loss


@register_loss("arcface")
class ArcFaceLoss(nn.Module):
    """ArcFace loss (Deng et al., CVPR 2019) via pytorch-metric-learning.

    Additive angular margin penalty applied to the ground-truth class in
    the cosine similarity space.  Internally handles L2-normalization of
    both embeddings and classifier weights, computes the angular margin,
    scales, and applies cross-entropy.

    Config params:
        margin: angular margin in radians (default 0.5, ≈28.6°)
        s:      logit scale factor (default 64.0)

    Requires num_classes and embed_dim at construction (passed by combiner).
    """

    def __init__(self, cfg, num_classes=None, embed_dim=None, **kwargs):
        super().__init__()
        if num_classes is None or embed_dim is None:
            raise ValueError(
                "ArcFaceLoss requires num_classes and embed_dim. "
                "These are passed automatically by ObjectiveCombiner."
            )
        margin_rad = getattr(cfg, "margin", 0.5)
        margin_deg = math.degrees(margin_rad)
        scale = getattr(cfg, "s", 64.0)

        self.loss_fn = PMLArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embed_dim,
            margin=margin_deg,
            scale=scale,
        )

    def forward(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        # Use BN-normalized embeddings (consistent with ArcFace convention)
        embeddings = model_outputs["bn_embeddings"]
        labels = batch["pids"].to(embeddings.device)
        return self.loss_fn(embeddings, labels)
