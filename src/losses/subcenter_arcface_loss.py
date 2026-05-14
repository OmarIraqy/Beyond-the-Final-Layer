"""Sub-center ArcFace loss using pytorch-metric-learning.

This is a classification-based loss with multiple learnable centers per class.
It is useful when the same identity can occupy distinct camera-specific modes.
When using this loss, use a BNNeck head (not ArcFaceHead) — the angular margin
is handled entirely within the loss function.
"""

import math

import torch
import torch.nn as nn
from pytorch_metric_learning.losses import SubCenterArcFaceLoss as PMLSubCenterArcFaceLoss

from .registry import register_loss


@register_loss("subcenter_arcface")
class SubCenterArcFaceLoss(nn.Module):
    """Sub-center ArcFace loss via pytorch-metric-learning.

    Each class is represented by multiple sub-centers. At training time, the
    classifier picks the closest sub-center for the ground-truth class before
    applying the ArcFace angular margin.

    Config params:
        margin: angular margin in radians (default 0.5, ≈28.6°)
        s: logit scale factor (default 64.0)
        sub_centers: centers per class. If omitted, defaults to the number of
            unique cameras in the training set.

    Requires num_classes and embed_dim at construction (passed by combiner).
    """

    def __init__(self, cfg, num_classes=None, embed_dim=None, num_train_cams=None, **kwargs):
        super().__init__()
        if num_classes is None or embed_dim is None:
            raise ValueError(
                "SubCenterArcFaceLoss requires num_classes and embed_dim. "
                "These are passed automatically by ObjectiveCombiner."
            )

        margin_rad = getattr(cfg, "margin", 0.5)
        margin_deg = math.degrees(margin_rad)
        scale = getattr(cfg, "s", 64.0)
        sub_centers = getattr(cfg, "sub_centers", None)
        if sub_centers is None:
            sub_centers = num_train_cams
        if sub_centers is None or int(sub_centers) < 1:
            raise ValueError(
                "SubCenterArcFaceLoss requires sub_centers >= 1. "
                "Set objectives[].sub_centers or provide num_train_cams."
            )

        self.loss_fn = PMLSubCenterArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embed_dim,
            margin=margin_deg,
            scale=scale,
            sub_centers=int(sub_centers),
        )

    def forward(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        embeddings = model_outputs["bn_embeddings"]
        labels = batch["pids"].to(embeddings.device)
        return self.loss_fn(embeddings, labels)