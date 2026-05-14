"""Circle loss — using pytorch-metric-learning implementation."""

import torch
import torch.nn as nn
from pytorch_metric_learning.losses import CircleLoss as PMLCircleLoss

from .registry import register_loss


@register_loss("circle")
class CircleLoss(nn.Module):
    """Circle loss (Sun et al., CVPR 2020) via pytorch-metric-learning.

    Pairwise similarity-based loss with adaptive weighting.
    PML params: m (margin), gamma (scale factor).
    Config params: margin (default 0.25), s (default 64.0).
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        m = getattr(cfg, "margin", 0.25)
        gamma = getattr(cfg, "s", 64.0)
        self.loss_fn = PMLCircleLoss(m=m, gamma=gamma)

    def forward(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        embeddings = model_outputs["embeddings"]
        labels = batch["pids"].to(embeddings.device)
        return self.loss_fn(embeddings, labels)
