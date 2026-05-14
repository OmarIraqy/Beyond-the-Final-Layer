"""Center loss — pulls embeddings toward class centers.

Note: pytorch-metric-learning does not include CenterLoss (Wen et al., ECCV 2016).
This is a custom implementation. The learnable centers are included in the
objective combiner's parameters and optimized by the main optimizer.
"""

import torch
import torch.nn as nn

from .registry import register_loss


@register_loss("center")
class CenterLoss(nn.Module):
    """Center loss (Wen et al., ECCV 2016).

    Maintains learnable class centers and minimizes intra-class distance.

    Requires num_classes and embed_dim at construction (passed by combiner).
    """

    def __init__(self, cfg, num_classes=None, embed_dim=None, **kwargs):
        super().__init__()
        if num_classes is None or embed_dim is None:
            raise ValueError(
                "CenterLoss requires num_classes and embed_dim. "
                "These are passed automatically by ObjectiveCombiner."
            )
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        embeddings = model_outputs["embeddings"]
        pids = batch["pids"].to(embeddings.device)

        centers_batch = self.centers[pids]  # [B, D]
        loss = ((embeddings - centers_batch) ** 2).sum(dim=1).mean()
        return loss
