"""Ranked List Loss via pytorch-metric-learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import RankedListLoss as PMLRankedListLoss

from .registry import register_loss


@register_loss("ranked_list")
class RankedListLoss(nn.Module):
    """Ranked List Loss (Wang et al., CVPR 2019) via pytorch-metric-learning.

    Instead of triplets, this loss operates on the full ranked list of
    positives and negatives for each anchor. It treats every positive
    individually and enforces a margin-separated ranking over the
    entire negative set, weighted by temperature-scaled softmax.

    PML signature:
        RankedListLoss(margin, Tn, imbalance=0.5, alpha=None, Tp=0, **kwargs)

    Config params:
        margin   (float): margin between positive and negative set (default 1.0)
        Tn       (float): temperature for negative pairs            (default 1.0)
        Tp       (float): temperature for positive pairs            (default 0.0)
        alpha    (float): smallest allowed neg distance             (default 1.2)
        imbalance(float): tradeoff between pos/neg sets             (default 0.5)
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        margin = getattr(cfg, "margin", 1.0)
        Tn = getattr(cfg, "Tn", 1.0)
        Tp = getattr(cfg, "Tp", 0.0)
        alpha = getattr(cfg, "alpha", 1.2)
        imbalance = getattr(cfg, "imbalance", 0.5)
        self.loss_fn = PMLRankedListLoss(
            margin=margin,
            Tn=Tn,
            Tp=Tp,
            alpha=alpha,
            imbalance=imbalance,
        )

    def forward(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        # L2-normalise the post-BN embeddings so that distances live in [0, 2]
        # (unit sphere), making margin and alpha settings well-defined.
        # Raw pre-BN features are unbounded, so margin=1.0 / alpha=1.2 would
        # have no consistent meaning there.
        bn_emb = model_outputs["bn_embeddings"]
        embeddings = F.normalize(bn_emb, p=2, dim=1)
        labels = batch["pids"].to(embeddings.device)
        return self.loss_fn(embeddings, labels)
