"""Auxiliary object-class classification loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss


@register_loss("class_ce")
class ClassCELoss(nn.Module):
    """Cross-entropy for auxiliary object-class prediction (Container, Crosswalk, etc.)."""

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.label_smooth = getattr(cfg, "label_smooth", 0.0)

    def forward(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        if "class_logits" not in model_outputs:
            return torch.tensor(0.0, requires_grad=True)

        logits = model_outputs["class_logits"]
        targets = batch["class_labels"].to(logits.device)

        # Filter out samples with unknown class (label == -1)
        valid = targets >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid]
        targets = targets[valid]

        return F.cross_entropy(logits, targets)
