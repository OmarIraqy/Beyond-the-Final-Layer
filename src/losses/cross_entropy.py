"""Label-smoothed cross-entropy loss for identity classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss


@register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):
    """Cross-entropy with optional label smoothing for ReID identity classification."""

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.label_smooth = getattr(cfg, "label_smooth", 0.0)

    def forward(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        logits = model_outputs["id_logits"]
        targets = batch["pids"].to(logits.device)

        if self.label_smooth > 0:
            num_classes = logits.size(1)
            log_probs = F.log_softmax(logits, dim=1)
            # Smooth labels
            smooth = self.label_smooth / num_classes
            targets_one_hot = torch.zeros_like(log_probs).fill_(smooth)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smooth + smooth)
            loss = (-targets_one_hot * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, targets)

        return loss
