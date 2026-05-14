"""Objective combiner — weighted sum of multiple losses."""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .registry import build_loss


class ObjectiveCombiner(nn.Module):
    """Combines multiple losses with configurable weights.

    Config example:
        objectives:
          - name: id_loss
            type: cross_entropy
            weight: 1.0
            label_smooth: 0.1
          - name: triplet_loss
            type: triplet
            weight: 1.0
            margin: 0.3
    """

    def __init__(
        self,
        objectives_cfg,
        num_classes: int = None,
        embed_dim: int = None,
        num_train_cams: int = None,
    ):
        super().__init__()
        self.objectives = nn.ModuleList()
        self.names = []
        self.weights = []

        for obj_cfg in objectives_cfg:
            loss_fn = build_loss(
                obj_cfg,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_train_cams=num_train_cams,
            )
            self.objectives.append(loss_fn)
            self.names.append(obj_cfg.name)
            self.weights.append(obj_cfg.weight)

    def forward(
        self, model_outputs: dict, batch: dict
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted sum of all objectives.

        Returns:
            total_loss: scalar tensor (for backward)
            loss_dict: {name: float} for logging (detached values)
        """
        total_loss = torch.tensor(0.0, device=next(iter(model_outputs.values())).device)
        loss_dict = {}

        for name, weight, loss_fn in zip(self.names, self.weights, self.objectives):
            loss = loss_fn(model_outputs, batch)
            loss_dict[name] = loss.item()
            total_loss = total_loss + weight * loss

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
