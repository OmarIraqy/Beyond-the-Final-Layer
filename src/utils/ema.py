"""Exponential Moving Average (EMA) of model weights for training stabilization."""

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("urban_reid")


class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        for batch in loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update(model)
        # At evaluation time:
        ema.apply_shadow(model)  # copy EMA weights into model
        evaluate(model)
        ema.restore(model)       # restore training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow(model)
        logger.info(f"EMA initialized with decay={decay}")

    def _init_shadow(self, model: nn.Module):
        """Copy current model params as initial shadow weights."""
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name] = param.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow weights with exponential moving average."""
        for name, param in model.state_dict().items():
            if name in self.shadow and param.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Copy EMA weights into the model (for evaluation). Backs up current weights."""
        self.backup = {}
        for name, param in model.state_dict().items():
            self.backup[name] = param.clone().detach()
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module):
        """Restore the training weights from backup."""
        if self.backup:
            model.load_state_dict(self.backup, strict=False)
            self.backup = {}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
