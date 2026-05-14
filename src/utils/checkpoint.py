"""Checkpoint save / load utilities."""

import os
import torch
from typing import Optional, Dict, Any


def save_checkpoint(
    state: Dict[str, Any],
    output_dir: str,
    filename: str = "checkpoint.pth",
) -> str:
    """Save training state to a checkpoint file. Returns the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load checkpoint and restore model / optimizer / scheduler state.

    Returns the full checkpoint dict (contains epoch, metrics, etc.).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
