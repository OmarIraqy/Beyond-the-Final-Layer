"""Device helpers for CPU/GPU execution."""

import torch
import torch.nn as nn


def get_default_device(rank: int = 0) -> torch.device:
    """Return the preferred execution device for this process."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def get_module_device(module: nn.Module) -> torch.device:
    """Return the device of the first parameter, or CPU for parameterless modules."""
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


def move_to_device(value, device: torch.device, non_blocking: bool = False):
    """Move a tensor-like value to the target device, leaving other values unchanged."""
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking and device.type == "cuda")
    return value