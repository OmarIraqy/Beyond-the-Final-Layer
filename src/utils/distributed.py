"""Distributed training helpers."""

import os
import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp():
    """Initialize DDP from environment variables (torchrun)."""
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if is_dist_initialized():
        dist.destroy_process_group()


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all processes."""
    if not is_dist_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)
