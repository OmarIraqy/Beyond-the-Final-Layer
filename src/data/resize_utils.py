"""Utilities for class-aware resizing, masks, and collate-time padding."""

import random
from typing import Mapping, Sequence, Tuple

from PIL import Image
import torch
import torch.nn.functional as F

from .dataset import IDX_TO_CLASS


def get_class_target_size(
    class_label: int,
    class_sizes: Mapping[str, Sequence[int]],
    fallback_size: Sequence[int],
) -> Tuple[int, int]:
    """Resolve the target canvas for a class label."""
    class_name = IDX_TO_CLASS.get(class_label)
    if class_name is None or class_name not in class_sizes:
        return int(fallback_size[0]), int(fallback_size[1])

    target_hw = class_sizes[class_name]
    return int(target_hw[0]), int(target_hw[1])


def _resolve_raw_pad_rgb(pad_value, pixel_mean: Sequence[float]):
    if isinstance(pad_value, str):
        key = pad_value.lower()
        if key == "mean":
            return [float(value) for value in pixel_mean]
        if key in {"zero", "black"}:
            return [0.0, 0.0, 0.0]
        raise ValueError(f"Unsupported input.pad_value: {pad_value}")

    if isinstance(pad_value, Sequence):
        values = [float(value) for value in pad_value]
        if len(values) != 3:
            raise ValueError("input.pad_value sequences must have exactly 3 channels")
        if max(values) > 1.0:
            return [value / 255.0 for value in values]
        return values

    value = float(pad_value)
    if value > 1.0:
        value /= 255.0
    return [value, value, value]


def get_pil_pad_fill(pad_value, pixel_mean: Sequence[float]):
    """Convert configured pad values to PIL RGB fill values."""
    raw_rgb = _resolve_raw_pad_rgb(pad_value, pixel_mean)
    return tuple(int(round(channel * 255.0)) for channel in raw_rgb)


def get_normalized_pad_fill(pad_value, pixel_mean: Sequence[float], pixel_std: Sequence[float]):
    """Convert configured pad values to normalized tensor-space fill values."""
    raw_rgb = _resolve_raw_pad_rgb(pad_value, pixel_mean)
    return [
        (raw_rgb[channel_idx] - float(pixel_mean[channel_idx])) / float(pixel_std[channel_idx])
        for channel_idx in range(3)
    ]


def letterbox_pil_image(
    image: Image.Image,
    target_hw: Sequence[int],
    fill,
    random_offset: bool = False,
):
    """Resize a PIL image to fit inside a target canvas and pad the remainder."""
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    width, height = image.size
    scale = min(target_w / width, target_h / height)

    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = image.resize((resized_w, resized_h), resample=Image.Resampling.BICUBIC)

    max_top = target_h - resized_h
    max_left = target_w - resized_w
    if random_offset and (max_top > 0 or max_left > 0):
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
    else:
        top = max_top // 2
        left = max_left // 2

    canvas = Image.new(image.mode, (target_w, target_h), color=fill)
    canvas.paste(resized, (left, top))
    return canvas, (resized_h, resized_w), (top, left)


def build_pixel_mask(
    target_hw: Sequence[int],
    valid_hw: Sequence[int],
    offset: Sequence[int],
) -> torch.Tensor:
    """Build a boolean valid-pixel mask for a padded image."""
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    valid_h, valid_w = int(valid_hw[0]), int(valid_hw[1])
    top, left = int(offset[0]), int(offset[1])

    mask = torch.zeros((target_h, target_w), dtype=torch.bool)
    mask[top : top + valid_h, left : left + valid_w] = True
    return mask


def pad_image_tensor(
    image: torch.Tensor,
    target_hw: Sequence[int],
    fill_values: Sequence[float],
) -> torch.Tensor:
    """Center-pad an image tensor to a common batch canvas."""
    channels, height, width = image.shape
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    top = (target_h - height) // 2
    left = (target_w - width) // 2

    padded = image.new_empty((channels, target_h, target_w))
    for channel_idx in range(channels):
        padded[channel_idx].fill_(fill_values[min(channel_idx, len(fill_values) - 1)])
    padded[:, top : top + height, left : left + width] = image
    return padded


def pad_spatial_tensor(tensor: torch.Tensor, target_hw: Sequence[int], fill_value) -> torch.Tensor:
    """Center-pad a 2D spatial tensor to a target height and width."""
    height, width = tensor.shape[-2:]
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    top = (target_h - height) // 2
    left = (target_w - width) // 2

    padded = tensor.new_full((target_h, target_w), fill_value)
    padded[top : top + height, left : left + width] = tensor
    return padded


def _normalize_spatial_stride(patch_stride) -> Tuple[int, int]:
    if isinstance(patch_stride, int):
        return patch_stride, patch_stride
    if isinstance(patch_stride, Sequence) and not isinstance(patch_stride, (str, bytes)):
        stride = tuple(int(value) for value in patch_stride)
        if len(stride) == 2:
            return stride
    raise ValueError(f"patch_stride must be an int or a two-element sequence, got {patch_stride!r}")


def build_patch_mask(pixel_mask: torch.Tensor, patch_stride) -> torch.Tensor:
    """Convert a pixel-validity mask into a patch-validity mask."""
    stride_h, stride_w = _normalize_spatial_stride(patch_stride)

    squeeze = False
    if pixel_mask.ndim == 2:
        pixel_mask = pixel_mask.unsqueeze(0)
        squeeze = True

    if pixel_mask.ndim != 3:
        raise ValueError(f"pixel_mask must have shape [H, W] or [B, H, W], got {tuple(pixel_mask.shape)}")

    pooled = F.max_pool2d(
        pixel_mask.unsqueeze(1).to(dtype=torch.float32),
        kernel_size=(stride_h, stride_w),
        stride=(stride_h, stride_w),
        ceil_mode=True,
    )
    patch_mask = pooled.squeeze(1) > 0
    return patch_mask.squeeze(0) if squeeze else patch_mask