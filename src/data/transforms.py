"""Train and test image transforms."""

import logging
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .resize_utils import (
    build_pixel_mask,
    get_class_target_size,
    get_pil_pad_fill,
    letterbox_pil_image,
)


logger = logging.getLogger("urban_reid")


def _build_fixed_resize_compose(cfg, is_train: bool):
    size = cfg.input.size_train if is_train else cfg.input.size_test
    tfms = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)]

    if is_train:
        if cfg.input.autoaug:
            tfms.append(T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET))

        if cfg.input.random_flip:
            tfms.append(T.RandomHorizontalFlip(p=0.5))

        if cfg.input.random_crop:
            tfms.append(T.Pad(cfg.input.pad))
            tfms.append(T.RandomCrop(cfg.input.size_train))

        if cfg.input.color_jitter and not cfg.input.autoaug:
            tfms.append(T.ColorJitter(
                brightness=cfg.input.cj_brightness,
                contrast=cfg.input.cj_contrast,
                saturation=cfg.input.cj_saturation,
                hue=cfg.input.cj_hue,
            ))

        if cfg.input.gaussian_blur:
            sigma = tuple(cfg.input.gaussian_blur_sigma)
            tfms.append(T.RandomApply(
                [T.GaussianBlur(kernel_size=cfg.input.gaussian_blur_kernel_size, sigma=sigma)],
                p=cfg.input.gaussian_blur_prob,
            ))

    tfms.append(T.ToTensor())
    tfms.append(T.Normalize(mean=cfg.input.pixel_mean, std=cfg.input.pixel_std))

    if is_train and cfg.input.random_erasing:
        tfms.append(T.RandomErasing(p=cfg.input.re_prob, value="random"))

    return T.Compose(tfms)


class FixedResizeTransform:
    """Legacy fixed-size transform wrapped in the new payload format."""

    def __init__(self, cfg, is_train: bool):
        self.return_masks = cfg.input.return_masks
        self.pipeline = _build_fixed_resize_compose(cfg, is_train=is_train)

    def __call__(self, img, sample=None, is_train=None):
        tensor = self.pipeline(img)
        target_hw = torch.tensor(tensor.shape[-2:], dtype=torch.long)
        output = {
            "images": tensor,
            "valid_hw": target_hw.clone(),
            "target_hw": target_hw.clone(),
        }
        if self.return_masks:
            output["pixel_mask"] = torch.ones(tuple(target_hw.tolist()), dtype=torch.bool)
        return output


class ClassAwareImageTransform:
    """Aspect-preserving class-aware resize with per-class canvases."""

    def __init__(self, cfg, is_train: bool):
        self.is_train = is_train
        self.keep_aspect_ratio = cfg.input.keep_aspect_ratio
        self.return_masks = cfg.input.return_masks
        self.random_flip = is_train and cfg.input.random_flip
        self.class_sizes = cfg.input.class_sizes_train if is_train else cfg.input.class_sizes_test
        self.fallback_size = cfg.input.size_train if is_train else cfg.input.size_test
        self.pixel_mean = list(cfg.input.pixel_mean)
        self.pixel_std = list(cfg.input.pixel_std)
        self.pad_fill = get_pil_pad_fill(cfg.input.pad_value, cfg.input.pixel_mean)

        self.autoaugment = None
        if is_train and cfg.input.autoaug:
            self.autoaugment = T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET)

        self.color_jitter = None
        self.color_jitter_no_hue = None  # 6B: for TrafficSign (preserve color semantics)
        if is_train and cfg.input.color_jitter and not cfg.input.autoaug:
            self.color_jitter = T.ColorJitter(
                brightness=cfg.input.cj_brightness,
                contrast=cfg.input.cj_contrast,
                saturation=cfg.input.cj_saturation,
                hue=cfg.input.cj_hue,
            )
            if getattr(cfg.input, "class_color_jitter", False):
                # TrafficSign: zero hue jitter — sign colors (red/yellow/green) carry semantic meaning
                self.color_jitter_no_hue = T.ColorJitter(
                    brightness=cfg.input.cj_brightness,
                    contrast=cfg.input.cj_contrast,
                    saturation=cfg.input.cj_saturation,
                    hue=0,
                )

        self.gaussian_blur = None
        if is_train and cfg.input.gaussian_blur:
            sigma = tuple(cfg.input.gaussian_blur_sigma)
            self.gaussian_blur = T.RandomApply(
                [T.GaussianBlur(kernel_size=cfg.input.gaussian_blur_kernel_size, sigma=sigma)],
                p=cfg.input.gaussian_blur_prob,
            )

        self.random_erasing = None
        if is_train and cfg.input.random_erasing:
            logger.warning(
                "Random erasing is disabled for class_letterbox because the current implementation "
                "does not restrict erasing to valid pixels."
            )

    def _apply_image_only_augmentations(self, img, class_label=-1):
        if self.autoaugment is not None:
            img = self.autoaugment(img)

        if self.random_flip and random.random() < 0.5:
            img = TF.hflip(img)

        if self.color_jitter is not None:
            # 6B: TrafficSign (class 3) gets no hue jitter — preserves color semantics
            if self.color_jitter_no_hue is not None and class_label == 3:
                img = self.color_jitter_no_hue(img)
            else:
                img = self.color_jitter(img)

        if self.gaussian_blur is not None:
            img = self.gaussian_blur(img)

        return img

    def __call__(self, img, sample=None, is_train=None):
        class_label = -1 if sample is None else sample.class_label
        img = self._apply_image_only_augmentations(img, class_label=class_label)
        target_hw = get_class_target_size(class_label, self.class_sizes, self.fallback_size)

        if self.keep_aspect_ratio:
            img, valid_hw, offset = letterbox_pil_image(
                img,
                target_hw=target_hw,
                fill=self.pad_fill,
                random_offset=self.is_train,
            )
        else:
            img = TF.resize(img, list(target_hw), interpolation=T.InterpolationMode.BICUBIC)
            valid_hw = target_hw
            offset = (0, 0)

        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=self.pixel_mean, std=self.pixel_std)

        if self.random_erasing is not None:
            tensor = self.random_erasing(tensor)

        output = {
            "images": tensor,
            "valid_hw": torch.tensor(valid_hw, dtype=torch.long),
            "target_hw": torch.tensor(target_hw, dtype=torch.long),
        }
        if self.return_masks:
            output["pixel_mask"] = build_pixel_mask(target_hw, valid_hw, offset)
        return output


def _build_transform(cfg, is_train: bool):
    strategy = cfg.input.resize_strategy.lower()
    if strategy in {"fixed", "fixed_resize"}:
        return FixedResizeTransform(cfg, is_train=is_train)
    if strategy == "class_letterbox":
        return ClassAwareImageTransform(cfg, is_train=is_train)
    raise ValueError(f"Unsupported input.resize_strategy: {cfg.input.resize_strategy}")


def build_train_transforms(cfg):
    """Build training augmentation pipeline from config."""
    return _build_transform(cfg, is_train=True)


def build_test_transforms(cfg):
    """Build test/inference transforms."""
    return _build_transform(cfg, is_train=False)
