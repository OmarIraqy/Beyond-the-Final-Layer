"""DataLoader builders for train / val / test splits."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import UrbanReIDDataset, ImageDataset
from .resize_utils import build_patch_mask, get_normalized_pad_fill, pad_image_tensor, pad_spatial_tensor
from .sampler import PKSampler, ClassAwarePKSampler, ClassBucketPKSampler
from .transforms import build_train_transforms, build_test_transforms


def build_collate_fn(cfg, *, force_batch_pad: bool = False):
    """Build a collate function for either direct stacking or batch-level padding."""
    normalized_pad_fill = get_normalized_pad_fill(
        cfg.input.pad_value,
        cfg.input.pixel_mean,
        cfg.input.pixel_std,
    )
    batching_strategy = "batch_pad" if force_batch_pad else cfg.dataloader.batching_strategy.lower()

    def _collate_fn(batch):
        images_list = [b["images"] for b in batch]
        image_shapes = {tuple(img.shape[-2:]) for img in images_list}
        use_batch_padding = batching_strategy == "batch_pad"

        if use_batch_padding:
            max_h = max(img.shape[-2] for img in images_list)
            max_w = max(img.shape[-1] for img in images_list)
            images = torch.stack([
                pad_image_tensor(img, (max_h, max_w), normalized_pad_fill)
                for img in images_list
            ])
        else:
            if len(image_shapes) != 1:
                raise ValueError(
                    "Batch contains mixed image shapes. Use "
                    "dataloader.batching_strategy=batch_pad or same_class with class-aware canvases."
                )
            images = torch.stack(images_list)

        collated = {
            "images": images,
            "pids": torch.tensor([b["pids"] for b in batch], dtype=torch.long),
            "camids": torch.tensor([b["camids"] for b in batch], dtype=torch.long),
            "class_labels": torch.tensor([b["class_labels"] for b in batch], dtype=torch.long),
            "img_paths": [b["img_paths"] for b in batch],
        }

        if all("valid_hw" in b for b in batch):
            collated["valid_hw"] = torch.stack([b["valid_hw"] for b in batch])

        if all("target_hw" in b for b in batch):
            collated["target_hw"] = torch.stack([b["target_hw"] for b in batch])

        if use_batch_padding or any("pixel_mask" in b for b in batch):
            if use_batch_padding:
                mask_target_hw = images.shape[-2:]
                pixel_masks = []
                for sample in batch:
                    mask = sample.get(
                        "pixel_mask",
                        torch.ones(sample["images"].shape[-2:], dtype=torch.bool),
                    )
                    pixel_masks.append(pad_spatial_tensor(mask, mask_target_hw, False))
                collated["pixel_mask"] = torch.stack(pixel_masks)
            else:
                collated["pixel_mask"] = torch.stack([
                    b.get("pixel_mask", torch.ones(b["images"].shape[-2:], dtype=torch.bool))
                    for b in batch
                ])

        if "pixel_mask" in collated:
            collated["patch_mask"] = build_patch_mask(
                collated["pixel_mask"],
                cfg.input.patch_mask_stride,
            )

        return collated

    return _collate_fn


def _build_train_sampler(cfg, dataset: UrbanReIDDataset):
    batching_strategy = cfg.dataloader.batching_strategy.lower()
    if batching_strategy == "same_class":
        return ClassBucketPKSampler(
            dataset.train,
            num_instances=cfg.dataloader.num_instances,
            batch_size=cfg.dataloader.batch_size,
        )

    sampler_name = cfg.dataloader.sampler.lower()
    if sampler_name == "pk":
        return PKSampler(
            dataset.train,
            num_instances=cfg.dataloader.num_instances,
            batch_size=cfg.dataloader.batch_size,
        )
    if sampler_name == "class_aware_pk":
        return ClassAwarePKSampler(
            dataset.train,
            num_instances=cfg.dataloader.num_instances,
            batch_size=cfg.dataloader.batch_size,
        )
    return None


def build_train_loader(cfg, dataset: UrbanReIDDataset) -> DataLoader:
    """Build training DataLoader with configurable sampler."""
    transform = build_train_transforms(cfg)
    train_set = ImageDataset(dataset.train, transform=transform, is_train=True)
    sampler = _build_train_sampler(cfg, dataset)
    collate_fn = build_collate_fn(cfg)

    if sampler is not None:
        loader = DataLoader(
            train_set,
            batch_size=cfg.dataloader.batch_size,
            sampler=sampler,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            drop_last=True,
            collate_fn=collate_fn,
        )
    else:
        loader = DataLoader(
            train_set,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            drop_last=True,
            collate_fn=collate_fn,
        )

    return loader


def _build_eval_loader(cfg, query_samples, gallery_samples) -> Tuple[DataLoader, int]:
    """Build evaluation loader: query + gallery concatenated.

    Returns (DataLoader, num_query) so features can be split later.
    """
    transform = build_test_transforms(cfg)
    combined = list(query_samples) + list(gallery_samples)
    eval_set = ImageDataset(combined, transform=transform, is_train=False)
    use_batch_pad = cfg.input.resize_strategy.lower() == "class_letterbox"
    loader = DataLoader(
        eval_set,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=False,
        collate_fn=build_collate_fn(cfg, force_batch_pad=use_batch_pad),
    )
    return loader, len(query_samples)


def build_val_loader(cfg, dataset: UrbanReIDDataset) -> Tuple[DataLoader, int]:
    """Build validation loader (val_query + val_gallery)."""
    return _build_eval_loader(cfg, dataset.val_query, dataset.val_gallery)


def build_test_loader(cfg, dataset: UrbanReIDDataset) -> Tuple[DataLoader, int]:
    """Build competition test loader (query + gallery)."""
    return _build_eval_loader(cfg, dataset.query, dataset.gallery)


def build_train_probe_loader(cfg, dataset: UrbanReIDDataset) -> DataLoader:
    """Build a sequential train loader with test transforms for train mAP computation.

    Uses test transforms (no augmentation) so features are deterministic.
    Order matches dataset.train exactly for pid alignment.
    """
    transform = build_test_transforms(cfg)
    train_set = ImageDataset(dataset.train, transform=transform, is_train=False)
    use_batch_pad = cfg.input.resize_strategy.lower() == "class_letterbox"
    loader = DataLoader(
        train_set,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=False,
        collate_fn=build_collate_fn(cfg, force_batch_pad=use_batch_pad),
    )
    return loader
