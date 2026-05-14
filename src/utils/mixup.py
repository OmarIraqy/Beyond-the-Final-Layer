"""Mixup and CutMix batch-level augmentation for ReID training."""

import numpy as np
import torch


def mixup_data(images: torch.Tensor, pids: torch.Tensor, alpha: float = 0.2):
    """Apply Mixup to a batch of images and return mixed images + lambda + shuffled indices.

    For ReID with triplet loss, we mix images but keep BOTH original and shuffled
    pid labels so the loss can handle them properly. The CE loss uses soft targets.

    Args:
        images: [B, C, H, W] input images.
        pids: [B] identity labels.
        alpha: Beta distribution parameter (higher = more mixing).

    Returns:
        mixed_images: [B, C, H, W]
        pids_a: [B] original labels
        pids_b: [B] shuffled labels
        lam: mixing coefficient (scalar)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    pids_b = pids[index]

    return mixed_images, pids, pids_b, lam


def cutmix_data(images: torch.Tensor, pids: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix to a batch of images.

    Cuts a rectangular patch from one image and pastes it onto another.
    The mixing ratio (lam) is proportional to the area of the patch.
    Unlike Mixup, CutMix forces the model to attend to non-local regions.

    Args:
        images: [B, C, H, W] input images.
        pids: [B] identity labels.
        alpha: Beta distribution parameter for box area sampling.

    Returns:
        mixed_images: [B, C, H, W]
        pids_a: [B] original labels
        pids_b: [B] shuffled labels
        lam: area ratio of the original image kept (scalar)
    """
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    _, _, h, w = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    # Uniform random center
    cx = np.random.randint(h)
    cy = np.random.randint(w)

    bbx1 = np.clip(cx - cut_h // 2, 0, h)
    bby1 = np.clip(cy - cut_w // 2, 0, w)
    bbx2 = np.clip(cx + cut_h // 2, 0, h)
    bby2 = np.clip(cy + cut_w // 2, 0, w)

    mixed_images = images.clone()
    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lam to match the exact area ratio
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
    pids_b = pids[index]

    return mixed_images, pids, pids_b, lam


def mixup_criterion_ce(logits: torch.Tensor, targets_a: torch.Tensor,
                       targets_b: torch.Tensor, lam: float,
                       label_smooth: float = 0.0) -> torch.Tensor:
    """Compute mixed cross-entropy loss for Mixup.

    Loss = lam * CE(logits, targets_a) + (1-lam) * CE(logits, targets_b)
    """
    import torch.nn.functional as F

    if label_smooth > 0:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        smooth = label_smooth / num_classes

        targets_a_oh = torch.zeros_like(log_probs).fill_(smooth)
        targets_a_oh.scatter_(1, targets_a.unsqueeze(1), 1.0 - label_smooth + smooth)
        loss_a = (-targets_a_oh * log_probs).sum(dim=1).mean()

        targets_b_oh = torch.zeros_like(log_probs).fill_(smooth)
        targets_b_oh.scatter_(1, targets_b.unsqueeze(1), 1.0 - label_smooth + smooth)
        loss_b = (-targets_b_oh * log_probs).sum(dim=1).mean()
    else:
        loss_a = F.cross_entropy(logits, targets_a)
        loss_b = F.cross_entropy(logits, targets_b)

    return lam * loss_a + (1 - lam) * loss_b
