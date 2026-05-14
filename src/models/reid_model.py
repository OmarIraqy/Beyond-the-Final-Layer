"""Full ReID model: backbone + projection (optional) + head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone, freeze_backbone_blocks
from .head import build_head
from ..data.resize_utils import build_patch_mask


class ReIDModel(nn.Module):
    """Modular ReID model.

    Composed of:
      1. A timm backbone (any model) producing [B, feat_dim] features
      2. An optional projection layer if embed_dim != feat_dim
      3. An embedding head (BNNeck, ArcFace, etc.) producing the output dict
    """

    def __init__(self, cfg, num_pids: int, num_obj_classes: int = 0):
        super().__init__()
        self.cfg = cfg
        self.backbone, feat_dim = build_backbone(cfg)

        embed_dim = cfg.head.embed_dim if cfg.head.embed_dim else feat_dim
        self.proj = None
        if embed_dim != feat_dim:
            self.proj = nn.Linear(feat_dim, embed_dim)

        # Optional dropout before head (regularization)
        dropout_rate = getattr(cfg.head, "dropout", 0.0)
        self.feat_drop = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.head = build_head(cfg, embed_dim, num_pids, num_obj_classes)
        self.neck_feat = cfg.head.neck_feat

        # Freeze a percentage of backbone blocks if requested
        freeze_pct = getattr(cfg.backbone, "freeze_percent", 0.0)
        if freeze_pct > 0:
            freeze_backbone_blocks(self.backbone, freeze_pct)

    def _resolve_patch_mask(
        self,
        pixel_mask: torch.Tensor = None,
        patch_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if patch_mask is not None:
            return patch_mask
        if pixel_mask is None:
            return None

        patch_stride = getattr(self.backbone, "patch_stride", None)
        if patch_stride is None:
            patch_stride = getattr(self.cfg.input, "patch_mask_stride", None)
        if patch_stride is None:
            return None
        return build_patch_mask(pixel_mask, patch_stride)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None,
        pixel_mask: torch.Tensor = None,
        patch_mask: torch.Tensor = None,
        class_labels: torch.Tensor = None,
        camids: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            x: [B, C, H, W] input images
            labels: optional [B] identity labels (for ArcFace)
            pixel_mask: optional [B, H, W] valid-pixel mask
            patch_mask: optional [B, Hp, Wp] or [B, N] valid-patch mask
            class_labels: optional [B] object-class indices (for VPT / class-weighted losses)
            camids: optional [B] camera IDs (for SIE)

        Returns:
            dict: {embeddings, bn_embeddings, id_logits, class_logits (optional)}
        """
        patch_mask = self._resolve_patch_mask(pixel_mask=pixel_mask, patch_mask=patch_mask)
        # Inject class labels into VPT backbone if supported
        if class_labels is not None and hasattr(self.backbone, "set_class_labels"):
            self.backbone.set_class_labels(class_labels)
        # Inject camera IDs into SIE backbone if supported
        if camids is not None and hasattr(self.backbone, "set_camids"):
            self.backbone.set_camids(camids)
        features = self.backbone(x, patch_mask=patch_mask, pixel_mask=pixel_mask)  # [B, feat_dim]
        if self.proj is not None:
            features = self.proj(features)
        if self.feat_drop is not None:
            features = self.feat_drop(features)

        if hasattr(self.head, 'forward') and 'labels' in self.head.forward.__code__.co_varnames:
            return self.head(features, labels=labels)
        return self.head(features)

    @torch.no_grad()
    def extract(
        self,
        x: torch.Tensor,
        pixel_mask: torch.Tensor = None,
        patch_mask: torch.Tensor = None,
        camids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inference mode: return embeddings (normalized)."""
        patch_mask = self._resolve_patch_mask(pixel_mask=pixel_mask, patch_mask=patch_mask)
        if camids is not None and hasattr(self.backbone, "set_camids"):
            self.backbone.set_camids(camids)
        features = self.backbone(x, patch_mask=patch_mask, pixel_mask=pixel_mask)
        if self.proj is not None:
            features = self.proj(features)
        emb = self.head.extract(features, neck_feat=self.neck_feat)
        return emb
