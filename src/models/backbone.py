"""Backbone registry — wraps timm for one-line backbone swaps."""

import logging
import math
from collections.abc import Sequence

import timm
import torch
import torch.nn as nn
from typing import List, Tuple

logger = logging.getLogger("urban_reid")


_PATCH_STRIDE_MODEL_HINTS = ("vit", "eva", "deit", "beit")


def _unwrap_backbone_module(backbone: nn.Module) -> nn.Module:
    return getattr(backbone, "model", backbone)


def unwrap_backbone_module(backbone: nn.Module) -> nn.Module:
    """Return the underlying timm backbone when wrapped."""
    return _unwrap_backbone_module(backbone)


def _normalize_patch_stride(patch_stride) -> Tuple[int, int]:
    if isinstance(patch_stride, int):
        return patch_stride, patch_stride
    if isinstance(patch_stride, Sequence) and not isinstance(patch_stride, (str, bytes)):
        stride = tuple(int(value) for value in patch_stride)
        if len(stride) == 2:
            return stride
    raise ValueError(
        "cfg.backbone.patch_stride must be an int or a two-element sequence, "
        f"got {patch_stride!r}"
    )


def _supports_patch_stride(model_name: str) -> bool:
    lower_name = model_name.lower()
    return any(hint in lower_name for hint in _PATCH_STRIDE_MODEL_HINTS)


def _apply_patch_stride(model: nn.Module, model_name: str, patch_stride: Tuple[int, int]) -> None:
    patch_embed = getattr(model, "patch_embed", None)
    proj = getattr(patch_embed, "proj", None)
    if proj is None or not hasattr(proj, "stride"):
        raise ValueError(
            f"cfg.backbone.patch_stride is only supported for ViT-like timm backbones with "
            f"patch_embed.proj; got {model_name!r}"
        )
    if not getattr(model, "dynamic_img_size", False):
        raise ValueError(
            f"cfg.backbone.patch_stride requires a timm backbone that supports dynamic_img_size; "
            f"got {model_name!r}"
        )

    proj.stride = patch_stride
    logger.info("Backbone patch stride override: %s -> %s", model_name, patch_stride)


def _infer_patch_stride(model: nn.Module):
    patch_embed = getattr(model, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", None)
    if patch_size is None:
        proj = getattr(patch_embed, "proj", None)
        patch_size = getattr(proj, "stride", None)
    if patch_size is None:
        return None
    return _normalize_patch_stride(patch_size)


class TimmBackboneAdapter(nn.Module):
    """Wrap timm backbones so optional mask inputs do not leak into call sites."""

    def __init__(self, model: nn.Module, model_name: str, patch_stride=None):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.num_features = model.num_features
        self.patch_stride = patch_stride
        self.supports_patch_mask = False
        self._warned_mask_ignored = False

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor = None, pixel_mask: torch.Tensor = None) -> torch.Tensor:
        if (patch_mask is not None or pixel_mask is not None) and not self._warned_mask_ignored:
            logger.warning(
                "Backbone %s ignores patch/pixel masks and will use the standard timm forward path.",
                self.model_name,
            )
            self._warned_mask_ignored = True
        return self.model(x)


class MaskedAveragePoolingBackbone(TimmBackboneAdapter):
    """Mask-aware wrapper for ViT-like backbones that use average token pooling."""

    def __init__(self, model: nn.Module, model_name: str, patch_stride=None):
        super().__init__(model, model_name, patch_stride=patch_stride)
        self.supports_patch_mask = True
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))

    def _flatten_patch_mask(self, patch_mask: torch.Tensor, num_tokens: int, device: torch.device):
        if patch_mask is None:
            return None

        if patch_mask.ndim == 3:
            flat_mask = patch_mask.reshape(patch_mask.shape[0], -1)
        elif patch_mask.ndim == 2:
            flat_mask = patch_mask
        else:
            logger.warning(
                "Backbone %s received patch_mask with unsupported shape %s; falling back to unmasked pooling.",
                self.model_name,
                tuple(patch_mask.shape),
            )
            return None

        if flat_mask.shape[1] != num_tokens:
            logger.warning(
                "Backbone %s received patch_mask with %d tokens, expected %d; falling back to unmasked pooling.",
                self.model_name,
                flat_mask.shape[1],
                num_tokens,
            )
            return None

        return flat_mask.to(device=device, dtype=torch.bool)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor = None, pixel_mask: torch.Tensor = None) -> torch.Tensor:
        if patch_mask is None:
            return self.model(x)

        token_embeddings = self.model.forward_features(x)
        if not isinstance(token_embeddings, torch.Tensor) or token_embeddings.ndim != 3:
            return super().forward(x, patch_mask=patch_mask, pixel_mask=pixel_mask)

        spatial_tokens = token_embeddings[:, self.num_prefix_tokens :, :]
        flat_mask = self._flatten_patch_mask(patch_mask, spatial_tokens.shape[1], x.device)
        if flat_mask is None:
            return super().forward(x, patch_mask=patch_mask, pixel_mask=pixel_mask)

        weights = flat_mask.to(dtype=spatial_tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (spatial_tokens * weights).sum(dim=1) / denom

        fc_norm = getattr(self.model, "fc_norm", None)
        if fc_norm is not None:
            pooled = fc_norm(pooled)

        head_drop = getattr(self.model, "head_drop", None)
        if head_drop is not None:
            pooled = head_drop(pooled)

        return pooled


class MeanVarPoolingBackbone(MaskedAveragePoolingBackbone):
    """Plan 1: mean + variance pooling over patch tokens → projected to same dim.

    Variance captures texture 'roughness' — the spread of local patch features.
    """

    def __init__(self, model: nn.Module, model_name: str, patch_stride=None):
        super().__init__(model, model_name, patch_stride=patch_stride)
        D = model.num_features
        self.var_proj = nn.Linear(2 * D, D)
        with torch.no_grad():
            nn.init.zeros_(self.var_proj.weight)
            # First D columns → identity (preserve mean behaviour at init)
            nn.init.eye_(self.var_proj.weight[:, :D])
            nn.init.zeros_(self.var_proj.bias)

    def _pool_mean_var(self, spatial_tokens: torch.Tensor, flat_mask=None):
        if flat_mask is None:
            mean_feat = spatial_tokens.mean(dim=1)
            var_feat = spatial_tokens.var(dim=1, unbiased=False)
        else:
            weights = flat_mask.to(dtype=spatial_tokens.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            mean_feat = (spatial_tokens * weights).sum(dim=1) / denom
            diff = spatial_tokens - mean_feat.unsqueeze(1)
            var_feat = ((diff ** 2) * weights).sum(dim=1) / denom
        return mean_feat, var_feat

    def forward(self, x: torch.Tensor, patch_mask=None, pixel_mask=None) -> torch.Tensor:
        token_embeddings = self.model.forward_features(x)
        if not isinstance(token_embeddings, torch.Tensor) or token_embeddings.ndim != 3:
            return super().forward(x, patch_mask=patch_mask, pixel_mask=pixel_mask)

        spatial_tokens = token_embeddings[:, self.num_prefix_tokens:, :]
        flat_mask = None
        if patch_mask is not None:
            flat_mask = self._flatten_patch_mask(patch_mask, spatial_tokens.shape[1], x.device)

        mean_feat, var_feat = self._pool_mean_var(spatial_tokens, flat_mask)

        fc_norm = getattr(self.model, "fc_norm", None)
        if fc_norm is not None:
            mean_feat = fc_norm(mean_feat)
        head_drop = getattr(self.model, "head_drop", None)
        if head_drop is not None:
            mean_feat = head_drop(mean_feat)

        combined = torch.cat([mean_feat, var_feat], dim=1)  # [B, 2D]
        return self.var_proj(combined)  # [B, D]


class MultiLayerViTBackbone(nn.Module):
    """Plan 2: extract patch tokens from multiple ViT depth levels, combine with learned weights.

    Shallow blocks encode texture/mid-level patterns; deep blocks encode semantics.
    Memory note: holds [B, N, D] activations for each hooked block simultaneously.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        hook_blocks: List[int],
        init_weights: List[float],
        patch_stride=None,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.num_features = model.num_features
        self.patch_stride = patch_stride
        self.supports_patch_mask = True
        self.hook_blocks = list(hook_blocks)
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
        self._warned_mask_ignored = False

        init_w = torch.tensor(list(init_weights), dtype=torch.float32)
        if len(init_w) != len(hook_blocks):
            logger.warning(
                "MultiLayerViT: length of init_weights (%d) does not match number of hook_blocks (%d); "
                "defaulting to equal weights",
                len(init_w),
                len(hook_blocks),
            )    
            init_w = torch.ones(len(hook_blocks)) / len(hook_blocks)
        self.layer_weights = nn.Parameter(init_w)

        self._hook_features: dict = {}
        self._hook_handles: List = []
        blocks = list(getattr(model, "blocks", []))
        for block_idx in hook_blocks:
            if block_idx >= len(blocks):
                logger.warning("MultiLayerViT: hook_block %d out of range (%d blocks)", block_idx, len(blocks))
                continue
            handle = blocks[block_idx].register_forward_hook(self._make_hook(block_idx))
            self._hook_handles.append(handle)

    def _make_hook(self, block_idx: int):
        def _hook(module, inp, output):
            # output may be a tuple (timm sometimes returns (tensor, ...) )
            self._hook_features[block_idx] = output[0] if isinstance(output, tuple) else output
        return _hook

    def _flatten_patch_mask(self, patch_mask, num_tokens, device):
        if patch_mask.ndim == 3:
            flat = patch_mask.reshape(patch_mask.shape[0], -1)
        elif patch_mask.ndim == 2:
            flat = patch_mask
        else:
            return None
        if flat.shape[1] != num_tokens:
            return None
        return flat.to(device=device, dtype=torch.bool)

    def _masked_mean(self, tokens, flat_mask=None):
        spatial = tokens[:, self.num_prefix_tokens:, :]
        if flat_mask is None:
            return spatial.mean(dim=1)
        w = flat_mask.to(dtype=spatial.dtype).unsqueeze(-1)
        return (spatial * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def forward(self, x: torch.Tensor, patch_mask=None, pixel_mask=None) -> torch.Tensor:
        self._hook_features = {}
        final_tokens = self.model.forward_features(x)

        flat_mask = None
        if patch_mask is not None and isinstance(final_tokens, torch.Tensor) and final_tokens.ndim == 3:
            num_spatial = final_tokens.shape[1] - self.num_prefix_tokens
            flat_mask = self._flatten_patch_mask(patch_mask, num_spatial, x.device)

        layer_feats = []
        for block_idx in self.hook_blocks:
            tokens = self._hook_features.get(block_idx)
            if tokens is None or not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                continue
            layer_feats.append(self._masked_mean(tokens, flat_mask))

        if not layer_feats:
            if isinstance(final_tokens, torch.Tensor) and final_tokens.ndim == 3:
                return self._masked_mean(final_tokens, flat_mask)
            return self.model(x)

        # Apply fc_norm to the final layer's pooled feature
        fc_norm = getattr(self.model, "fc_norm", None)
        if fc_norm is not None:
            layer_feats[-1] = fc_norm(layer_feats[-1])

        stacked = torch.stack(layer_feats, dim=1)  # [B, L, D]
        weights = torch.softmax(self.layer_weights[: len(layer_feats)], dim=0)  # [L]
        return (stacked * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # [B, D]

def build_backbone(cfg) -> Tuple[nn.Module, int]:
    """Create a timm model as a feature extractor.

    Returns (backbone, feature_dim).
    Setting num_classes=0 removes the classification head so the model
    returns a [B, feature_dim] feature tensor.

    Backbone modes (mutually exclusive, checked in priority order):
      stripe_pooling_enabled      → PartPoolingBackbone              (Plan 3)
      multi_layer_enabled + sa    → MultiLayerSelfAttentionBackbone  (Plan 2 SA)
      multi_layer_enabled + cam   → MultiLayerViTBackboneWithCamera
      multi_layer_enabled + depth → MultiLayerViTBackboneWithDepth
      multi_layer_enabled         → MultiLayerViTBackbone            (Plan 2)
      prompts_enabled             → ViTWithPrompts                   (Plan 4)
      sie_cameras > 0             → CameraAwareSIEBackbone           (S4)
      pool_mode == "mean_var"     → MeanVarPoolingBackbone           (Plan 1)
    """
    extra = dict(cfg.backbone.extra_kwargs) if cfg.backbone.extra_kwargs else {}
    raw_patch_stride = getattr(cfg.backbone, "patch_stride", None)
    patch_stride = None
    if raw_patch_stride is not None:
        if not _supports_patch_stride(cfg.backbone.name):
            raise ValueError(
                "cfg.backbone.patch_stride currently supports ViT/EVA-style timm backbones; "
                f"got {cfg.backbone.name!r}"
            )
        patch_stride = _normalize_patch_stride(raw_patch_stride)
        extra.setdefault("dynamic_img_size", True)

    model = timm.create_model(
        cfg.backbone.name,
        pretrained=cfg.backbone.pretrained,
        num_classes=0,
        drop_path_rate=cfg.backbone.drop_path_rate,
        **extra,
    )
    if patch_stride is not None:
        _apply_patch_stride(model, cfg.backbone.name, patch_stride)
    # Ensure patch_embed supports dynamic image padding (needed for non-standard sizes)
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "dynamic_img_pad"):
        model.patch_embed.dynamic_img_pad = True
    inferred_patch_stride = patch_stride or _infer_patch_stride(model)

    is_vit_like = hasattr(model, "forward_features") and getattr(model, "global_pool", None) == "avg"

    # Parse optional manual camid_map once for all camera-aware branches
    camid_map = None
    if getattr(cfg.backbone, "camid_map", None):
        camid_map = {int(k): int(v) for k, v in cfg.backbone.camid_map.items()}

    # --- Plan 3: stripe pooling (highest priority for ViT-like models) ---
    if getattr(cfg.backbone, "stripe_pooling_enabled", False):
        if not is_vit_like:
            raise ValueError("stripe_pooling requires a ViT-like backbone with global_pool='avg'")
        backbone = PartPoolingBackbone(
            model, cfg.backbone.name,
            num_stripes=cfg.backbone.stripe_pooling_num_stripes,
            patch_stride=inferred_patch_stride,
        )
        logger.info("Backbone: PartPoolingBackbone (%d stripes)", cfg.backbone.stripe_pooling_num_stripes)

    # --- Plan 2: multi-layer aggregation ---
    elif getattr(cfg.backbone, "multi_layer_enabled", False):
        if not hasattr(model, "forward_features") or not hasattr(model, "blocks"):
            raise ValueError("multi_layer_enabled requires a ViT-like backbone with .blocks")
        if getattr(cfg.backbone, "multi_layer_sa_blocks", 0) > 0:
            backbone = MultiLayerSelfAttentionBackbone(
                model, cfg.backbone.name,
                hook_blocks=list(cfg.backbone.multi_layer_hook_blocks),
                num_sa_blocks=cfg.backbone.multi_layer_sa_blocks,
                sa_nheads=cfg.backbone.multi_layer_sa_nheads,
                sa_dropout=cfg.backbone.multi_layer_sa_dropout,
                patch_stride=inferred_patch_stride,
            )
            logger.info(
                "Backbone: MultiLayerSelfAttentionBackbone, hooks=%s, sa_blocks=%d, nheads=%d",
                list(cfg.backbone.multi_layer_hook_blocks),
                cfg.backbone.multi_layer_sa_blocks,
                cfg.backbone.multi_layer_sa_nheads,
            )
        elif getattr(cfg.backbone, "sie_cameras", 0) > 0:
            if getattr(cfg.backbone, "depth_enabled", False):
                raise ValueError("depth_enabled + sie_cameras is not yet supported; use one at a time")
            backbone = MultiLayerViTBackboneWithCamera(
                model, cfg.backbone.name,
                hook_blocks=list(cfg.backbone.multi_layer_hook_blocks),
                init_weights=list(cfg.backbone.multi_layer_init_weights),
                num_cameras=cfg.backbone.sie_cameras,
                camid_map=camid_map,
                patch_stride=inferred_patch_stride,
            )
            logger.info("Backbone: MultiLayerViTBackboneWithCamera, hooks=%s, num_cameras=%d", list(cfg.backbone.multi_layer_hook_blocks), cfg.backbone.sie_cameras)
        elif getattr(cfg.backbone, "depth_enabled", False):
            backbone = MultiLayerViTBackboneWithDepth(
                model, cfg.backbone.name,
                hook_blocks=list(cfg.backbone.multi_layer_hook_blocks),
                init_weights=list(cfg.backbone.multi_layer_init_weights),
                depth_model_variant=cfg.backbone.depth_model_variant,
                depth_checkpoint=cfg.backbone.depth_checkpoint,
                depth_init_weight=cfg.backbone.depth_init_weight,
                patch_stride=inferred_patch_stride,
            )
            logger.info("Backbone: MultiLayerViTBackboneWithDepth, hooks=%s, depth_variant=%s",
                        list(cfg.backbone.multi_layer_hook_blocks), cfg.backbone.depth_model_variant)
        else:
            backbone = MultiLayerViTBackbone(
                model, cfg.backbone.name,
                hook_blocks=list(cfg.backbone.multi_layer_hook_blocks),
                init_weights=list(cfg.backbone.multi_layer_init_weights),
                patch_stride=inferred_patch_stride,
            )
            logger.info("Backbone: MultiLayerViTBackbone, hooks=%s", list(cfg.backbone.multi_layer_hook_blocks))

    # --- Plan 4: visual prompt tuning ---
    elif getattr(cfg.backbone, "prompts_enabled", False):
        if not hasattr(model, "forward_features") or not hasattr(model, "blocks"):
            raise ValueError("prompts_enabled requires a ViT-like backbone with .blocks")
        backbone = ViTWithPrompts(
            model, cfg.backbone.name,
            num_obj_classes=cfg.backbone.prompts_num_classes,
            n_prompts=cfg.backbone.prompts_n,
            patch_stride=inferred_patch_stride,
        )
        logger.info("Backbone: ViTWithPrompts (n_prompts=%d, num_classes=%d)",
                    cfg.backbone.prompts_n, cfg.backbone.prompts_num_classes)

    # --- S4: camera-aware SIE ---
    elif getattr(cfg.backbone, "sie_cameras", 0) > 0:
        if not is_vit_like or not hasattr(model, "blocks"):
            raise ValueError("sie_cameras requires a ViT-like backbone with .blocks")
        backbone = CameraAwareSIEBackbone(
            model, cfg.backbone.name,
            num_cameras=cfg.backbone.sie_cameras,
            camid_map=camid_map,
            patch_stride=inferred_patch_stride,
        )
        logger.info("Backbone: CameraAwareSIEBackbone (num_cameras=%d)", cfg.backbone.sie_cameras)

    # --- Plan 1: mean+variance pooling ---
    elif getattr(cfg.backbone, "pool_mode", "mean") == "mean_var":
        if not is_vit_like:
            raise ValueError("pool_mode='mean_var' requires a ViT-like backbone with global_pool='avg'")
        backbone = MeanVarPoolingBackbone(model, cfg.backbone.name, patch_stride=inferred_patch_stride)
        logger.info("Backbone: MeanVarPoolingBackbone")

    # --- Default ---
    elif is_vit_like:
        backbone = MaskedAveragePoolingBackbone(model, cfg.backbone.name, patch_stride=inferred_patch_stride)
    else:
        backbone = TimmBackboneAdapter(model, cfg.backbone.name, patch_stride=inferred_patch_stride)

    feature_dim = backbone.num_features
    return backbone, feature_dim


def get_backbone_blocks(backbone: nn.Module) -> List[nn.Module]:
    """Return an ordered list of the main repeating blocks from a timm backbone.

    Supports the common timm architectures:
      - ViT / DeiT / EVA / DINOv2/v3 →  backbone.blocks  (Sequential or ModuleList)
      - Swin / SwinV2                 →  backbone.layers
      - ConvNeXt / ConvNeXtV2         →  backbone.stages
      - EfficientNet / EfficientNetV2 →  backbone.blocks  (Sequential)
      - ResNet / ResNeXt              →  backbone.layer1 … layer4
    Falls back to all direct children if none of the above match.
    """
    backbone = _unwrap_backbone_module(backbone)
    _CONTAINER = (nn.Sequential, nn.ModuleList)

    # ViT-like & EfficientNet: .blocks is Sequential or ModuleList of transformer / conv blocks
    if hasattr(backbone, "blocks") and isinstance(backbone.blocks, _CONTAINER):
        return list(backbone.blocks)

    # Swin / SwinV2: flatten stages → individual transformer blocks for fine-grained control
    # backbone.layers is a Sequential of SwinTransformerStage objects, each with .blocks
    if hasattr(backbone, "layers") and isinstance(backbone.layers, _CONTAINER):
        flat = []
        for stage in backbone.layers:
            if hasattr(stage, "blocks") and isinstance(stage.blocks, _CONTAINER):
                flat.extend(list(stage.blocks))
            else:
                flat.append(stage)
        return flat

    # ConvNeXt / ConvNeXtV2
    if hasattr(backbone, "stages") and isinstance(backbone.stages, _CONTAINER):
        return list(backbone.stages)

    # ResNet-like: layer1 … layerN
    layers = []
    for i in range(1, 10):
        name = f"layer{i}"
        if hasattr(backbone, name):
            layers.append(getattr(backbone, name))
    if layers:
        return layers

    # Fallback
    return list(backbone.children())


def freeze_backbone_blocks(backbone: nn.Module, freeze_percent: float) -> int:
    """Freeze the first *freeze_percent* of backbone blocks (and the stem).

    Args:
        backbone: timm backbone module.
        freeze_percent: value in [0, 1]. 0 freezes nothing; 1 freezes all blocks.

    Returns:
        Number of blocks frozen.
    """
    if freeze_percent <= 0.0:
        return 0

    backbone = _unwrap_backbone_module(backbone)
    blocks = get_backbone_blocks(backbone)
    num_blocks = len(blocks)
    num_to_freeze = int(math.ceil(num_blocks * min(freeze_percent, 1.0)))

    # --- freeze stem / patch-embed / positional embedding ---
    _STEM_ATTRS = [
        "patch_embed", "cls_token", "pos_embed", "pos_drop",
        "stem", "conv1", "bn1", "maxpool", "norm_pre",
        "downsample_layers",
    ]
    for attr in _STEM_ATTRS:
        obj = getattr(backbone, attr, None)
        if obj is None:
            continue
        if isinstance(obj, nn.Parameter):
            obj.requires_grad_(False)
        elif isinstance(obj, nn.Module):
            for p in obj.parameters():
                p.requires_grad = False

    # --- freeze the first N blocks ---
    for block in blocks[:num_to_freeze]:
        for p in block.parameters():
            p.requires_grad = False

    # --- Swin/SwinV2: also freeze each stage's downsample (PatchMerging) when
    #     ALL blocks in that stage are frozen, and the final backbone norm when
    #     all blocks are frozen. ---
    if hasattr(backbone, "layers") and isinstance(backbone.layers, (nn.Sequential, nn.ModuleList)):
        for stage in backbone.layers:
            if hasattr(stage, "blocks") and hasattr(stage, "downsample") and stage.downsample is not None:
                all_frozen = all(
                    not p.requires_grad
                    for blk in stage.blocks
                    for p in blk.parameters()
                )
                if all_frozen:
                    for p in stage.downsample.parameters():
                        p.requires_grad = False
        if num_to_freeze == num_blocks and hasattr(backbone, "norm") and isinstance(backbone.norm, nn.Module):
            for p in backbone.norm.parameters():
                p.requires_grad = False

    total_params = sum(p.numel() for p in backbone.parameters())
    frozen_params = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    logger.info(
        f"Backbone freeze: {num_to_freeze}/{num_blocks} blocks frozen "
        f"({frozen_params/total_params*100:.1f}% of backbone params)"
    )
    return num_to_freeze
