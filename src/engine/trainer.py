"""Training engine — single-GPU and DDP-safe training loop."""

import os
import time
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from ..config import save_config
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.meters import AverageMeter
from ..utils.distributed import is_main_process, get_rank
from ..utils.device import get_module_device, move_to_device
from ..utils.ema import ModelEMA
from ..utils.mixup import mixup_data, cutmix_data, mixup_criterion_ce
from .evaluator import ReIDEvaluator, eval_train_map

logger = logging.getLogger("urban_reid")


def build_optimizer(cfg, model: nn.Module, objective_combiner: nn.Module = None) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Includes parameters from both the model and (optionally) the objective
    combiner — needed when losses contain learnable weights (e.g. ArcFace,
    CenterLoss).

    Supports layer-wise LR decay (LLRD) when cfg.solver.llrd < 1.0:
    deeper backbone layers get progressively lower learning rates.
    """
    opt_name = cfg.solver.optimizer.lower()
    base_lr = cfg.solver.lr
    wd = cfg.solver.weight_decay
    llrd = getattr(cfg.solver, "llrd", 1.0)

    if llrd < 1.0:
        param_groups = _build_llrd_param_groups(model, base_lr, wd, llrd)
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                         "lr": base_lr, "weight_decay": wd}]

    # Add objective combiner params (center loss embeddings, etc.)
    if objective_combiner is not None:
        loss_params = [p for p in objective_combiner.parameters() if p.requires_grad]
        if loss_params:
            param_groups.append({"params": loss_params, "lr": base_lr, "weight_decay": wd})

    if opt_name == "adam":
        return torch.optim.Adam(param_groups)
    elif opt_name == "adamw":
        return torch.optim.AdamW(param_groups)
    elif opt_name == "sgd":
        return torch.optim.SGD(param_groups, momentum=cfg.solver.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def _build_llrd_param_groups(model: nn.Module, base_lr: float, wd: float, decay: float):
    """Build per-layer parameter groups with decaying LR.

    Layer N (deepest) gets base_lr, layer N-1 gets base_lr * decay, etc.
    Head/projection/non-backbone params get base_lr.
    """
    from ..models.backbone import get_backbone_blocks

    # Get the raw model (unwrap DDP if needed)
    raw_model = model.module if hasattr(model, "module") else model
    backbone = raw_model.backbone

    blocks = get_backbone_blocks(backbone)
    num_layers = len(blocks)

    # Map each parameter to its layer depth
    param_to_group = {}

    # Backbone stem params → layer 0 (deepest = lowest LR)
    _STEM_ATTRS = ["patch_embed", "cls_token", "pos_embed", "pos_drop",
                   "stem", "conv1", "bn1", "norm_pre", "downsample_layers"]
    for attr in _STEM_ATTRS:
        obj = getattr(backbone, attr, None)
        if obj is None:
            continue
        if isinstance(obj, nn.Parameter):
            if obj.requires_grad:
                param_to_group[id(obj)] = 0
        elif isinstance(obj, nn.Module):
            for p in obj.parameters():
                if p.requires_grad:
                    param_to_group[id(p)] = 0

    # Backbone blocks → layers 1..num_layers
    for layer_idx, block in enumerate(blocks):
        for p in block.parameters():
            if p.requires_grad:
                param_to_group[id(p)] = layer_idx + 1

    # Backbone norm (final) → top layer
    if hasattr(backbone, "norm") and isinstance(backbone.norm, nn.Module):
        for p in backbone.norm.parameters():
            if p.requires_grad:
                param_to_group[id(p)] = num_layers

    # Build LR for each depth: layer 0 gets base_lr * decay^num_layers,
    # layer num_layers gets base_lr
    total_depth = num_layers + 1  # stem(0) + blocks(1..N)
    layer_lrs = {}
    for depth in range(total_depth):
        layer_lrs[depth] = base_lr * (decay ** (total_depth - 1 - depth))

    # Group params
    groups = {}
    # Backbone params with LLRD
    for p in backbone.parameters():
        if not p.requires_grad:
            continue
        depth = param_to_group.get(id(p), num_layers)
        lr = layer_lrs.get(depth, base_lr)
        lr_key = f"{lr:.10f}"
        if lr_key not in groups:
            groups[lr_key] = {"params": [], "lr": lr, "weight_decay": wd}
        groups[lr_key]["params"].append(p)

    # Non-backbone params (head, proj, etc.) get base_lr
    backbone_param_ids = {id(p) for p in backbone.parameters()}
    head_params = [p for p in raw_model.parameters()
                   if p.requires_grad and id(p) not in backbone_param_ids]
    if head_params:
        groups["head"] = {"params": head_params, "lr": base_lr, "weight_decay": wd}

    result = list(groups.values())
    # Tag each group with its target LR for warmup compatibility
    for g in result:
        g["lr_target"] = g["lr"]
    logger.info(f"LLRD: {len(result)} param groups, decay={decay}, "
                f"LR range [{min(g['lr'] for g in result):.2e}, {max(g['lr'] for g in result):.2e}]")
    return result


def build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    """Build LR scheduler from config."""
    sched_name = cfg.solver.scheduler.lower()

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.solver.max_epochs - cfg.solver.warmup_epochs, eta_min=cfg.solver.min_lr
        )
    elif sched_name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(cfg.solver.step_milestones), gamma=cfg.solver.gamma
        )
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.solver.gamma, patience=10
        )
    elif sched_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def _warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float, warmup_lr: float):
    """Linear warmup from warmup_lr to base_lr (or per-group base LR for LLRD)."""
    if epoch >= warmup_epochs:
        return
    progress = epoch / warmup_epochs
    for pg in optimizer.param_groups:
        # Each group may have its own target LR (from LLRD)
        target_lr = pg.get("lr_target", base_lr)
        pg["lr"] = warmup_lr + (target_lr - warmup_lr) * progress


class Trainer:
    """Main training loop with AMP, gradient accumulation, checkpointing, and evaluation."""

    def __init__(
        self,
        cfg,
        model: nn.Module,
        train_loader,
        objective_combiner: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        val_loader=None,
        num_val_query: int = 0,
        val_pids=None,
        val_camids=None,
        val_class_labels=None,
        train_probe_loader=None,
        train_pids=None,
        train_class_labels=None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.objective_combiner = objective_combiner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.val_loader = val_loader
        self.num_val_query = num_val_query
        self.val_pids = val_pids
        self.val_camids = val_camids
        self.val_class_labels = val_class_labels
        self.train_probe_loader = train_probe_loader
        self.train_pids = train_pids
        self.train_class_labels = train_class_labels

        self.max_epochs = cfg.solver.max_epochs
        self.warmup_epochs = cfg.solver.warmup_epochs
        self.eval_period = cfg.trainer.eval_period
        self.checkpoint_period = cfg.trainer.checkpoint_period
        self.log_period = cfg.trainer.log_period
        self.output_dir = cfg.trainer.output_dir

        self.device = get_module_device(model)
        self.amp = cfg.solver.amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp)
        self.grad_accum_steps = cfg.solver.grad_accum_steps
        self.clip_grad_norm = cfg.solver.clip_grad_norm

        self.evaluator = ReIDEvaluator(cfg)
        self.best_mAP = 0.0
        self.start_epoch = 0
        # S2: which metric drives best-model checkpointing
        self.checkpoint_metric = getattr(cfg.trainer, "checkpoint_metric", "mAP")

        # EMA
        self.ema = None
        if getattr(cfg.solver, "ema", False):
            raw_model = model.module if hasattr(model, "module") else model
            self.ema = ModelEMA(raw_model, decay=cfg.solver.ema_decay)

        # Mixup / CutMix config
        self.mixup_alpha = getattr(cfg.input, "mixup_alpha", 0.0)
        self.cutmix_alpha = getattr(cfg.input, "cutmix_alpha", 0.0)
        self.mixup_prob = getattr(cfg.input, "mixup_prob", 1.0)
        self.aug_mode = getattr(cfg.input, "aug_mode", "none").lower()

        # TensorBoard
        self.writer = None
        if is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb_logs"))
            # Save config for reproducibility
            save_config(cfg, self.output_dir)

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        ckpt = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_mAP = ckpt.get("best_mAP", 0.0)
        if self.amp and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        logger.info(f"Resumed from epoch {self.start_epoch}, best mAP={self.best_mAP:.4f}")

    def train(self):
        """Run the full training loop."""
        logger.info(f"Start training for {self.max_epochs} epochs")
        logger.info(f"Output dir: {self.output_dir}")

        for epoch in range(self.start_epoch, self.max_epochs):
            # Warmup
            if epoch < self.warmup_epochs:
                _warmup_lr(
                    self.optimizer, epoch, self.warmup_epochs,
                    self.cfg.solver.lr, self.cfg.solver.warmup_lr,
                )

            loss_dict_avg = self._train_one_epoch(epoch)

            # Log to tensorboard
            if self.writer:
                for k, v in loss_dict_avg.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Step scheduler (after warmup)
            if epoch >= self.warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    pass  # stepped after evaluation
                else:
                    self.scheduler.step()

            # Evaluation
            if self.val_loader and (epoch + 1) % self.eval_period == 0:
                # Apply EMA weights for evaluation if enabled
                if self.ema is not None:
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    self.ema.apply_shadow(raw_model)

                metrics = self.evaluator.evaluate(
                    self.model, self.val_loader, self.num_val_query,
                    q_pids=self.val_pids[:self.num_val_query] if self.val_pids is not None else None,
                    g_pids=self.val_pids[self.num_val_query:] if self.val_pids is not None else None,
                    q_camids=self.val_camids[:self.num_val_query] if self.val_camids is not None else None,
                    g_camids=self.val_camids[self.num_val_query:] if self.val_camids is not None else None,
                    q_class_labels=self.val_class_labels[:self.num_val_query] if self.val_class_labels is not None else None,
                    g_class_labels=self.val_class_labels[self.num_val_query:] if self.val_class_labels is not None else None,
                )
                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"val/{k}", v, epoch)

                # Train mAP — overfitting indicator
                if self.train_probe_loader is not None and self.train_pids is not None:
                    train_metrics = eval_train_map(
                        self.model,
                        self.train_probe_loader,
                        self.train_pids,
                        train_class_labels=self.train_class_labels,
                        flip=self.cfg.test.flip_test,
                        feat_norm=self.cfg.test.feat_norm,
                    )
                    if self.writer:
                        for k, v in train_metrics.items():
                            self.writer.add_scalar(f"train/{k}", v, epoch)
                    logger.info(
                        f"Epoch [{epoch+1}]  val_mAP={metrics['mAP']:.4f}  "
                        f"train_mAP={train_metrics['train_mAP']:.4f}  "
                        f"gap={train_metrics['train_mAP'] - metrics['mAP']:.4f}"
                    )

                # Plateau scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics["mAP"])

                # S2: save best checkpoint by the configured primary metric
                primary = (
                    metrics.get("test-weighted-mAP", metrics["mAP"])
                    if self.checkpoint_metric == "test_weighted_map"
                    else metrics["mAP"]
                )
                if is_main_process() and primary > self.best_mAP:
                    self.best_mAP = primary
                    self._save(epoch, metrics, filename="best_model.pth")
                    logger.info(
                        f"New best {self.checkpoint_metric}: {self.best_mAP:.4f} at epoch {epoch}"
                    )

                # Restore training weights after EMA evaluation
                if self.ema is not None:
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    self.ema.restore(raw_model)

            # Periodic checkpoint
            if is_main_process() and (epoch + 1) % self.checkpoint_period == 0:
                self._save(epoch, loss_dict_avg, filename=f"checkpoint_ep{epoch+1}.pth")

        # Save final
        if is_main_process():
            self._save(self.max_epochs - 1, {}, filename="final_model.pth")

        if self.writer:
            self.writer.close()

        logger.info(f"Training complete. Best mAP: {self.best_mAP:.4f}")

    def _train_one_epoch(self, epoch: int) -> dict:
        """Train for one epoch. Returns average loss dict."""
        self.model.train()
        meters = {}
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        for i, batch in enumerate(self.train_loader):
            images = move_to_device(batch["images"], self.device, non_blocking=True)
            pids = move_to_device(batch["pids"], self.device, non_blocking=True)
            pixel_mask = move_to_device(batch.get("pixel_mask"), self.device, non_blocking=True)
            patch_mask = move_to_device(batch.get("patch_mask"), self.device, non_blocking=True)
            class_labels = move_to_device(batch.get("class_labels"), self.device, non_blocking=True)
            camids = move_to_device(batch.get("camids"), self.device, non_blocking=True)
            masks_present = pixel_mask is not None or patch_mask is not None

            # Batch-level augmentation (Mixup / CutMix)
            aug_active = False
            pids_a = pids
            pids_b = pids
            lam = 1.0

            if not masks_present and self.aug_mode != "none" and torch.rand(1).item() < self.mixup_prob:
                if self.aug_mode == "mixup" and self.mixup_alpha > 0:
                    images, pids_a, pids_b, lam = mixup_data(images, pids, self.mixup_alpha)
                    aug_active = True
                elif self.aug_mode == "cutmix" and self.cutmix_alpha > 0:
                    images, pids_a, pids_b, lam = cutmix_data(images, pids, self.cutmix_alpha)
                    aug_active = True
                elif self.aug_mode == "random":
                    if torch.rand(1).item() < 0.5 and self.mixup_alpha > 0:
                        images, pids_a, pids_b, lam = mixup_data(images, pids, self.mixup_alpha)
                        aug_active = True
                    elif self.cutmix_alpha > 0:
                        images, pids_a, pids_b, lam = cutmix_data(images, pids, self.cutmix_alpha)
                        aug_active = True

            if masks_present and self.aug_mode != "none" and not hasattr(self, "_logged_mask_aug_disable"):
                logger.warning(
                    "Masks are present in the batch; disabling batch-level aug so padded regions stay well-defined."
                )
                self._logged_mask_aug_disable = True

            with autocast(enabled=self.amp):
                model_outputs = self.model(
                    images,
                    labels=pids_a,
                    pixel_mask=pixel_mask,
                    patch_mask=patch_mask,
                    class_labels=class_labels,
                    camids=camids,
                )

                if aug_active:
                    # Mixed-batch loss: interpolate CE loss, use original pids for triplet/metric losses
                    total_loss, loss_dict = self._compute_mixed_batch_loss(
                        model_outputs, batch, pids_a, pids_b, lam
                    )
                else:
                    total_loss, loss_dict = self.objective_combiner(model_outputs, batch)

                total_loss = total_loss / self.grad_accum_steps

            self.scaler.scale(total_loss).backward()

            # Gradient accumulation
            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == num_batches:
                if self.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # EMA update after each optimizer step
                if self.ema is not None:
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    self.ema.update(raw_model)

            # Update meters
            for k, v in loss_dict.items():
                if k not in meters:
                    meters[k] = AverageMeter(k)
                meters[k].update(v)

            # Log
            if (i + 1) % self.log_period == 0 and is_main_process():
                loss_str = " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch [{epoch+1}/{self.max_epochs}] "
                    f"Iter [{i+1}/{num_batches}] "
                    f"lr={lr:.6f} {loss_str}"
                )

        return {k: m.avg for k, m in meters.items()}

    def _compute_mixed_batch_loss(self, model_outputs, batch, pids_a, pids_b, lam):
        """Compute loss with Mixup/CutMix: interpolated CE + triplet on original pids."""
        total_loss = torch.tensor(0.0, device=pids_a.device)
        loss_dict = {}

        for name, weight, loss_fn in zip(
            self.objective_combiner.names,
            self.objective_combiner.weights,
            self.objective_combiner.objectives,
        ):
            if name == "id_loss" or "cross_entropy" in type(loss_fn).__name__.lower():
                # Mixed-batch CE: interpolate between both target sets
                label_smooth = getattr(loss_fn, "label_smooth", 0.0)
                loss = mixup_criterion_ce(
                    model_outputs["id_logits"], pids_a, pids_b, lam, label_smooth
                )
            else:
                # Triplet/metric losses: use original pids (spatial mixing doesn't change identity structure)
                loss = loss_fn(model_outputs, batch)

            loss_dict[name] = loss.item()
            total_loss = total_loss + weight * loss

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def _save(self, epoch: int, metrics: dict, filename: str):
        """Save checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_mAP": self.best_mAP,
            "metrics": metrics,
        }
        if self.amp:
            state["scaler_state_dict"] = self.scaler.state_dict()
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        save_checkpoint(state, self.output_dir, filename)
        logger.info(f"Saved {filename} (epoch {epoch+1})")
