"""Structured configuration for the Urban ReID modular repo.

Every experiment is fully defined by a single YAML config file.
The config is saved alongside checkpoints for reproducibility.
"""

import os
import copy
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from omegaconf import OmegaConf, DictConfig, MISSING


# ---------------------------------------------------------------------------
# Structured config dataclasses
# ---------------------------------------------------------------------------


def _default_class_sizes() -> Dict[str, List[int]]:
    return {
        "Container": [256, 128],
        "Crosswalk": [256, 128],
        "RubbishBins": [256, 128],
        "TrafficSign": [256, 128],
    }

@dataclass
class DatasetConfig:
    root: str = "/scratch/dr/Urban-ReID/Combined_dataset"
    train_csv: str = "train.csv"
    train_classes_csv: str = "train_classes.csv"
    query_csv: str = "query.csv"
    query_classes_csv: str = "query_classes.csv"
    test_csv: str = "test.csv"
    test_classes_csv: str = "test_classes.csv"
    val_query_csv: str = "val_query.csv"
    val_test_csv: str = "val_test.csv"
    val_query_classes_csv: str = "val_query_classes.csv"
    val_test_classes_csv: str = "val_test_classes.csv"
    num_obj_classes: int = 4


@dataclass
class InputConfig:
    size_train: List[int] = field(default_factory=lambda: [256, 128])
    size_test: List[int] = field(default_factory=lambda: [256, 128])
    resize_strategy: str = "fixed"  # fixed | class_letterbox
    keep_aspect_ratio: bool = True
    pad_value: str = "mean"  # mean | zero
    class_sizes_train: Dict[str, List[int]] = field(default_factory=_default_class_sizes)
    class_sizes_test: Dict[str, List[int]] = field(default_factory=_default_class_sizes)
    return_masks: bool = False
    patch_mask_stride: int = 16
    pixel_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    pixel_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    random_flip: bool = True
    random_erasing: bool = True
    re_prob: float = 0.5
    pad: int = 10
    random_crop: bool = True
    autoaug: bool = False
    color_jitter: bool = False
    # 6B: per-class color jitter (TrafficSign gets hue=0 to preserve sign color semantics)
    class_color_jitter: bool = False
    cj_brightness: float = 0.2
    cj_contrast: float = 0.15
    cj_saturation: float = 0.15
    cj_hue: float = 0.05
    gaussian_blur: bool = False
    gaussian_blur_prob: float = 0.5
    gaussian_blur_kernel_size: int = 3
    gaussian_blur_sigma: List[float] = field(default_factory=lambda: [0.1, 1.0])
    # Batch-level augmentation (applied in training loop, not transform pipeline)
    mixup_alpha: float = 0.0       # Mixup alpha (0 = disabled)
    cutmix_alpha: float = 0.0      # CutMix alpha (0 = disabled)
    mixup_prob: float = 1.0        # probability of applying mixup/cutmix per batch
    aug_mode: str = "none"         # "none" | "mixup" | "cutmix" | "random" (random picks one per batch)


@dataclass
class DataloaderConfig:
    batch_size: int = 64
    num_workers: int = 8
    sampler: str = "pk"  # random | pk | class_aware_pk
    batching_strategy: str = "standard"  # standard | same_class | batch_pad
    num_instances: int = 4
    pin_memory: bool = True


@dataclass
class BackboneConfig:
    name: str = "resnet50"
    pretrained: bool = True
    drop_path_rate: float = 0.1
    freeze_percent: float = 0.0  # fraction of backbone blocks to freeze (0.0 = none, 1.0 = all)
    patch_stride: Optional[List[int]] = None  # optional [stride_h, stride_w] for overlapping ViT/EVA patches
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Plan 1: second-order pooling (mean + variance)
    pool_mode: str = "mean"  # "mean" | "mean_var"
    # Plan 2: multi-layer intermediate aggregation
    multi_layer_enabled: bool = False
    multi_layer_hook_blocks: List[int] = field(default_factory=lambda: [5, 11, 17, 23])
    multi_layer_init_weights: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    # Plan 3: horizontal stripe pooling
    stripe_pooling_enabled: bool = False
    stripe_pooling_num_stripes: int = 4
    # Plan 4: class-conditional visual prompt tuning
    prompts_enabled: bool = False
    prompts_n: int = 16
    prompts_num_classes: int = 4
    # S4: camera-aware side information embedding
    sie_cameras: int = 0  # 0 = disabled; set to num training cameras to enable
    # Manual mapping from raw camera ID -> embedding index.
    # Example: {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 8: 6, 4: 6}
    # This lets test cameras share embeddings with training cameras.
    camid_map: Optional[Dict] = None
    # Depth-Anything-V2: depth-aware feature branch (frozen, combined with learned weight)
    depth_enabled: bool = False
    depth_model_variant: str = "vitl"  # vits | vitb | vitl
    depth_init_weight: float = 0.1
    depth_checkpoint: str = ""


@dataclass
class HeadConfig:
    type: str = "bnneck"  # bnneck | linear | arcface
    embed_dim: Optional[int] = None  # auto-set from backbone
    neck_feat: str = "after"  # before | after BN for inference
    s: float = 64.0  # ArcFace feature scale
    m: float = 0.5   # ArcFace angular margin
    dropout: float = 0.0  # dropout rate before head (0 = disabled)


@dataclass
class ObjectiveConfig:
    name: str = MISSING
    type: str = MISSING
    weight: float = 1.0
    # Loss-specific optional params
    label_smooth: float = 0.0
    margin: float = 0.3
    hard_mining: bool = True
    temperature: float = 0.07
    s: float = 64.0  # ArcFace scale
    m: float = 0.5   # ArcFace margin
    sub_centers: Optional[int] = None  # SubCenterArcFace centers per class (None = use train camera count)
    # 7C: per-class triplet weights (indexed by class label 0-3; empty = uniform)
    class_triplet_weights: List[float] = field(default_factory=list)
    # RankedListLoss params
    Tn: float = 1.0        # temperature for negative pair weighting
    Tp: float = 0.0        # temperature for positive pair weighting
    alpha: Optional[float] = None  # upper-bound for hard negatives (None = all negatives contribute)
    imbalance: float = 0.5 # tradeoff between pos and neg set contributions


@dataclass
class SolverConfig:
    optimizer: str = "adamw"  # adam | adamw | sgd
    lr: float = 3.5e-4
    weight_decay: float = 0.01
    momentum: float = 0.9
    max_epochs: int = 120
    warmup_epochs: int = 10
    warmup_lr: float = 1.0e-6
    scheduler: str = "cosine"  # cosine | step | plateau
    step_milestones: List[int] = field(default_factory=lambda: [40, 70])
    gamma: float = 0.1
    amp: bool = True
    grad_accum_steps: int = 1
    clip_grad_norm: Optional[float] = None
    min_lr: float = 1.0e-6
    # EMA (Exponential Moving Average)
    ema: bool = False
    ema_decay: float = 0.999
    # Layer-wise LR decay (LLRD)
    llrd: float = 1.0            # decay factor per layer (1.0 = uniform, <1.0 = deeper layers get lower LR)


@dataclass
class TrainerConfig:
    eval_period: int = 10
    checkpoint_period: int = 10
    log_period: int = 50
    output_dir: str = "./outputs"
    seed: int = 42
    resume: Optional[str] = None  # path to checkpoint to resume from
    eval_train_map: bool = False  # compute train mAP as overfitting indicator
    # S2: primary metric for best-model checkpoint selection
    checkpoint_metric: str = "mAP"  # "mAP" | "test_weighted_map"


@dataclass
class TestConfig:
    batch_size: int = 256
    weight: str = ""
    feat_norm: bool = True
    flip_test: bool = True
    rerank: bool = True
    rerank_k1: int = 20
    rerank_k2: int = 6
    rerank_lambda: float = 0.3
    class_mask: bool = False
    # Multi-scale test-time augmentation
    scales: Optional[List] = None  # e.g. [[224,224], [256,256], [288,288]]; None = single scale
    # Query Expansion (LQE)
    lqe_k: int = 0                 # 0 = disabled; top-k gallery neighbors to expand query
    lqe_alpha: float = 3.0         # expansion weight
    # Semantic classifier confidence bonus
    semantic_classifier: str = ""  # path to SemanticClassifier checkpoint; empty = disabled
    semantic_backbone: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    semantic_img_size: int = 224
    semantic_alpha: float = 0.3
    # Query majority voting
    query_majority_vote_window: int = 0  # 0 = disabled


@dataclass
class SubmissionConfig:
    top_k: int = 100
    output_path: str = "./submission.csv"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    input: InputConfig = field(default_factory=InputConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    objectives: List[ObjectiveConfig] = field(default_factory=lambda: [
        ObjectiveConfig(name="id_loss", type="cross_entropy", weight=1.0, label_smooth=0.1),
        ObjectiveConfig(name="triplet_loss", type="triplet", weight=1.0, margin=0.3),
    ])
    solver: SolverConfig = field(default_factory=SolverConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    test: TestConfig = field(default_factory=TestConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)


# ---------------------------------------------------------------------------
# Config loading / saving helpers
# ---------------------------------------------------------------------------

def get_default_cfg() -> DictConfig:
    """Return a fresh default config as OmegaConf DictConfig."""
    schema = OmegaConf.structured(Config)
    return schema


def load_config(path: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load a YAML config, merge with defaults, and apply CLI overrides."""
    defaults = get_default_cfg()
    if path and os.path.isfile(path):
        user_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(defaults, user_cfg)
    else:
        cfg = defaults

    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli)

    return cfg


def save_config(cfg: DictConfig, output_dir: str) -> str:
    """Save the resolved config to output_dir for reproducibility.
    Returns the path to the saved config file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"config_{timestamp}.yaml")
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    # Also save as config.yaml (latest)
    latest = os.path.join(output_dir, "config.yaml")
    with open(latest, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    return path
