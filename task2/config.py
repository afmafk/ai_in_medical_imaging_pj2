from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class ModelConfig:
    name: str = "simple_unet"
    input_channels: int = 3
    in_modalities: int = 4
    num_classes: int = 4
    base_channels: int = 32
    encoder_channels: Optional[Sequence[int]] = None
    bottleneck_channels: Optional[int] = None
    dropout: float = 0.1
    bilinear: bool = False
    fusion_mode: str = "concat"
    deep_supervision: bool = False


@dataclass
class LossConfig:
    name: str = "dice_ce"
    dice_smooth: float = 1e-5
    dice_include_background: bool = True
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    bce_weight: float = 1.0


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    name: Optional[str] = "plateau"
    mode: str = "min"
    t_max: int = 100
    eta_min: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1
    power: float = 0.9
    factor: float = 0.5
    patience: int = 10
    threshold: float = 1e-4
    min_lr: float = 1e-6


@dataclass
class AugmentationConfig:
    enabled: bool = True
    pad_pixels: int = 24
    rotation_degrees: float = 15.0
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    scale_range: Tuple[float, float] = (0.9, 1.1)
    gaussian_noise_prob: float = 0.3
    gaussian_noise_std: float = 0.05


@dataclass
class TrainConfig:
    device: str = "cuda"
    amp: bool = True
    max_epochs: int = 100
    grad_clip_norm: Optional[float] = 1.0
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    log_every: int = 10
    save_dir: str = "checkpoints"
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
