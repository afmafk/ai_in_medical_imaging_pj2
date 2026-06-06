from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from .config import ModelConfig, OptimizerConfig, SchedulerConfig
    from .models import MultiModalUNet, SimpleUNet
except ImportError:
    from config import ModelConfig, OptimizerConfig, SchedulerConfig
    from models import MultiModalUNet, SimpleUNet


def build_model(config: ModelConfig) -> nn.Module:
    model_name = config.name.lower()
    if model_name == "simple_unet":
        return SimpleUNet(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
            encoder_channels=config.encoder_channels,
            bottleneck_channels=config.bottleneck_channels,
            dropout=config.dropout,
            bilinear=config.bilinear,
            deep_supervision=config.deep_supervision,
        )
    if model_name == "multimodal_unet":
        return MultiModalUNet(
            in_modalities=config.in_modalities,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
            encoder_channels=config.encoder_channels,
            bottleneck_channels=config.bottleneck_channels,
            dropout=config.dropout,
            bilinear=config.bilinear,
            fusion_mode=config.fusion_mode,
            deep_supervision=config.deep_supervision,
        )
    raise ValueError(f"Unsupported model name: {config.name}")


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    name = config.name.lower()
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Optional[SchedulerConfig],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if config is None or config.name is None or str(config.name).lower() == "none":
        return None

    name = config.name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.t_max,
            eta_min=config.eta_min,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            min_lr=config.min_lr,
        )
    if name == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1.0 - min(epoch, config.t_max) / max(config.t_max, 1)) ** config.power,
        )
    raise ValueError(f"Unsupported scheduler: {config.name}")
