from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import LossConfig
except ImportError:
    from config import LossConfig


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    if target.ndim == 4 and target.size(1) == num_classes:
        return target.float()
    if target.ndim == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    if target.ndim != 3:
        raise ValueError(f"Expected target shape [B, H, W] or [B, 1, H, W], got {tuple(target.shape)}")
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def _ensure_channel_first_binary_target(target: torch.Tensor, num_channels: int) -> torch.Tensor:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4:
        raise ValueError(f"Expected target shape [B, C, H, W] or [B, H, W], got {tuple(target.shape)}")
    if target.size(1) != num_channels:
        raise ValueError(
            f"Expected target with {num_channels} channels, but received shape {tuple(target.shape)}"
        )
    return target.float()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, include_background: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_one_hot = _one_hot(target, num_classes=logits.size(1)).to(device=logits.device, dtype=logits.dtype)

        if not self.include_background and logits.size(1) > 1:
            probs = probs[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        dims: Tuple[int, ...] = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        denominator = torch.sum(probs + target_one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = True,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(smooth=smooth, include_background=include_background)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        if target.ndim == 4 and target.size(1) == logits.size(1):
            target = target.argmax(dim=1)
        dice_term = self.dice(logits, target)
        ce_term = self.ce(logits, target.long())
        return self.dice_weight * dice_term + self.ce_weight * ce_term


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = _ensure_channel_first_binary_target(target, num_channels=logits.size(1)).to(
            device=logits.device,
            dtype=logits.dtype,
        )

        dims: Tuple[int, ...] = (0, 2, 3)
        intersection = torch.sum(probs * target, dim=dims)
        denominator = torch.sum(probs + target, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BinaryDiceBCELoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = BinaryDiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _ensure_channel_first_binary_target(target, num_channels=logits.size(1))
        dice_term = self.dice(logits, target)
        bce_term = self.bce(logits, target.to(device=logits.device, dtype=logits.dtype))
        return self.dice_weight * dice_term + self.bce_weight * bce_term


def build_loss(config: LossConfig) -> nn.Module:
    name = config.name.lower()
    if name == "dice":
        return DiceLoss(
            smooth=config.dice_smooth,
            include_background=config.dice_include_background,
        )
    if name == "binary_dice":
        return BinaryDiceLoss(smooth=config.dice_smooth)
    if name in {"dice_ce", "dice+ce", "hybrid"}:
        return DiceCrossEntropyLoss(
            dice_weight=config.dice_weight,
            ce_weight=config.ce_weight,
            smooth=config.dice_smooth,
            include_background=config.dice_include_background,
        )
    if name in {"binary_dice_bce", "dice_bce", "bce_dice"}:
        return BinaryDiceBCELoss(
            dice_weight=config.dice_weight,
            bce_weight=config.bce_weight,
            smooth=config.dice_smooth,
        )
    raise ValueError(f"Unsupported loss name: {config.name}")
