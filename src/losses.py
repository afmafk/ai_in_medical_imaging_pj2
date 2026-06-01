from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean Dice over foreground classes 1..C-1."""
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 4, 1, 2, 3).float()
    dices = []
    for c in range(1, num_classes):
        pred_c = probs[:, c]
        tgt_c = target_oh[:, c]
        inter = (pred_c * tgt_c).sum()
        denom = pred_c.sum() + tgt_c.sum()
        dices.append((2 * inter + eps) / (denom + eps))
    return torch.stack(dices).mean()


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        ce = self.ce(logits, target)
        dice_loss = 1.0 - dice_per_class(logits, target, self.num_classes)
        loss = self.ce_weight * ce + self.dice_weight * dice_loss
        return loss, {"ce": ce.item(), "dice_loss": dice_loss.item()}


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft Dice loss over a batch of binary region logits."""
    probs = torch.sigmoid(logits)
    target = target.float()
    reduce_dims = tuple(range(1, logits.ndim))
    inter = (probs * target).sum(dim=reduce_dims)
    denom = probs.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class RegionSupervisedDiceCELoss(nn.Module):
    """Four-class segmentation loss plus WT/TC/ET auxiliary supervision."""

    REGION_NAMES = ("WT", "TC", "ET")

    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        aux_weight: float = 0.3,
        nested_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.main_loss = DiceCELoss(num_classes, ce_weight, dice_weight)
        self.aux_weight = aux_weight
        self.nested_weight = nested_weight

    @staticmethod
    def region_targets(target: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "WT": ((target == 1) | (target == 2) | (target == 3)).unsqueeze(1).float(),
            "TC": ((target == 1) | (target == 3)).unsqueeze(1).float(),
            "ET": (target == 3).unsqueeze(1).float(),
        }

    def forward(
        self,
        outputs: Mapping[str, object],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"]
        aux_logits = outputs["aux_logits"]
        if not isinstance(logits, torch.Tensor) or not isinstance(aux_logits, Mapping):
            raise TypeError("RSF outputs must contain tensor logits and mapping aux_logits")

        main, details = self.main_loss(logits, target)
        targets = self.region_targets(target)
        aux_losses: dict[str, torch.Tensor] = {}
        aux_probs: dict[str, torch.Tensor] = {}
        for region in self.REGION_NAMES:
            region_logits = aux_logits[region]
            if not isinstance(region_logits, torch.Tensor):
                raise TypeError(f"missing tensor auxiliary logits for {region}")
            aux_losses[region] = binary_dice_loss(region_logits, targets[region]) + F.binary_cross_entropy_with_logits(
                region_logits,
                targets[region],
            )
            aux_probs[region] = torch.sigmoid(region_logits)

        aux = torch.stack([aux_losses[region] for region in self.REGION_NAMES]).mean()
        nested = F.relu(aux_probs["ET"] - aux_probs["TC"]).mean() + F.relu(
            aux_probs["TC"] - aux_probs["WT"]
        ).mean()
        loss = main + self.aux_weight * aux + self.nested_weight * nested
        return loss, {
            **details,
            "aux_loss": aux.item(),
            "nested_loss": nested.item(),
        }


class TAFDiceCELoss(nn.Module):
    """Four-class segmentation loss plus deepest cross-modality KL regularization."""

    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        correlation_weight: float = 0.05,
        correlation_start_epoch: int = 20,
        correlation_warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.main_loss = DiceCELoss(num_classes, ce_weight, dice_weight)
        self.correlation_weight = correlation_weight
        self.correlation_start_epoch = max(0, int(correlation_start_epoch))
        self.correlation_warmup_epochs = max(0, int(correlation_warmup_epochs))
        self.current_correlation_weight = 0.0

    def set_epoch(self, epoch: int) -> None:
        active_epoch = int(epoch) - self.correlation_start_epoch
        if active_epoch <= 0:
            self.current_correlation_weight = 0.0
        elif self.correlation_warmup_epochs:
            progress = min(active_epoch / self.correlation_warmup_epochs, 1.0)
            self.current_correlation_weight = self.correlation_weight * progress
        else:
            self.current_correlation_weight = self.correlation_weight

    def forward(
        self,
        outputs: Mapping[str, object],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"]
        correlation_loss = outputs["correlation_loss"]
        if not isinstance(logits, torch.Tensor) or not isinstance(correlation_loss, torch.Tensor):
            raise TypeError("TAF outputs must contain tensor logits and correlation_loss")
        main, details = self.main_loss(logits, target)
        weighted_correlation = self.current_correlation_weight * correlation_loss
        loss = main + weighted_correlation
        return loss, {
            **details,
            "correlation_loss": correlation_loss.item(),
            "weighted_correlation_loss": weighted_correlation.item(),
        }
