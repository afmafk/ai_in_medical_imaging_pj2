from __future__ import annotations

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
