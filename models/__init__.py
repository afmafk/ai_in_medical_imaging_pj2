import torch.nn as nn
from typing import Any

from .attention_unet3d import AttentionUNet3D
from .swinunetr_v2 import build_swinunetr
from .transbts import TransBTS

__all__ = ["AttentionUNet3D", "TransBTS", "build_model"]

MODEL_NAMES = ("attention_unet", "transbts", "swinunetr")


def build_model(
    name: str,
    in_channels: int = 4,
    num_classes: int = 4,
    model_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    name = name.lower()
    if name in ("attention_unet", "attention_unet3d", "attunet"):
        return AttentionUNet3D(in_channels, num_classes)
    if name in ("transbts", "trans_bts"):
        return TransBTS(in_channels, num_classes)
    if name in ("swinunetr", "swinunetr_v2", "swin_unetr"):
        return build_swinunetr(in_channels, num_classes, model_cfg)
    raise ValueError(f"unknown model: {name}. Choose from {MODEL_NAMES}")
