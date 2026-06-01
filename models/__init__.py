import torch.nn as nn
from typing import Any

from .attention_unet3d import AttentionUNet3D
from .taf_transbts import TAFTransBTS
from .transbts import MSATransBTS, RAMSTransBTS, RAMTransBTS, RSFTransBTS, TransBTS

__all__ = [
    "AttentionUNet3D",
    "TransBTS",
    "MSATransBTS",
    "RAMTransBTS",
    "RAMSTransBTS",
    "RSFTransBTS",
    "TAFTransBTS",
    "build_model",
]

MODEL_NAMES = (
    "attention_unet",
    "transbts",
    "msa_transbts",
    "ram_transbts",
    "rams_transbts",
    "rsf_transbts",
    "taf_transbts",
    "swinunetr",
)


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
    if name in ("msa_transbts", "msa-transbts", "msatransbts"):
        return MSATransBTS(in_channels, num_classes)
    if name in ("ram_transbts", "ram-transbts", "ramtransbts"):
        return RAMTransBTS(in_channels, num_classes)
    if name in ("rams_transbts", "rams-transbts", "ramstransbts"):
        return RAMSTransBTS(in_channels, num_classes)
    if name in ("rsf_transbts", "rsf-transbts", "rsftransbts"):
        return RSFTransBTS(in_channels, num_classes)
    if name in ("taf_transbts", "taf-transbts", "taftransbts"):
        return TAFTransBTS(in_channels, num_classes, **(model_cfg or {}))
    if name in ("swinunetr", "swinunetr_v2", "swin_unetr"):
        from .swinunetr_v2 import build_swinunetr

        return build_swinunetr(in_channels, num_classes, model_cfg)
    raise ValueError(f"unknown model: {name}. Choose from {MODEL_NAMES}")
