"""MONAI SwinUNETR with V2 transformer blocks (aligned with standalone_swinunetr_v2).

Uses the same 4-class BraTS labels and processed_2d -> 3D volume pipeline as Attention U-Net / TransBTS.
Reference: Hatamizadeh et al., SwinUNETR-V2, MICCAI 2023.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from monai.networks.nets import SwinUNETR


def ensure_patch_size_divisible_by_32(patch_size: tuple[int, int, int]) -> None:
    for dim in patch_size:
        if dim % 32 != 0:
            raise ValueError(
                f"SwinUNETR patch size {patch_size} invalid: each dimension must be divisible by 32."
            )


def build_swinunetr(
    in_channels: int = 4,
    num_classes: int = 4,
    model_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    """Build MONAI SwinUNETR; default ``use_v2=True`` matches standalone_swinunetr_v2."""
    cfg = model_cfg or {}
    feature_size = int(cfg.get("feature_size", 12))
    use_v2 = bool(cfg.get("use_v2", True))
    use_checkpoint = bool(cfg.get("use_checkpoint", True))

    kwargs: dict[str, Any] = dict(
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
        use_v2=use_v2,
    )

    if "depths" in cfg:
        depths = cfg["depths"]
        kwargs["depths"] = tuple(depths) if isinstance(depths, list) else depths
    if "num_heads" in cfg:
        num_heads = cfg["num_heads"]
        kwargs["num_heads"] = tuple(num_heads) if isinstance(num_heads, list) else num_heads
    if "window_size" in cfg:
        kwargs["window_size"] = int(cfg["window_size"])

    return SwinUNETR(**kwargs)
