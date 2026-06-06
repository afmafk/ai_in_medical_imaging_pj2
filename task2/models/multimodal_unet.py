from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2
        self.conv = ConvBlock(up_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionFusion(nn.Module):
    def __init__(self, num_modalities: int, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.num_modalities = num_modalities

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, modalities, channels, _, _ = x.shape
        pooled = x.mean(dim=(-1, -2))
        weights = self.score(pooled.view(batch_size * modalities, channels))
        weights = weights.view(batch_size, modalities, 1, 1, 1)
        weights = torch.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)


class ModalityFusion(nn.Module):
    def __init__(self, num_modalities: int, channels: int, fusion_mode: str) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode.lower()
        self.num_modalities = num_modalities
        if self.fusion_mode == "concat":
            self.project = nn.Conv2d(num_modalities * channels, channels, kernel_size=1, bias=False)
        elif self.fusion_mode == "attention":
            self.attention = AttentionFusion(num_modalities, channels)
        elif self.fusion_mode not in {"sum", "mean", "max"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "concat":
            batch_size, modalities, channels, height, width = x.shape
            x = x.view(batch_size, modalities * channels, height, width)
            return self.project(x)
        if self.fusion_mode == "sum":
            return x.sum(dim=1)
        if self.fusion_mode == "mean":
            return x.mean(dim=1)
        if self.fusion_mode == "max":
            return x.max(dim=1).values
        return self.attention(x)


class ModalityStem(nn.Module):
    def __init__(self, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ConvBlock(1, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiModalUNet(nn.Module):
    """2D UNet with explicit modality stems for BraTS-style multimodal fusion."""

    def __init__(
        self,
        in_modalities: int = 4,
        num_classes: int = 4,
        base_channels: int = 32,
        encoder_channels: Optional[Sequence[int]] = None,
        bottleneck_channels: Optional[int] = None,
        dropout: float = 0.1,
        bilinear: bool = False,
        fusion_mode: str = "concat",
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        else:
            encoder_channels = list(encoder_channels)
        if bottleneck_channels is None:
            bottleneck_channels = encoder_channels[-1] * 2
        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must contain at least two stages.")

        self.in_modalities = in_modalities
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        first_channels = encoder_channels[0]
        self.modality_stems = nn.ModuleList(
            [ModalityStem(first_channels, dropout=dropout) for _ in range(in_modalities)]
        )
        self.fusion = ModalityFusion(in_modalities, first_channels, fusion_mode)

        self.encoder_blocks = nn.ModuleList()
        prev_channels = first_channels
        for out_channels in encoder_channels[1:]:
            self.encoder_blocks.append(DownBlock(prev_channels, out_channels, dropout=dropout))
            prev_channels = out_channels

        self.bottleneck = DownBlock(prev_channels, bottleneck_channels, dropout=dropout)

        decoder_specs = list(reversed(encoder_channels))
        self.decoder_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in decoder_specs:
            out_channels = skip_channels
            self.decoder_blocks.append(
                UpBlock(current_channels, skip_channels, out_channels, bilinear=bilinear)
            )
            current_channels = out_channels

        self.seg_head = nn.Conv2d(current_channels, num_classes, kernel_size=1)

        if deep_supervision:
            aux_channels = decoder_specs[:-1]
            self.aux_heads = nn.ModuleList(
                [nn.Conv2d(channels, num_classes, kernel_size=1) for channels in aux_channels]
            )
        else:
            self.aux_heads = None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, M, H, W], got {tuple(x.shape)}")
        if x.size(1) != self.in_modalities:
            raise ValueError(
                f"Expected {self.in_modalities} modalities, but received tensor with {x.size(1)} channels."
            )

        modality_features = []
        for modality_idx, stem in enumerate(self.modality_stems):
            modality_features.append(stem(x[:, modality_idx : modality_idx + 1]))
        fused = self.fusion(torch.stack(modality_features, dim=1))

        skips = [fused]
        current = fused
        for encoder in self.encoder_blocks:
            current = encoder(current)
            skips.append(current)

        current = self.bottleneck(current)

        aux_outputs = []
        for decoder_idx, decoder in enumerate(self.decoder_blocks):
            skip = skips[-(decoder_idx + 1)]
            current = decoder(current, skip)
            if self.deep_supervision and decoder_idx < len(self.decoder_blocks) - 1:
                aux_outputs.append(self.aux_heads[decoder_idx](current))

        logits = self.seg_head(current)
        if self.deep_supervision:
            return logits, aux_outputs
        return logits
