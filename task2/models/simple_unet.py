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


class SimpleUNet(nn.Module):
    """Plain 2D UNet that consumes concatenated modalities as input channels."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 32,
        encoder_channels: Optional[Sequence[int]] = None,
        bottleneck_channels: Optional[int] = None,
        dropout: float = 0.1,
        bilinear: bool = False,
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

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.stem = ConvBlock(input_channels, encoder_channels[0], dropout=dropout)

        self.encoder_blocks = nn.ModuleList()
        prev_channels = encoder_channels[0]
        for out_channels in encoder_channels[1:]:
            self.encoder_blocks.append(DownBlock(prev_channels, out_channels, dropout=dropout))
            prev_channels = out_channels

        self.bottleneck = DownBlock(prev_channels, bottleneck_channels, dropout=dropout)

        decoder_specs = list(reversed(encoder_channels))
        self.decoder_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in decoder_specs:
            self.decoder_blocks.append(
                UpBlock(current_channels, skip_channels, skip_channels, bilinear=bilinear)
            )
            current_channels = skip_channels

        self.seg_head = nn.Conv2d(current_channels, num_classes, kernel_size=1)

        if deep_supervision:
            aux_channels = decoder_specs[:-1]
            self.aux_heads = nn.ModuleList(
                [nn.Conv2d(channels, num_classes, kernel_size=1) for channels in aux_channels]
            )
        else:
            self.aux_heads = None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, but received tensor with {x.size(1)} channels."
            )

        current = self.stem(x)
        skips = [current]
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
