"""TransBTS-lite: 3D CNN encoder + Transformer bottleneck + CNN decoder.

Reference: Wang et al., TransBTS, MICCAI 2021. https://arxiv.org/abs/2103.04430
Simplified for single-GPU course training.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class TransformerBottleneck(nn.Module):
    def __init__(self, dim: int, depth: int = 2, heads: int = 8) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


class TransBTS(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base: int = 32,
        transformer_depth: int = 2,
    ) -> None:
        super().__init__()
        self.enc1 = ResConv3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResConv3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResConv3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ResConv3D(base * 4, base * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.transformer = TransformerBottleneck(base * 8, depth=transformer_depth)

        self.up4 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec4 = ResConv3D(base * 12, base * 4)
        self.up3 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec3 = ResConv3D(base * 6, base * 2)
        self.up2 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec2 = ResConv3D(base * 3, base)
        self.up1 = nn.ConvTranspose3d(base, base, 2, stride=2)
        self.dec1 = ResConv3D(base * 2, base)

        self.out = nn.Conv3d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.transformer(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)
