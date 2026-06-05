"""3D Attention U-Net with attention gates on skip connections."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate3D(nn.Module):
    """Attention gate (MIDL 2018) adapted to 3D."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int | None = None) -> None:
        super().__init__()
        inter_ch = inter_ch or max(skip_ch // 2, 1)
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, 1, bias=False),
            nn.InstanceNorm3d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, 1, bias=False),
            nn.InstanceNorm3d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=False)
        a = self.psi(torch.relu(g1 + x1))
        return x * a


class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(base * 4, base * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(base * 8, base * 16)

        self.up4 = nn.ConvTranspose3d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate3D(base * 8, base * 8)
        self.dec4 = ConvBlock3D(base * 16, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate3D(base * 4, base * 4)
        self.dec3 = ConvBlock3D(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate3D(base * 2, base * 2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate3D(base, base)
        self.dec1 = ConvBlock3D(base * 2, base)

        self.out = nn.Conv3d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        e4a = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4a], dim=1))

        d3 = self.up3(d4)
        e3a = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3a], dim=1))

        d2 = self.up2(d3)
        e2a = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2a], dim=1))

        d1 = self.up1(d2)
        e1a = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1a], dim=1))

        return self.out(d1)
