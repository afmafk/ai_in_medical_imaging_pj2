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


class ModalitySpatialAttention3D(nn.Module):
    """Lightweight modality + spatial attention for 4-modal MRI input."""

    def __init__(
        self,
        in_channels: int = 4,
        modality_hidden: int | None = None,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        hidden = modality_hidden or max(in_channels * 2, 8)
        self.modality_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv3d(2, 1, spatial_kernel, padding=padding)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        last = self.modality_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        nn.init.zeros_(self.spatial_conv.weight)
        nn.init.zeros_(self.spatial_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_abs = x.abs()
        mod_mean = x_abs.mean(dim=(2, 3, 4))
        mod_max = x_abs.amax(dim=(2, 3, 4))
        mod_logits = self.modality_mlp(torch.cat([mod_mean, mod_max], dim=1))
        mod_scale = (2.0 * torch.sigmoid(mod_logits)).view(x.shape[0], x.shape[1], 1, 1, 1)
        x = x * mod_scale

        x_abs = x.abs()
        spatial_mean = x_abs.mean(dim=1, keepdim=True)
        spatial_max = x_abs.amax(dim=1, keepdim=True)
        spatial_logits = self.spatial_conv(torch.cat([spatial_mean, spatial_max], dim=1))
        spatial_scale = 2.0 * torch.sigmoid(spatial_logits)
        return x * spatial_scale


class SharedSpatialAttention3D(nn.Module):
    """Shared 3D spatial attention over all input feature channels."""

    def __init__(self, spatial_kernel: int = 7) -> None:
        super().__init__()
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv3d(2, 1, spatial_kernel, padding=padding)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.spatial_conv.weight)
        nn.init.zeros_(self.spatial_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_abs = x.abs()
        spatial_mean = x_abs.mean(dim=1, keepdim=True)
        spatial_max = x_abs.amax(dim=1, keepdim=True)
        spatial_logits = self.spatial_conv(torch.cat([spatial_mean, spatial_max], dim=1))
        spatial_scale = 2.0 * torch.sigmoid(spatial_logits)
        return x * spatial_scale


class ModalityFeatureStem3D(nn.Module):
    """Extract shallow modality-specific features before cross-modal fusion."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegionSupervisedFeatureFusion3D(nn.Module):
    """Spatially varying WT/TC/ET modality fusion with auxiliary region heads."""

    REGION_NAMES = ("WT", "TC", "ET")

    def __init__(
        self,
        in_channels: int = 4,
        stem_channels: int = 4,
        out_channels: int = 32,
        num_regions: int = 3,
    ) -> None:
        super().__init__()
        if num_regions != len(self.REGION_NAMES):
            raise ValueError(f"expected {len(self.REGION_NAMES)} regions, got {num_regions}")
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_regions = num_regions
        self.stems = nn.ModuleList(
            [ModalityFeatureStem3D(stem_channels) for _ in range(in_channels)]
        )

        stacked_channels = in_channels * stem_channels
        self.modality_attention = nn.Sequential(
            nn.Conv3d(stacked_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, num_regions * in_channels, 1),
        )
        self.spatial_attention = nn.ModuleList(
            [nn.Conv3d(2, 1, 7, padding=3) for _ in range(num_regions)]
        )
        self.aux_heads = nn.ModuleList(
            [nn.Conv3d(stem_channels, 1, 1) for _ in range(num_regions)]
        )
        self.compress = nn.Sequential(
            nn.Conv3d((in_channels + num_regions) * stem_channels, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.reset_attention_parameters()

    def reset_attention_parameters(self) -> None:
        # Equal modality weights and identity spatial scales are a stable starting point.
        final_modality_conv = self.modality_attention[-1]
        nn.init.zeros_(final_modality_conv.weight)
        nn.init.zeros_(final_modality_conv.bias)
        for spatial_conv in self.spatial_attention:
            nn.init.zeros_(spatial_conv.weight)
            nn.init.zeros_(spatial_conv.bias)

    def forward_with_aux(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        modal_features = torch.stack(
            [stem(x[:, idx : idx + 1]) for idx, stem in enumerate(self.stems)],
            dim=1,
        )  # (B, M, C, D, H, W)
        b, m, c, d, h, w = modal_features.shape
        flat_features = modal_features.reshape(b, m * c, d, h, w)

        modality_logits = self.modality_attention(flat_features)
        modality_weights = modality_logits.view(b, self.num_regions, m, d, h, w)
        modality_weights = torch.softmax(modality_weights, dim=2)
        fused = (modality_weights.unsqueeze(3) * modal_features.unsqueeze(1)).sum(dim=2)

        region_features: list[torch.Tensor] = []
        spatial_scales: list[torch.Tensor] = []
        aux_logits: dict[str, torch.Tensor] = {}
        for idx, region_name in enumerate(self.REGION_NAMES):
            region = fused[:, idx]
            spatial_stats = torch.cat(
                [region.abs().mean(dim=1, keepdim=True), region.abs().amax(dim=1, keepdim=True)],
                dim=1,
            )
            spatial_scale = 2.0 * torch.sigmoid(self.spatial_attention[idx](spatial_stats))
            region = region * spatial_scale
            region_features.append(region)
            spatial_scales.append(spatial_scale)
            aux_logits[region_name] = self.aux_heads[idx](region)

        fused_features = torch.cat([flat_features, *region_features], dim=1)
        return (
            self.compress(fused_features),
            aux_logits,
            modality_weights,
            torch.stack(spatial_scales, dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _, _, _ = self.forward_with_aux(x)
        return features


class RegionAwareModalityFusion3D(nn.Module):
    """Create WT/TC/ET-oriented modality fusion channels from 4-modal MRI input."""

    def __init__(
        self,
        in_channels: int = 4,
        num_regions: int = 3,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_regions = num_regions
        hidden = hidden or max(in_channels * num_regions * 2, 16)
        self.weight_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_regions * in_channels),
        )
        self.reset_parameters()

    @property
    def out_channels(self) -> int:
        return self.in_channels + self.num_regions

    def reset_parameters(self) -> None:
        last = self.weight_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_abs = x.abs()
        mod_mean = x_abs.mean(dim=(2, 3, 4))
        mod_max = x_abs.amax(dim=(2, 3, 4))
        logits = self.weight_mlp(torch.cat([mod_mean, mod_max], dim=1))
        weights = logits.view(x.shape[0], self.num_regions, self.in_channels)
        weights = torch.softmax(weights, dim=-1)
        fused = torch.einsum("brc,bcdhw->brdhw", weights, x)
        return torch.cat([x, fused], dim=1)


class TransBTS(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base: int = 32,
        transformer_depth: int = 2,
        use_input_attention: bool = False,
        input_module: nn.Module | None = None,
        encoder_in_channels: int | None = None,
    ) -> None:
        super().__init__()
        if input_module is not None and use_input_attention:
            raise ValueError("Use either input_module or use_input_attention, not both.")
        self.input_attention = input_module or (
            ModalitySpatialAttention3D(in_channels) if use_input_attention else nn.Identity()
        )
        encoder_in_channels = encoder_in_channels or in_channels
        self.enc1 = ResConv3D(encoder_in_channels, base)
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

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_backbone(self.input_attention(x))


class MSATransBTS(TransBTS):
    """TransBTS with modality-spatial attention on the multi-modal input."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base: int = 32,
        transformer_depth: int = 2,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base=base,
            transformer_depth=transformer_depth,
            use_input_attention=True,
        )


class RAMTransBTS(TransBTS):
    """TransBTS with region-aware modality fusion input channels."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base: int = 32,
        transformer_depth: int = 2,
    ) -> None:
        fusion = RegionAwareModalityFusion3D(in_channels=in_channels, num_regions=3)
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base=base,
            transformer_depth=transformer_depth,
            input_module=fusion,
            encoder_in_channels=fusion.out_channels,
        )


class RAMSTransBTS(TransBTS):
    """TransBTS with region-aware modality fusion plus shared spatial attention."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base: int = 32,
        transformer_depth: int = 2,
    ) -> None:
        fusion = RegionAwareModalityFusion3D(in_channels=in_channels, num_regions=3)
        input_module = nn.Sequential(fusion, SharedSpatialAttention3D())
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base=base,
            transformer_depth=transformer_depth,
            input_module=input_module,
            encoder_in_channels=fusion.out_channels,
        )


class RSFTransBTS(TransBTS):
    """TransBTS with region-supervised, spatially varying feature-level fusion."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base: int = 32,
        transformer_depth: int = 2,
        stem_channels: int = 4,
    ) -> None:
        feature_fusion = RegionSupervisedFeatureFusion3D(
            in_channels=in_channels,
            stem_channels=stem_channels,
            out_channels=base,
        )
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base=base,
            transformer_depth=transformer_depth,
            input_module=feature_fusion,
            encoder_in_channels=base,
        )

    def forward_with_aux(self, x: torch.Tensor) -> dict[str, object]:
        if not isinstance(self.input_attention, RegionSupervisedFeatureFusion3D):
            raise TypeError("RSFTransBTS requires RegionSupervisedFeatureFusion3D")
        features, aux_logits, modality_weights, spatial_scales = self.input_attention.forward_with_aux(x)
        return {
            "logits": self.forward_backbone(features),
            "aux_logits": aux_logits,
            "modality_weights": modality_weights,
            "spatial_scales": spatial_scales,
        }
