"""Tri-attention fusion adaptation for the TransBTS-lite backbone.

The fusion layout follows Zhou et al. (Pattern Recognition 2022):
independent modality encoders, dual modality/spatial attention, and a deepest
correlation-attention constraint. The decoder and transformer remain aligned
with the local TransBTS baseline so the experiment isolates the fusion change.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transbts import ResConv3D, TransformerBottleneck


class ModalityEncoder3D(nn.Module):
    """One independent encoder path for one MRI modality."""

    def __init__(self, base: int) -> None:
        super().__init__()
        self.enc1 = ResConv3D(1, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResConv3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResConv3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ResConv3D(base * 4, base * 8)
        self.pool4 = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        return e1, e2, e3, e4, self.pool4(e4)


class DualAttentionFusion3D(nn.Module):
    """Fuse modality-specific features along modality and spatial paths.

    Softmax normalization keeps both paths as convex modality combinations.
    This avoids the constant feature amplification observed in the earlier
    sigmoid-gated RSF experiment while preserving spatially varying weights.
    """

    def __init__(self, channels: int, num_modalities: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_modalities = num_modalities
        hidden = max(num_modalities * 2, 8)
        self.modality_gate = nn.Sequential(
            nn.Linear(num_modalities * 2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, num_modalities),
        )
        self.spatial_gate = nn.Conv3d(num_modalities * channels, num_modalities, 3, padding=1)
        self.reset_attention_parameters()

    def reset_attention_parameters(self) -> None:
        nn.init.zeros_(self.modality_gate[-1].weight)
        nn.init.zeros_(self.modality_gate[-1].bias)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != self.num_modalities:
            raise ValueError(f"expected {self.num_modalities} modalities, got {len(features)}")
        stacked = torch.stack(list(features), dim=1)  # (B, M, C, D, H, W)
        modality_stats = torch.cat(
            [
                stacked.abs().mean(dim=(2, 3, 4, 5)),
                stacked.abs().amax(dim=(2, 3, 4, 5)),
            ],
            dim=1,
        )
        modality_weights = torch.softmax(self.modality_gate(modality_stats), dim=1)
        modality_repr = (
            stacked * modality_weights[:, :, None, None, None, None]
        ).sum(dim=1)

        b, m, c, d, h, w = stacked.shape
        spatial_logits = self.spatial_gate(stacked.reshape(b, m * c, d, h, w))
        spatial_weights = torch.softmax(spatial_logits, dim=1)
        spatial_features = stacked * spatial_weights[:, :, None]
        spatial_repr = spatial_features.sum(dim=1)

        fused = 0.5 * (modality_repr + spatial_repr)
        return fused, spatial_features, modality_weights, spatial_weights


class CorrelationDescription3D(nn.Module):
    """Learn a nonlinear source-to-target modality correlation.

    The paper models nonlinear cross-modality relationships with a two-layer
    correlation-description block. Here the predicted target is a calibrated
    quadratic transformation of the source feature map. Zero initialization
    begins at an identity mapping and lets the KL term introduce correlation
    pressure gradually.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, channels * 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_context = source.mean(dim=(2, 3, 4))
        # The target modality is a teacher signal. Detaching its context avoids
        # a shortcut where the encoder deforms both sides instead of learning
        # the source-to-target correlation in this adapter.
        target_context = target.detach().mean(dim=(2, 3, 4))
        linear, quadratic, bias = self.mlp(
            torch.cat([source_context, target_context], dim=1)
        ).chunk(3, dim=1)
        shape = (source.shape[0], source.shape[1], 1, 1, 1)
        linear = 1.0 + 0.1 * torch.tanh(linear).view(shape)
        quadratic = 0.1 * torch.tanh(quadratic).view(shape)
        bias = 0.1 * torch.tanh(bias).view(shape)
        return linear * source + quadratic * source.square() + bias


def _symmetric_kl(
    predicted: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric KL against a stop-gradient target feature distribution."""
    predicted = predicted.float().flatten(1)
    target = target.detach().float().flatten(1)
    predicted = (predicted - predicted.mean(dim=1, keepdim=True)) / (
        predicted.std(dim=1, keepdim=True) + 1e-6
    )
    target = (target - target.mean(dim=1, keepdim=True)) / (
        target.std(dim=1, keepdim=True) + 1e-6
    )
    pred_log = F.log_softmax(predicted / temperature, dim=1)
    tgt_log = F.log_softmax(target / temperature, dim=1)
    pred_prob = pred_log.exp()
    tgt_prob = tgt_log.exp()
    return 0.5 * (
        F.kl_div(pred_log, tgt_prob, reduction="batchmean")
        + F.kl_div(tgt_log, pred_prob, reduction="batchmean")
    )


class TriAttentionFusion3D(nn.Module):
    """Deepest dual-attention fusion plus correlation-attention supervision."""

    DEFAULT_PAIRS = ((0, 1), (0, 2), (2, 3))  # T1-T1c, T1-T2, T2-FLAIR

    def __init__(
        self,
        channels: int,
        num_modalities: int = 4,
        correlation_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.dual_attention = DualAttentionFusion3D(channels, num_modalities)
        self.correlation_temperature = float(correlation_temperature)
        self.pairs = self.DEFAULT_PAIRS
        self.adapters = nn.ModuleDict()
        for left, right in self.pairs:
            self.adapters[f"{left}_to_{right}"] = CorrelationDescription3D(channels)
            self.adapters[f"{right}_to_{left}"] = CorrelationDescription3D(channels)

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, spatial_features, modality_weights, spatial_weights = self.dual_attention(features)
        losses: list[torch.Tensor] = []
        for left, right in self.pairs:
            left_feature = spatial_features[:, left]
            right_feature = spatial_features[:, right]
            left_to_right = self.adapters[f"{left}_to_{right}"](left_feature, right_feature)
            right_to_left = self.adapters[f"{right}_to_{left}"](right_feature, left_feature)
            losses.append(
                _symmetric_kl(left_to_right, right_feature, self.correlation_temperature)
            )
            losses.append(
                _symmetric_kl(right_to_left, left_feature, self.correlation_temperature)
            )
        correlation_loss = torch.stack(losses).mean()
        return fused, correlation_loss, modality_weights, spatial_weights


class TAFTransBTS(nn.Module):
    """TransBTS with modality-independent encoders and deepest tri-attention."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base: int = 24,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        correlation_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if in_channels != 4:
            raise ValueError("TAFTransBTS currently expects T1, T1c, T2, and FLAIR inputs")
        if (base * 8) % transformer_heads != 0:
            raise ValueError("base * 8 must be divisible by transformer_heads")
        self.in_channels = in_channels
        self.encoders = nn.ModuleList([ModalityEncoder3D(base) for _ in range(in_channels)])
        self.skip_fusions = nn.ModuleList(
            [
                DualAttentionFusion3D(base, in_channels),
                DualAttentionFusion3D(base * 2, in_channels),
                DualAttentionFusion3D(base * 4, in_channels),
                DualAttentionFusion3D(base * 8, in_channels),
            ]
        )
        self.tri_attention = TriAttentionFusion3D(
            base * 8,
            in_channels,
            correlation_temperature=correlation_temperature,
        )
        self.transformer = TransformerBottleneck(
            base * 8,
            depth=transformer_depth,
            heads=transformer_heads,
        )

        self.up4 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec4 = ResConv3D(base * 12, base * 4)
        self.up3 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec3 = ResConv3D(base * 6, base * 2)
        self.up2 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec2 = ResConv3D(base * 3, base)
        self.up1 = nn.ConvTranspose3d(base, base, 2, stride=2)
        self.dec1 = ResConv3D(base * 2, base)
        self.out = nn.Conv3d(base, num_classes, 1)

    def forward_with_aux(self, x: torch.Tensor) -> dict[str, object]:
        modality_levels = [
            encoder(x[:, idx : idx + 1]) for idx, encoder in enumerate(self.encoders)
        ]
        skip_features = [
            self.skip_fusions[level]([features[level] for features in modality_levels])[0]
            for level in range(4)
        ]
        bottom, correlation_loss, modality_weights, spatial_weights = self.tri_attention(
            [features[4] for features in modality_levels]
        )
        bottleneck = self.transformer(bottom)
        e1, e2, e3, e4 = skip_features
        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return {
            "logits": self.out(d1),
            "correlation_loss": correlation_loss,
            "taf_modality_weights": modality_weights,
            "taf_spatial_weights": spatial_weights,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward_with_aux(x)
        logits = outputs["logits"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("TAFTransBTS logits must be a tensor")
        return logits
