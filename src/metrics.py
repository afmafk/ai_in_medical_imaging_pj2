from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def seg_to_regions(seg: np.ndarray) -> dict[str, np.ndarray]:
    """Map label map (0,1,2,3) to WT, TC, ET binary masks."""
    return {
        "WT": (seg == 1) | (seg == 2) | (seg == 3),
        "TC": (seg == 1) | (seg == 3),
        "ET": seg == 3,
    }


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return float((2 * inter + eps) / (denom + eps))


def hd95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """95% Hausdorff distance between binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    struct = np.ones((3, 3, 3), dtype=bool)

    def surface(mask: np.ndarray) -> np.ndarray:
        eroded = binary_erosion(mask, structure=struct)
        return mask & ~eroded

    surf_pred = surface(pred)
    surf_tgt = surface(target)
    if not surf_pred.any() or not surf_tgt.any():
        return float("inf")

    dt_pred = distance_transform_edt(~surf_pred)
    dt_tgt = distance_transform_edt(~surf_tgt)
    d1 = dt_tgt[surf_pred]
    d2 = dt_pred[surf_tgt]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


@torch.no_grad()
def compute_region_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Compute Dice for WT, TC, ET from 4-class logits and label map."""
    pred = logits.argmax(dim=1).cpu().numpy()
    tgt = target.cpu().numpy()
    metrics: dict[str, float] = {}
    dices = []
    for i in range(pred.shape[0]):
        pr = seg_to_regions(pred[i])
        gt = seg_to_regions(tgt[i])
        for name in ("WT", "TC", "ET"):
            d = dice_binary(pr[name], gt[name])
            metrics[f"dice_{name}"] = metrics.get(f"dice_{name}", 0.0) + d
            dices.append(d)
    n = pred.shape[0]
    for name in ("WT", "TC", "ET"):
        metrics[f"dice_{name}"] /= n
    metrics["dice_mean"] = float(np.mean([metrics["dice_WT"], metrics["dice_TC"], metrics["dice_ET"]]))
    return metrics


def _metrics_from_seg_arrays(pred: np.ndarray, tgt: np.ndarray) -> tuple[dict[str, float], list[float]]:
    if pred.ndim == 3:
        pred = pred[np.newaxis, ...]
        tgt = tgt[np.newaxis, ...]
    metrics: dict[str, float] = {}
    dices: list[float] = []
    for i in range(pred.shape[0]):
        pr = seg_to_regions(pred[i])
        gt = seg_to_regions(tgt[i])
        for name in ("WT", "TC", "ET"):
            d = dice_binary(pr[name], gt[name])
            metrics[f"dice_{name}"] = metrics.get(f"dice_{name}", 0.0) + d
            dices.append(d)
    n = pred.shape[0]
    for name in ("WT", "TC", "ET"):
        metrics[f"dice_{name}"] /= n
    metrics["dice_mean"] = float(np.mean(dices))
    return metrics, dices


def compute_region_metrics_from_seg(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    """Dice for WT/TC/ET from label maps (D,H,W) or (B,D,H,W)."""
    metrics, _ = _metrics_from_seg_arrays(pred, tgt)
    return metrics


def compute_region_metrics_hd95_from_seg(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    base = compute_region_metrics_from_seg(pred, tgt)
    if pred.ndim == 3:
        pred = pred[np.newaxis, ...]
        tgt = tgt[np.newaxis, ...]
    for i in range(pred.shape[0]):
        pr = seg_to_regions(pred[i])
        gt = seg_to_regions(tgt[i])
        for name in ("WT", "TC", "ET"):
            h = hd95_binary(pr[name], gt[name])
            key = f"hd95_{name}"
            base[key] = base.get(key, 0.0) + (h if np.isfinite(h) else 999.0)
    n = pred.shape[0]
    for name in ("WT", "TC", "ET"):
        base[f"hd95_{name}"] /= n
    return base


@torch.no_grad()
def compute_region_metrics_hd95(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    base = compute_region_metrics(logits, target)
    pred = logits.argmax(dim=1).cpu().numpy()
    tgt = target.cpu().numpy()
    for i in range(pred.shape[0]):
        pr = seg_to_regions(pred[i])
        gt = seg_to_regions(tgt[i])
        for name in ("WT", "TC", "ET"):
            h = hd95_binary(pr[name], gt[name])
            key = f"hd95_{name}"
            base[key] = base.get(key, 0.0) + (h if np.isfinite(h) else 999.0)
    n = pred.shape[0]
    for name in ("WT", "TC", "ET"):
        base[f"hd95_{name}"] /= n
    return base
