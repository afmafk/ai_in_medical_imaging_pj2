"""Full-volume inference via overlapping 3D patches (BraTS / nnU-Net style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset.brats_multimodal import load_patient_volume_from_root
from metrics import compute_region_metrics_from_seg, compute_region_metrics_hd95_from_seg


def patch_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    stride: int,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """Average overlapping patch logits; return label map (D, H, W)."""
    model.eval()
    pd, ph, pw = patch_size
    _, depth, height, width = volume.shape
    logits_sum = np.zeros((num_classes, depth, height, width), dtype=np.float32)
    counts = np.zeros((depth, height, width), dtype=np.float32)

    z_starts = patch_starts(depth, pd, stride)
    y_starts = patch_starts(height, ph, stride)
    x_starts = patch_starts(width, pw, stride)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = volume[:, z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
                patch = np.array(patch, dtype=np.float32, copy=True, order="C")
                logits = model(torch.from_numpy(patch).unsqueeze(0).to(device))[0]
                logits_np = logits.cpu().numpy()
                logits_sum[:, z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += logits_np
                counts[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += 1.0

    counts = np.maximum(counts, 1.0)
    return logits_sum.argmax(axis=0).astype(np.uint8)


@torch.no_grad()
def evaluate_patients_sliding_window(
    model: torch.nn.Module,
    patient_ids: list[str],
    data_root: str,
    patch_size: tuple[int, int, int],
    stride: int,
    num_classes: int,
    device: torch.device,
    with_hd95: bool = False,
) -> dict[str, float]:
    """Per-patient full-volume sliding window; mean metrics over patients."""
    root = Path(data_root)
    agg: dict[str, float] = {}
    n = 0
    for pid in tqdm(patient_ids, desc="sliding-window"):
        image, seg = load_patient_volume_from_root(root, pid)
        pred = sliding_window_predict(model, image, patch_size, stride, num_classes, device)
        if with_hd95:
            m = compute_region_metrics_hd95_from_seg(pred, seg)
        else:
            m = compute_region_metrics_from_seg(pred, seg)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}
