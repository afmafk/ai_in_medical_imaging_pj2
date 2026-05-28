from __future__ import annotations

import random
from collections.abc import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, zoom


DEFAULT_AUGMENT_CONFIG = {
    "flip_prob_hw": 0.5,
    "flip_prob_depth": 0.15,
    "rotate_prob": 0.3,
    "rotate_degrees": 10.0,
    "scale_prob": 0.25,
    "scale_range": (0.9, 1.1),
    "intensity_prob": 1.0,
    "intensity_scale_range": (0.9, 1.1),
    "intensity_shift_range": (-0.1, 0.1),
    "noise_prob": 0.4,
    "noise_std_range": (0.0, 0.05),
    "blur_prob": 0.2,
    "blur_sigma_range": (0.5, 1.0),
}


def _range_pair(value: object) -> tuple[float, float]:
    lo, hi = value  # type: ignore[misc]
    return float(lo), float(hi)


def _center_crop_or_pad_3d(array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Center crop/pad a 3D array to target_shape."""
    result = array
    for axis, target in enumerate(target_shape):
        size = result.shape[axis]
        if size > target:
            start = (size - target) // 2
            end = start + target
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(start, end)
            result = result[tuple(slices)]
        elif size < target:
            before = (target - size) // 2
            after = target - size - before
            pad = [(0, 0)] * result.ndim
            pad[axis] = (before, after)
            result = np.pad(result, pad, mode="constant")
    return result


def _scale_volume(
    image: np.ndarray,
    seg: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_shape = seg.shape
    image_scaled = zoom(image, (1.0, scale, scale, scale), order=1, mode="constant", cval=0.0)
    seg_scaled = zoom(seg, (scale, scale, scale), order=0, mode="constant", cval=0)
    image_out = np.stack(
        [_center_crop_or_pad_3d(ch, target_shape) for ch in image_scaled],
        axis=0,
    )
    seg_out = _center_crop_or_pad_3d(seg_scaled, target_shape)
    return image_out.astype(np.float32, copy=False), seg_out.astype(seg.dtype, copy=False)


def _rotate_axial(
    image: np.ndarray,
    seg: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    image_rot = rotate(
        image,
        angle,
        axes=(2, 3),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )
    seg_rot = rotate(
        seg,
        angle,
        axes=(1, 2),
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
    )
    return image_rot.astype(np.float32, copy=False), seg_rot.astype(seg.dtype, copy=False)


def augment_volume(
    image: np.ndarray,
    seg: np.ndarray,
    rng: random.Random,
    config: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """3D augmentation on image (4,D,H,W) and seg (D,H,W)."""
    cfg = {**DEFAULT_AUGMENT_CONFIG, **(dict(config) if config else {})}
    np_rng = np.random.default_rng(rng.randrange(2**32))
    image = image.copy()
    seg = seg.copy()

    # Synchronized spatial transforms for all modalities and the label map.
    if rng.random() < float(cfg["flip_prob_depth"]):
        image = np.flip(image, axis=1).copy()
        seg = np.flip(seg, axis=0).copy()
    for axis in (2, 3):  # H, W in image tensor
        if rng.random() < float(cfg["flip_prob_hw"]):
            image = np.flip(image, axis=axis).copy()
            seg = np.flip(seg, axis=axis - 1).copy()

    if rng.random() < float(cfg["rotate_prob"]):
        angle = rng.uniform(-float(cfg["rotate_degrees"]), float(cfg["rotate_degrees"]))
        image, seg = _rotate_axial(image, seg, angle)

    if rng.random() < float(cfg["scale_prob"]):
        scale_min, scale_max = _range_pair(cfg["scale_range"])
        image, seg = _scale_volume(image, seg, rng.uniform(scale_min, scale_max))

    # Modality-wise intensity transforms for images only.
    if rng.random() < float(cfg["intensity_prob"]):
        scale_min, scale_max = _range_pair(cfg["intensity_scale_range"])
        shift_min, shift_max = _range_pair(cfg["intensity_shift_range"])
        for c in range(image.shape[0]):
            brain = image[c] != 0
            if brain.any():
                scale = rng.uniform(scale_min, scale_max)
                shift = rng.uniform(shift_min, shift_max)
                image[c, brain] = image[c, brain] * scale + shift

    if rng.random() < float(cfg["noise_prob"]):
        std_min, std_max = _range_pair(cfg["noise_std_range"])
        std = rng.uniform(std_min, std_max)
        brain = image != 0
        image[brain] = image[brain] + np_rng.normal(0.0, std, size=image[brain].shape).astype(np.float32)

    if rng.random() < float(cfg["blur_prob"]):
        sigma_min, sigma_max = _range_pair(cfg["blur_sigma_range"])
        sigma = rng.uniform(sigma_min, sigma_max)
        for c in range(image.shape[0]):
            brain = image[c] != 0
            blurred = gaussian_filter(image[c], sigma=sigma)
            image[c, brain] = blurred[brain]

    return image.astype(np.float32, copy=False), seg
