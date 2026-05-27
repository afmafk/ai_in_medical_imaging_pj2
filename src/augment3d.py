from __future__ import annotations

import random

import numpy as np


def augment_volume(
    image: np.ndarray,
    seg: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """3D augmentation on image (4,D,H,W) and seg (D,H,W)."""
    image = image.copy()
    seg = seg.copy()

    # random flips
    for axis in (1, 2, 3):  # D, H, W in image tensor
        if rng.random() < 0.5:
            image = np.flip(image, axis=axis).copy()
            seg = np.flip(seg, axis=axis - 1).copy()

    # intensity shift / scale (image only)
    shift = rng.uniform(-0.1, 0.1)
    scale = rng.uniform(0.9, 1.1)
    brain = image != 0
    image[brain] = image[brain] * scale + shift

    # gaussian noise on brain voxels
    if rng.random() < 0.5:
        noise = rng.gauss(0, 0.05)
        image[brain] = image[brain] + np.random.normal(0, 0.05, size=image[brain].shape).astype(
            np.float32
        )

    return image, seg
