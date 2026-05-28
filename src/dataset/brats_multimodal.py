from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from augment3d import augment_volume


def _slice_path(patient_dir: Path, z: int) -> Path:
    return patient_dir / f"slice_{z:03d}.npz"


def _volume_path(data_root: Path, patient_id: str) -> Path:
    return data_root / f"{patient_id}.npz"


def get_patient_depth(patient_dir: Path) -> int:
    """Fast depth count without loading arrays."""
    n = 0
    while _slice_path(patient_dir, n).exists():
        n += 1
    if n == 0:
        raise FileNotFoundError(f"No slices in {patient_dir}")
    return n


def get_patient_depth_from_root(data_root: Path, patient_id: str) -> int:
    """Depth for either processed_3d .npz files or processed_2d slice folders."""
    volume_path = _volume_path(data_root, patient_id)
    if volume_path.exists():
        data = np.load(volume_path)
        return int(data["image"].shape[-1])
    return get_patient_depth(data_root / patient_id)


def _load_one_slice(
    patient_dir: Path,
    zi: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(_slice_path(patient_dir, zi))
    return (
        data["image"][:, y0:y1, x0:x1].astype(np.float32),
        data["seg"][y0:y1, x0:x1].astype(np.uint8),
    )


def load_z_patch(
    patient_dir: Path,
    z0: int,
    depth: int,
    y0: int,
    height: int,
    x0: int,
    width: int,
    slice_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load only [z0, z0+depth) slices and crop (y,x)."""
    y1, x1 = y0 + height, x0 + width
    z_indices = list(range(z0, z0 + depth))

    if slice_threads <= 1 or depth <= 4:
        images: list[np.ndarray] = []
        segs: list[np.ndarray] = []
        for zi in z_indices:
            img, seg = _load_one_slice(patient_dir, zi, y0, y1, x0, x1)
            images.append(img)
            segs.append(seg)
    else:
        workers = min(slice_threads, depth)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunks = pool.map(
                lambda zi: _load_one_slice(patient_dir, zi, y0, y1, x0, x1),
                z_indices,
            )
        images, segs = zip(*chunks)

    volume = np.stack(images, axis=1)  # (4, D, h, w)
    seg = np.stack(segs, axis=0)
    return volume, seg


def load_patient_volume(patient_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load full preprocessed volume: image (4,D,H,W), seg (D,H,W)."""
    depth = get_patient_depth(patient_dir)
    d0 = np.load(_slice_path(patient_dir, 0))
    _, h, w = d0["image"].shape
    return load_z_patch(patient_dir, 0, depth, 0, h, 0, w)


def load_patient_volume_from_root(data_root: str | Path, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a full volume from processed_3d .npz or processed_2d slice folders."""
    root = Path(data_root)
    volume_path = _volume_path(root, patient_id)
    if volume_path.exists():
        data = np.load(volume_path)
        image = data["image"].astype(np.float32)  # (4,H,W,D)
        seg = data["seg"].astype(np.uint8)  # (H,W,D)
        return image.transpose(0, 3, 1, 2), seg.transpose(2, 0, 1)
    return load_patient_volume(root / patient_id)


def load_z_patch_from_root(
    data_root: str | Path,
    patient_id: str,
    z0: int,
    depth: int,
    y0: int,
    height: int,
    x0: int,
    width: int,
    slice_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cropped 3D patch from either storage layout."""
    root = Path(data_root)
    volume_path = _volume_path(root, patient_id)
    if volume_path.exists():
        image, seg = load_patient_volume_from_root(root, patient_id)
        return (
            image[:, z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width],
            seg[z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width],
        )
    return load_z_patch(root / patient_id, z0, depth, y0, height, x0, width, slice_threads)


def pad_volume(
    image: np.ndarray,
    seg: np.ndarray,
    factor: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad D,H,W so each dim is divisible by factor (for 4× U-Net pooling)."""
    _, d, h, w = image.shape
    pd = (factor - d % factor) % factor
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    if pd == ph == pw == 0:
        return image, seg
    image = np.pad(image, ((0, 0), (0, pd), (0, ph), (0, pw)), mode="constant")
    seg = np.pad(seg, ((0, pd), (0, ph), (0, pw)), mode="constant")
    return image, seg


def random_crop_coords(
    depth: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    pd, ph, pw = patch_size
    if depth < pd or height < ph or width < pw:
        raise ValueError(f"patch {patch_size} larger than volume {(depth, height, width)}")
    z0 = rng.randint(0, depth - pd)
    y0 = rng.randint(0, height - ph)
    x0 = rng.randint(0, width - pw)
    return z0, y0, x0


class BraTS3DPatchDataset(Dataset):
    """BraTS 3D volumes: full brain (default) or legacy fixed/random patches."""

    def __init__(
        self,
        patient_ids: list[str],
        data_root: str | Path,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        split: Literal["train", "val", "test"] = "train",
        samples_per_patient: int = 4,
        seed: int = 42,
        require_tumor: bool = False,
        volume_shape: tuple[int, int, int] = (155, 177, 219),
        use_full_volume: bool = True,
        pad_factor: int = 16,
        slice_load_threads: int = 8,
        augment_config: Mapping[str, object] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.patient_ids = list(patient_ids)
        self.patch_size = tuple(patch_size)
        self.split = split
        self.use_full_volume = use_full_volume
        self.pad_factor = pad_factor
        self.slice_load_threads = max(1, int(slice_load_threads))
        self.augment_config = dict(augment_config) if augment_config else None
        self.require_tumor = require_tumor and split == "train"
        self.rng = random.Random(seed + hash(split) % 10000)
        self.vol_depth, self.vol_h, self.vol_w = volume_shape

        if not self.patient_ids:
            raise ValueError("empty patient list")

        self._patient_depths: dict[str, int] | None = None

        if use_full_volume:
            self.samples_per_patient = 1
        else:
            self.samples_per_patient = samples_per_patient if split == "train" else 1
            pd, ph, pw = self.patch_size
            if split != "train":
                self._fixed_z = max(0, (self.vol_depth - pd) // 2)
                self._fixed_y = max(0, (self.vol_h - ph) // 2)
                self._fixed_x = max(0, (self.vol_w - pw) // 2)

    def __len__(self) -> int:
        return len(self.patient_ids) * self.samples_per_patient

    def _patient_depth(self, patient_id: str) -> int:
        """Lazy depth cache (safe with Windows DataLoader worker pickle)."""
        if self._patient_depths is None:
            self._patient_depths = {
                pid: get_patient_depth_from_root(self.data_root, pid) for pid in self.patient_ids
            }
        return self._patient_depths[patient_id]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._patient_depths is None and self.patient_ids:
            self._patient_depths = {
                pid: get_patient_depth_from_root(self.data_root, pid) for pid in self.patient_ids
            }

    def _load_full_volume(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        image, seg = load_patient_volume_from_root(self.data_root, patient_id)
        image, seg = pad_volume(image, seg, self.pad_factor)
        if self.split == "train":
            image, seg = augment_volume(image, seg, self.rng, self.augment_config)
        return image, seg

    def _load_patch(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        patient_dir = self.data_root / patient_id
        pd, ph, pw = self.patch_size
        depth = self._patient_depth(patient_id)
        load_kw = dict(slice_threads=self.slice_load_threads)

        if self.split == "train":
            for _ in range(12):
                z0, y0, x0 = random_crop_coords(
                    depth, self.vol_h, self.vol_w, self.patch_size, self.rng
                )
                image, seg = load_z_patch_from_root(
                    self.data_root, patient_id, z0, pd, y0, ph, x0, pw, **load_kw
                )
                if not self.require_tumor or (seg > 0).any():
                    break
            image, seg = augment_volume(image, seg, self.rng, self.augment_config)
        else:
            z0 = min(self._fixed_z, max(0, depth - pd))
            image, seg = load_z_patch_from_root(
                self.data_root,
                patient_id,
                z0,
                pd,
                self._fixed_y,
                ph,
                self._fixed_x,
                pw,
                **load_kw,
            )
        return image, seg

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patient_id = self.patient_ids[index % len(self.patient_ids)]
        if self.use_full_volume:
            image, seg = self._load_full_volume(patient_id)
        else:
            image, seg = self._load_patch(patient_id)
        return {
            "image": torch.from_numpy(image),
            "seg": torch.from_numpy(seg).long(),
            "patient_id": patient_id,
        }


def load_split_patient_ids(splits_path: str | Path, split: str) -> list[str]:
    with Path(splits_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data[split])
