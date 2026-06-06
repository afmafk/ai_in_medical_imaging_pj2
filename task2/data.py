from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .config import AugmentationConfig
except ImportError:
    from config import AugmentationConfig


DEFAULT_MODALITIES = ("t2f", "t1c", "t2w")
DEFAULT_REGION_NAMES = ("WT", "TC", "ET")


def _affine_matrix(angle_deg: float, scale: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad) * scale
    sin_a = math.sin(angle_rad) * scale
    return torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
        dtype=dtype,
        device=device,
    )


def _apply_spatial_transform(
    tensor: torch.Tensor,
    angle_deg: float,
    scale: float,
    mode: str,
) -> torch.Tensor:
    original_ndim = tensor.ndim
    if original_ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape [H, W] or [C, H, W], got {tuple(tensor.shape)}")

    batch = tensor.unsqueeze(0)
    theta = _affine_matrix(angle_deg, scale, batch.device, batch.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, size=batch.shape, align_corners=False)
    transformed = F.grid_sample(
        batch,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)
    if original_ndim == 2:
        transformed = transformed.squeeze(0)
    return transformed


def _pad_spatial(tensor: torch.Tensor, pad_pixels: int) -> torch.Tensor:
    if pad_pixels <= 0:
        return tensor
    if tensor.ndim == 2:
        return F.pad(tensor.unsqueeze(0), (pad_pixels, pad_pixels, pad_pixels, pad_pixels), mode="constant", value=0).squeeze(0)
    if tensor.ndim == 3:
        return F.pad(tensor, (pad_pixels, pad_pixels, pad_pixels, pad_pixels), mode="constant", value=0)
    raise ValueError(f"Expected tensor shape [H, W] or [C, H, W], got {tuple(tensor.shape)}")


def _center_crop_spatial(tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    if tensor.ndim == 2:
        height, width = tensor.shape
        start_y = max((height - target_height) // 2, 0)
        start_x = max((width - target_width) // 2, 0)
        return tensor[start_y : start_y + target_height, start_x : start_x + target_width]
    if tensor.ndim == 3:
        _, height, width = tensor.shape
        start_y = max((height - target_height) // 2, 0)
        start_x = max((width - target_width) // 2, 0)
        return tensor[:, start_y : start_y + target_height, start_x : start_x + target_width]
    raise ValueError(f"Expected tensor shape [H, W] or [C, H, W], got {tuple(tensor.shape)}")


class BraTSTrainAugment:
    """Applies simple paired augmentations to image/mask samples."""

    def __init__(self, config: Optional[AugmentationConfig] = None) -> None:
        self.config = config or AugmentationConfig()

    def __call__(self, item: Dict[str, torch.Tensor | str | int]) -> Dict[str, torch.Tensor | str | int]:
        if not self.config.enabled:
            return item

        image = item["image"]
        mask = item["mask"]
        if not torch.is_tensor(image) or not torch.is_tensor(mask):
            raise TypeError("Augmentation expects tensor 'image' and 'mask' entries.")

        target_height, target_width = image.shape[-2:]
        angle = float(torch.empty(1).uniform_(-self.config.rotation_degrees, self.config.rotation_degrees).item())
        scale = float(torch.empty(1).uniform_(*self.config.scale_range).item())

        image = _pad_spatial(image.float(), self.config.pad_pixels)
        mask = _pad_spatial(mask, self.config.pad_pixels)

        image = _apply_spatial_transform(image.float(), angle_deg=angle, scale=scale, mode="bilinear")
        if mask.ndim == 2:
            mask = _apply_spatial_transform(mask.unsqueeze(0).float(), angle_deg=angle, scale=scale, mode="nearest")
            mask = _center_crop_spatial(mask.squeeze(0), target_height, target_width).round().long()
        else:
            mask = _apply_spatial_transform(mask.float(), angle_deg=angle, scale=scale, mode="nearest")
            mask = _center_crop_spatial(mask, target_height, target_width).round().clamp_(0.0, 1.0)

        image = _center_crop_spatial(image, target_height, target_width)

        if torch.rand(1).item() < self.config.horizontal_flip_prob:
            image = torch.flip(image, dims=(-1,))
            if mask.ndim == 2:
                mask = torch.flip(mask, dims=(-1,))
            else:
                mask = torch.flip(mask, dims=(-1,))

        if torch.rand(1).item() < self.config.vertical_flip_prob:
            image = torch.flip(image, dims=(-2,))
            if mask.ndim == 2:
                mask = torch.flip(mask, dims=(-2,))
            else:
                mask = torch.flip(mask, dims=(-2,))

        if torch.rand(1).item() < self.config.gaussian_noise_prob:
            noise = torch.randn_like(image) * self.config.gaussian_noise_std
            image = image + noise

        item["image"] = image
        item["mask"] = mask
        return item


class Processed2DSegmentationDataset(Dataset):
    """Dataset for processed_2d BraTS slices stored as .npz files."""

    def __init__(
        self,
        root: str | Path = "processed_2d",
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        target_mode: str = "multiclass",
        region_names: Sequence[str] = DEFAULT_REGION_NAMES,
        include_empty_slices: bool = True,
        remove_black_slices: bool = True,
        sample_paths: Optional[Sequence[str | Path]] = None,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.root = Path(root)
        self.modalities = tuple(modalities)
        self.target_mode = target_mode.lower()
        self.region_names = tuple(region_names)
        self.include_empty_slices = include_empty_slices
        self.remove_black_slices = remove_black_slices
        self.transform = transform

        if self.target_mode not in {"multiclass", "regions"}:
            raise ValueError("target_mode must be either 'multiclass' or 'regions'.")
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        if sample_paths is not None:
            self.samples = [Path(sample_path) for sample_path in sample_paths]
        else:
            self.samples = self._discover_samples()
        if not self.samples:
            raise RuntimeError(f"No .npz samples found under {self.root}")

    def _discover_samples(self) -> List[Path]:
        samples: List[Path] = []
        for sample_path in sorted(self.root.rglob("*.npz")):
            if self.include_empty_slices:
                samples.append(sample_path)
                continue

            with np.load(sample_path, allow_pickle=True) as sample:
                if self.remove_black_slices and np.all(sample["image"] == 0):
                    continue

                if self.target_mode == "multiclass":
                    keep = np.any(sample["seg"] > 0)
                else:
                    keep = np.any(sample["regions"] > 0)
            if keep:
                samples.append(sample_path)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _modality_indices(self, available_modalities: np.ndarray) -> List[int]:
        modality_list = [str(modality) for modality in available_modalities.tolist()]
        indices: List[int] = []
        for modality in self.modalities:
            if modality not in modality_list:
                raise ValueError(f"Requested modality '{modality}' not found in sample modalities {modality_list}")
            indices.append(modality_list.index(modality))
        return indices

    def _region_indices(self, available_regions: np.ndarray) -> List[int]:
        region_list = [str(region) for region in available_regions.tolist()]
        indices: List[int] = []
        for region in self.region_names:
            if region not in region_list:
                raise ValueError(f"Requested region '{region}' not found in sample regions {region_list}")
            indices.append(region_list.index(region))
        return indices

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        sample_path = self.samples[index]
        with np.load(sample_path, allow_pickle=True) as sample:
            image = sample["image"].astype(np.float32)
            modality_indices = self._modality_indices(sample["modalities"])
            image = image[modality_indices]

            if self.target_mode == "multiclass":
                mask = sample["seg"].astype(np.int64)
                mask_tensor = torch.from_numpy(mask)
            else:
                region_indices = self._region_indices(sample["region_names"])
                mask = sample["regions"][region_indices].astype(np.float32)
                mask_tensor = torch.from_numpy(mask)

            item: Dict[str, torch.Tensor | str | int] = {
                "image": torch.from_numpy(image),
                "mask": mask_tensor,
                "path": str(sample_path),
                "patient_id": sample_path.parent.name,
                "slice_idx": int(sample_path.stem.split("_")[-1]),
            }

        if self.transform is not None:
            item = self.transform(item)
        return item


def list_patient_ids(root: str | Path = "processed_2d") -> List[str]:
    return sorted([path.name for path in Path(root).iterdir() if path.is_dir()])


def split_patients(
    patient_ids: Sequence[str],
    ratios: Sequence[float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> Dict[str, List[str]]:
    if len(ratios) != 3:
        raise ValueError("ratios must contain exactly three values for train, val, and test.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    patient_ids = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    total = len(patient_ids)
    train_count = int(total * ratios[0])
    val_count = int(total * ratios[1])
    if train_count <= 0 or val_count <= 0 or total - train_count - val_count <= 0:
        raise ValueError("Split produced an empty subset. Check patient count and ratios.")

    train_ids = patient_ids[:train_count]
    val_ids = patient_ids[train_count : train_count + val_count]
    test_ids = patient_ids[train_count + val_count :]
    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def collect_sample_paths_for_patients(
    root: str | Path,
    patient_ids: Sequence[str],
) -> List[Path]:
    root = Path(root)
    sample_paths: List[Path] = []
    for patient_id in patient_ids:
        patient_dir = root / patient_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
        sample_paths.extend(sorted(patient_dir.glob("*.npz")))
    return sample_paths


def build_patient_split_datasets(
    root: str | Path = "processed_2d",
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    target_mode: str = "multiclass",
    region_names: Sequence[str] = DEFAULT_REGION_NAMES,
    ratios: Sequence[float] = (0.7, 0.1, 0.2),
    seed: int = 42,
    include_empty_slices: bool = True,
    remove_black_slices: bool = True,
    train_transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    eval_transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
) -> Dict[str, Processed2DSegmentationDataset]:
    patient_ids = list_patient_ids(root)
    split = split_patients(patient_ids, ratios=ratios, seed=seed)

    datasets: Dict[str, Processed2DSegmentationDataset] = {}
    for split_name, split_patient_ids in split.items():
        sample_paths = collect_sample_paths_for_patients(root, split_patient_ids)
        datasets[split_name] = Processed2DSegmentationDataset(
            root=root,
            modalities=modalities,
            target_mode=target_mode,
            region_names=region_names,
            include_empty_slices=include_empty_slices,
            remove_black_slices=remove_black_slices,
            sample_paths=sample_paths,
            transform=train_transform if split_name == "train" else eval_transform,
        )
    return datasets


def load_metadata(metadata_path: str | Path = "metadata.json") -> Dict:
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        return json.load(f)
