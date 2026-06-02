"""Measure connected components for selected full-volume prediction masks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from dataset.brats_multimodal import load_patient_volume_from_root
from device_utils import configure_cuda, resolve_device
from metrics import seg_to_regions
from models import MODEL_NAMES, build_model
from sliding_window import sliding_window_predict


def component_stats(mask: np.ndarray) -> dict[str, int]:
    components, count = label(mask)
    if count == 0:
        return {
            "components": 0,
            "largest_component_voxels": 0,
            "non_largest_component_voxels": 0,
        }
    sizes = np.bincount(components.ravel())[1:]
    largest = int(sizes.max())
    return {
        "components": int(count),
        "largest_component_voxels": largest,
        "non_largest_component_voxels": int(sizes.sum() - largest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patient-id", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    configure_cuda(device)
    patch_size = tuple(cfg.get("patch_size", [96, 96, 96]))
    stride = int(cfg.get("sliding_window_stride", patch_size[0] // 2))
    model = build_model(
        args.model,
        in_channels=4,
        num_classes=int(cfg["num_classes"]),
        model_cfg=cfg.get(args.model),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows: list[dict[str, object]] = []
    for patient_id in tqdm(args.patient_id, desc="analyze-components"):
        image, gt = load_patient_volume_from_root(Path(cfg["data_root"]), patient_id)
        pred = sliding_window_predict(
            model,
            image,
            patch_size,
            stride,
            int(cfg["num_classes"]),
            device,
        )
        pred_regions = seg_to_regions(pred)
        gt_regions = seg_to_regions(gt)
        row: dict[str, object] = {"patient_id": patient_id}
        for region in ("WT", "TC", "ET"):
            for prefix, mask in (("pred", pred_regions[region]), ("gt", gt_regions[region])):
                for key, value in component_stats(mask).items():
                    row[f"{prefix}_{region}_{key}"] = value
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
