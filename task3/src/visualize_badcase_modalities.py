"""Render four MRI modalities with GT and prediction for selected bad cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from analyze_cases import CLASS_COLORS, best_display_slice, normalize_slice, overlay_labels
from config import load_config
from dataset.brats_multimodal import load_patient_volume_from_root
from device_utils import configure_cuda, resolve_device
from models import MODEL_NAMES, build_model
from sliding_window import sliding_window_predict


MODALITY_NAMES = ("T1", "T1c", "T2", "FLAIR")


def render_case(
    patient_id: str,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
) -> None:
    z_idx = best_display_slice(gt, pred)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    for idx, name in enumerate(MODALITY_NAMES):
        axes[0, idx].imshow(normalize_slice(image[idx, z_idx]), cmap="gray", vmin=0, vmax=1)
        axes[0, idx].set_title(name)
        axes[0, idx].axis("off")

    flair = normalize_slice(image[3, z_idx])
    overlay_labels(axes[1, 0], flair, gt[z_idx])
    axes[1, 0].set_title("Ground truth")
    overlay_labels(axes[1, 1], flair, pred[z_idx])
    axes[1, 1].set_title("Prediction")

    wt_diff = np.logical_xor(gt[z_idx] > 0, pred[z_idx] > 0)
    axes[1, 2].imshow(flair, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].imshow(np.ma.masked_where(~wt_diff, wt_diff), cmap="cool", alpha=0.75)
    axes[1, 2].set_title("WT disagreement")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(flair, cmap="gray", vmin=0, vmax=1)
    axes[1, 3].set_title("FLAIR reference")
    axes[1, 3].axis("off")

    legend = [
        Patch(facecolor=CLASS_COLORS[1][:3], alpha=CLASS_COLORS[1][3], label="NCR/NET (1)"),
        Patch(facecolor=CLASS_COLORS[2][:3], alpha=CLASS_COLORS[2][3], label="ED (2)"),
        Patch(facecolor=CLASS_COLORS[3][:3], alpha=CLASS_COLORS[3][3], label="ET (3)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{patient_id} | z={z_idx}", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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

    for patient_id in tqdm(args.patient_id, desc="render-multimodal-badcases"):
        image, gt = load_patient_volume_from_root(Path(cfg["data_root"]), patient_id)
        pred = sliding_window_predict(
            model,
            image,
            patch_size,
            stride,
            int(cfg["num_classes"]),
            device,
        )
        render_case(patient_id, image, gt, pred, args.output_dir / f"{patient_id}.png")


if __name__ == "__main__":
    main()
