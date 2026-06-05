"""Export per-case metrics and worst-case full-volume visualizations."""

from __future__ import annotations

import argparse
import csv
import json
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

from config import PROJECT_ROOT as ROOT, load_config
from dataset.brats_multimodal import load_patient_volume_from_root, load_split_patient_ids
from device_utils import configure_cuda, resolve_device
from metrics import dice_binary, hd95_binary, seg_to_regions
from models import MODEL_NAMES, build_model
from sliding_window import sliding_window_predict

CLASS_COLORS = {
    1: (1.0, 0.2, 0.2, 0.55),   # NCR/NET
    2: (0.2, 0.85, 0.35, 0.50),  # ED
    3: (1.0, 0.85, 0.1, 0.60),   # ET
}


def normalize_slice(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def overlay_labels(ax, base: np.ndarray, labels: np.ndarray) -> None:
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        rgba[labels == cls] = color
    ax.imshow(rgba)
    ax.axis("off")


def best_display_slice(gt: np.ndarray, pred: np.ndarray) -> int:
    """Pick a slice showing the largest combined GT/pred tumor discrepancy."""
    diff = np.logical_xor(gt > 0, pred > 0).sum(axis=(1, 2))
    if diff.max() > 0:
        return int(diff.argmax())
    tumor = (gt > 0).sum(axis=(1, 2))
    return int(tumor.argmax()) if tumor.max() > 0 else int(gt.shape[0] // 2)


def case_metrics(patient_id: str, pred: np.ndarray, gt: np.ndarray) -> dict[str, object]:
    row: dict[str, object] = {"patient_id": patient_id}
    pr_regions = seg_to_regions(pred)
    gt_regions = seg_to_regions(gt)
    dices: list[float] = []
    for region in ("WT", "TC", "ET"):
        pr = pr_regions[region]
        target = gt_regions[region]
        dice = dice_binary(pr, target)
        hd95_raw = hd95_binary(pr, target)
        row[f"dice_{region}"] = dice
        row[f"hd95_{region}"] = hd95_raw if np.isfinite(hd95_raw) else 999.0
        row[f"hd95_{region}_is_inf"] = not np.isfinite(hd95_raw)
        row[f"pred_voxels_{region}"] = int(pr.sum())
        row[f"gt_voxels_{region}"] = int(target.sum())
        row[f"pred_empty_{region}"] = not bool(pr.any())
        row[f"gt_empty_{region}"] = not bool(target.any())
        dices.append(dice)
    row["dice_mean"] = float(np.mean(dices))
    return row


def save_case_figure(
    patient_id: str,
    flair: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    metrics: dict[str, object],
    out_path: Path,
    z_idx: int | None = None,
) -> None:
    z_idx = best_display_slice(gt, pred) if z_idx is None else z_idx
    if not 0 <= z_idx < gt.shape[0]:
        raise ValueError(f"slice z={z_idx} is outside valid range [0, {gt.shape[0] - 1}]")
    flair_sl = normalize_slice(flair[z_idx])
    gt_sl = gt[z_idx]
    pred_sl = pred[z_idx]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    axes[0].imshow(flair_sl, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("FLAIR")
    axes[0].axis("off")
    overlay_labels(axes[1], flair_sl, gt_sl)
    axes[1].set_title("Ground truth")
    overlay_labels(axes[2], flair_sl, pred_sl)
    axes[2].set_title("Prediction")

    diff = np.logical_xor(gt_sl > 0, pred_sl > 0)
    axes[3].imshow(flair_sl, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(~diff, diff), cmap="cool", alpha=0.75)
    axes[3].set_title("WT disagreement")
    axes[3].axis("off")

    legend = [
        Patch(facecolor=CLASS_COLORS[1][:3], alpha=CLASS_COLORS[1][3], label="NCR/NET (1)"),
        Patch(facecolor=CLASS_COLORS[2][:3], alpha=CLASS_COLORS[2][3], label="ED (2)"),
        Patch(facecolor=CLASS_COLORS[3][:3], alpha=CLASS_COLORS[3][3], label="ET (3)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{patient_id} | z={z_idx} | "
        f"Dice WT/TC/ET={metrics['dice_WT']:.3f}/{metrics['dice_TC']:.3f}/{metrics['dice_ET']:.3f} | "
        f"HD95 WT/TC/ET={metrics['hd95_WT']:.2f}/{metrics['hd95_TC']:.2f}/{metrics['hd95_ET']:.2f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--worst-k", type=int, default=5)
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    patient_ids = load_split_patient_ids(cfg["splits_path"], args.split)
    if args.max_patients:
        patient_ids = patient_ids[: args.max_patients]

    output_dir = Path(args.output_dir)
    rows: list[dict[str, object]] = []
    for patient_id in tqdm(patient_ids, desc="analyze-cases"):
        image, gt = load_patient_volume_from_root(cfg["data_root"], patient_id)
        pred = sliding_window_predict(
            model,
            image,
            patch_size,
            stride,
            int(cfg["num_classes"]),
            device,
        )
        rows.append(case_metrics(patient_id, pred, gt))

    write_csv(output_dir / f"per_case_metrics_{args.split}.csv", rows)

    worst_mean = sorted(rows, key=lambda row: float(row["dice_mean"]))[: args.worst_k]
    worst_wt_hd95 = sorted(rows, key=lambda row: float(row["hd95_WT"]), reverse=True)[: args.worst_k]
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in [*worst_mean, *worst_wt_hd95]:
        patient_id = str(row["patient_id"])
        if patient_id not in seen:
            selected.append(row)
            seen.add(patient_id)

    write_csv(output_dir / f"worst_cases_{args.split}.csv", selected)
    figure_dir = output_dir / f"worst_case_figures_{args.split}"
    for row in tqdm(selected, desc="render-worst-cases"):
        patient_id = str(row["patient_id"])
        image, gt = load_patient_volume_from_root(cfg["data_root"], patient_id)
        pred = sliding_window_predict(
            model,
            image,
            patch_size,
            stride,
            int(cfg["num_classes"]),
            device,
        )
        save_case_figure(patient_id, image[3], gt, pred, row, figure_dir / f"{patient_id}.png")

    summary = {
        "model": args.model,
        "split": args.split,
        "num_cases": len(rows),
        "num_inf_hd95_WT": sum(bool(row["hd95_WT_is_inf"]) for row in rows),
        "num_inf_hd95_TC": sum(bool(row["hd95_TC_is_inf"]) for row in rows),
        "num_inf_hd95_ET": sum(bool(row["hd95_ET_is_inf"]) for row in rows),
        "worst_mean_dice_cases": [row["patient_id"] for row in worst_mean],
        "worst_wt_hd95_cases": [row["patient_id"] for row in worst_wt_hd95],
    }
    with (output_dir / f"analysis_summary_{args.split}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
