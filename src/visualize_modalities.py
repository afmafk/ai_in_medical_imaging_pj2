"""Visualize four MRI modalities (and optional seg) from Task 1 processed_2d slices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as ROOT, load_config

MODALITY_TITLES = {
    "t1n": "T1 native (t1n)",
    "t1c": "T1 contrast (t1c)",
    "t2w": "T2 weighted (t2w)",
    "t2f": "T2-FLAIR (t2f)",
}

CLASS_COLORS = {
    1: (1.0, 0.2, 0.2, 0.55),
    2: (0.2, 0.85, 0.35, 0.50),
    3: (1.0, 0.85, 0.1, 0.60),
}


def normalize_slice(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def overlay_seg(ax, base: np.ndarray, seg: np.ndarray) -> None:
    ax.imshow(base, cmap="gray", vmin=0, vmax=1, aspect="equal")
    rgb = np.zeros((*seg.shape, 4), dtype=np.float32)
    for cls, rgba in CLASS_COLORS.items():
        rgb[seg == cls] = rgba
    ax.imshow(rgb, aspect="equal")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 4 modalities from processed_2d slice npz.")
    parser.add_argument("--patient-id", type=str, required=True)
    parser.add_argument("--z-slice", type=int, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--with-seg", action="store_true", help="Add GT segmentation overlay on t2f")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg.get("data_root", "outputs_task1/outputs_task1/processed_2d"))
    slice_path = data_root / args.patient_id / f"slice_{args.z_slice:03d}.npz"
    if not slice_path.exists():
        raise FileNotFoundError(f"Slice not found: {slice_path}")

    data = np.load(slice_path)
    image = data["image"].astype(np.float32)  # (4, H, W)
    seg = data["seg"].astype(np.uint8)
    mods = [str(m) for m in data["modalities"]] if "modalities" in data else ["t1n", "t1c", "t2w", "t2f"]

    ncols = len(mods) + (1 if args.with_seg else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    for ax, mod, ch in zip(axes[: len(mods)], mods, range(image.shape[0])):
        sl = normalize_slice(image[ch])
        ax.imshow(sl, cmap="gray", vmin=0, vmax=1, aspect="equal")
        ax.set_title(MODALITY_TITLES.get(mod, mod), fontsize=11)
        ax.axis("off")

    if args.with_seg:
        ax = axes[-1]
        flair = normalize_slice(image[mods.index("t2f")] if "t2f" in mods else image[-1])
        overlay_seg(ax, flair, seg)
        ax.set_title("GT overlay (on t2f)", fontsize=11)
        ax.axis("off")
        legend = [
            Patch(facecolor=CLASS_COLORS[1][:3], alpha=CLASS_COLORS[1][3], label="NCR (1)"),
            Patch(facecolor=CLASS_COLORS[2][:3], alpha=CLASS_COLORS[2][3], label="ED (2)"),
            Patch(facecolor=CLASS_COLORS[3][:3], alpha=CLASS_COLORS[3][3], label="ET (3)"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9, frameon=False)

    raw_z = int(data["raw_z"]) if "raw_z" in data else args.z_slice
    fig.suptitle(
        f"{args.patient_id}  |  slice z={args.z_slice}  (raw_z={raw_z})  |  Task 1 preprocessed",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = ROOT / "outputs_task1" / "outputs_task1" / "visualizations_modalities"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_seg" if args.with_seg else ""
        out_path = out_dir / f"{args.patient_id}_z{args.z_slice:03d}_modalities{suffix}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
