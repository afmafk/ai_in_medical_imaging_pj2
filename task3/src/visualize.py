"""Save axial-slice overlay figures: FLAIR + GT vs model predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as ROOT, load_config
from dataset.brats_multimodal import (
    BraTS3DPatchDataset,
    load_patient_volume,
    load_split_patient_ids,
    pad_volume,
)
from device_utils import configure_cuda, resolve_device
from metrics import seg_to_regions
from models import build_model
from sliding_window import sliding_window_predict

# BraTS-style class colors on grayscale background (RGBA)
CLASS_COLORS = {
    1: (1.0, 0.2, 0.2, 0.55),   # necrosis — red
    2: (0.2, 0.85, 0.35, 0.50),  # edema — green
    3: (1.0, 0.85, 0.1, 0.60),   # ET — yellow
}


def normalize_slice(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def overlay_labels(ax, base: np.ndarray, labels: np.ndarray) -> None:
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    rgb = np.zeros((*labels.shape, 4), dtype=np.float32)
    for cls, rgba in CLASS_COLORS.items():
        m = labels == cls
        rgb[m] = rgba
    ax.imshow(rgb)


@torch.no_grad()
def predict_patch(model, image: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(image.unsqueeze(0).to(device))
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def best_tumor_slice(seg: np.ndarray) -> int:
    """Axial index with the largest tumor area (for display)."""
    per_slice = (seg > 0).sum(axis=(1, 2))
    if per_slice.max() == 0:
        return int(seg.shape[0] // 2)
    return int(per_slice.argmax())


def save_patient_figure(
    patient_id: str,
    flair: np.ndarray,
    gt: np.ndarray,
    preds: dict[str, np.ndarray],
    out_path: Path,
    z_idx: int,
    full_slice: bool = False,
) -> None:
    gt_sl = gt[z_idx]
    flair_sl = normalize_slice(flair[z_idx])
    h, w = flair_sl.shape

    ncols = 2 + len(preds)
    if full_slice:
        panel_h = 4.5
        panel_w = panel_h * (w / h)
        fig, axes = plt.subplots(1, ncols, figsize=(panel_w * ncols, panel_h))
    else:
        fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.2))
    if ncols == 1:
        axes = [axes]

    titles = ["FLAIR (t2f)"] + ["Ground truth"] + [f"{name}" for name in preds]
    panels: list[np.ndarray] = [flair_sl, gt_sl] + [preds[k][z_idx] for k in preds]

    for ax, title, labels in zip(axes, titles, panels):
        if title.startswith("FLAIR"):
            ax.imshow(flair_sl, cmap="gray", vmin=0, vmax=1, aspect="equal")
        else:
            overlay_labels(ax, flair_sl, labels)
            ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    legend = [
        Patch(facecolor=CLASS_COLORS[1][:3], alpha=CLASS_COLORS[1][3], label="NCR (1)"),
        Patch(facecolor=CLASS_COLORS[2][:3], alpha=CLASS_COLORS[2][3], label="ED (2)"),
        Patch(facecolor=CLASS_COLORS[3][:3], alpha=CLASS_COLORS[3][3], label="ET (3)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9, frameon=False)
    slice_note = f"full slice {h}×{w}" if full_slice else f"patch crop {h}×{w}"
    fig.suptitle(f"{patient_id}  |  axial z={z_idx}  |  {slice_note}", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_region_comparison(
    patient_id: str,
    flair: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    model_name: str,
    out_path: Path,
    z_idx: int,
) -> None:
    """WT / TC / ET binary masks side-by-side (GT vs pred)."""
    flair_sl = normalize_slice(flair[z_idx])
    gt_r = {k: v[z_idx] for k, v in seg_to_regions(gt).items()}
    pr_r = {k: v[z_idx] for k, v in seg_to_regions(pred).items()}

    cmap = ListedColormap(["none", (0.2, 0.75, 1.0, 0.55)])
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for j, region in enumerate(("WT", "TC", "ET")):
        for i, (src, sl) in enumerate((("GT", gt_r), (model_name, pr_r))):
            ax = axes[i, j]
            ax.imshow(flair_sl, cmap="gray", vmin=0, vmax=1)
            mask = sl[region].astype(float)
            ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"{region} — {src}")
            ax.axis("off")
    fig.suptitle(f"{patient_id}  |  {model_name}  |  z={z_idx}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_full_slice(
    patient_ids: list[str],
    cfg: dict,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    out_dir: Path,
    stride: int,
    z_slice: int | None = None,
) -> None:
    """Full axial slice; prediction uses one full-volume forward when use_full_volume."""
    patch_size = tuple(cfg.get("patch_size", [96, 96, 96]))
    num_classes = int(cfg["num_classes"])
    data_root = Path(cfg["data_root"])
    flair_ch = ["t1n", "t1c", "t2w", "t2f"].index("t2f")
    use_full = bool(cfg.get("use_full_volume", False))
    pad_factor = int(cfg.get("pad_factor", 16))

    for pid in patient_ids:
        image, gt = load_patient_volume(data_root / pid)
        if use_full:
            image, gt = pad_volume(image, gt, pad_factor)
        z_idx = int(z_slice) if z_slice is not None else best_tumor_slice(gt)
        z_idx = max(0, min(z_idx, gt.shape[0] - 1))
        flair = image[flair_ch]
        h, w = int(gt.shape[1]), int(gt.shape[2])

        preds: dict[str, np.ndarray] = {}
        for name, model in models.items():
            if use_full:
                print(f"  {pid}: full-volume forward {name} ...", flush=True)
                preds[name] = predict_patch(model, torch.from_numpy(image), device)
            else:
                print(f"  {pid}: sliding-window {name} ...", flush=True)
                preds[name] = sliding_window_predict(
                    model, image, patch_size, stride, num_classes, device
                )

        save_patient_figure(
            pid,
            flair,
            gt,
            preds,
            out_dir / f"{pid}_compare.png",
            z_idx,
            full_slice=True,
        )
        print(f"wrote {out_dir / f'{pid}_compare.png'}  (slice {h}×{w} at z={z_idx})")


def run_patch(
    patient_ids: list[str],
    cfg: dict,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    out_dir: Path,
) -> None:
    patch_size = tuple(cfg["patch_size"])
    flair_ch = ["t1n", "t1c", "t2w", "t2f"].index("t2f")

    ds = BraTS3DPatchDataset(
        patient_ids,
        cfg["data_root"],
        patch_size=patch_size,
        split="val",
        samples_per_patient=1,
        volume_shape=tuple(cfg.get("volume_shape", [155, 177, 219])),
        use_full_volume=bool(cfg.get("use_full_volume", True)),
        pad_factor=int(cfg.get("pad_factor", 16)),
    )
    z_idx = best_tumor_slice(ds[0]["seg"].numpy()) if len(ds) else 0

    for i in range(len(ds)):
        batch = ds[i]
        pid = batch["patient_id"]
        image = batch["image"]
        gt = batch["seg"].numpy()
        flair = image[flair_ch].numpy()
        preds = {name: predict_patch(m, image, device) for name, m in models.items()}

        save_patient_figure(
            pid, flair, gt, preds, out_dir / f"{pid}_compare.png", z_idx
        )
        for name, pred in preds.items():
            save_region_comparison(
                pid, flair, gt, pred, name, out_dir / f"{pid}_{name}_regions.png", z_idx
            )
        print(f"wrote {pid} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-patients", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["attention_unet", "transbts"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--full-slice",
        "--full-brain",
        action="store_true",
        dest="full_slice",
        help="full axial slice (177×219) + sliding-window prediction; not 96×96 patch",
    )
    parser.add_argument("--stride", type=int, default=48, help="sliding-window stride")
    parser.add_argument("--patient-id", type=str, default=None, help="single patient ID")
    parser.add_argument("--z-slice", type=int, default=None, help="axial slice index (default: max tumor)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device, require_gpu=False)
    configure_cuda(device)

    if args.patient_id:
        patient_ids = [args.patient_id]
    else:
        patient_ids = load_split_patient_ids(cfg["splits_path"], args.split)[: args.num_patients]
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.full_slice:
        out_dir = ROOT / "outputs" / "comparison" / "visualizations_fullslice"
    else:
        out_dir = ROOT / "outputs" / "comparison" / "visualizations"

    models: dict[str, torch.nn.Module] = {}
    for name in args.models:
        ckpt = torch.load(
            ROOT / "outputs" / name / "checkpoints" / "best.ckpt",
            map_location=device,
            weights_only=False,
        )
        mcfg = cfg.get("swinunetr") if name == "swinunetr" else None
        model = build_model(
            name, in_channels=4, num_classes=int(cfg["num_classes"]), model_cfg=mcfg
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        models[name] = model

    if args.full_slice:
        run_full_slice(patient_ids, cfg, models, device, out_dir, args.stride, z_slice=args.z_slice)
    else:
        run_patch(patient_ids, cfg, models, device, out_dir)

    print(f"done: {len(patient_ids)} patients in {out_dir}")


if __name__ == "__main__":
    main()
