from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from builders import build_model
from data import Processed2DSegmentationDataset, build_patient_split_datasets, collect_sample_paths_for_patients
from engine import compute_loss, compute_region_dice_stats, finalize_region_dice, move_batch_to_device
from losses import build_loss

matplotlib.use("Agg")

LABEL_COLORS = np.array(
    [
        [255, 51, 51],
        [51, 217, 89],
        [255, 217, 26],
    ],
    dtype=np.uint8,
)

MULTICLASS_COLORS = np.array(
    [
        [255, 51, 51],   # label 1
        [51, 217, 89],   # label 2
        [255, 217, 26],  # label 3
    ],
    dtype=np.uint8,
)

MODEL_SPECS = [
    {
        "name": "multiclass_t2w_t2f",
        "checkpoint": "unet_multiclass/best_model.pt",
        "run_config": "unet_multiclass/run_config.json",
        "save_json": "evaluation_outputs/multiclass_t2w_t2f/test_metrics.json",
    },
    {
        "name": "multiclass_all_modalities",
        "checkpoint": "unet_multiclass_all_modalities/best_model.pt",
        "run_config": "unet_multiclass_all_modalities/run_config.json",
        "save_json": "evaluation_outputs/multiclass_all_modalities/test_metrics.json",
    },
    {
        "name": "multiclass_t1n",
        "checkpoint": "unet_multiclass_t1/best_model.pt",
        "run_config": "unet_multiclass_t1/run_config.json",
        "save_json": "evaluation_outputs/multiclass_t1n/test_metrics.json",
    },
    {
        "name": "multiclass_t1c",
        "checkpoint": "unet_multiclass_t1ce/best_model.pt",
        "run_config": "unet_multiclass_t1ce/run_config.json",
        "save_json": "evaluation_outputs/multiclass_t1c/test_metrics.json",
    },
    {
        "name": "regions_t2f_t1c_t2w",
        "checkpoint": "unet_regions/best_model.pt",
        "run_config": "unet_regions/run_config.json",
        "save_json": "evaluation_outputs/regions_t2f_t1c_t2w/test_metrics.json",
    },
    {
        "name": "multiclass_t2f_t1c_t2w",
        "checkpoint": "unet_multiclass_t2f_t1c_t2w/best_model.pt",
        "run_config": "unet_multiclass_t2f_t1c_t2w/run_config.json",
        "save_json": "evaluation_outputs/multiclass_t2f_t1c_t2w/test_metrics.json",
    },
]

REGION_LEGEND = [
    ("WT", LABEL_COLORS[0] / 255.0),
    ("TC", LABEL_COLORS[1] / 255.0),
    ("ET", LABEL_COLORS[2] / 255.0),
]

MULTICLASS_LEGEND = [
    ("NCR/NET (1)", MULTICLASS_COLORS[0] / 255.0),
    ("ED (2)", MULTICLASS_COLORS[1] / 255.0),
    ("ET (3)", MULTICLASS_COLORS[2] / 255.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-set evaluation for a trained UNet checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pt")
    parser.add_argument(
        "--run-config",
        type=str,
        default=None,
        help="Optional path to run_config.json. Defaults to checkpoint sibling run_config.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save test metrics as JSON",
    )
    parser.add_argument(
        "--patient-ids-file",
        type=str,
        default=None,
        help="Optional text file with one patient ID per line for extra visualizations.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run evaluation/visualization for all known UNet checkpoints.",
    )
    parser.add_argument(
        "--visuals-only",
        action="store_true",
        help="Only render selected-patient visualizations; do not rerun test metrics or overwrite test_metrics.json.",
    )
    return parser.parse_args()


def load_run_config(run_config_path: Path) -> Dict:
    with run_config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --all-models is used.")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.run_config is None:
        run_config_path = checkpoint_path.with_name("run_config.json")
    else:
        run_config_path = Path(args.run_config)

    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json not found: {run_config_path}")

    return checkpoint_path, run_config_path


def build_test_loader(run_cfg: Dict, batch_size: int, num_workers: int) -> DataLoader:
    args_cfg = run_cfg["args"]
    datasets = build_patient_split_datasets(
        root=args_cfg["data_root"],
        modalities=args_cfg["modalities"],
        target_mode=args_cfg["target_mode"],
        ratios=(0.7, 0.1, 0.2),
        seed=args_cfg["seed"],
        include_empty_slices=args_cfg.get("include_empty_slices", False),
        remove_black_slices=not args_cfg.get("keep_black_slices", False),
        train_transform=None,
        eval_transform=None,
    )
    test_dataset = datasets["test"]
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=num_workers > 0,
    )


def multiclass_to_three_region_mask(mask: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            (mask > 0).astype(np.float32),
            np.logical_or(mask == 1, mask == 3).astype(np.float32),
            (mask == 3).astype(np.float32),
        ],
        axis=0,
    )


def regions_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
    for channel_idx, color in enumerate(LABEL_COLORS):
        rgb[mask[channel_idx] > 0.5] = np.maximum(rgb[mask[channel_idx] > 0.5], color)
    return rgb


def multiclass_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_idx, color in enumerate(MULTICLASS_COLORS, start=1):
        rgb[mask == label_idx] = color
    return rgb


def normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = image - image.min()
    max_value = image.max()
    if max_value > 0:
        image = image / max_value
    return image


def get_three_region_target(mask: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "regions":
        return mask.astype(np.float32)
    return multiclass_to_three_region_mask(mask)


def extract_prediction(outputs: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "regions":
        return (torch.sigmoid(outputs) > 0.5).float()
    return outputs.argmax(dim=1)


def load_selected_patient_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_best_slice_indices(dataset: Processed2DSegmentationDataset, target_mode: str) -> list[int]:
    best_by_patient: dict[str, tuple[int, float]] = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        patient_id = str(item["patient_id"])
        mask_np = item["mask"].cpu().numpy() if torch.is_tensor(item["mask"]) else np.asarray(item["mask"])
        region_mask = get_three_region_target(mask_np, target_mode=target_mode)
        tumor_pixels = float(region_mask.sum())
        current = best_by_patient.get(patient_id)
        if current is None or tumor_pixels > current[1]:
            best_by_patient[patient_id] = (idx, tumor_pixels)
    return [idx for idx, _ in best_by_patient.values()]


def save_visualization(
    model: torch.nn.Module,
    dataset: Processed2DSegmentationDataset,
    sample_idx: int,
    target_mode: str,
    modalities: list[str],
    device: torch.device,
    amp_enabled: bool,
    output_path: Path,
) -> dict[str, str | int]:
    item = dataset[sample_idx]
    image = item["image"].unsqueeze(0).to(device)
    target = item["mask"]

    with torch.no_grad():
        with autocast(enabled=amp_enabled):
            outputs = model(image)

    main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
    prediction = extract_prediction(main_outputs, target_mode=target_mode)

    image_np = item["image"].cpu().numpy()
    target_np = target.cpu().numpy()
    prediction_np = prediction.squeeze(0).detach().cpu().numpy()

    base_image = normalize_slice(image_np[0])
    image_title = f"Image ({modalities[0]})" if modalities else "Image"

    if target_mode == "multiclass":
        target_rgb = multiclass_to_rgb(target_np)
        pred_rgb = multiclass_to_rgb(prediction_np)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        for ax, (title, panel_image) in zip(
            axes,
            [
                (image_title, base_image),
                ("Ground Truth", target_rgb),
                ("Prediction", pred_rgb),
            ],
        ):
            if panel_image.ndim == 2:
                ax.imshow(panel_image, cmap="gray")
            else:
                ax.imshow(panel_image)
            ax.set_title(title, fontsize=14)
            ax.axis("off")

        legend_handles = [Patch(color=color, label=label) for label, color in MULTICLASS_LEGEND]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
        fig.suptitle(
            f"patient={item['patient_id']} slice={item['slice_idx']}",
            fontsize=16,
        )
    else:
        target_regions = get_three_region_target(target_np, target_mode=target_mode)
        pred_regions = get_three_region_target(prediction_np, target_mode=target_mode)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        region_names = ("WT", "TC", "ET")
        region_colors = [LABEL_COLORS[i] / 255.0 for i in range(3)]
        for col, (region_name, region_color) in enumerate(zip(region_names, region_colors)):
            for row, (title_prefix, region_mask) in enumerate(
                (
                    ("Ground Truth", target_regions[col]),
                    ("Prediction", pred_regions[col]),
                )
            ):
                ax = axes[row, col]
                ax.imshow(base_image, cmap="gray")
                overlay = np.zeros((*region_mask.shape, 4), dtype=np.float32)
                overlay[..., :3] = region_color
                overlay[..., 3] = (region_mask > 0.5).astype(np.float32) * 0.65
                ax.imshow(overlay)
                ax.set_title(f"{title_prefix} {region_name}", fontsize=13)
                ax.axis("off")

        legend_handles = [Patch(color=color, label=label) for label, color in REGION_LEGEND]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
        fig.suptitle(
            f"patient={item['patient_id']} slice={item['slice_idx']}",
            fontsize=16,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "patient_id": str(item["patient_id"]),
        "slice_idx": int(item["slice_idx"]),
        "visualization": str(output_path),
    }


def render_selected_patient_visuals(
    model: torch.nn.Module,
    run_cfg: Dict,
    device: torch.device,
    amp_enabled: bool,
    output_root: Path,
    patient_ids_file: str | Path,
) -> list[dict[str, str | int]]:
    args_cfg = run_cfg["args"]
    patient_ids = load_selected_patient_ids(patient_ids_file)
    sample_paths = collect_sample_paths_for_patients(args_cfg["data_root"], patient_ids)
    dataset = Processed2DSegmentationDataset(
        root=args_cfg["data_root"],
        modalities=args_cfg["modalities"],
        target_mode=args_cfg["target_mode"],
        include_empty_slices=args_cfg.get("include_empty_slices", False),
        remove_black_slices=not args_cfg.get("keep_black_slices", False),
        sample_paths=sample_paths,
        transform=None,
    )
    selected_indices = select_best_slice_indices(dataset, target_mode=args_cfg["target_mode"])
    visuals_dir = output_root / "visuals_selected_patients"
    visuals = []
    for sample_idx in tqdm(selected_indices, desc="Rendering selected visuals", unit="patient"):
        item = dataset[sample_idx]
        output_path = visuals_dir / f"{item['patient_id']}_slice_{int(item['slice_idx']):03d}.png"
        visuals.append(
            save_visualization(
                model=model,
                dataset=dataset,
                sample_idx=sample_idx,
                target_mode=args_cfg["target_mode"],
                modalities=args_cfg["modalities"],
                device=device,
                amp_enabled=amp_enabled,
                output_path=output_path,
            )
        )
    return visuals


def run_single_evaluation(
    checkpoint_path: Path,
    run_config_path: Path,
    save_json_path: Path | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    run_cfg = load_run_config(run_config_path)

    model_cfg_dict = run_cfg["config"]["model"]
    loss_cfg_dict = run_cfg["config"]["loss"]
    args_cfg = run_cfg["args"]

    device_str = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)
    amp_enabled = (not args.disable_amp) and device.type == "cuda"

    model = build_model(type("ModelConfigShim", (), model_cfg_dict)()).to(device)
    criterion = None if args.visuals_only else build_loss(type("LossConfigShim", (), loss_cfg_dict)())

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    summary: dict[str, object]
    if args.visuals_only:
        summary = {
            "checkpoint": str(checkpoint_path),
            "run_config": str(run_config_path),
            "target_mode": args_cfg["target_mode"],
            "modalities": args_cfg["modalities"],
        }
        if save_json_path is not None and save_json_path.exists():
            with save_json_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                summary = loaded
    else:
        test_loader = build_test_loader(
            run_cfg=run_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        running_loss = 0.0
        region_stats = {
            "wt": {"intersection": 0.0, "denominator": 0.0},
            "tc": {"intersection": 0.0, "denominator": 0.0},
            "et": {"intersection": 0.0, "denominator": 0.0},
        }

        progress = tqdm(test_loader, desc=f"Testing {checkpoint_path.parent.name}", unit="batch")
        with torch.no_grad():
            for batch in progress:
                batch = move_batch_to_device(batch, device)
                images = batch["image"]
                target = batch["mask"]

                with autocast(enabled=amp_enabled):
                    outputs = model(images)
                    loss = compute_loss(criterion, outputs, target)

                running_loss += loss.item()
                main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                batch_region_stats = compute_region_dice_stats(main_outputs, target)
                for region_name in ("wt", "tc", "et"):
                    region_stats[region_name]["intersection"] += batch_region_stats[region_name]["intersection"]
                    region_stats[region_name]["denominator"] += batch_region_stats[region_name]["denominator"]

                progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(len(test_loader), 1)
        metrics = finalize_region_dice(region_stats)
        summary = {
            "checkpoint": str(checkpoint_path),
            "run_config": str(run_config_path),
            "target_mode": args_cfg["target_mode"],
            "modalities": args_cfg["modalities"],
            "test_loss": mean_loss,
            "test_dice_wt": metrics["dice_wt"],
            "test_dice_tc": metrics["dice_tc"],
            "test_dice_et": metrics["dice_et"],
            "test_dice_mean": metrics["dice_mean"],
        }

    if args.patient_ids_file is not None:
        visuals_root = save_json_path.parent if save_json_path is not None else checkpoint_path.parent
        selected_visuals = render_selected_patient_visuals(
            model=model,
            run_cfg=run_cfg,
            device=device,
            amp_enabled=amp_enabled,
            output_root=visuals_root,
            patient_ids_file=args.patient_ids_file,
        )
        manifest_path = visuals_root / "visuals_selected_patients" / "selected_visuals_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(selected_visuals, f, indent=2)
        summary["selected_visuals_manifest"] = str(manifest_path)

    print(json.dumps(summary, indent=2))

    if (not args.visuals_only) and save_json_path is not None:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with save_json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    if args.all_models:
        all_results = []
        for spec in MODEL_SPECS:
            checkpoint_path = Path(spec["checkpoint"])
            run_config_path = Path(spec["run_config"])
            save_json_path = Path(spec["save_json"])
            if not checkpoint_path.exists() or not run_config_path.exists():
                continue
            all_results.append(
                run_single_evaluation(
                    checkpoint_path=checkpoint_path,
                    run_config_path=run_config_path,
                    save_json_path=save_json_path,
                    args=args,
                )
            )
        print(json.dumps(all_results, indent=2))
        return

    checkpoint_path, run_config_path = resolve_paths(args)
    save_json_path = Path(args.save_json) if args.save_json is not None else None
    run_single_evaluation(
        checkpoint_path=checkpoint_path,
        run_config_path=run_config_path,
        save_json_path=save_json_path,
        args=args,
    )


if __name__ == "__main__":
    main()
