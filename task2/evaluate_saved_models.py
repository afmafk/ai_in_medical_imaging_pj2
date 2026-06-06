from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from builders import build_model
from data import build_patient_split_datasets
from engine import compute_loss, compute_region_dice_stats, finalize_region_dice, move_batch_to_device
from losses import build_loss


MODEL_SPECS = [
    {
        "name": "multiclass_t2w_t2f",
        "checkpoint": "unet_multiclass/best_model.pt",
        "run_config": "unet_multiclass/run_config.json",
    },
    {
        "name": "multiclass_all_modalities",
        "checkpoint": "unet_multiclass_all_modalities/best_model.pt",
        "run_config": "unet_multiclass_all_modalities/run_config.json",
    },
    {
        "name": "multiclass_t1n",
        "checkpoint": "unet_multiclass_t1/best_model.pt",
        "run_config": "unet_multiclass_t1/run_config.json",
    },
    {
        "name": "multiclass_t1c",
        "checkpoint": "unet_multiclass_t1ce/best_model.pt",
        "run_config": "unet_multiclass_t1ce/run_config.json",
    },
    {
        "name": "regions_t2f_t1c_t2w",
        "checkpoint": "unet_regions/best_model.pt",
        "run_config": "unet_regions/run_config.json",
    },
]


LABEL_COLORS = np.array(
    [
        [255, 51, 51],
        [51, 217, 89],
        [255, 217, 26],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved BraTS models and export visualizations.")
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--num-visuals", type=int, default=3)
    parser.add_argument("--visuals-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=None,
        help="Optional model name to evaluate. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def log_step(message: str) -> None:
    tqdm.write(message)


def build_dataset_and_loader(run_cfg: dict[str, Any], batch_size: int, num_workers: int) -> tuple[Any, DataLoader]:
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return test_dataset, test_loader


def build_model_and_loss(run_cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    model_cfg = run_cfg["config"]["model"]
    loss_cfg = run_cfg["config"]["loss"]
    model = build_model(type("ModelConfigShim", (), model_cfg)()).to(device)
    criterion = build_loss(type("LossConfigShim", (), loss_cfg)())

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, criterion


def extract_prediction(outputs: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "regions":
        return (torch.sigmoid(outputs) > 0.5).float()
    return outputs.argmax(dim=1)


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


def normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = image - image.min()
    max_value = image.max()
    if max_value > 0:
        image = image / max_value
    return image


def get_dataset_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item["patient_id"]), int(item["slice_idx"])


def get_three_region_target(mask: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "regions":
        mask = mask.astype(np.float32)
        if mask.ndim == 3 and mask.shape[0] == 3:
            return mask
        if mask.ndim == 4 and mask.shape[1] == 3:
            return np.transpose(mask, (1, 0, 2, 3))
        if mask.ndim == 4 and mask.shape[0] == 3:
            return mask
        raise ValueError(f"Unsupported regions mask shape: {mask.shape}")
    return multiclass_to_three_region_mask(mask)


def build_index_lookup(dataset: Any) -> dict[tuple[str, int], int]:
    lookup: dict[tuple[str, int], int] = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        lookup[get_dataset_key(item)] = idx
    return lookup


def empty_region_stats() -> dict[str, dict[str, float]]:
    return {
        "wt": {"intersection": 0.0, "denominator": 0.0},
        "tc": {"intersection": 0.0, "denominator": 0.0},
        "et": {"intersection": 0.0, "denominator": 0.0},
    }


def compute_mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def hd95_binary(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    if not pred_mask.any() and not target_mask.any():
        return 0.0
    if not pred_mask.any() or not target_mask.any():
        return float(np.linalg.norm(np.asarray(pred_mask.shape, dtype=np.float64)))

    footprint = generate_binary_structure(pred_mask.ndim, 1)
    pred_surface = np.logical_xor(pred_mask, binary_erosion(pred_mask, structure=footprint, border_value=0))
    target_surface = np.logical_xor(target_mask, binary_erosion(target_mask, structure=footprint, border_value=0))

    pred_to_target = distance_transform_edt(~target_surface)[pred_surface]
    target_to_pred = distance_transform_edt(~pred_surface)[target_surface]
    distances = np.concatenate([pred_to_target, target_to_pred]).astype(np.float64)
    return float(np.percentile(distances, 95))


def compute_patient_hd95(pred_volume: np.ndarray, target_volume: np.ndarray, target_mode: str) -> dict[str, float]:
    pred_regions = get_three_region_target(pred_volume, target_mode)
    target_regions = get_three_region_target(target_volume, target_mode)

    hd95_wt = hd95_binary(pred_regions[0], target_regions[0])
    hd95_tc = hd95_binary(pred_regions[1], target_regions[1])
    hd95_et = hd95_binary(pred_regions[2], target_regions[2])
    return {
        "hd95_wt": hd95_wt,
        "hd95_tc": hd95_tc,
        "hd95_et": hd95_et,
        "hd95_mean": float(np.mean([hd95_wt, hd95_tc, hd95_et])),
    }


def select_shared_visual_keys(
    dataset_infos: list[dict[str, Any]],
    num_visuals: int,
) -> list[tuple[str, int]]:
    common_keys: set[tuple[str, int]] | None = None
    for info in dataset_infos:
        keys = set(info["index_lookup"].keys())
        common_keys = keys if common_keys is None else (common_keys & keys)

    if not common_keys:
        return []

    scored_keys: list[tuple[float, tuple[str, int]]] = []
    reference_info = dataset_infos[0]
    reference_dataset = reference_info["dataset"]
    reference_mode = reference_info["target_mode"]

    scoring_progress = tqdm(
        sorted(common_keys),
        desc="Scoring shared slices",
        unit="slice",
    )
    for key in scoring_progress:
        item = reference_dataset[reference_info["index_lookup"][key]]
        mask_np = item["mask"].cpu().numpy() if torch.is_tensor(item["mask"]) else np.asarray(item["mask"])
        region_mask = get_three_region_target(mask_np, reference_mode)
        score = float(region_mask.sum())
        scored_keys.append((score, key))

    scored_keys.sort(key=lambda x: (x[0], x[1][0], x[1][1]), reverse=True)
    selected = [key for score, key in scored_keys if score > 0][:num_visuals]
    if len(selected) < num_visuals:
        filler = [key for _, key in scored_keys if key not in selected]
        selected.extend(filler[: max(0, num_visuals - len(selected))])
    return selected


def save_visualization(
    model: torch.nn.Module,
    dataset: Any,
    sample_idx: int,
    target_mode: str,
    modalities: list[str],
    device: torch.device,
    amp_enabled: bool,
    output_path: Path,
) -> dict[str, Any]:
    item = dataset[sample_idx]
    image = item["image"].unsqueeze(0).to(device)
    target = item["mask"]

    with torch.no_grad():
        with autocast(enabled=amp_enabled and device.type == "cuda"):
            outputs = model(image)

    main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
    prediction = extract_prediction(main_outputs, target_mode=target_mode)

    image_np = item["image"].cpu().numpy()
    target_np = target.cpu().numpy()
    prediction_np = prediction.squeeze(0).detach().cpu().numpy()

    # Keep the source image grayscale; for multimodal inputs we show the first requested modality.
    base_image = normalize_slice(image_np[0])
    image_title = f"Image ({modalities[0]})" if modalities else "Image"

    if target_mode == "regions":
        target_region_mask = target_np.astype(np.float32)
        pred_region_mask = prediction_np.astype(np.float32)
    else:
        target_region_mask = multiclass_to_three_region_mask(target_np)
        pred_region_mask = multiclass_to_three_region_mask(prediction_np)

    target_rgb = regions_to_rgb(target_region_mask)
    pred_rgb = regions_to_rgb(pred_region_mask)
    tumor_area = float(target_region_mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    panels = [
        (image_title, base_image),
        ("Ground Truth", target_rgb),
        ("Prediction", pred_rgb),
    ]

    for ax, (title, panel_image) in zip(axes, panels):
        if panel_image.ndim == 2:
            ax.imshow(panel_image, cmap="gray")
        else:
            ax.imshow(panel_image)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        rect = patches.Rectangle((0, 0), panel_image.shape[1] - 1, panel_image.shape[0] - 1, linewidth=1.5, edgecolor="white", facecolor="none")
        ax.add_patch(rect)

    fig.suptitle(
        f"patient={item['patient_id']} slice={item['slice_idx']} tumor_pixels={int(tumor_area)}",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "sample_index": sample_idx,
        "patient_id": item["patient_id"],
        "slice_idx": int(item["slice_idx"]),
        "path": item["path"],
        "tumor_pixels": int(tumor_area),
        "visualization": str(output_path),
    }


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    run_config_path: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    num_visuals: int,
    run_full_eval: bool,
    shared_visual_keys: list[tuple[str, int]],
) -> dict[str, Any]:
    log_step(f"[{model_name}] Loading run config")
    run_cfg = load_json(run_config_path)
    log_step(f"[{model_name}] Building test dataset/loader")
    test_dataset, test_loader = build_dataset_and_loader(run_cfg, batch_size=batch_size, num_workers=num_workers)
    log_step(f"[{model_name}] Loading checkpoint")
    model, criterion = build_model_and_loss(run_cfg, checkpoint_path=checkpoint_path, device=device)

    mean_loss = None
    metrics = {
        "dice_wt": None,
        "dice_tc": None,
        "dice_et": None,
        "dice_mean": None,
    }
    if run_full_eval:
        running_loss = 0.0
        region_stats = empty_region_stats()
        patient_region_stats: dict[str, dict[str, dict[str, float]]] = {}
        patient_predictions: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}

        with torch.no_grad():
            eval_progress = tqdm(
                test_loader,
                desc=f"[{model_name}] test",
                unit="batch",
            )
            for batch in eval_progress:
                batch = move_batch_to_device(batch, device)
                images = batch["image"]
                target = batch["mask"]

                with autocast(enabled=amp_enabled and device.type == "cuda"):
                    outputs = model(images)
                    loss = compute_loss(criterion, outputs, target)

                running_loss += loss.item()
                main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                batch_region_stats = compute_region_dice_stats(main_outputs, target)
                for region_name in ("wt", "tc", "et"):
                    region_stats[region_name]["intersection"] += batch_region_stats[region_name]["intersection"]
                    region_stats[region_name]["denominator"] += batch_region_stats[region_name]["denominator"]

                patient_ids = batch["patient_id"]
                if isinstance(patient_ids, str):
                    patient_ids = [patient_ids]
                for sample_idx, patient_id in enumerate(patient_ids):
                    if patient_id not in patient_region_stats:
                        patient_region_stats[patient_id] = empty_region_stats()
                    sample_region_stats = compute_region_dice_stats(
                        main_outputs[sample_idx : sample_idx + 1],
                        target[sample_idx : sample_idx + 1],
                    )
                    for region_name in ("wt", "tc", "et"):
                        patient_region_stats[patient_id][region_name]["intersection"] += sample_region_stats[region_name]["intersection"]
                        patient_region_stats[patient_id][region_name]["denominator"] += sample_region_stats[region_name]["denominator"]

                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = []
                    sample_pred = extract_prediction(
                        main_outputs[sample_idx : sample_idx + 1],
                        target_mode=run_cfg["args"]["target_mode"],
                    )[0].detach().cpu().numpy()
                    sample_target = target[sample_idx].detach().cpu().numpy()
                    sample_slice_idx = int(batch["slice_idx"][sample_idx])
                    patient_predictions[patient_id].append((sample_slice_idx, sample_pred, sample_target))
                eval_progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(len(test_loader), 1)
        metrics = finalize_region_dice(region_stats)
        log_step(f"[{model_name}] Aggregating patient Dice")
        patient_metrics = {
            patient_id: finalize_region_dice(stats)
            for patient_id, stats in patient_region_stats.items()
        }
        patient_hd95 = {}
        hd95_progress = tqdm(
            sorted(patient_predictions.items(), key=lambda x: x[0]),
            desc=f"[{model_name}] hd95",
            unit="patient",
        )
        for patient_id, samples in hd95_progress:
            samples.sort(key=lambda x: x[0])
            pred_volume = np.stack([sample[1] for sample in samples], axis=0)
            target_volume = np.stack([sample[2] for sample in samples], axis=0)
            patient_hd95[patient_id] = compute_patient_hd95(
                pred_volume=pred_volume,
                target_volume=target_volume,
                target_mode=run_cfg["args"]["target_mode"],
            )
        patient_dice_summary = {
            "dice_wt": compute_mean_std([metric["dice_wt"] for metric in patient_metrics.values()]),
            "dice_tc": compute_mean_std([metric["dice_tc"] for metric in patient_metrics.values()]),
            "dice_et": compute_mean_std([metric["dice_et"] for metric in patient_metrics.values()]),
            "dice_mean": compute_mean_std([metric["dice_mean"] for metric in patient_metrics.values()]),
        }
        patient_hd95_summary = {
            "hd95_wt": compute_mean_std([metric["hd95_wt"] for metric in patient_hd95.values()]),
            "hd95_tc": compute_mean_std([metric["hd95_tc"] for metric in patient_hd95.values()]),
            "hd95_et": compute_mean_std([metric["hd95_et"] for metric in patient_hd95.values()]),
            "hd95_mean": compute_mean_std([metric["hd95_mean"] for metric in patient_hd95.values()]),
        }
    else:
        patient_metrics = {}
        patient_hd95 = {}
        patient_dice_summary = {
            "dice_wt": {"mean": None, "std": None},
            "dice_tc": {"mean": None, "std": None},
            "dice_et": {"mean": None, "std": None},
            "dice_mean": {"mean": None, "std": None},
        }
        patient_hd95_summary = {
            "hd95_wt": {"mean": None, "std": None},
            "hd95_tc": {"mean": None, "std": None},
            "hd95_et": {"mean": None, "std": None},
            "hd95_mean": {"mean": None, "std": None},
        }

    index_lookup = build_index_lookup(test_dataset)
    selected_indices = [index_lookup[key] for key in shared_visual_keys if key in index_lookup][:num_visuals]
    visuals = []
    visuals_dir = output_dir / model_name / "visuals"
    if num_visuals > 0:
        log_step(f"[{model_name}] Rendering visualizations")
        visual_progress = tqdm(
            list(enumerate(selected_indices, start=1)),
            desc=f"[{model_name}] visuals",
            unit="slice",
        )
        for rank, sample_idx in visual_progress:
            output_path = visuals_dir / f"slice_{rank:02d}.png"
            visuals.append(
                save_visualization(
                    model=model,
                    dataset=test_dataset,
                    sample_idx=sample_idx,
                    target_mode=run_cfg["args"]["target_mode"],
                    modalities=run_cfg["args"]["modalities"],
                    device=device,
                    amp_enabled=amp_enabled,
                    output_path=output_path,
                )
            )

    summary = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "run_config": str(run_config_path),
        "target_mode": run_cfg["args"]["target_mode"],
        "modalities": run_cfg["args"]["modalities"],
        "test_loss": mean_loss,
        "test_dice_wt": patient_dice_summary["dice_wt"]["mean"],
        "test_dice_tc": patient_dice_summary["dice_tc"]["mean"],
        "test_dice_et": patient_dice_summary["dice_et"]["mean"],
        "test_dice_mean": patient_dice_summary["dice_mean"]["mean"],
        "global_micro_dice_wt": metrics["dice_wt"],
        "global_micro_dice_tc": metrics["dice_tc"],
        "global_micro_dice_et": metrics["dice_et"],
        "global_micro_dice_mean": metrics["dice_mean"],
        "test_hd95_wt": patient_hd95_summary["hd95_wt"]["mean"],
        "test_hd95_tc": patient_hd95_summary["hd95_tc"]["mean"],
        "test_hd95_et": patient_hd95_summary["hd95_et"]["mean"],
        "test_hd95_mean": patient_hd95_summary["hd95_mean"]["mean"],
        "patient_dice_summary": patient_dice_summary,
        "patient_metrics": patient_metrics,
        "patient_hd95_summary": patient_hd95_summary,
        "patient_hd95": patient_hd95,
        "visuals": visuals,
    }

    summary_path = output_dir / model_name / "test_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_step(f"[{model_name}] Writing {summary_path}")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    if args.visuals_only and args.metrics_only:
        raise ValueError("--visuals-only and --metrics-only cannot be used together.")

    output_dir = Path(args.output_dir)
    device_str = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)
    amp_enabled = (not args.disable_amp) and device.type == "cuda"
    selected_model_names = set(args.model) if args.model else None
    selected_specs = [
        spec for spec in MODEL_SPECS
        if selected_model_names is None or spec["name"] in selected_model_names
    ]
    if not selected_specs:
        available = ", ".join(spec["name"] for spec in MODEL_SPECS)
        raise ValueError(f"No matching models selected. Available models: {available}")

    dataset_infos: list[dict[str, Any]] = []
    shared_visual_keys: list[tuple[str, int]] = []
    if not args.metrics_only:
        setup_progress = tqdm(selected_specs, desc="Preparing datasets", unit="model", leave=False)
        for spec in setup_progress:
            checkpoint_path = Path(spec["checkpoint"])
            run_config_path = Path(spec["run_config"])
            if not checkpoint_path.exists() or not run_config_path.exists():
                continue
            setup_progress.set_postfix(current=spec["name"])
            run_cfg = load_json(run_config_path)
            test_dataset, _ = build_dataset_and_loader(run_cfg, batch_size=1, num_workers=0)
            dataset_infos.append(
                {
                    "name": spec["name"],
                    "dataset": test_dataset,
                    "target_mode": run_cfg["args"]["target_mode"],
                    "index_lookup": build_index_lookup(test_dataset),
                }
            )

        log_step("Selecting shared slices for visualization")
        shared_visual_keys = select_shared_visual_keys(dataset_infos, num_visuals=args.num_visuals)

    all_results = []
    model_progress = tqdm(selected_specs, desc="Evaluating models", unit="model")
    for spec in model_progress:
        checkpoint_path = Path(spec["checkpoint"])
        run_config_path = Path(spec["run_config"])
        if not checkpoint_path.exists() or not run_config_path.exists():
            continue
        model_progress.set_postfix(current=spec["name"])

        result = evaluate_model(
            model_name=spec["name"],
            checkpoint_path=checkpoint_path,
            run_config_path=run_config_path,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp_enabled=amp_enabled,
            num_visuals=0 if args.metrics_only else args.num_visuals,
            run_full_eval=not args.visuals_only,
            shared_visual_keys=shared_visual_keys,
        )
        all_results.append(result)

    summary_path = output_dir / "all_model_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Writing {summary_path}")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
