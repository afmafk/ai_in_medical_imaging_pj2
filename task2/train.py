from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AugmentationConfig, LossConfig, ModelConfig, OptimizerConfig, SchedulerConfig, TrainConfig
from data import BraTSTrainAugment, build_patient_split_datasets
from engine import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D BraTS UNet on processed_2d.")
    parser.add_argument("--data-root", type=str, default="processed_2d")
    parser.add_argument("--save-dir", type=str, default="checkpoints/default_run")
    parser.add_argument("--target-mode", type=str, choices=["multiclass", "regions"], default="multiclass")
    parser.add_argument("--modalities", type=str, nargs="+", default=["t2f", "t1c", "t2w"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--include-empty-slices", action="store_true")
    parser.add_argument("--keep-black-slices", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--rotation-degrees", type=float, default=15.0)
    parser.add_argument("--pad-pixels", type=int, default=24)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--vertical-flip-prob", type=float, default=0.5)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--gaussian-noise-prob", type=float, default=0.3)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "cosine", "step", "poly", "plateau"])
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_dice_mean", "val_dice_wt", "val_dice_tc", "val_dice_et"],
    )
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    monitor_mode = "min" if args.monitor_metric == "val_loss" else "max"

    if args.target_mode == "multiclass":
        num_classes = 4
        loss_name = "dice_ce"
    else:
        num_classes = 3
        loss_name = "binary_dice_bce"

    model_config = ModelConfig(
        name="simple_unet",
        input_channels=len(args.modalities),
        num_classes=num_classes,
        base_channels=args.base_channels,
        dropout=args.dropout,
    )
    loss_config = LossConfig(name=loss_name)
    optimizer_config = OptimizerConfig(
        name="adamw",
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler_name = None if args.scheduler == "none" else args.scheduler
    scheduler_config = SchedulerConfig(
        name=scheduler_name,
        mode=monitor_mode,
        t_max=args.epochs,
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        threshold=args.plateau_threshold,
        min_lr=args.min_lr,
    )
    augmentation_config = AugmentationConfig(
        enabled=not args.disable_augmentation,
        pad_pixels=args.pad_pixels,
        rotation_degrees=args.rotation_degrees,
        horizontal_flip_prob=args.horizontal_flip_prob,
        vertical_flip_prob=args.vertical_flip_prob,
        scale_range=(args.scale_min, args.scale_max),
        gaussian_noise_prob=args.gaussian_noise_prob,
        gaussian_noise_std=args.gaussian_noise_std,
    )
    return TrainConfig(
        device=args.device,
        amp=not args.disable_amp,
        max_epochs=args.epochs,
        grad_clip_norm=args.grad_clip_norm,
        monitor_metric=args.monitor_metric,
        monitor_mode=monitor_mode,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        save_dir=args.save_dir,
        model=model_config,
        loss=loss_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        augmentation=augmentation_config,
    )


def build_loaders(
    args: argparse.Namespace,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], dict[str, list[str]]]:
    train_transform = BraTSTrainAugment(config.augmentation) if config.augmentation.enabled else None
    datasets = build_patient_split_datasets(
        root=args.data_root,
        modalities=args.modalities,
        target_mode=args.target_mode,
        ratios=(0.7, 0.1, 0.2),
        seed=args.seed,
        include_empty_slices=args.include_empty_slices,
        remove_black_slices=not args.keep_black_slices,
        train_transform=train_transform,
        eval_transform=None,
    )

    loaders = {}
    for split_name, dataset in datasets.items():
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=(config.device == "cuda"),
            persistent_workers=args.num_workers > 0,
        )

    split_sizes = {split_name: len(dataset) for split_name, dataset in datasets.items()}
    split_patients = {
        split_name: sorted({sample.parent.name for sample in dataset.samples})
        for split_name, dataset in datasets.items()
    }
    return loaders["train"], loaders["val"], loaders["test"], split_sizes, split_patients


def save_run_metadata(
    save_dir: Path,
    args: argparse.Namespace,
    config: TrainConfig,
    split_sizes: dict[str, int],
    split_patients: dict[str, list[str]],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "config": asdict(config),
        "split_sizes": split_sizes,
        "split_patient_counts": {name: len(ids) for name, ids in split_patients.items()},
        "split_patients": split_patients,
    }
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = build_train_config(args)
    save_dir = Path(config.save_dir)

    train_loader, val_loader, test_loader, split_sizes, split_patients = build_loaders(args, config)
    save_run_metadata(save_dir, args, config, split_sizes, split_patients)

    print("Training configuration")
    print(json.dumps(asdict(config), indent=2))
    print("Dataset split sizes:", split_sizes)
    print("Patient counts:", {name: len(ids) for name, ids in split_patients.items()})

    fit(train_loader=train_loader, val_loader=val_loader, config=config)

    # Build test loader eagerly so the split is serialized and visible in logs.
    print(f"Test loader ready with {len(test_loader.dataset)} slices.")


if __name__ == "__main__":
    main()
