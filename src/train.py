from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as ROOT, load_config
from device_utils import configure_cuda, resolve_device
from dataset.brats_multimodal import BraTS3DPatchDataset, load_split_patient_ids
from early_stopping import EarlyStopping
from losses import DiceCELoss, RegionSupervisedDiceCELoss, TAFDiceCELoss
from metrics import compute_region_metrics
from models import MODEL_NAMES, build_model
from sliding_window import evaluate_patients_sliding_window


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_info = get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "reseed"):
        worker_info.dataset.reseed(worker_seed)


def limit_patients(ids: list[str], max_patients: int | None) -> list[str]:
    if max_patients is None:
        return ids
    return ids[: int(max_patients)]


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            logits = model(image)
            m = compute_region_metrics(logits, seg)
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    accum_steps: int,
    channels_last: bool = False,
    max_grad_norm: float = 0.0,
) -> tuple[float, dict[str, float]]:
    model.train()
    running = 0.0
    detail_totals: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        image = batch["image"].to(device, non_blocking=True)
        seg = batch["seg"].to(device, non_blocking=True)
        if channels_last and image.ndim == 5:
            image = image.to(memory_format=torch.channels_last_3d)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model.forward_with_aux(image) if hasattr(model, "forward_with_aux") else model(image)
            loss, details = criterion(outputs, seg)
            loss = loss / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )
                detail_totals["grad_norm"] = detail_totals.get("grad_norm", 0.0) + float(
                    grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item() * accum_steps
        for key, value in details.items():
            detail_totals[key] = detail_totals.get(key, 0.0) + float(value)
    n = max(len(loader), 1)
    return running / n, {key: value / n for key, value in detail_totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="initialize model weights from a checkpoint but start a fresh optimizer and schedule",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu, auto")
    parser.add_argument("--require-gpu", action="store_true", help="fail if CUDA unavailable")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", cfg.get("split_seed", 42)))
    set_global_seed(seed)
    max_patients = args.max_patients or cfg.get("max_patients")

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    splits_path = cfg["splits_path"]
    if not Path(splits_path).exists():
        from make_splits import main as make_splits_main

        print("splits not found, generating...")
        make_splits_main()

    train_ids = limit_patients(load_split_patient_ids(splits_path, "train"), max_patients)
    val_ids = limit_patients(load_split_patient_ids(splits_path, "val"), max_patients)

    patch_size = tuple(cfg.get("patch_size", [96, 96, 96]))
    if args.model == "swinunetr":
        from models.swinunetr_v2 import ensure_patch_size_divisible_by_32

        ensure_patch_size_divisible_by_32(patch_size)
    vol_shape = tuple(cfg.get("volume_shape", [155, 177, 219]))
    ds_kw = dict(
        patch_size=patch_size,
        volume_shape=vol_shape,
        use_full_volume=bool(cfg.get("use_full_volume", False)),
        pad_factor=int(cfg.get("pad_factor", 16)),
        slice_load_threads=int(cfg.get("slice_load_threads", 8)),
        augment_config=cfg.get("augmentation"),
        region_sampling_config=cfg.get("region_balanced_sampling"),
        volume_cache_size=int(cfg.get("volume_cache_size", 0)),
    )
    if ds_kw["use_full_volume"]:
        print(f"dataset: full volume (pad_factor={ds_kw['pad_factor']}), batch_size=1 recommended")
    else:
        print(f"dataset: random {patch_size} patches (train) / center patch (val)")
    train_ds = BraTS3DPatchDataset(
        train_ids,
        cfg["data_root"],
        split="train",
        samples_per_patient=int(cfg.get("train_samples_per_patient", 1)),
        seed=int(cfg.get("split_seed", 42)),
        require_tumor=True,
        **ds_kw,
    )
    val_ds = BraTS3DPatchDataset(
        val_ids,
        cfg["data_root"],
        split="val",
        samples_per_patient=1,
        seed=int(cfg.get("split_seed", 42)),
        **ds_kw,
    )

    device_name = args.device or cfg.get("device", "cuda")
    require_gpu = args.require_gpu or bool(cfg.get("require_gpu", True))
    device = resolve_device(device_name, require_gpu=require_gpu)
    configure_cuda(device)
    pin_memory = device.type == "cuda"

    nw = int(cfg.get("num_workers", 0))
    loader_kw: dict = dict(
        batch_size=int(cfg["batch_size"]),
        num_workers=nw,
        pin_memory=pin_memory,
    )
    if nw > 0:
        loader_kw["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))
        loader_kw["persistent_workers"] = bool(cfg.get("persistent_workers", True))

    train_loader_kw = {**loader_kw, "drop_last": len(train_ds) > int(cfg["batch_size"])}
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
        **train_loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=val_generator,
        **loader_kw,
    )
    if device.type == "cuda":
        print(
            f"loader: batch={cfg['batch_size']} workers={nw} prefetch={loader_kw.get('prefetch_factor', 0)} "
            f"slice_threads={ds_kw['slice_load_threads']}"
        )

    model_cfg = cfg.get(args.model)
    model = build_model(
        args.model,
        in_channels=4,
        num_classes=int(cfg["num_classes"]),
        model_cfg=model_cfg,
    ).to(device)
    use_channels_last = bool(cfg.get("channels_last", False)) and args.model != "swinunetr"
    if use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)
    use_compile = bool(cfg.get("use_compile", False)) and args.model != "swinunetr"
    if use_compile and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled")
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state"])
        print(
            f"initialized model weights from {args.init_checkpoint} "
            f"(source_epoch={init_ckpt.get('epoch', 'unknown')})"
        )
    if args.model == "rsf_transbts":
        criterion = RegionSupervisedDiceCELoss(
            num_classes=int(cfg["num_classes"]),
            ce_weight=float(cfg.get("loss_ce_weight", 1.0)),
            dice_weight=float(cfg.get("loss_dice_weight", 1.0)),
            aux_weight=float(cfg.get("loss_aux_weight", 0.3)),
            nested_weight=float(cfg.get("loss_nested_weight", 0.05)),
        )
    elif args.model == "taf_transbts":
        criterion = TAFDiceCELoss(
            num_classes=int(cfg["num_classes"]),
            ce_weight=float(cfg.get("loss_ce_weight", 1.0)),
            dice_weight=float(cfg.get("loss_dice_weight", 1.0)),
            correlation_weight=float(cfg.get("loss_correlation_weight", 0.05)),
            correlation_start_epoch=int(cfg.get("loss_correlation_start_epoch", 20)),
            correlation_warmup_epochs=int(cfg.get("loss_correlation_warmup_epochs", 10)),
        )
    else:
        criterion = DiceCELoss(
            num_classes=int(cfg["num_classes"]),
            ce_weight=float(cfg.get("loss_ce_weight", 1.0)),
            dice_weight=float(cfg.get("loss_dice_weight", 1.0)),
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg.get("scheduler_factor", 0.5)),
        patience=int(cfg.get("scheduler_patience", 10)),
    )
    early = EarlyStopping(
        patience=int(cfg.get("early_stop_patience", 30)),
        min_delta=float(cfg.get("early_stop_min_delta", 0.001)),
        mode="max",
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    log_path = out_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_correlation_loss",
                "train_weighted_correlation_loss",
                "train_grad_norm",
                "val_dice_wt",
                "val_dice_tc",
                "val_dice_et",
                "val_mean_dice",
                "full_val_dice_wt",
                "full_val_dice_tc",
                "full_val_dice_et",
                "full_val_mean_dice",
                "selection_mean_dice",
                "lr",
                "early_stop_counter",
            ]
        )

    max_epochs = int(args.max_epochs or cfg.get("max_epochs", 300))
    full_val_interval = int(cfg.get("full_val_interval", 0))
    best_path = ckpt_dir / "best.ckpt"

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        train_loss, train_details = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            int(cfg.get("accum_steps", 1)),
            channels_last=use_channels_last,
            max_grad_norm=float(cfg.get("max_grad_norm", 0.0)),
        )
        val_metrics = evaluate(model, val_loader, device)
        quick_val_mean = val_metrics["dice_mean"]

        full_val_metrics: dict[str, float] | None = None
        run_full_val = full_val_interval > 0 and (
            epoch % full_val_interval == 0 or epoch == max_epochs
        )
        if run_full_val:
            full_val_metrics = evaluate_patients_sliding_window(
                model,
                val_ids,
                cfg["data_root"],
                patch_size,
                int(cfg.get("sliding_window_stride", patch_size[0] // 2)),
                int(cfg["num_classes"]),
                device,
            )

        selection_metrics = full_val_metrics or (val_metrics if full_val_interval <= 0 else None)
        is_best = False
        if selection_metrics is not None:
            selection_mean = selection_metrics["dice_mean"]
            scheduler.step(selection_mean)
            is_best = early.step(selection_mean, epoch)
        else:
            selection_mean = None

        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": selection_metrics,
                    "quick_val_metrics": val_metrics,
                    "full_val_metrics": full_val_metrics,
                    "config": cfg,
                    "model_name": args.model,
                },
                best_path,
            )

        lr = optimizer.param_groups[0]["lr"]
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    (
                        ""
                        if "correlation_loss" not in train_details
                        else f"{train_details['correlation_loss']:.6f}"
                    ),
                    (
                        ""
                        if "weighted_correlation_loss" not in train_details
                        else f"{train_details['weighted_correlation_loss']:.6f}"
                    ),
                    "" if "grad_norm" not in train_details else f"{train_details['grad_norm']:.6f}",
                    f"{val_metrics['dice_WT']:.4f}",
                    f"{val_metrics['dice_TC']:.4f}",
                    f"{val_metrics['dice_ET']:.4f}",
                    f"{quick_val_mean:.4f}",
                    "" if full_val_metrics is None else f"{full_val_metrics['dice_WT']:.4f}",
                    "" if full_val_metrics is None else f"{full_val_metrics['dice_TC']:.4f}",
                    "" if full_val_metrics is None else f"{full_val_metrics['dice_ET']:.4f}",
                    "" if full_val_metrics is None else f"{full_val_metrics['dice_mean']:.4f}",
                    "" if selection_mean is None else f"{selection_mean:.4f}",
                    lr,
                    early.counter,
                ]
            )

        print(
            f"epoch {epoch}/{max_epochs} "
            f"loss={train_loss:.4f} quick_val_mean_dice={quick_val_mean:.4f} "
            + (
                ""
                if "correlation_loss" not in train_details
                else (
                    f"correlation_loss={train_details['correlation_loss']:.4f} "
                    f"weighted_correlation_loss={train_details['weighted_correlation_loss']:.4f} "
                    + (
                        ""
                        if "grad_norm" not in train_details
                        else f"grad_norm={train_details['grad_norm']:.3f} "
                    )
                )
            )
            + (
                f"WT={val_metrics['dice_WT']:.3f} TC={val_metrics['dice_TC']:.3f} "
                f"ET={val_metrics['dice_ET']:.3f} "
            )
            + (
                ""
                if full_val_metrics is None
                else (
                    f"full_val_mean_dice={full_val_metrics['dice_mean']:.4f} "
                    f"WT={full_val_metrics['dice_WT']:.3f} TC={full_val_metrics['dice_TC']:.3f} "
                    f"ET={full_val_metrics['dice_ET']:.3f} "
                )
            )
            + f"es={early.counter}/{early.patience} ({time.time()-t0:.0f}s)"
        )

        if early.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    early.save_json(out_dir / "early_stop.json")
    summary = {
        "model": args.model,
        "best_epoch": early.best_epoch,
        "best_val_mean_dice": early.best_score,
        "stopped_epoch": early.stopped_epoch or epoch,
        "checkpoint": str(best_path),
    }
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("done:", summary)


if __name__ == "__main__":
    main()
