from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

try:
    from .builders import build_model, build_optimizer, build_scheduler
    from .config import TrainConfig
    from .losses import build_loss
except ImportError:
    from builders import build_model, build_optimizer, build_scheduler
    from config import TrainConfig
    from losses import build_loss


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def compute_loss(
    criterion: nn.Module,
    outputs: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
    target: torch.Tensor,
) -> torch.Tensor:
    if isinstance(outputs, tuple):
        main_logits, aux_logits = outputs
        loss = criterion(main_logits, target)
        if aux_logits:
            aux_weight = 0.4 / len(aux_logits)
            for logits in aux_logits:
                upsampled = nn.functional.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = loss + aux_weight * criterion(upsampled, target)
        return loss
    return criterion(outputs, target)


def _multiclass_labels_to_regions(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    if labels.ndim == 4 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    if labels.ndim != 3:
        raise ValueError(f"Expected multiclass labels with shape [B, H, W], got {tuple(labels.shape)}")
    return {
        "wt": (labels > 0),
        "tc": ((labels == 1) | (labels == 3)),
        "et": (labels == 3),
    }


def _binary_channels_to_regions(channels: torch.Tensor) -> Dict[str, torch.Tensor]:
    if channels.ndim != 4 or channels.size(1) != 3:
        raise ValueError(f"Expected region channels with shape [B, 3, H, W], got {tuple(channels.shape)}")
    return {
        "wt": channels[:, 0].bool(),
        "tc": channels[:, 1].bool(),
        "et": channels[:, 2].bool(),
    }


def _extract_region_predictions_and_targets(
    outputs: torch.Tensor,
    target: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if target.ndim == 4 and target.size(1) == 3:
        pred_regions = _binary_channels_to_regions(torch.sigmoid(outputs) > 0.5)
        target_regions = _binary_channels_to_regions(target > 0.5)
        return pred_regions, target_regions

    pred_labels = outputs.argmax(dim=1)
    if target.ndim == 4 and target.size(1) == outputs.size(1):
        target = target.argmax(dim=1)
    elif target.ndim == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    pred_regions = _multiclass_labels_to_regions(pred_labels)
    target_regions = _multiclass_labels_to_regions(target.long())
    return pred_regions, target_regions


def compute_region_dice_stats(
    outputs: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    pred_regions, target_regions = _extract_region_predictions_and_targets(outputs, target)
    stats: Dict[str, Dict[str, float]] = {}
    for region_name in ("wt", "tc", "et"):
        pred = pred_regions[region_name].float()
        truth = target_regions[region_name].float()
        intersection = torch.sum(pred * truth).item()
        denominator = torch.sum(pred).item() + torch.sum(truth).item()
        stats[region_name] = {
            "intersection": intersection,
            "denominator": denominator,
        }
    return stats


def finalize_region_dice(region_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for region_name in ("wt", "tc", "et"):
        intersection = region_stats[region_name]["intersection"]
        denominator = region_stats[region_name]["denominator"]
        if denominator == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / denominator
        metrics[f"dice_{region_name}"] = dice
    metrics["dice_mean"] = (
        metrics["dice_wt"] + metrics["dice_tc"] + metrics["dice_et"]
    ) / 3.0
    return metrics


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def get_monitor_value(val_loss: float, val_metrics: Dict[str, float], monitor_metric: str) -> float:
    if monitor_metric == "val_loss":
        return val_loss
    if monitor_metric == "val_dice_mean":
        return val_metrics["dice_mean"]
    if monitor_metric == "val_dice_wt":
        return val_metrics["dice_wt"]
    if monitor_metric == "val_dice_tc":
        return val_metrics["dice_tc"]
    if monitor_metric == "val_dice_et":
        return val_metrics["dice_et"]
    raise ValueError(f"Unsupported monitor metric: {monitor_metric}")


def is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < (best - min_delta)
    if mode == "max":
        return current > (best + min_delta)
    raise ValueError(f"Unsupported monitor mode: {mode}")


def append_history_row(history_path: Path, row: Dict[str, float | int]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "val_dice_wt",
        "val_dice_tc",
        "val_dice_et",
        "val_dice_mean",
    ]
    write_header = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool,
    grad_clip_norm: Optional[float] = None,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        images = batch["image"]
        target = batch["mask"]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled and device.type == "cuda"):
            outputs = model(images)
            loss = compute_loss(criterion, outputs, target)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    region_stats = {
        "wt": {"intersection": 0.0, "denominator": 0.0},
        "tc": {"intersection": 0.0, "denominator": 0.0},
        "et": {"intersection": 0.0, "denominator": 0.0},
    }

    for batch in loader:
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

    metrics = finalize_region_dice(region_stats)
    return running_loss / max(len(loader), 1), metrics


def fit(
    train_loader: Iterable[Dict[str, torch.Tensor]],
    val_loader: Iterable[Dict[str, torch.Tensor]],
    config: TrainConfig,
) -> nn.Module:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config.model).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    criterion = build_loss(config.loss)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")
    history_path = save_dir / "history.csv"

    best_monitor_value = float("inf") if config.monitor_mode == "min" else float("-inf")
    epochs_without_improvement = 0
    for epoch in range(config.max_epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            amp_enabled=config.amp,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_loss, val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=config.amp,
        )
        monitor_value = get_monitor_value(val_loss, val_metrics, config.monitor_metric)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_value)
            else:
                scheduler.step()

        lr = get_current_lr(optimizer)
        append_history_row(
            history_path,
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "val_dice_wt": val_metrics["dice_wt"],
                "val_dice_tc": val_metrics["dice_tc"],
                "val_dice_et": val_metrics["dice_et"],
                "val_dice_mean": val_metrics["dice_mean"],
            },
        )

        improved = is_improvement(
            current=monitor_value,
            best=best_monitor_value,
            mode=config.monitor_mode,
            min_delta=config.early_stopping_min_delta,
        )
        if improved:
            best_monitor_value = monitor_value
            epochs_without_improvement = 0
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "config": config,
                    "best_monitor_metric": config.monitor_metric,
                    "best_monitor_mode": config.monitor_mode,
                    "best_monitor_value": best_monitor_value,
                    "best_val_loss": val_loss,
                    "best_val_metrics": val_metrics,
                },
                save_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch [{epoch + 1}/{config.max_epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_dice_wt={val_metrics['dice_wt']:.4f} "
            f"val_dice_tc={val_metrics['dice_tc']:.4f} "
            f"val_dice_et={val_metrics['dice_et']:.4f} "
            f"{config.monitor_metric}={monitor_value:.4f} "
            f"lr={lr:.6g}"
        )

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"Early stopping triggered at epoch {epoch + 1} "
                f"after {epochs_without_improvement} epochs without improvement in "
                f"{config.monitor_metric}."
            )
            break

    return model
