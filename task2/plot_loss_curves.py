from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")


EPOCH_PATTERN = re.compile(r"^Epoch \[(?P<epoch>\d+)/\d+\] (?P<body>.+)$")
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
CONFIG_START = "Training configuration"
CONFIG_END_PREFIX = "Dataset split sizes:"
DEFAULT_MONITOR_METRIC = "val_loss"
DEFAULT_MONITOR_MODE = "min"
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4


def parse_config(log_text: str) -> dict:
    lines = log_text.splitlines()
    try:
        start_idx = lines.index(CONFIG_START) + 1
    except ValueError:
        return {}

    json_lines: list[str] = []
    for line in lines[start_idx:]:
        if line.startswith(CONFIG_END_PREFIX):
            break
        json_lines.append(line)

    if not json_lines:
        return {}

    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return {}


def get_monitor_settings(config: dict) -> tuple[str, str, int | None, float]:
    monitor_metric = config.get("monitor_metric", DEFAULT_MONITOR_METRIC)
    monitor_mode = config.get("monitor_mode", DEFAULT_MONITOR_MODE)
    patience = config.get("early_stopping_patience")
    min_delta = config.get("early_stopping_min_delta")

    if min_delta is None:
        min_delta = DEFAULT_EARLY_STOPPING_MIN_DELTA

    return monitor_metric, monitor_mode, (None if patience is None else int(patience)), float(min_delta)


def parse_losses(log_path: Path, drop_nan: bool = False) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    monitor_history: list[tuple[int, float]] = []
    log_text = log_path.read_text(encoding="utf-8")
    config = parse_config(log_text)
    monitor_metric, monitor_mode, patience, min_delta = get_monitor_settings(config)

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        match = EPOCH_PATTERN.match(line)
        if not match:
            continue

        epoch = int(match.group("epoch"))
        metrics = {key: float(value) for key, value in KEY_VALUE_PATTERN.findall(match.group("body"))}
        train_loss = metrics["train_loss"]
        val_loss = metrics["val_loss"]
        monitor_value = metrics.get(monitor_metric)

        if drop_nan and (math.isnan(train_loss) or math.isnan(val_loss)):
            continue

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if monitor_value is not None and not math.isnan(monitor_value):
            monitor_history.append((epoch, monitor_value))

    stop_epoch = infer_early_stop_epoch(
        monitor_history=monitor_history,
        mode=monitor_mode,
        patience=patience,
        min_delta=min_delta,
    )

    if stop_epoch is None:
        return epochs, train_losses, val_losses

    filtered_epochs: list[int] = []
    filtered_train_losses: list[float] = []
    filtered_val_losses: list[float] = []
    for epoch, train_loss, val_loss in zip(epochs, train_losses, val_losses):
        if epoch > stop_epoch:
            break
        filtered_epochs.append(epoch)
        filtered_train_losses.append(train_loss)
        filtered_val_losses.append(val_loss)

    return filtered_epochs, filtered_train_losses, filtered_val_losses


def infer_early_stop_epoch(
    monitor_history: list[tuple[int, float]],
    mode: str,
    patience: int | None,
    min_delta: float,
) -> int | None:
    if not monitor_history or patience is None or patience <= 0:
        return None

    best = float("inf") if mode == "min" else float("-inf")
    epochs_without_improvement = 0

    for epoch, value in monitor_history:
        improved = False
        if mode == "min":
            improved = value < (best - min_delta)
        elif mode == "max":
            improved = value > (best + min_delta)

        if improved:
            best = value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                return epoch

    return None


def main() -> None:
    root = Path(".")
    runs = [
        {
            "title": "Regions | modalities: t2f + t1c + t2w",
            "filename": "loss_curve_regions_t2f_t1c_t2w.png",
            "log_path": root / "3648048.out",
            "drop_nan": True,
        },
        {
            "title": "Multiclass | modalities: t1c",
            "filename": "loss_curve_multiclass_t1c.png",
            "log_path": root / "3648915.out",
            "drop_nan": False,
        },
        {
            "title": "Multiclass | modalities: t1n",
            "filename": "loss_curve_multiclass_t1n.png",
            "log_path": root / "3648916.out",
            "drop_nan": False,
        },
        {
            "title": "Multiclass | modalities: t2f + t1c + t2w",
            "filename": "loss_curve_multiclass_t2f_t1c_t2w.png",
            "log_path": root / "3652106.out",
            "drop_nan": False,
        },
        {
            "title": "Multiclass | modalities: t1n + t1c + t2w + t2f",
            "filename": "loss_curve_multiclass_all_modalities.png",
            "log_path": root / "3648917.out",
            "drop_nan": False,
        },
    ]

    for run in runs:
        epochs, train_losses, val_losses = parse_losses(run["log_path"], drop_nan=run["drop_nan"])
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(epochs, train_losses, label="train_loss", linewidth=2)
        ax.plot(epochs, val_losses, label="val_loss", linewidth=2)
        ax.set_title(run["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / run["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
