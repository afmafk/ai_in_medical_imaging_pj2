from __future__ import annotations

import json
from pathlib import Path


class EarlyStopping:
    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def step(self, score: float, epoch: int) -> bool:
        """Returns True if this is a new best score."""
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            self.stopped_epoch = epoch
        return False

    def state_dict(self) -> dict:
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "stopped_epoch": self.stopped_epoch,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)
