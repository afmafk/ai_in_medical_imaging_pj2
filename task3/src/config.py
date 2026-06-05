from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "task3_compare.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("data_root", "metadata_path", "splits_path"):
        if key in cfg and cfg[key]:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str((PROJECT_ROOT / p).resolve())
            else:
                cfg[key] = str(p.resolve())
    return cfg
