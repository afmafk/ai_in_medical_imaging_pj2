from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as ROOT, load_config

DEFAULT_MODELS = ("multimodal_unet", "attention_unet", "transbts", "swinunetr")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--hd95", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="models to evaluate (default: all three)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    models = tuple(args.models) if args.models else DEFAULT_MODELS
    out_dir = ROOT / "outputs" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    default_config = str(ROOT / "configs" / "task3_compare.yaml")
    for model in models:
        config_path = args.config
        if config_path is None:
            config_map = {
                "swinunetr": ROOT / "configs" / "task3_swinunetr.yaml",
                "transbts_mm_msca_af": ROOT / "configs" / "task3_transbts_mm_msca_af.yaml",
                "transbts_tri_attention": ROOT / "configs" / "task3_transbts_tri_attention.yaml",
            }
            config_path = str(config_map.get(model, default_config))
        config_path = config_path or default_config

        if model == "multimodal_unet":
            cmd = [
                sys.executable,
                str(ROOT / "src" / "evaluate_2d.py"),
                "--split",
                args.split,
                "--config",
                str(ROOT / "configs" / "task2_multimodal_unet.yaml"),
            ]
        else:
            cmd = [
                sys.executable,
                str(ROOT / "src" / "evaluate.py"),
                "--model",
                model,
                "--split",
                args.split,
                "--config",
                config_path,
            ]
        if args.hd95:
            cmd.append("--hd95")
        if args.max_patients:
            cmd.extend(["--max-patients", str(args.max_patients)])
        subprocess.run(cmd, check=True)

        metrics_path = ROOT / "outputs" / model / f"metrics_{args.split}.json"
        with metrics_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "model": model,
            "dice_wt": m.get("dice_WT"),
            "dice_tc": m.get("dice_TC"),
            "dice_et": m.get("dice_ET"),
            "dice_mean": m.get("dice_mean"),
        }
        if args.hd95:
            row.update(
                {
                    "hd95_wt": m.get("hd95_WT"),
                    "hd95_tc": m.get("hd95_TC"),
                    "hd95_et": m.get("hd95_ET"),
                }
            )
        rows.append(row)

    table_path = out_dir / f"metrics_table_{args.split}.csv"
    fieldnames = list(rows[0].keys())
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {table_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
