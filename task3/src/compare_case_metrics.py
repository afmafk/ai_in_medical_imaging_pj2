"""Compare per-case segmentation metrics exported by analyze_cases.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REGIONS = ("WT", "TC", "ET")


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["patient_id"]: row for row in rows}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def compare_rows(
    baseline: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "patient sets differ: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )

    rows: list[dict[str, object]] = []
    for patient_id in sorted(baseline):
        base_row = baseline[patient_id]
        candidate_row = candidate[patient_id]
        row: dict[str, object] = {"patient_id": patient_id}
        for name in (*REGIONS, "mean"):
            metric_name = f"dice_{name}"
            base_value = metric(base_row, metric_name)
            candidate_value = metric(candidate_row, metric_name)
            row[f"baseline_{metric_name}"] = base_value
            row[f"candidate_{metric_name}"] = candidate_value
            row[f"delta_{metric_name}"] = candidate_value - base_value
        for region in REGIONS:
            metric_name = f"hd95_{region}"
            base_value = metric(base_row, metric_name)
            candidate_value = metric(candidate_row, metric_name)
            row[f"baseline_{metric_name}"] = base_value
            row[f"candidate_{metric_name}"] = candidate_value
            row[f"delta_{metric_name}"] = candidate_value - base_value

            base_voxels = int(base_row[f"pred_voxels_{region}"])
            candidate_voxels = int(candidate_row[f"pred_voxels_{region}"])
            row[f"baseline_pred_voxels_{region}"] = base_voxels
            row[f"candidate_pred_voxels_{region}"] = candidate_voxels
            row[f"delta_pred_voxels_{region}"] = candidate_voxels - base_voxels
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]], top_k: int) -> dict[str, object]:
    def patient_ids(sorted_rows: list[dict[str, object]]) -> list[str]:
        return [str(row["patient_id"]) for row in sorted_rows[:top_k]]

    improved_mean = [row for row in rows if float(row["delta_dice_mean"]) > 0.0]
    regressed_mean = [row for row in rows if float(row["delta_dice_mean"]) < 0.0]
    summary: dict[str, object] = {
        "num_cases": len(rows),
        "num_mean_dice_improved": len(improved_mean),
        "num_mean_dice_regressed": len(regressed_mean),
        "num_mean_dice_unchanged": len(rows) - len(improved_mean) - len(regressed_mean),
        "mean_delta_dice": {
            name: sum(float(row[f"delta_dice_{name}"]) for row in rows) / len(rows)
            for name in (*REGIONS, "mean")
        },
        "mean_delta_hd95": {
            region: sum(float(row[f"delta_hd95_{region}"]) for row in rows) / len(rows)
            for region in REGIONS
        },
        "largest_mean_dice_improvements": patient_ids(
            sorted(rows, key=lambda row: float(row["delta_dice_mean"]), reverse=True)
        ),
        "largest_mean_dice_regressions": patient_ids(
            sorted(rows, key=lambda row: float(row["delta_dice_mean"]))
        ),
    }
    for region in REGIONS:
        tradeoffs = [
            row
            for row in rows
            if float(row[f"delta_dice_{region}"]) > 0.0
            and float(row[f"delta_hd95_{region}"]) > 0.0
        ]
        summary[f"num_{region}_dice_improved_hd95_regressed"] = len(tradeoffs)
        summary[f"largest_{region}_hd95_regressions"] = patient_ids(
            sorted(rows, key=lambda row: float(row[f"delta_hd95_{region}"]), reverse=True)
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = compare_rows(read_rows(args.baseline), read_rows(args.candidate))
    summary = summarize(rows, args.top_k)
    write_rows(args.output_dir / "paired_case_metrics_test.csv", rows)
    with (args.output_dir / "paired_analysis_summary_test.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
