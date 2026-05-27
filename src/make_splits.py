"""Generate train/val/test splits (7:1:2) at patient or case level."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from config import PROJECT_ROOT, load_config

PATIENT_GROUP_RE = re.compile(r"^(BraTS-GLI-\d+)")


def patient_group_id(case_id: str) -> str:
    """Map case ID to patient group, e.g. BraTS-GLI-00014-000 -> BraTS-GLI-00014."""
    m = PATIENT_GROUP_RE.match(case_id)
    if not m:
        raise ValueError(f"cannot parse patient group from case id: {case_id}")
    return m.group(1)


def group_cases_by_patient(case_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in case_ids:
        groups[patient_group_id(cid)].append(cid)
    for gid in groups:
        groups[gid] = sorted(groups[gid])
    return dict(groups)


def make_splits(
    items: list[str],
    ratios: list[float],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
    if len(ratios) != 3:
        raise ValueError("expected three ratios: train, val, test")

    items = sorted(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    assert len(test) == n_test
    return train, val, test


def make_splits_by_patient(
    case_ids: list[str],
    ratios: list[float],
    seed: int,
) -> dict:
    groups = group_cases_by_patient(case_ids)
    group_ids = sorted(groups.keys())
    train_g, val_g, test_g = make_splits(group_ids, ratios, seed)

    train_cases = [c for g in train_g for c in groups[g]]
    val_cases = [c for g in val_g for c in groups[g]]
    test_cases = [c for g in test_g for c in groups[g]]

    train_set, val_set, test_set = set(train_g), set(val_g), set(test_g)
    assert not (train_set & val_set or train_set & test_set or val_set & test_set)

    return {
        "seed": seed,
        "split_level": "patient",
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "counts": {
            "patients": {
                "train": len(train_g),
                "val": len(val_g),
                "test": len(test_g),
                "total": len(group_ids),
            },
            "cases": {
                "train": len(train_cases),
                "val": len(val_cases),
                "test": len(test_cases),
                "total": len(case_ids),
            },
        },
        "patient_groups": {"train": train_g, "val": val_g, "test": test_g},
        "train": sorted(train_cases),
        "val": sorted(val_cases),
        "test": sorted(test_cases),
    }


def make_splits_by_case(
    case_ids: list[str],
    ratios: list[float],
    seed: int,
) -> dict:
    train, val, test = make_splits(case_ids, ratios, seed)
    return {
        "seed": seed,
        "split_level": "case",
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(case_ids),
        },
        "train": train,
        "val": val,
        "test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--by-case",
        action="store_true",
        help="split by case ID (legacy); default is by patient group",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata_path = Path(cfg["metadata_path"])
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    patients = meta["patients"]

    data_root = Path(cfg["data_root"])
    available = {p.name for p in data_root.iterdir() if p.is_dir()}
    patients = [p for p in patients if p in available]
    missing = len(meta["patients"]) - len(patients)
    if missing:
        print(f"warning: {missing} patients in metadata but no processed_2d folder")

    ratios = cfg.get("split_ratios", [0.7, 0.1, 0.2])
    seed = int(cfg.get("split_seed", 42))

    if args.by_case:
        splits = make_splits_by_case(patients, ratios, seed)
    else:
        splits = make_splits_by_patient(patients, ratios, seed)

    out = Path(args.output) if args.output else PROJECT_ROOT / cfg.get(
        "splits_path", "splits_patient_seed42.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"wrote {out}")
    print(f"split_level: {splits['split_level']}")
    print(f"counts: {splits['counts']}")


if __name__ == "__main__":
    main()
