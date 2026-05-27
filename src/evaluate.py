from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as ROOT, load_config
from device_utils import configure_cuda, resolve_device
from dataset.brats_multimodal import BraTS3DPatchDataset, load_split_patient_ids
from metrics import compute_region_metrics, compute_region_metrics_hd95
from models import MODEL_NAMES, build_model
from models.swinunetr_v2 import ensure_patch_size_divisible_by_32
from sliding_window import evaluate_patients_sliding_window


@torch.no_grad()
def evaluate_split(model, loader, device, with_hd95: bool = False) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    n = 0
    for batch in tqdm(loader, desc="eval-patch"):
        image = batch["image"].to(device, non_blocking=True)
        seg = batch["seg"].to(device, non_blocking=True)
        logits = model(image)
        if with_hd95:
            m = compute_region_metrics_hd95(logits, seg)
        else:
            m = compute_region_metrics(logits, seg)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hd95", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="center-crop patch metrics (fast, not BraTS-standard for reporting)",
    )
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="full-volume sliding-window metrics (default when eval_sliding_window in config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(
        args.device or cfg.get("device", "cuda"),
        require_gpu=args.require_gpu or bool(cfg.get("require_gpu", True)),
    )
    configure_cuda(device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else ROOT / "outputs" / args.model / "checkpoints" / "best.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    patient_ids = load_split_patient_ids(cfg["splits_path"], args.split)
    if args.max_patients:
        patient_ids = patient_ids[: args.max_patients]

    patch_size = tuple(cfg.get("patch_size", [96, 96, 96]))
    if args.model == "swinunetr":
        ensure_patch_size_divisible_by_32(patch_size)
    use_sliding = args.sliding_window or (
        not args.patch_only and bool(cfg.get("eval_sliding_window", True)) and not cfg.get("use_full_volume", False)
    )

    model_cfg = cfg.get("swinunetr") if args.model == "swinunetr" else None
    model = build_model(
        args.model,
        in_channels=4,
        num_classes=int(cfg["num_classes"]),
        model_cfg=model_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    if use_sliding:
        stride = int(cfg.get("sliding_window_stride", patch_size[0] // 2))
        print(f"eval: sliding-window patch={patch_size} stride={stride} patients={len(patient_ids)}")
        metrics = evaluate_patients_sliding_window(
            model,
            patient_ids,
            cfg["data_root"],
            patch_size,
            stride,
            int(cfg["num_classes"]),
            device,
            with_hd95=args.hd95,
        )
    else:
        ds = BraTS3DPatchDataset(
            patient_ids,
            cfg["data_root"],
            patch_size=patch_size,
            split="val" if args.split != "train" else "train",
            samples_per_patient=1,
            volume_shape=tuple(cfg.get("volume_shape", [155, 177, 219])),
            use_full_volume=bool(cfg.get("use_full_volume", False)),
            pad_factor=int(cfg.get("pad_factor", 16)),
        )
        loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
        print(f"eval: center patch {patch_size} patients={len(patient_ids)}")
        metrics = evaluate_split(model, loader, device, with_hd95=args.hd95)

    out_dir = ROOT / "outputs" / args.model
    out_path = out_dir / f"metrics_{args.split}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
