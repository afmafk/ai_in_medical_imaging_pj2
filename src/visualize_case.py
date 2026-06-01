"""Render one full-volume prediction at a requested axial slice."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from analyze_cases import case_metrics, save_case_figure
from config import load_config
from dataset.brats_multimodal import load_patient_volume_from_root
from device_utils import configure_cuda, resolve_device
from models import MODEL_NAMES, build_model
from sliding_window import sliding_window_predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--slice-z", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    configure_cuda(device)
    patch_size = tuple(cfg.get("patch_size", [96, 96, 96]))
    stride = int(cfg.get("sliding_window_stride", patch_size[0] // 2))

    model = build_model(args.model, in_channels=4, num_classes=int(cfg["num_classes"])).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image, gt = load_patient_volume_from_root(cfg["data_root"], args.patient_id)
    pred = sliding_window_predict(
        model,
        image,
        patch_size,
        stride,
        int(cfg["num_classes"]),
        device,
    )
    metrics = case_metrics(args.patient_id, pred, gt)
    output_path = Path(args.output)
    save_case_figure(args.patient_id, image[3], gt, pred, metrics, output_path, z_idx=args.slice_z)
    print(f"wrote {output_path.resolve()}")
    print(metrics)


if __name__ == "__main__":
    main()
