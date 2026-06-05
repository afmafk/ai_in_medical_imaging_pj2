from __future__ import annotations

import torch


def resolve_device(requested: str = "cuda", require_gpu: bool = False) -> torch.device:
    """Pick training device; default CUDA when available."""
    req = requested.lower().strip()
    if req == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif req.startswith("cuda"):
        if not torch.cuda.is_available():
            if require_gpu:
                raise RuntimeError("CUDA requested but no GPU is available.")
            return torch.device("cpu")
        index = 0
        if ":" in req:
            index = int(req.split(":")[1])
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    else:
        device = torch.device(req)

    if require_gpu and device.type != "cuda":
        raise RuntimeError(f"GPU required but using {device}.")
    return device


def configure_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} | {props.total_memory / 1024**3:.1f} GB | cuda:{device.index}")
