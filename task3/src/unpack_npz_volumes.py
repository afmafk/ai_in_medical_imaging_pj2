"""Convert compressed processed_3d NPZ volumes into mmap-friendly NPY arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Directory containing patient .npz files")
    parser.add_argument("--output", required=True, help="Directory for *_image.npy and *_seg.npy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz volumes found in {source}")

    for path in tqdm(paths, desc="unpack-npz"):
        image_path = output / f"{path.stem}_image.npy"
        seg_path = output / f"{path.stem}_seg.npy"
        if not args.overwrite and image_path.exists() and seg_path.exists():
            continue
        with np.load(path) as data:
            image = data["image"].astype(np.float32, copy=False)
            seg = data["seg"].astype(np.uint8, copy=False)
        np.save(image_path, image, allow_pickle=False)
        np.save(seg_path, seg, allow_pickle=False)

    print(f"wrote mmap-friendly arrays to {output}")


if __name__ == "__main__":
    main()
