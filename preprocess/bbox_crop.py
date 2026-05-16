#!/usr/bin/env python3
"""BBox-crop preprocessing: crop original images to meteorite bbox (+mask), resize to square."""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


Margin = float


def find_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def expand_and_square_bbox(x1: int, y1: int, x2: int, y2: int, margin: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    # Make square (expand shorter side to match longer side)
    if bw > bh:
        expand = (bw - bh) // 2
        y1 -= expand
        y2 = y1 + bw
        pad_y = int(bw * margin)
    else:
        expand = (bh - bw) // 2
        x1 -= expand
        x2 = x1 + bh
        pad_x = int(bh * margin)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)
    return x1, y1, x2, y2


def process_one(mask_path: Path, original_path: Path, output_path: Path, margin: float, output_size: int) -> None:
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img)
    h, w = mask.shape

    x1, y1, x2, y2 = find_bbox(mask)
    x1, y1, x2, y2 = expand_and_square_bbox(x1, y1, x2, y2, margin, w, h)

    original = Image.open(original_path).convert("RGB")
    if original.size != (w, h):
        original = original.resize((w, h), Image.LANCZOS)
    original_np = np.array(original)

    # Crop both original and mask
    crop_original = original_np[y1:y2+1, x1:x2+1]
    crop_mask = mask[y1:y2+1, x1:x2+1]

    # Apply mask: keep meteorite, zero out background
    masked = crop_original.copy()
    masked[crop_mask == 0] = 0

    # Resize to output_size
    result = Image.fromarray(masked).resize((output_size, output_size), Image.LANCZOS)
    result.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="BBox-crop images using SAM masks")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory containing mask PNGs")
    parser.add_argument("--original-dir", type=Path, required=True, help="Directory containing original images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for cropped images")
    parser.add_argument("--margin", type=float, default=0.1, help="Padding margin around bbox (fraction of bbox size)")
    parser.add_argument("--output-size", type=int, default=224, help="Output image size")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build index: stem -> original path
    original_index = {}
    for fpath in args.original_dir.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in (".jpg", ".jpeg", ".png"):
            original_index[fpath.stem] = fpath

    # Collect mask files (only _mask_000.png for each image)
    mask_files = sorted(
        f for f in args.mask_dir.glob("*_mask_*.png") if f.name.endswith("_mask_000.png")
    )
    print(f"Found {len(mask_files)} mask_000 files in {args.mask_dir}")

    skipped = 0
    processed = 0
    for mask_path in tqdm(mask_files, desc="BBox cropping"):
        stem = mask_path.name.split("_mask_")[0]
        original_path = original_index.get(stem)
        if original_path is None:
            skipped += 1
            continue
        out_name = f"{stem}_mask_000.png"
        out_path = args.output_dir / out_name
        process_one(mask_path, original_path, out_path, args.margin, args.output_size)
        processed += 1

    print(f"Done: {processed} processed, {skipped} skipped (no original image)")
    for f in sorted(args.mask_dir.glob("*_nomask.done")):
        stem = f.name.replace("_nomask.done", "")
        marker = args.output_dir / f"{stem}_nomask.done"
        marker.touch()


if __name__ == "__main__":
    main()
