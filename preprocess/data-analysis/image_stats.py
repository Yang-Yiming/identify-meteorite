#!/usr/bin/env python3
"""Compute mean & variance of image dimensions, and pixel stats after resize to 224x224.
Splits: train, test, total."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path("/root/project/data")
TRAIN_DIR = DATA_DIR / "train_images" / "train_images"
TEST_DIR = DATA_DIR / "test_images" / "test_images"
OUTPUT_DIR = Path("/root/project/preprocess/data-analysis")
TARGET_SIZE = (224, 224)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image_paths(root: Path):
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    )


def compute_size_stats(paths):
    widths = []
    heights = []
    for p in tqdm(paths, desc="Reading sizes"):
        with Image.open(p) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    widths = np.array(widths, dtype=np.float64)
    heights = np.array(heights, dtype=np.float64)
    areas = widths * heights
    return {
        "n": len(paths),
        "width_mean": float(np.mean(widths)),
        "width_var": float(np.var(widths)),
        "width_std": float(np.std(widths)),
        "width_min": int(np.min(widths)),
        "width_max": int(np.max(widths)),
        "height_mean": float(np.mean(heights)),
        "height_var": float(np.var(heights)),
        "height_std": float(np.std(heights)),
        "height_min": int(np.min(heights)),
        "height_max": int(np.max(heights)),
        "area_mean": float(np.mean(areas)),
        "area_var": float(np.var(areas)),
        "area_std": float(np.std(areas)),
    }


def compute_pixel_stats(paths):
    n_pixels = 0
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    sum_sq_r = 0.0
    sum_sq_g = 0.0
    sum_sq_b = 0.0

    for p in tqdm(paths, desc="Computing pixel stats (224x224)"):
        with Image.open(p) as img:
            arr = np.array(img.convert("RGB").resize(TARGET_SIZE, Image.BILINEAR), dtype=np.float64)

        h, w, _ = arr.shape
        pixel_count = h * w
        n_pixels += pixel_count

        sum_r += arr[:, :, 0].sum()
        sum_g += arr[:, :, 1].sum()
        sum_b += arr[:, :, 2].sum()
        sum_sq_r += (arr[:, :, 0] ** 2).sum()
        sum_sq_g += (arr[:, :, 1] ** 2).sum()
        sum_sq_b += (arr[:, :, 2] ** 2).sum()

    mean_r = sum_r / n_pixels
    mean_g = sum_g / n_pixels
    mean_b = sum_b / n_pixels

    var_r = sum_sq_r / n_pixels - mean_r ** 2
    var_g = sum_sq_g / n_pixels - mean_g ** 2
    var_b = sum_sq_b / n_pixels - mean_b ** 2

    return {
        "n_images": len(paths),
        "n_pixels": n_pixels,
        "image_size": list(TARGET_SIZE),
        "mean_r": float(mean_r),
        "mean_g": float(mean_g),
        "mean_b": float(mean_b),
        "mean_r_norm": float(mean_r / 255.0),
        "mean_g_norm": float(mean_g / 255.0),
        "mean_b_norm": float(mean_b / 255.0),
        "var_r": float(var_r),
        "var_g": float(var_g),
        "var_b": float(var_b),
        "var_r_norm": float(var_r / (255.0 ** 2)),
        "var_g_norm": float(var_g / (255.0 ** 2)),
        "var_b_norm": float(var_b / (255.0 ** 2)),
        "std_r": float(np.sqrt(max(var_r, 0))),
        "std_g": float(np.sqrt(max(var_g, 0))),
        "std_b": float(np.sqrt(max(var_b, 0))),
        "std_r_norm": float(np.sqrt(max(var_r, 0)) / 255.0),
        "std_g_norm": float(np.sqrt(max(var_g, 0)) / 255.0),
        "std_b_norm": float(np.sqrt(max(var_b, 0)) / 255.0),
        "mean_gray": float((mean_r + mean_g + mean_b) / 3.0),
        "mean_gray_norm": float((mean_r + mean_g + mean_b) / 3.0 / 255.0),
        "var_gray": float((var_r + var_g + var_b) / 3.0),
        "var_gray_norm": float((var_r + var_g + var_b) / 3.0 / (255.0 ** 2)),
        "std_gray_norm": float(np.sqrt(max((var_r + var_g + var_b) / 3.0, 0)) / 255.0),
    }


def main():
    train_paths = load_image_paths(TRAIN_DIR)
    test_paths = load_image_paths(TEST_DIR)
    all_paths = train_paths + test_paths

    print(f"Train images: {len(train_paths)}")
    print(f"Test images:  {len(test_paths)}")
    print(f"Total images: {len(all_paths)}")

    results = {}

    print("\n=== Size Stats ===")
    results["train_size"] = compute_size_stats(train_paths)
    results["test_size"] = compute_size_stats(test_paths)
    results["total_size"] = compute_size_stats(all_paths)

    print("\n=== Pixel Stats (224x224) ===")
    results["train_pixel_224"] = compute_pixel_stats(train_paths)
    results["test_pixel_224"] = compute_pixel_stats(test_paths)
    results["total_pixel_224"] = compute_pixel_stats(all_paths)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "image_stats.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")

    for key in ["train_size", "test_size", "total_size"]:
        s = results[key]
        print(f"\n{key} (n={s['n']}):")
        print(f"  width:  mean={s['width_mean']:.1f}  var={s['width_var']:.1f}  std={s['width_std']:.1f}  [{s['width_min']}, {s['width_max']}]")
        print(f"  height: mean={s['height_mean']:.1f}  var={s['height_var']:.1f}  std={s['height_std']:.1f}  [{s['height_min']}, {s['height_max']}]")
        print(f"  area:   mean={s['area_mean']:.1f}  var={s['area_var']:.1f}  std={s['area_std']:.1f}")

    for key in ["train_pixel_224", "test_pixel_224", "total_pixel_224"]:
        s = results[key]
        print(f"\n{key}:")
        print(f"  R: mean={s['mean_r_norm']:.4f}  var={s['var_r_norm']:.4f}  std={s['std_r_norm']:.4f}")
        print(f"  G: mean={s['mean_g_norm']:.4f}  var={s['var_g_norm']:.4f}  std={s['std_g_norm']:.4f}")
        print(f"  B: mean={s['mean_b_norm']:.4f}  var={s['var_b_norm']:.4f}  std={s['std_b_norm']:.4f}")
        print(f"  overall: mean={s['mean_gray_norm']:.4f}  var={s['var_gray_norm']:.4f}  std={s['std_gray_norm']:.4f}")


if __name__ == "__main__":
    main()
