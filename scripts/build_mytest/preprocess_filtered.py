#!/usr/bin/env python3
"""
Build mytest images from SAM-filtered meteorite/rock data.

Steps:
1. Detect the non-black foreground left by filter_raw/filter.py.
2. Crop to the foreground bbox with padding.
3. Place the crop on a centered square canvas and resize.
4. Remove exact and near-duplicate images using image-only hashes/features.
"""

import argparse
import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRS = {
    "meteorite": PROJECT_ROOT / "build_mytest" / "data" / "meteorite-filtered",
    "rock": PROJECT_ROOT / "build_mytest" / "data" / "rock-filtered",
}
OUTPUT_ROOT = PROJECT_ROOT / "mytest"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ProcessedImage:
    category: str
    source_path: Path
    output_path: Path
    original_size: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    bbox_area_ratio: float
    foreground_ratio: float
    exact_hash: str
    ahash: int
    dhash: int
    phash: int
    colorhash: int
    feature: np.ndarray
    duplicate_of: Optional[str] = None
    duplicate_reason: Optional[str] = None


def iter_images(path: Path) -> Iterable[Path]:
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            yield child


def foreground_bbox(image: Image.Image, threshold: int) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(image.convert("RGB"))
    mask = (arr > threshold).any(axis=2)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return left, top, right, bottom


def padded_bbox(
    bbox: Tuple[int, int, int, int],
    size: Tuple[int, int],
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    width, height = size
    left, top, right, bottom = bbox
    box_w = right - left
    box_h = bottom - top
    pad = int(round(max(box_w, box_h) * padding_ratio))
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def center_on_square(image: Image.Image, fill: Tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def resize_image(image: Image.Image, image_size: int) -> Image.Image:
    if image_size <= 0:
        return image
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def image_bytes_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.astype(bool).ravel():
        value = (value << 1) | int(bit)
    return value


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = ImageOps.grayscale(image).resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    return bits_to_int(pixels >= pixels.mean())


def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = ImageOps.grayscale(image).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    return bits_to_int(pixels[:, 1:] >= pixels[:, :-1])


def dct_matrix(size: int) -> np.ndarray:
    matrix = np.empty((size, size), dtype=np.float32)
    factor = np.pi / (2.0 * size)
    scale0 = np.sqrt(1.0 / size)
    scale = np.sqrt(2.0 / size)
    for k in range(size):
        row_scale = scale0 if k == 0 else scale
        for n in range(size):
            matrix[k, n] = row_scale * np.cos((2 * n + 1) * k * factor)
    return matrix


DCT_32 = dct_matrix(32)


def perceptual_hash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    img_size = hash_size * highfreq_factor
    gray = ImageOps.grayscale(image).resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct = DCT_32 @ pixels @ DCT_32.T
    low_freq = dct[:hash_size, :hash_size].copy()
    comparable = low_freq.ravel()[1:]
    return bits_to_int(low_freq >= np.median(comparable))


def color_hash(image: Image.Image, bins_per_channel: int = 4) -> int:
    small = image.resize((32, 32), Image.Resampling.BILINEAR).convert("RGB")
    arr = np.asarray(small, dtype=np.uint8)
    bins = arr // (256 // bins_per_channel)
    hist = np.zeros((bins_per_channel, bins_per_channel, bins_per_channel), dtype=np.float32)
    for r, g, b in bins.reshape(-1, 3):
        hist[int(r), int(g), int(b)] += 1.0
    return bits_to_int(hist.ravel() >= hist.mean())


def lowres_feature(image: Image.Image, size: int = 48) -> np.ndarray:
    gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    pixels -= pixels.mean()
    std = float(pixels.std())
    if std > 1e-6:
        pixels /= std
    return pixels.ravel()


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def correlation_distance(left: np.ndarray, right: np.ndarray) -> float:
    corr = float(np.dot(left, right) / left.size)
    return 1.0 - corr


def similar_enough(current: ProcessedImage, kept: ProcessedImage, strict_hamming: int, loose_hamming: int) -> Optional[str]:
    if current.exact_hash == kept.exact_hash:
        return "exact_sha256"

    p_dist = hamming(current.phash, kept.phash)
    d_dist = hamming(current.dhash, kept.dhash)
    a_dist = hamming(current.ahash, kept.ahash)
    c_dist = hamming(current.colorhash, kept.colorhash)

    if p_dist <= strict_hamming and d_dist <= strict_hamming:
        corr_dist = correlation_distance(current.feature, kept.feature)
        if corr_dist <= 0.08:
            return f"near_duplicate_phash={p_dist};dhash={d_dist};corr_dist={corr_dist:.4f}"

    if p_dist <= loose_hamming and d_dist <= loose_hamming and a_dist <= loose_hamming and c_dist <= 10:
        corr_dist = correlation_distance(current.feature, kept.feature)
        if corr_dist <= 0.04:
            return (
                f"near_duplicate_hash_combo=phash:{p_dist},dhash:{d_dist},"
                f"ahash:{a_dist},color:{c_dist};corr_dist={corr_dist:.4f}"
            )

    return None


def output_name(category: str, source_path: Path) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:8]
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source_path.stem)
    return f"{category}_{safe_stem}_{digest}.png"


def process_one(
    category: str,
    source_path: Path,
    output_dir: Path,
    image_size: int,
    padding_ratio: float,
    foreground_threshold: int,
    fill: Tuple[int, int, int],
) -> Optional[ProcessedImage]:
    try:
        image = Image.open(source_path).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        print(f"[WARN] Failed to open {source_path}: {exc}")
        return None

    original_size = image.size
    bbox = foreground_bbox(image, foreground_threshold)
    if bbox is None:
        print(f"[WARN] No foreground found: {source_path}")
        return None

    arr = np.asarray(image)
    foreground_ratio = float((arr > foreground_threshold).any(axis=2).mean())
    padded = padded_bbox(bbox, image.size, padding_ratio)
    cropped = image.crop(padded)
    squared = center_on_square(cropped, fill)
    processed = resize_image(squared, image_size)

    output_path = output_dir / output_name(category, source_path)
    processed.save(output_path, "PNG", compress_level=1)

    left, top, right, bottom = bbox
    bbox_area_ratio = ((right - left) * (bottom - top)) / float(original_size[0] * original_size[1])
    return ProcessedImage(
        category=category,
        source_path=source_path,
        output_path=output_path,
        original_size=original_size,
        bbox=bbox,
        bbox_area_ratio=bbox_area_ratio,
        foreground_ratio=foreground_ratio,
        exact_hash=image_bytes_hash(processed),
        ahash=average_hash(processed),
        dhash=difference_hash(processed),
        phash=perceptual_hash(processed),
        colorhash=color_hash(processed),
        feature=lowres_feature(processed),
    )


def deduplicate(
    records: Sequence[ProcessedImage],
    strict_hamming: int,
    loose_hamming: int,
) -> Tuple[List[ProcessedImage], List[ProcessedImage]]:
    kept: List[ProcessedImage] = []
    removed: List[ProcessedImage] = []
    exact_index: Dict[str, ProcessedImage] = {}
    phash_index: Dict[int, List[ProcessedImage]] = {}
    dhash_index: Dict[int, List[ProcessedImage]] = {}

    def hash_keys(value: int) -> Tuple[int, int, int, int]:
        return (
            (value >> 48) & 0xFFFF,
            (value >> 32) & 0xFFFF,
            (value >> 16) & 0xFFFF,
            value & 0xFFFF,
        )

    def add_to_index(index: Dict[int, List[ProcessedImage]], value: int, record: ProcessedImage) -> None:
        for key in hash_keys(value):
            index.setdefault(key, []).append(record)

    def candidates(record: ProcessedImage) -> List[ProcessedImage]:
        seen: Set[int] = set()
        result: List[ProcessedImage] = []
        for index, value in ((phash_index, record.phash), (dhash_index, record.dhash)):
            for key in hash_keys(value):
                for candidate in index.get(key, []):
                    ident = id(candidate)
                    if ident not in seen:
                        seen.add(ident)
                        result.append(candidate)
        return result

    for record in records:
        exact_match = exact_index.get(record.exact_hash)
        if exact_match is not None:
            record.duplicate_of = str(exact_match.output_path)
            record.duplicate_reason = "exact_sha256"
            removed.append(record)
            continue

        duplicate_of = None
        duplicate_reason = None
        for candidate in candidates(record):
            reason = similar_enough(record, candidate, strict_hamming, loose_hamming)
            if reason is not None:
                duplicate_of = candidate
                duplicate_reason = reason
                break

        if duplicate_of is None:
            kept.append(record)
            exact_index[record.exact_hash] = record
            add_to_index(phash_index, record.phash, record)
            add_to_index(dhash_index, record.dhash, record)
        else:
            record.duplicate_of = str(duplicate_of.output_path)
            record.duplicate_reason = duplicate_reason
            removed.append(record)

    return kept, removed


def write_manifest(path: Path, records: Sequence[ProcessedImage], status: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "status",
                "category",
                "source_path",
                "output_path",
                "duplicate_of",
                "duplicate_reason",
                "original_width",
                "original_height",
                "bbox_left",
                "bbox_top",
                "bbox_right",
                "bbox_bottom",
                "bbox_area_ratio",
                "foreground_ratio",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    status,
                    record.category,
                    record.source_path,
                    record.output_path,
                    record.duplicate_of or "",
                    record.duplicate_reason or "",
                    record.original_size[0],
                    record.original_size[1],
                    record.bbox[0],
                    record.bbox[1],
                    record.bbox[2],
                    record.bbox[3],
                    f"{record.bbox_area_ratio:.8f}",
                    f"{record.foreground_ratio:.8f}",
                ]
            )


def parse_fill(value: str) -> Tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--fill must be R,G,B")
    rgb = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("--fill channels must be in [0, 255]")
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop, center, resize, and deduplicate filtered mytest images.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=384, help="Final square image size. Use 0 to keep crop size.")
    parser.add_argument("--padding-ratio", type=float, default=0.08)
    parser.add_argument("--foreground-threshold", type=int, default=8)
    parser.add_argument("--fill", type=parse_fill, default=(0, 0, 0), help="Square canvas fill color as R,G,B.")
    parser.add_argument("--strict-hamming", type=int, default=4)
    parser.add_argument("--loose-hamming", type=int, default=8)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.dry_run:
        shutil.rmtree(args.output_root)
    for category in INPUT_DIRS:
        if not args.dry_run:
            (args.output_root / category).mkdir(parents=True, exist_ok=True)

    records: List[ProcessedImage] = []
    for category, input_dir in INPUT_DIRS.items():
        if not input_dir.is_dir():
            raise FileNotFoundError(input_dir)
        output_dir = args.output_root / category
        paths = list(iter_images(input_dir))
        print(f"[INFO] {category}: processing {len(paths)} images from {input_dir}")
        if args.dry_run:
            continue
        for idx, path in enumerate(paths, start=1):
            record = process_one(
                category=category,
                source_path=path,
                output_dir=output_dir,
                image_size=args.image_size,
                padding_ratio=args.padding_ratio,
                foreground_threshold=args.foreground_threshold,
                fill=args.fill,
            )
            if record is not None:
                records.append(record)
            if idx % 250 == 0 or idx == len(paths):
                print(f"[INFO] {category}: processed {idx}/{len(paths)}", flush=True)

    if args.dry_run:
        return

    if args.no_dedup:
        kept = records
        removed: List[ProcessedImage] = []
    else:
        print(f"[INFO] Deduplicating {len(records)} processed images")
        kept, removed = deduplicate(records, args.strict_hamming, args.loose_hamming)
        for record in removed:
            if record.output_path.exists():
                record.output_path.unlink()

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_root / "kept.csv", kept, "kept")
    write_manifest(args.output_root / "removed_duplicates.csv", removed, "removed")

    kept_by_category = {category: 0 for category in INPUT_DIRS}
    removed_by_category = {category: 0 for category in INPUT_DIRS}
    for record in kept:
        kept_by_category[record.category] += 1
    for record in removed:
        removed_by_category[record.category] += 1

    print("[DONE]")
    for category in INPUT_DIRS:
        print(
            f"  {category}: kept={kept_by_category[category]} "
            f"removed_duplicates={removed_by_category[category]}"
        )
    print(f"  output_root={args.output_root}")
    print(f"  kept_manifest={args.output_root / 'kept.csv'}")
    print(f"  duplicate_manifest={args.output_root / 'removed_duplicates.csv'}")


if __name__ == "__main__":
    main()
