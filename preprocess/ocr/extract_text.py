#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract text from all images under ../data/test_images/test_images (or nearby
test_images dirs) using PaddleOCR, and save to test_text.csv as:

    filename, [text]

Only writes rows for images with non-empty extracted text.
"""

# ── must be set before any paddle / paddleocr import ──────────────────────
import os as _os
_os.environ["FLAGS_use_mkldnn"] = "0"
_os.environ["FLAGS_use_onednn"] = "0"
_os.environ["FLAGS_enable_pir_api"] = "0"
del _os
# ───────────────────────────────────────────────────────────────────────────

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from paddleocr import PaddleOCR
except ImportError:
    print(
        "Missing dependency: paddleocr. Install with: pip install paddleocr",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print(
        "Missing dependency: numpy. Install with: pip install numpy",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from PIL import Image, ImageOps
except ImportError:
    print(
        "Missing dependency: pillow. Install with: pip install pillow",
        file=sys.stderr,
    )
    sys.exit(1)


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

Image.MAX_IMAGE_PIXELS = 120_000_000


# ── image discovery ────────────────────────────────────────────────────────


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def find_default_image_dir(script_dir: Path) -> Optional[Path]:
    candidates = [
        script_dir.parent / "data" / "test_images" / "test_images",
        script_dir.parent / "data" / "test_images",
        script_dir / "test_images",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


# ── image loading & resizing ───────────────────────────────────────────────


def _resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    return Image.LANCZOS  # type: ignore[attr-defined]


def _target_size(
    w: int,
    h: int,
    max_long: Optional[int],
    max_px: Optional[int],
    min_short: int,
) -> Tuple[int, int]:
    w, h = max(1, int(w)), max(1, int(h))

    if max_long and max_long > 0:
        long_edge = max(w, h)
        if long_edge > max_long:
            r = max_long / float(long_edge)
            w, h = max(1, int(round(w * r))), max(1, int(round(h * r)))

    if max_px and max_px > 0:
        area = w * h
        if area > max_px:
            r = (max_px / float(area)) ** 0.5
            w, h = max(1, int(round(w * r))), max(1, int(round(h * r)))

    short_edge = min(w, h)
    if short_edge < min_short and min(w, h) > 0:
        r = min_short / float(short_edge)
        w, h = max(1, int(round(w * r))), max(1, int(round(h * r)))

    return w, h


def load_image(img_path: Path, max_long: int = 1280, max_px: int = 1_400_000) -> "np.ndarray":
    """
    Load, EXIF-orient, resize, return BGR uint8 array for PaddleOCR.
    """
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        src_w, src_h = im.size
        dst_w, dst_h = _target_size(src_w, src_h, max_long, max_px, min_short=32)
        if (dst_w, dst_h) != (src_w, src_h):
            im = im.resize((dst_w, dst_h), _resample_filter())
        rgb = np.array(im)
    return rgb[:, :, ::-1].copy()  # RGB → BGR


# ── result parsing (PaddleOCR 3.x / paddlex format) ───────────────────────


def extract_text(result, *, min_score: float = 0.0) -> str:
    """
    Parse PaddleOCR 3.x ``predict()`` output into a single newline-joined string.

    Expected format: ``[page_dict, ...]`` where each *page_dict* contains::

        {
            "rec_texts":  ["text1", "text2", ...],
            "rec_scores": [0.99, 0.87, ...],
            ...
        }
    """
    if not result:
        return ""
    lines: List[str] = []
    for page in result:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        for text, score in zip(texts, scores):
            if not isinstance(text, str):
                continue
            if isinstance(score, (float, int)) and score < min_score:
                continue
            text = text.strip()
            if text:
                lines.append(text)
    return "\n".join(lines).strip()


# ── main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract OCR text from images via PaddleOCR and write CSV."
    )
    parser.add_argument("--images-dir", default=None, help="Directory containing images.")
    parser.add_argument("--output", default=None, help="Output CSV path (default: preprocess/test_text.csv)")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    parser.add_argument("--lang", default="ch", help="PaddleOCR language (default: ch)")
    parser.add_argument("--use-angle-cls", action="store_true", help="Enable textline orientation classifier")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU (requires paddlepaddle-gpu)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop results below this confidence")
    parser.add_argument("--max-long-edge", type=int, default=1280, help="Clamp long edge (0 to disable)")
    parser.add_argument("--max-pixels", type=int, default=1_400_000, help="Clamp total pixels (0 to disable)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.images_dir:
        images_dir = Path(args.images_dir).resolve()
    else:
        auto_dir = find_default_image_dir(script_dir)
        if auto_dir is None:
            print("Cannot auto-detect images directory. Please pass --images-dir.", file=sys.stderr)
            sys.exit(2)
        images_dir = auto_dir.resolve()

    output_csv = Path(args.output).resolve() if args.output else (script_dir / "test_text.csv").resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(images_dir))
    if not images:
        print(f"No images found in: {images_dir}", file=sys.stderr)
        sys.exit(0)

    max_long = args.max_long_edge if args.max_long_edge > 0 else None
    max_px = args.max_pixels if args.max_pixels > 0 else None

    ocr_cfg = {
        "lang": args.lang,
        "enable_mkldnn": False,
    }
    if args.use_gpu:
        ocr_cfg["device"] = "gpu"
    if args.use_angle_cls:
        ocr_cfg["use_textline_orientation"] = True

    ocr = PaddleOCR(**ocr_cfg)

    written = 0
    processed = 0
    failed = 0

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for img in images:
            processed += 1
            if args.verbose:
                print(f"[{processed}/{len(images)}] OCR: {img.name}")

            try:
                bgr = load_image(img, max_long=max_long, max_px=max_px)
                result = ocr.predict(bgr)
                text = extract_text(result, min_score=args.min_score)
            except Exception as e:
                failed += 1
                print(f"[WARN] Failed {img.name}: {e}", file=sys.stderr)
                continue

            if not text:
                continue

            writer.writerow([img.name, f"[{text.strip()}]"])
            written += 1

    print(f"Done. Processed={processed}, Written={written}, Failed={failed}, Output={output_csv}")


if __name__ == "__main__":
    main()
