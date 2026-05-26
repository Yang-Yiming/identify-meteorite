#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Keyword-based meteorite detector using PaddleOCR.

This script scans images in a directory, performs OCR, and marks an image as
positive ("meteorite") if any configured keyword is detected in recognized text.

Default behavior matches the user's request:
- Input images: data/test_images/
- Output file: txt with only positive image names (one per line), e.g.:
    000001.jpg
    000005.jpg

Example:
    python detect_keyword.py \
        --input_dir ../../data/test_images \
        --output_txt ./keyword_positive.txt \
        --keywords "陨石,meteorite,meteorit"

Requirements:
    pip install paddleocr paddlepaddle opencv-python
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import cv2
from paddleocr import PaddleOCR


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    default_input_dir = Path(__file__).resolve().parent.parent / "data" / "test_images"
    default_output_txt = Path(__file__).resolve().parent / "keyword_positive.txt"

    parser = argparse.ArgumentParser(
        description="Use PaddleOCR to detect keywords and output positive image filenames."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(default_input_dir),
        help="Directory containing test images. Default: ../../data/test_images",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=str(default_output_txt),
        help="Output txt path containing positive image names only.",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="陨石,meteorite,meteorit,陨,隕石,流星石",
        help="Comma-separated keyword list. Any match => positive.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ch",
        help="PaddleOCR language model (e.g. ch, en, korean, japan). Default: ch",
    )
    parser.add_argument(
        "--use_angle_cls",
        action="store_true",
        help="Enable angle classifier in PaddleOCR.",
    )
    parser.add_argument(
        "--det_db_box_thresh",
        type=float,
        default=0.3,
        help="Text detection box threshold.",
    )
    parser.add_argument(
        "--det_db_thresh",
        type=float,
        default=0.2,
        help="Text detection binary threshold.",
    )
    parser.add_argument(
        "--rec_score_thresh",
        type=float,
        default=0.2,
        help="OCR recognition score threshold (line-level).",
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="Enable case-sensitive keyword matching (default: case-insensitive).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directory.",
    )
    return parser.parse_args()


def split_keywords(raw: str) -> List[str]:
    kws = [k.strip() for k in raw.split(",")]
    kws = [k for k in kws if k]
    if not kws:
        raise ValueError("Keyword list is empty. Please provide at least one keyword.")
    return kws


def list_images(input_dir: Path, recursive: bool = False) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if recursive:
        items = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        items = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    items.sort(key=lambda p: p.name)
    return items


def normalize_text(s: str, case_sensitive: bool) -> str:
    s = s.strip()
    # Remove obvious separators/noise to improve recall for OCR artifacts.
    s = re.sub(r"[\s\|\-_/\\]+", "", s)
    return s if case_sensitive else s.lower()


def compile_keywords(keywords: Sequence[str], case_sensitive: bool) -> List[str]:
    return [normalize_text(k, case_sensitive=case_sensitive) for k in keywords]


def extract_texts_from_ocr_result(result) -> Iterable[Tuple[str, float]]:
    """
    PaddleOCR output format (single image):
      result = [
          [
            [box_points], (text, score)
          ],
          ...
      ]
    Sometimes wrappers vary slightly; this parser is tolerant.
    """
    if not result:
        return []

    # Common shape: [ [line1, line2, ...] ]
    first = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result
    lines = first if isinstance(first, (list, tuple)) else result

    out: List[Tuple[str, float]] = []
    for line in lines:
        if not line:
            continue
        # Expected line: [box, (text, score)]
        text, score = None, None
        if isinstance(line, (list, tuple)) and len(line) >= 2:
            rec = line[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text, score = rec[0], rec[1]
        if isinstance(text, str):
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0
            out.append((text, score_f))
    return out


def has_keyword(
    texts_with_scores: Iterable[Tuple[str, float]],
    kw_list_norm: Sequence[str],
    rec_score_thresh: float,
    case_sensitive: bool,
) -> bool:
    for text, score in texts_with_scores:
        if score < rec_score_thresh:
            continue
        t_norm = normalize_text(text, case_sensitive=case_sensitive)
        if not t_norm:
            continue
        for kw in kw_list_norm:
            if kw and kw in t_norm:
                return True
    return False


def run_detection(
    image_paths: Sequence[Path],
    keywords: Sequence[str],
    lang: str,
    use_angle_cls: bool,
    det_db_box_thresh: float,
    det_db_thresh: float,
    rec_score_thresh: float,
    case_sensitive: bool,
) -> List[str]:
    ocr = PaddleOCR(
        use_angle_cls=use_angle_cls,
        lang=lang,
        det_db_box_thresh=det_db_box_thresh,
        det_db_thresh=det_db_thresh,
        show_log=False,
    )

    kw_list_norm = compile_keywords(keywords, case_sensitive=case_sensitive)
    positives: List[str] = []

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image, skip: {img_path}")
            continue

        try:
            result = ocr.ocr(img, cls=use_angle_cls)
        except Exception as e:
            print(f"[WARN] OCR failed on {img_path.name}: {e}")
            continue

        texts = list(extract_texts_from_ocr_result(result))
        is_pos = has_keyword(
            texts_with_scores=texts,
            kw_list_norm=kw_list_norm,
            rec_score_thresh=rec_score_thresh,
            case_sensitive=case_sensitive,
        )

        if is_pos:
            positives.append(img_path.name)

        if idx % 50 == 0 or idx == total:
            print(f"[INFO] Processed {idx}/{total} images, current positives: {len(positives)}")

    return positives


def save_positive_list(names: Sequence[str], output_txt: Path) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        for n in names:
            f.write(f"{n}\n")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_txt = Path(args.output_txt).resolve()
    keywords = split_keywords(args.keywords)

    print(f"[INFO] Input dir     : {input_dir}")
    print(f"[INFO] Output txt    : {output_txt}")
    print(f"[INFO] Keywords      : {keywords}")
    print(f"[INFO] OCR lang      : {args.lang}")
    print(f"[INFO] Angle cls     : {args.use_angle_cls}")
    print(f"[INFO] Case sensitive: {args.case_sensitive}")

    image_paths = list_images(input_dir, recursive=args.recursive)
    print(f"[INFO] Found images  : {len(image_paths)}")

    positives = run_detection(
        image_paths=image_paths,
        keywords=keywords,
        lang=args.lang,
        use_angle_cls=args.use_angle_cls,
        det_db_box_thresh=args.det_db_box_thresh,
        det_db_thresh=args.det_db_thresh,
        rec_score_thresh=args.rec_score_thresh,
        case_sensitive=args.case_sensitive,
    )
    positives = sorted(set(positives))

    save_positive_list(positives, output_txt)

    print(f"[DONE] Positive images: {len(positives)}")
    print(f"[DONE] Saved to       : {output_txt}")


if __name__ == "__main__":
    main()
