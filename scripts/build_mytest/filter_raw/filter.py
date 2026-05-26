#!/usr/bin/env python3
"""
Filter raw meteorite/rock images using SAM3 masks:
1. Generate mask using SAM3 (same method as preprocess/sam/batch_stone_mask.py)
2. Filter out images with no mask or stone area ratio < 0.005
3. Apply mask to remove background and save
"""

import argparse
import os
import sys

# Fix NVRTC library path for CUDA driver 13.0 + torch cu128 compatibility
try:
    import nvidia.cuda_nvrtc

    _nvrtc_lib = os.path.join(nvidia.cuda_nvrtc.__path__[0], "lib")
    os.environ["LD_LIBRARY_PATH"] = (
        _nvrtc_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
except ImportError:
    pass

import numpy as np
import sam3
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Enable TF32 for Ampere GPUs (required by SAM3)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable bfloat16 autocast and inference mode (required by SAM3)
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAM3_DIR = os.path.dirname(sam3.__file__)
BPE_PATH = os.path.join(SAM3_DIR, "assets", "bpe_simple_vocab_16e6.txt.gz")
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "preprocess", "sam", "model_cache", "facebook", "sam3", "sam3.pt"
)

PROMPT = "The main stone or rock in picture"
CONFIDENCE_THRESHOLD = 0.5
AREA_RATIO_THRESHOLD = 0.005
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".JPG", ".JPEG")

INPUT_DIRS = {
    "meteorite": os.path.join(PROJECT_ROOT, "build_mytest", "data", "meteorite-raw"),
    "rock": os.path.join(PROJECT_ROOT, "build_mytest", "data", "rock-raw"),
}
OUTPUT_DIRS = {
    "meteorite": os.path.join(PROJECT_ROOT, "build_mytest", "data", "meteorite-filtered"),
    "rock": os.path.join(PROJECT_ROOT, "build_mytest", "data", "rock-filtered"),
}


def build_model():
    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        device="cuda",
        checkpoint_path=CHECKPOINT_PATH,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
    return model, processor


def process_directory(processor, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"[WARN] Input directory does not exist: {input_dir}")
        return 0, 0, 0

    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not image_files:
        print(f"[WARN] No images found in: {input_dir}")
        return 0, 0, 0

    total = len(image_files)
    kept = 0
    filtered_no_mask = 0
    filtered_small = 0

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        stem = os.path.splitext(img_name)[0]
        out_path = os.path.join(output_dir, f"{stem}.png")

        print(f"[{i + 1}/{total}] {img_name}", end="")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] Failed to open image: {e}")
            continue

        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=PROMPT, state=state)

        masks = state.get("masks")
        scores = state.get("scores")

        if masks is None or masks.numel() == 0:
            print(f"  -> FILTERED (no mask)")
            filtered_no_mask += 1
            continue

        # Take the first (highest-confidence) mask
        mask = masks[0].cpu().float().numpy().squeeze()
        score = float(scores[0])

        # Compute stone area ratio
        area_ratio = (mask > 0.5).mean()

        if area_ratio < AREA_RATIO_THRESHOLD:
            print(f"  -> FILTERED (area ratio {area_ratio:.5f} < {AREA_RATIO_THRESHOLD})")
            filtered_small += 1
            continue

        # Apply mask to remove background
        img_np = np.array(image)
        binary_mask = (mask > 0.5).astype(np.uint8)
        masked = img_np.copy()
        masked[binary_mask == 0] = 0

        Image.fromarray(masked).save(out_path)
        kept += 1
        print(f"  -> KEPT (score={score:.4f}, ratio={area_ratio:.4f})")

    return kept, filtered_no_mask, filtered_small


def main():
    parser = argparse.ArgumentParser(
        description="Filter raw images by SAM3 mask quality and apply masks"
    )
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=AREA_RATIO_THRESHOLD,
        help=f"Minimum stone area ratio (default: {AREA_RATIO_THRESHOLD})",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["meteorite", "rock"],
        help="Categories to process (default: meteorite rock)",
    )
    args = parser.parse_args()

    print(f"Building SAM3 model from {CHECKPOINT_PATH} ...")
    model, processor = build_model()

    for cat in args.categories:
        input_dir = INPUT_DIRS.get(cat)
        output_dir = OUTPUT_DIRS.get(cat)
        if input_dir is None or output_dir is None:
            print(f"[WARN] Unknown category: {cat}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {cat}")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}")

        kept, no_mask, small = process_directory(processor, input_dir, output_dir)
        total = kept + no_mask + small
        print(f"\nResults for {cat}:")
        print(f"  Total images:    {total}")
        print(f"  Kept:            {kept}")
        print(f"  Filtered (no mask):     {no_mask}")
        print(f"  Filtered (area < {args.area_threshold}): {small}")

    print("\nDone.")


if __name__ == "__main__":
    main()
