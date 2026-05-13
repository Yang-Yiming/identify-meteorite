#!/usr/bin/env python3
"""
Batch process images with SAM3 text prompt "stone" using ModelScope model.
Saves binary masks to output/test and output/train.
Supports resume: skips already-processed images by default (use --no-resume to disable).
"""

import argparse
import glob
import os
import sys

# Fix NVRTC library path for CUDA driver 13.0 + torch cu128 compatibility
# Required when system CUDA driver is 13.x but torch is built for CUDA 12.x
_this_dir = os.path.dirname(os.path.abspath(__file__))
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM3_DIR = os.path.dirname(sam3.__file__)
BPE_PATH = os.path.join(SAM3_DIR, "assets", "bpe_simple_vocab_16e6.txt.gz")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "model_cache", "facebook", "sam3", "sam3.pt")

PROMPT = "The main stone or rock in picture"
CONFIDENCE_THRESHOLD = 0.5
INPUT_DIRS = {
    "test": os.path.join(SCRIPT_DIR, "data", "test_images", "test_images"),
    "train": os.path.join(SCRIPT_DIR, "data", "train_images", "train_images"),
}
OUTPUT_DIRS = {
    "test": os.path.join(SCRIPT_DIR, "output", "test"),
    "train": os.path.join(SCRIPT_DIR, "output", "train"),
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def is_already_processed(stem, output_dir):
    existing = glob.glob(os.path.join(output_dir, f"{stem}_*"))
    return len(existing) > 0


def build_model():
    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        device="cuda",
        checkpoint_path=CHECKPOINT_PATH,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
    return model, processor


def process_directory(processor, input_dir, output_dir, resume=True):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"[WARN] Input directory does not exist: {input_dir}")
        return 0, 0

    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not image_files:
        print(f"[WARN] No images found in: {input_dir}")
        return 0, 0

    total_masks = 0
    skipped = 0
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        stem = os.path.splitext(img_name)[0]

        if resume and is_already_processed(stem, output_dir):
            skipped += 1
            continue

        print(f"[{i + 1}/{len(image_files)}] {img_name}")

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
            print(f"  No masks found")
            # touch a marker so resume skips this image next time
            marker_path = os.path.join(output_dir, f"{stem}_nomask.done")
            open(marker_path, "w").close()
            continue

        masks_np = masks.cpu().float().numpy()
        scores_np = scores.cpu().float().numpy()

        for j in range(masks_np.shape[0]):
            mask = masks_np[j].squeeze()
            score = float(scores_np[j])
            mask_img = (mask * 255).astype(np.uint8)
            out_name = f"{stem}_mask_{j:03d}.png"
            out_path = os.path.join(output_dir, out_name)
            Image.fromarray(mask_img).save(out_path)
            total_masks += 1

        print(f"  Found {masks_np.shape[0]} mask(s), top score: {scores_np[0]:.4f}")

    return total_masks, skipped


def main():
    parser = argparse.ArgumentParser(description="SAM3 batch stone mask inference")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (re-process all images)",
    )
    args = parser.parse_args()
    resume = not args.no_resume

    print(f"Building SAM3 model from {CHECKPOINT_PATH} ...")
    print(f"Resume mode: {'ON' if resume else 'OFF'}")
    model, processor = build_model()

    for split in ["test", "train"]:
        print(f"\n{'=' * 60}")
        print(f"Processing: {split}")
        print(f"{'=' * 60}")
        n, skipped = process_directory(
            processor, INPUT_DIRS[split], OUTPUT_DIRS[split], resume=resume
        )
        if skipped:
            print(f"Skipped {skipped} already-processed image(s)")
        print(f"Saved {n} mask(s) to {OUTPUT_DIRS[split]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
