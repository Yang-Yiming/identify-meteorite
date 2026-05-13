#!/usr/bin/env python3
"""Delete SAM mask images with white pixel ratio below a threshold (default 0.01)."""

import os
import re
import argparse
from PIL import Image
import numpy as np


def white_ratio(img_path):
    try:
        img = Image.open(img_path)
        arr = np.array(img)
        if arr.ndim == 3:
            white = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 255)
        else:
            white = arr == 255
        return white.mean()
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--masks-dir', default='masks')
    parser.add_argument('--dry-run', action='store_true', help='Only print, do not delete')
    args = parser.parse_args()
    threshold = args.threshold

    deleted = 0
    kept = 0
    for subdir in ['train', 'test']:
        subdir_path = os.path.join(args.masks_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in sorted(os.listdir(subdir_path)):
            if not fname.endswith('.png'):
                continue
            filepath = os.path.join(subdir_path, fname)
            ratio = white_ratio(filepath)
            if ratio < threshold:
                if args.dry_run:
                    print(f'[DRY-RUN] {subdir}/{fname} {ratio:.4f}')
                else:
                    os.remove(filepath)
                    print(f'{subdir}/{fname} {ratio:.4f}')
                deleted += 1
            else:
                kept += 1

    print(f'\nDeleted: {deleted}, Kept: {kept}')


if __name__ == '__main__':
    main()
