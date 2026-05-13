#!/usr/bin/env python3
"""
Filter SAM mask images by white pixel ratio.
Outputs mask files whose white-pixel proportion is below a threshold,
along with the best mask ratio for the same source image.
"""

import argparse
import os
import re
from collections import defaultdict
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


def parse_base_name(filename):
    """Extract base name like '000486' from '000486_mask_001.png'."""
    m = re.match(r'^(\d{6})_mask_\d+\.png$', filename)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(description='Filter SAM masks by white pixel ratio')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='White pixel ratio threshold (default: 0.4)')
    parser.add_argument('--masks-dir', default='masks',
                        help='Path to the masks directory (default: masks)')
    args = parser.parse_args()

    masks_dir = args.masks_dir
    threshold = args.threshold

    for subdir in ['train', 'test']:
        subdir_path = os.path.join(masks_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        group_files = defaultdict(list)  # base_name -> list of (filename, ratio)
        group_best = {}                  # base_name -> (best_filename, best_ratio)

        for fname in sorted(os.listdir(subdir_path)):
            if not fname.endswith('.png'):
                continue
            base = parse_base_name(fname)
            if base is None:
                continue
            filepath = os.path.join(subdir_path, fname)
            ratio = white_ratio(filepath)
            group_files[base].append((fname, ratio))

            if base not in group_best or ratio > group_best[base][1]:
                group_best[base] = (fname, ratio)

        print(f'[{subdir}]')
        for base in sorted(group_files):
            best_name, best_ratio = group_best[base]
            for fname, ratio in group_files[base]:
                if ratio < threshold:
                    if (fname, ratio) == (best_name, best_ratio):
                        print(f'{fname} {ratio:.4f}')
                    else:
                        print(f'{fname} {ratio:.4f} --- {best_ratio:.4f} {best_name}')
        print()


if __name__ == '__main__':
    main()
