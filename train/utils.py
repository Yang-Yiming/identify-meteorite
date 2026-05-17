import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from wandb_utils import add_wandb_args

DEFAULT_BACKBONE = "convnext_tiny"
DEFAULT_DATA_DIR = Path("../data")
DEFAULT_LABELS_CSV = DEFAULT_DATA_DIR / "train_labels.csv"
DEFAULT_VAL_ROOT = DEFAULT_DATA_DIR / "myval"
DEFAULT_VAL_MASK_SPLIT = "myval"
DEFAULT_PSEUDO_IMAGES_DIR = DEFAULT_DATA_DIR / "test_images"
DEFAULT_OUTPUT_DIR = Path("./outputs/convnextv2_tiny_finetune")
DEFAULT_TARGET_NEG_POS_RATIO = 4.06
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvNeXt Tiny finetuning entrypoint")
    parser.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_BACKBONE,
        help="timm model name (default: convnext_tiny)",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=None,
        help="Optional local checkpoint for backbone weights.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable timm pretrained backbone weights.",
    )
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument(
        "--val-root",
        type=Path,
        default=DEFAULT_VAL_ROOT,
        help="Root directory for external validation labels. Defaults to ../data/myval.",
    )
    parser.add_argument(
        "--val-labels-csv",
        type=Path,
        default=None,
        help="Validation labels CSV. Defaults to <val-root>/labels.csv.",
    )
    parser.add_argument(
        "--val-mask-split",
        type=str,
        default=DEFAULT_VAL_MASK_SPLIT,
        help="Mask subdirectory under --mask-dir for external validation. Defaults to myval.",
    )
    parser.add_argument(
        "--val-split-ratio",
        type=float,
        default=0.0,
        help="When > 0, split this fraction of training data as validation (ignores --val-root).",
    )
    parser.add_argument("--pseudo-prob-csv", type=Path, default=None)
    parser.add_argument(
        "--pseudo-images-dir", type=Path, default=DEFAULT_PSEUDO_IMAGES_DIR
    )
    parser.add_argument(
        "--pseduo-prop",
        "--pseudo-prop",
        dest="pseudo_prop",
        type=float,
        default=0.95,
        help="Confidence threshold for pseudo-label selection from prob_pos_corrected CSVs.",
    )
    parser.add_argument(
        "--pseudo-weight",
        type=float,
        default=1.0,
        help="Per-sample loss weight assigned to pseudo-labeled samples (original labels remain 1.0).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-image-ids-txt", type=Path, default=None)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("../mask"),
        help="Directory containing SAM mask images, e.g. train/, myval/, test/.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--head-only-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--train-sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llrd-decay", type=float, default=0.8)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--image-mean", type=float, nargs=3, default=None)
    parser.add_argument("--image-std", type=float, nargs=3, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--target-neg-pos-ratio", type=float, default=DEFAULT_TARGET_NEG_POS_RATIO
    )
    parser.add_argument("--threshold-metric", type=str, default="f1")
    parser.add_argument("--threshold-search-ratio", type=float, default=0.5)
    parser.add_argument(
        "--open-threshold-search",
        action="store_true",
        help="Enable validation threshold search; otherwise use fixed threshold 0.5.",
    )
    parser.add_argument("--disable-bayes-correction", action="store_true")
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--rotate-degrees", type=float, default=15.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.7)
    parser.add_argument("--cutmix-prob", type=float, default=0.2)
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Mixup alpha parameter (0.0 disables mixup).")
    parser.add_argument("--mixup-prob", type=float, default=0.5,
                        help="Per-batch probability of applying mixup.")
    parser.add_argument("--randaugment-n", type=int, default=0,
                        help="Number of RandAugment operations (0 disables).")
    parser.add_argument("--randaugment-m", type=int, default=9,
                        help="RandAugment magnitude.")
    parser.add_argument("--color-jitter-prob", type=float, default=0.0,
                        help="Probability of applying ColorJitter (0.0 disables).")
    parser.add_argument("--color-jitter-brightness", type=float, default=0.4)
    parser.add_argument("--color-jitter-contrast", type=float, default=0.4)
    parser.add_argument("--color-jitter-saturation", type=float, default=0.4)
    parser.add_argument("--color-jitter-hue", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--early-stop",
        type=int,
        default=None,
        help="Stop after N consecutive epochs without val F1 improvement; disabled when omitted.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Max gradient norm for clipping (0.0 disables). Default: 0.0.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay rate (0.0 disables EMA). Default: 0.0.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        choices=("constant", "cosine"),
        help="LR scheduler type. Default: constant.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum LR for cosine scheduler. Default: 1e-6.",
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    add_wandb_args(parser, default_job_type="finetune")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_image_size(image_size: object) -> int:
    if isinstance(image_size, int):
        return image_size
    if isinstance(image_size, Sequence) and image_size:
        return int(image_size[0])
    return 224


def normalize_stats(
    values: Optional[Sequence[float]], fallback: List[float]
) -> List[float]:
    if isinstance(values, Sequence) and len(values) == 3:
        return [float(v) for v in values]
    return fallback


def save_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
