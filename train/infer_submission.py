import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from modeling import ConvNeXtClassifier, build_transforms
from tta import predict_probabilities_with_tta, resolve_tta_views
from utils import DEFAULT_BACKBONE, DEFAULT_MEAN, DEFAULT_STD, normalize_image_size, normalize_stats


DEFAULT_DATA_DIR = Path("../data")
DEFAULT_TEST_IMAGES_DIR = DEFAULT_DATA_DIR / "test_images"
DEFAULT_SAMPLE_SUBMISSION = DEFAULT_DATA_DIR / "sample_submission.csv"
DEFAULT_OUTPUT_CSV = Path("./output.csv")
POSITIVE_LABEL = 1


MaskSample = Tuple[str, Path]


def build_image_index(images_root: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.is_file():
            image_index[path.name] = path
    return image_index


def apply_bayes_prior_correction(
    prob_pos: torch.Tensor,
    train_pos_prior: float,
    target_pos_prior: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob_pos = prob_pos.clamp(min=eps, max=1.0 - eps)
    train_pos_prior = min(max(float(train_pos_prior), eps), 1.0 - eps)
    target_pos_prior = min(max(float(target_pos_prior), eps), 1.0 - eps)

    prior_logit_delta = math.log(target_pos_prior / (1.0 - target_pos_prior)) - math.log(
        train_pos_prior / (1.0 - train_pos_prior)
    )
    corrected_logits = torch.logit(prob_pos) + prior_logit_delta
    return torch.sigmoid(corrected_logits)


class TestImageDataset(Dataset):
    def __init__(
        self,
        image_ids: Sequence[str],
        image_index: Dict[str, Path],
        transform,
        samples: Optional[Sequence[MaskSample]] = None,
        original_image_index: Optional[Dict[str, Path]] = None,
        apply_mask: bool = False,
        flip_mask: bool = False,
    ) -> None:
        self.image_ids = list(image_ids)
        self.image_index = image_index
        self.transform = transform
        self.samples = list(samples) if samples is not None else None
        self.original_image_index = original_image_index
        self.apply_mask = apply_mask
        self.flip_mask = flip_mask

    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        if self.samples is not None:
            image_id, image_path = self.samples[index]
        else:
            image_id = self.image_ids[index]
            image_path = self.image_index.get(image_id)
            if image_path is None:
                raise FileNotFoundError(f"Image not found under test_images: {image_id}")

        if (self.apply_mask or self.flip_mask) and self.original_image_index is not None:
            original_path = self.original_image_index.get(image_id)
            if original_path is None:
                raise FileNotFoundError(f"Original image not found: {image_id}")
            with Image.open(original_path) as original_file, Image.open(image_path) as mask_file:
                original = np.array(original_file.convert("RGB"))
                mask_image = mask_file.convert("L")
                if mask_image.size != original_file.size:
                    nearest = getattr(Image, "Resampling", Image).NEAREST
                    mask_image = mask_image.resize(original_file.size, resample=nearest)
                mask = np.array(mask_image)
            mask_area = mask > 0
            masked = original.copy()
            if self.flip_mask:
                masked[mask_area] = 0
            else:
                masked[~mask_area] = 0
            image = Image.fromarray(masked)
        else:
            image = Image.open(image_path).convert("RGB")

        pixel_values = self.transform(image)
        return pixel_values, image_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ConvNeXt checkpoint inference and write a Kaggle submission CSV")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a training checkpoint such as outputs/convnext_tiny_finetune/best.pt",
    )
    parser.add_argument("--test-images-dir", type=Path, default=DEFAULT_TEST_IMAGES_DIR)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--output-prob-csv",
        type=Path,
        default=None,
        help="Optional CSV with id, prob_pos, prob_pos_corrected, label",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", type=str, default=None, help="Override timm backbone name used to rebuild model")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--image-mean", type=float, nargs=3, default=None)
    parser.add_argument("--image-std", type=float, nargs=3, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override classification threshold. Defaults to checkpoint threshold or 0.5",
    )
    parser.add_argument("--disable-bayes-correction", action="store_true")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable deterministic test-time augmentation and average probabilities across views.",
    )
    parser.add_argument(
        "--tta-mode",
        type=str,
        choices=("4way", "8way"),
        default="4way",
        help="TTA view set to use when --tta is enabled.",
    )
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--use-mask",
        dest="use_mask",
        action="store_true",
        default=True,
        help="Use pre-computed mask images for inference (default)",
    )
    mask_group.add_argument(
        "--no-use-mask",
        dest="use_mask",
        action="store_false",
        help="Disable mask inference and run on original test images",
    )
    parser.add_argument("--mask-dir", type=Path, default=Path("../mask"), help="Directory containing mask images")
    parser.add_argument("--flip-mask", action="store_true", help="Invert mask: keep only background, remove meteorite area")
    return parser.parse_args()


def load_json_if_exists(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def read_submission_ids(sample_submission_path: Path) -> List[str]:
    with sample_submission_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"{sample_submission_path} must contain an 'id' column")
        return [str(row["id"]) for row in reader]


def collate_test_batch(batch: Iterable[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    pixel_values, image_ids = zip(*batch)
    return torch.stack(list(pixel_values), dim=0), list(image_ids)


def build_mask_inference_samples(
    mask_dir: Path,
    image_ids: Sequence[str],
) -> Tuple[List[MaskSample], Dict[str, int], List[str], List[str]]:
    samples: List[MaskSample] = []
    mask_counts: Dict[str, int] = {}
    nomask_ids: List[str] = []
    missing_marker_ids: List[str] = []

    for image_id in image_ids:
        stem = Path(image_id).stem
        mask_paths = sorted(mask_dir.glob(f"{stem}_mask_*.png"))
        mask_counts[image_id] = len(mask_paths)
        if mask_paths:
            samples.extend((image_id, mask_path) for mask_path in mask_paths)
            continue

        nomask_ids.append(image_id)
        if not (mask_dir / f"{stem}_nomask.done").is_file():
            missing_marker_ids.append(image_id)

    return samples, mask_counts, nomask_ids, missing_marker_ids


def resolve_runtime_settings(
    args: argparse.Namespace,
    metadata: Dict[str, object],
    train_args: Dict[str, object],
    checkpoint_payload: Dict[str, object],
) -> Dict[str, object]:
    image_size = args.image_size or metadata.get("image_size")
    image_mean = args.image_mean or metadata.get("image_mean")
    image_std = args.image_std or metadata.get("image_std")
    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint_payload.get("threshold", 0.5)
    dropout = args.dropout
    if dropout is None:
        dropout = train_args.get("dropout", 0.0)
    backbone_name = args.backbone
    if backbone_name is None:
        backbone_name = train_args.get("backbone") or metadata.get("backbone_name") or DEFAULT_BACKBONE

    return {
        "image_size": normalize_image_size(image_size),
        "image_mean": normalize_stats(image_mean, DEFAULT_MEAN),
        "image_std": normalize_stats(image_std, DEFAULT_STD),
        "threshold": float(threshold),
        "dropout": float(dropout),
        "backbone_name": str(backbone_name),
    }


def maybe_correct_probabilities(
    probabilities: torch.Tensor,
    metadata: Dict[str, object],
    disable_bayes_correction: bool,
) -> torch.Tensor:
    if disable_bayes_correction:
        return probabilities
    if not metadata.get("bayes_correction_enabled", False):
        return probabilities

    train_priors = metadata.get("train_priors")
    target_priors = metadata.get("target_priors")
    if not isinstance(train_priors, dict) or not isinstance(target_priors, dict):
        return probabilities

    train_prior_values = train_priors.get("priors")
    target_prior_values = target_priors.get("priors")
    if not isinstance(train_prior_values, dict) or not isinstance(target_prior_values, dict):
        return probabilities

    train_pos_prior = float(train_prior_values[str(POSITIVE_LABEL)])
    target_pos_prior = float(target_prior_values[str(POSITIVE_LABEL)])
    return apply_bayes_prior_correction(
        probabilities,
        train_pos_prior=train_pos_prior,
        target_pos_prior=target_pos_prior,
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve()
    checkpoint_dir = checkpoint_path.parent
    metadata = load_json_if_exists(checkpoint_dir / "metadata.json") or {}
    train_args = load_json_if_exists(checkpoint_dir / "train_args.json") or {}

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_payload, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = checkpoint_payload.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint does not contain a model state_dict: {checkpoint_path}")

    settings = resolve_runtime_settings(args, metadata, train_args, checkpoint_payload)
    _, eval_transform = build_transforms(
        settings["image_size"],
        settings["image_mean"],
        settings["image_std"],
        hflip_prob=0.0,
        rotate_degrees=0.0,
    )

    idx_to_label = metadata.get("idx_to_label")
    num_classes = len(idx_to_label) if isinstance(idx_to_label, dict) and idx_to_label else 2
    model = ConvNeXtClassifier(
        backbone_name=settings["backbone_name"],
        backbone_checkpoint=None,
        num_classes=num_classes,
        dropout=settings["dropout"],
        pretrained_backbone=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Loaded classifier checkpoint | missing_keys={len(missing)} | unexpected_keys={len(unexpected)}")

    if args.flip_mask and not args.use_mask:
        raise ValueError("--flip-mask requires mask inference. Remove --no-use-mask or --flip-mask.")

    test_images_dir = args.test_images_dir.resolve()
    image_index = build_image_index(test_images_dir)
    test_original_image_index = image_index
    if not image_index:
        raise RuntimeError(f"No images found under {test_images_dir}")

    sample_submission_path = args.sample_submission.resolve()
    if sample_submission_path.is_file():
        image_ids = read_submission_ids(sample_submission_path)
    else:
        image_ids = sorted(image_index.keys())

    missing_original_ids = [image_id for image_id in image_ids if image_id not in test_original_image_index]
    if missing_original_ids:
        raise RuntimeError(f"Missing test images for sample submission ids; examples: {missing_original_ids[:5]}")

    mask_samples: Optional[List[MaskSample]] = None
    mask_counts: Dict[str, int] = {}
    nomask_ids: List[str] = []
    if args.use_mask:
        mask_test_dir = args.mask_dir.resolve() / "test"
        if not mask_test_dir.is_dir():
            raise FileNotFoundError(f"Mask test directory not found: {mask_test_dir}")
        mask_samples, mask_counts, nomask_ids, missing_marker_ids = build_mask_inference_samples(mask_test_dir, image_ids)
        print(
            f"Mask inference | mask_dir={mask_test_dir} | "
            f"masked_images={sum(count > 0 for count in mask_counts.values())} | "
            f"mask_instances={len(mask_samples)} | nomask_images={len(nomask_ids)}"
        )
        if missing_marker_ids:
            print(
                f"Warning: {len(missing_marker_ids)} ids have no mask files and no *_nomask.done marker; "
                f"examples={missing_marker_ids[:5]}. They will be predicted as 0."
            )

    flip_mask_test = args.use_mask and args.flip_mask
    apply_mask_test = args.use_mask and not args.flip_mask
    test_original = test_original_image_index if flip_mask_test else None
    if apply_mask_test:
        test_original = test_original_image_index
    dataset = TestImageDataset(
        image_ids=image_ids, image_index=image_index, transform=eval_transform,
        samples=mask_samples,
        original_image_index=test_original, apply_mask=apply_mask_test, flip_mask=flip_mask_test,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_test_batch,
    )

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    tta_views = resolve_tta_views(args.tta_mode) if args.tta else ["identity"]
    print(f"Inference config | tta_enabled={args.tta} | tta_mode={args.tta_mode if args.tta else 'off'} | num_views={len(tta_views)}")

    ordered_ids: List[str] = []
    raw_probabilities: List[torch.Tensor] = []
    autocast_enabled = device.type == "cuda"

    with torch.no_grad():
        for pixel_values, batch_ids in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            if args.tta:
                prob_pos = predict_probabilities_with_tta(
                    model,
                    pixel_values,
                    positive_label=POSITIVE_LABEL,
                    views=tta_views,
                    device_type=device.type,
                    autocast_enabled=autocast_enabled,
                )
            else:
                with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                    logits = model(pixel_values)
                prob_pos = torch.softmax(logits, dim=1)[:, POSITIVE_LABEL]
            ordered_ids.extend(batch_ids)
            raw_probabilities.append(prob_pos.detach().cpu())

    raw_instance_prob_pos = torch.cat(raw_probabilities) if raw_probabilities else torch.empty(0, dtype=torch.float32)
    if args.use_mask:
        if len(ordered_ids) != len(raw_instance_prob_pos):
            raise RuntimeError("Mask inference output length mismatch; aborting to avoid writing corrupted probabilities")
        prob_sums = {image_id: 0.0 for image_id in image_ids}
        prob_counts = {image_id: 0 for image_id in image_ids}
        for image_id, probability in zip(ordered_ids, raw_instance_prob_pos.tolist()):
            prob_sums[image_id] += float(probability)
            prob_counts[image_id] += 1
        raw_prob_pos = torch.tensor(
            [
                prob_sums[image_id] / prob_counts[image_id] if prob_counts[image_id] > 0 else 0.0
                for image_id in image_ids
            ],
            dtype=torch.float32,
        )
        ordered_ids = list(image_ids)
    else:
        if ordered_ids != image_ids:
            raise RuntimeError("Inference order mismatch; aborting to avoid writing a corrupted submission")
        raw_prob_pos = raw_instance_prob_pos

    corrected_prob_pos = maybe_correct_probabilities(
        raw_prob_pos,
        metadata=metadata,
        disable_bayes_correction=args.disable_bayes_correction,
    )
    labels = (corrected_prob_pos >= settings["threshold"]).to(torch.int64)
    if args.use_mask and nomask_ids:
        nomask_id_set = set(nomask_ids)
        nomask_positions = [idx for idx, image_id in enumerate(ordered_ids) if image_id in nomask_id_set]
        if nomask_positions:
            corrected_prob_pos[nomask_positions] = 0.0
            labels[nomask_positions] = 0

    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for image_id, label in zip(ordered_ids, labels.tolist()):
            writer.writerow([image_id, int(label)])

    df = pd.read_csv(output_csv)
    print(df["label"].value_counts())

    if args.output_prob_csv is not None:
        output_prob_csv = args.output_prob_csv.resolve()
        output_prob_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_prob_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prob_pos", "prob_pos_corrected", "label"])
            for image_id, raw_prob, corrected_prob, label in zip(
                ordered_ids,
                raw_prob_pos.tolist(),
                corrected_prob_pos.tolist(),
                labels.tolist(),
            ):
                writer.writerow([image_id, raw_prob, corrected_prob, int(label)])

    positive_predictions = int(labels.sum().item())
    print(
        f"Wrote {len(ordered_ids)} predictions to {output_csv} | "
        f"threshold={settings['threshold']:.6f} | positive_predictions={positive_predictions}"
    )


if __name__ == "__main__":
    main()
