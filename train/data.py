from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


def build_image_index(images_root: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    stem_to_paths: Dict[str, List[Path]] = {}
    for path in images_root.rglob("*"):
        if path.is_file():
            image_index[path.name] = path
            stem_to_paths.setdefault(path.stem, []).append(path)

    # Add extension fallback aliases (e.g., r1.jpg -> r1.jpeg) when stem is unique.
    extension_aliases = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
    for stem, paths in stem_to_paths.items():
        if len(paths) != 1:
            continue
        only_path = paths[0]
        for suffix in extension_aliases:
            alias = f"{stem}{suffix}"
            image_index.setdefault(alias, only_path)
    return image_index


def load_skip_ids(skip_ids_txt: Path) -> List[str]:
    if not skip_ids_txt.is_file():
        raise FileNotFoundError(f"Skip id file not found: {skip_ids_txt}")

    skip_ids: List[str] = []
    with skip_ids_txt.open("r", encoding="utf-8") as f:
        for line in f:
            image_id = line.strip()
            if image_id:
                skip_ids.append(image_id)
    return list(dict.fromkeys(skip_ids))


def filter_dataframe_by_skip_ids(
    df: pd.DataFrame,
    skip_ids_txt: Path,
    id_column: str = "id",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if id_column not in df.columns:
        raise ValueError(f"Missing required id column: {id_column}")
    unique_skip_ids = load_skip_ids(skip_ids_txt)
    dataset_ids = set(df[id_column].astype(str).tolist())
    matched_skip_ids = [image_id for image_id in unique_skip_ids if image_id in dataset_ids]
    unmatched_skip_ids = [image_id for image_id in unique_skip_ids if image_id not in dataset_ids]

    filtered_df = df.loc[~df[id_column].astype(str).isin(set(matched_skip_ids))].reset_index(drop=True)
    stats: Dict[str, object] = {
        "skip_ids_txt": str(skip_ids_txt),
        "requested_skip_count": len(unique_skip_ids),
        "matched_skip_count": len(matched_skip_ids),
        "unmatched_skip_count": len(unmatched_skip_ids),
        "remaining_total_count": len(filtered_df),
        "unmatched_examples": unmatched_skip_ids[:10],
    }
    return filtered_df, stats


def build_unsupervised_image_list(
    image_roots: Sequence[Path],
    skip_ids_txt: Optional[Path] = None,
) -> Tuple[List[Path], Dict[str, object]]:
    resolved_roots = [root.resolve() for root in image_roots]
    missing_roots = [str(root) for root in resolved_roots if not root.is_dir()]
    if missing_roots:
        raise FileNotFoundError(f"Image roots not found: {missing_roots}")

    skip_ids = load_skip_ids(skip_ids_txt.resolve()) if skip_ids_txt is not None else []
    skip_id_set = set(skip_ids)
    image_paths: List[Path] = []
    skipped_paths: List[Path] = []

    for root in resolved_roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.name in skip_id_set:
                skipped_paths.append(path)
                continue
            image_paths.append(path)

    if not image_paths:
        raise RuntimeError("No images found for unsupervised training after filtering.")

    basename_counts = Counter(path.name for path in image_paths)
    duplicate_basenames = [name for name, count in basename_counts.items() if count > 1]
    matched_skip_ids = sorted({path.name for path in skipped_paths})
    unmatched_skip_ids = [image_id for image_id in skip_ids if image_id not in matched_skip_ids]
    stats: Dict[str, object] = {
        "image_roots": [str(root) for root in resolved_roots],
        "skip_ids_txt": str(skip_ids_txt.resolve()) if skip_ids_txt is not None else None,
        "requested_skip_count": len(skip_ids),
        "matched_skip_count": len(matched_skip_ids),
        "unmatched_skip_count": len(unmatched_skip_ids),
        "unmatched_examples": unmatched_skip_ids[:10],
        "kept_image_count": len(image_paths),
        "skipped_image_count": len(skipped_paths),
        "duplicate_basename_count": len(duplicate_basenames),
        "duplicate_basename_examples": duplicate_basenames[:10],
    }
    return image_paths, stats


def stratified_split(
    df: pd.DataFrame,
    label_column: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1).")

    grouped = []
    for _, group in df.groupby(label_column):
        group = group.sample(frac=1.0, random_state=seed)
        val_count = max(1, int(round(len(group) * val_ratio)))
        val_count = min(val_count, len(group) - 1) if len(group) > 1 else 0
        grouped.append((group.iloc[val_count:], group.iloc[:val_count]))

    train_parts = [train for train, _ in grouped if len(train) > 0]
    val_parts = [val for _, val in grouped if len(val) > 0]
    if not train_parts or not val_parts:
        raise RuntimeError("Failed to build a non-empty train/val split.")

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train_df, val_df


def stratified_subsplit(
    df: pd.DataFrame,
    label_column: str,
    first_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < first_ratio < 1.0):
        raise ValueError("--threshold-search-ratio must be in (0, 1).")

    grouped = []
    for _, group in df.groupby(label_column):
        group = group.sample(frac=1.0, random_state=seed)
        first_count = max(1, int(round(len(group) * first_ratio)))
        first_count = min(first_count, len(group) - 1) if len(group) > 1 else 0
        grouped.append((group.iloc[:first_count], group.iloc[first_count:]))

    first_parts = [first for first, _ in grouped if len(first) > 0]
    second_parts = [second for _, second in grouped if len(second) > 0]
    if not first_parts or not second_parts:
        raise RuntimeError("Failed to build a non-empty threshold/model-selection split.")

    first_df = pd.concat(first_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    second_df = pd.concat(second_parts, axis=0).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return first_df, second_df


def rebalance_binary_subset_to_ratio(
    df: pd.DataFrame,
    label_column: str,
    target_neg_pos_ratio: float,
    seed: int,
    negative_label: int = 0,
    positive_label: int = 1,
) -> pd.DataFrame:
    if target_neg_pos_ratio <= 0.0:
        raise ValueError("target_neg_pos_ratio must be positive.")
    if label_column not in df.columns:
        raise ValueError(f"Missing required label column: {label_column}")

    negative_df = df[df[label_column] == negative_label]
    positive_df = df[df[label_column] == positive_label]
    if negative_df.empty or positive_df.empty:
        raise ValueError("Binary rebalancing requires at least one sample from each class.")

    current_ratio = len(negative_df) / len(positive_df)
    if current_ratio < target_neg_pos_ratio:
        target_positive_count = max(1, int(round(len(negative_df) / target_neg_pos_ratio)))
        sampled_negative_df = negative_df
        sampled_positive_df = positive_df.sample(n=min(len(positive_df), target_positive_count), random_state=seed)
    else:
        target_negative_count = max(1, int(round(len(positive_df) * target_neg_pos_ratio)))
        sampled_negative_df = negative_df.sample(n=min(len(negative_df), target_negative_count), random_state=seed)
        sampled_positive_df = positive_df

    rebalanced_df = pd.concat([sampled_negative_df, sampled_positive_df], axis=0)
    return rebalanced_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def build_pseudo_labeled_dataframe(
    prob_csv: Path,
    confidence_threshold: float,
    positive_label: int = POSITIVE_LABEL,
    negative_label: int = NEGATIVE_LABEL,
) -> pd.DataFrame:
    if not (0.5 <= confidence_threshold <= 1.0):
        raise ValueError("Pseudo-label confidence threshold must be in [0.5, 1.0].")

    df = pd.read_csv(prob_csv, dtype={"id": str})
    required_columns = {"id", "prob_pos_corrected"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{prob_csv} is missing required columns: {sorted(missing_columns)}")

    df = df.copy()
    df["prob_pos_corrected"] = df["prob_pos_corrected"].astype(float)
    df["pseudo_confidence"] = df["prob_pos_corrected"].where(
        df["prob_pos_corrected"] >= 0.5,
        1.0 - df["prob_pos_corrected"],
    )
    selected_df = df[df["pseudo_confidence"] >= float(confidence_threshold)].copy()
    if selected_df.empty:
        return selected_df.assign(label_idx=pd.Series(dtype="int64"), label=pd.Series(dtype="int64"))

    selected_df["label_idx"] = selected_df["prob_pos_corrected"].ge(0.5).map({True: positive_label, False: negative_label})
    selected_df["label"] = selected_df["label_idx"]
    selected_df["source"] = "pseudo"
    return selected_df[["id", "label", "label_idx", "prob_pos_corrected", "pseudo_confidence", "source"]].reset_index(drop=True)


def build_mask_image_index(mask_dir: Path, image_ids: List[str]) -> tuple:
    mask_index: Dict[str, Path] = {}
    masked_ids: List[str] = []
    skipped_ids: List[str] = []
    for image_id in image_ids:
        stem = Path(image_id).stem
        mask_path = mask_dir / f"{stem}_mask_000.png"
        if mask_path.is_file():
            mask_index[image_id] = mask_path
            mask_index.setdefault(stem, mask_path)
            masked_ids.append(image_id)
        else:
            skipped_ids.append(image_id)
    return mask_index, masked_ids, skipped_ids


class MeteoriteDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_index: Dict[str, Path],
        transform: transforms.Compose,
        original_image_index: Optional[Dict[str, Path]] = None,
        flip_mask: bool = False,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform
        self.original_image_index = original_image_index
        self.flip_mask = flip_mask

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_id = str(row["id"])
        image_path = self.image_index.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Image not found under train_images: {image_id}")

        if self.flip_mask and self.original_image_index is not None:
            original_path = self.original_image_index.get(image_id)
            if original_path is None:
                raise FileNotFoundError(f"Original image not found: {image_id}")
            with Image.open(original_path) as original_file, Image.open(image_path) as mask_file:
                original = np.array(original_file.convert("RGB"))
                mask = np.array(mask_file.convert("RGB"))
            meteorite_area = mask.sum(axis=2) > 30
            flipped = original.copy()
            flipped[meteorite_area] = 0
            image = Image.fromarray(flipped)
        else:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")

        pixel_values = self.transform(image)
        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        sample_weight = torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32)
        return pixel_values, label, sample_weight


class UnsupervisedImageDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        transform: Callable,
    ) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        crops = self.transform(image)
        return crops, image_path.name
