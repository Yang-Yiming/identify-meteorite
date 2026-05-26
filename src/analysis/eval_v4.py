#!/usr/bin/env python3
"""Evaluate key checkpoints on new testlike_dino_train_v4."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from modeling import ConvNeXtClassifier, build_transforms
from utils import DEFAULT_BACKBONE, DEFAULT_MEAN, DEFAULT_STD, normalize_image_size, normalize_stats


class PathDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), int(row["label"]), row["sample_id"]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    train_args_path = checkpoint_path.parent / "train_args.json"
    metadata_path = checkpoint_path.parent / "metadata.json"
    train_args = json.loads(train_args_path.read_text()) if train_args_path.is_file() else {}
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    backbone_name = train_args.get("backbone", DEFAULT_BACKBONE)
    dropout = float(train_args.get("dropout", 0.0))
    image_size = normalize_image_size(metadata.get("image_size", 224))
    image_mean = normalize_stats(metadata.get("image_mean"), DEFAULT_MEAN)
    image_std = normalize_stats(metadata.get("image_std"), DEFAULT_STD)
    num_classes = len(metadata.get("idx_to_label", {})) if metadata.get("idx_to_label") else 2
    _, eval_transform = build_transforms(
        image_size=image_size, image_mean=image_mean, image_std=image_std,
        hflip_prob=0.0, rotate_degrees=0.0,
    )
    model = ConvNeXtClassifier(
        backbone_name=backbone_name, backbone_checkpoint=None,
        num_classes=num_classes, dropout=dropout, pretrained_backbone=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, eval_transform


def infer_probs(model, df, transform, device, batch_size):
    loader = DataLoader(
        PathDataset(df, transform), batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=device.type == "cuda",
    )
    probs = []
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs)


def f1_at(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoints with known test F1
    checkpoints = [
        ("soup_reduced_notstone", Path("train/outputs/myval_v13_hi288_seed42_soup/soup.pt"), 0.71962),
        ("mytest_strict_dino_v1", Path("train/outputs/mytest_strict_dino_v1/best.pt"), None),
        ("mytest_split_protocol", Path("train/outputs/mytest_v1_s42/best.pt"), 0.65979),
        ("mytest_pretrain_finetune", Path("train/outputs/mytest_pretrain_finetune_v2/best.pt"), 0.55214),
        ("mytest_augment_soup", Path("train/outputs/mytest_augment_v2/soup_top3.pt"), 0.67021),
        ("splitval_augment_soup", Path("train/outputs/splitval_augment_v1/soup_top3.pt"), 0.63212),
    ]

    # New v4 datasets
    datasets = {
        "v3_cluster": Path("analysis/testlike_dino_myval_v3/test_like_val_cluster.csv"),
        "v3_top": Path("analysis/testlike_dino_myval_v3/test_like_val_top.csv"),
        "v4_cluster": Path("analysis/testlike_dino_train_v4/test_like_val_cluster.csv"),
        "v4_top": Path("analysis/testlike_dino_train_v4/test_like_val_top.csv"),
    }

    results = []
    for name, cp, test_f1 in checkpoints:
        print(f"Evaluating {name}...")
        model, transform = load_model(cp, device)
        for ds_name, ds_path in datasets.items():
            df = pd.read_csv(ds_path)
            labels = df["label"].astype(int).to_numpy()
            probs = infer_probs(model, df, transform, device, 128)
            f1 = f1_at(probs, labels, 0.5)
            pos_pred = int((probs >= 0.5).sum())
            results.append({
                "run": name,
                "test_f1": test_f1,
                "dataset": ds_name,
                "f1_at_0_5": f1,
                "pos_pred": pos_pred,
                "prob_mean": float(probs.mean()),
            })

    results_df = pd.DataFrame(results)

    print("\n=== V3 (myval candidates) vs V4 (train candidates) ===\n")
    for ds_group in [("v3_cluster", "v4_cluster"), ("v3_top", "v4_top")]:
        print(f"--- {ds_group[0]} ---")
        sub = results_df[results_df["dataset"] == ds_group[0]].sort_values("f1_at_0_5", ascending=False)
        print(sub[["run", "test_f1", "f1_at_0_5", "pos_pred"]].to_string(index=False))
        print(f"\n--- {ds_group[1]} ---")
        sub = results_df[results_df["dataset"] == ds_group[1]].sort_values("f1_at_0_5", ascending=False)
        print(sub[["run", "test_f1", "f1_at_0_5", "pos_pred"]].to_string(index=False))
        print()

    # Compute rank correlation
    for ds_name in datasets:
        sub = results_df[results_df["dataset"] == ds_name].dropna(subset=["test_f1"])
        if len(sub) >= 3:
            corr = sub["f1_at_0_5"].rank().corr(sub["test_f1"].rank())
            print(f"Rank correlation {ds_name}: {corr:.4f}")
        else:
            print(f"Rank correlation {ds_name}: N/A (too few points)")

    results_df.to_csv("analysis/testlike_v4_eval/v4_eval_results.csv", index=False)


if __name__ == "__main__":
    Path("analysis/testlike_v4_eval").mkdir(parents=True, exist_ok=True)
    main()
