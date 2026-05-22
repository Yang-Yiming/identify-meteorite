#!/usr/bin/env python3
"""Quick evaluation of testlike_cluster_val_v1 checkpoint vs soup."""
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


def infer_probs(model, df, transform, device, batch_size, num_workers):
    loader = DataLoader(
        PathDataset(df, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
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

    # Checkpoints to evaluate
    checkpoints = {
        "soup": Path("train/outputs/myval_v13_hi288_seed42_soup/soup.pt"),
        "testlike_val_trained": Path("train/outputs/testlike_cluster_val_v1/best.pt"),
    }

    # Datasets to evaluate
    datasets = {
        "testlike_cluster_dino_v3": Path("analysis/testlike_dino_myval_v3/test_like_val_cluster.csv"),
        "testlike_top_dino_v3": Path("analysis/testlike_dino_myval_v3/test_like_val_top.csv"),
    }

    # Also add myval masked
    manifest = pd.read_csv("analysis/testlike_dino_myval_v3/manifest.csv")
    myval_masked = manifest[(manifest["source"] == "myval") & (manifest["has_image"])].copy()
    myval_masked = myval_masked[["sample_id", "source", "image_id", "label", "path"]]

    results = []
    for name, cp in checkpoints.items():
        print(f"Loading {name}: {cp}")
        model, transform = load_model(cp, device)

        # Evaluate on testlike sets
        for ds_name, ds_path in datasets.items():
            df = pd.read_csv(ds_path)
            labels = df["label"].astype(int).to_numpy()
            probs = infer_probs(model, df, transform, device, 128, 4)
            f1 = f1_at(probs, labels, 0.5)
            pos_pred = int((probs >= 0.5).sum())
            results.append({
                "model": name,
                "dataset": ds_name,
                "f1_at_0_5": f1,
                "pos_pred": pos_pred,
                "prob_mean": float(probs.mean()),
            })

        # Evaluate on myval masked
        probs = infer_probs(model, myval_masked, transform, device, 128, 4)
        labels = myval_masked["label"].astype(int).to_numpy()
        f1 = f1_at(probs, labels, 0.5)
        pos_pred = int((probs >= 0.5).sum())
        results.append({
            "model": name,
            "dataset": "myval_masked",
            "f1_at_0_5": f1,
            "pos_pred": pos_pred,
            "prob_mean": float(probs.mean()),
        })

    results_df = pd.DataFrame(results)
    print("\n=== Comparison ===")
    for ds_name in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == ds_name]
        print(f"\n{ds_name}:")
        for _, row in sub.iterrows():
            print(f"  {row['model']}: F1@0.5={row['f1_at_0_5']:.4f}  pos_pred={row['pos_pred']}  prob_mean={row['prob_mean']:.4f}")

    # Compute delta
    soup = results_df[results_df["model"] == "soup"].set_index("dataset")
    new = results_df[results_df["model"] == "testlike_val_trained"].set_index("dataset")
    print("\n=== Delta (testlike_trained - soup) ===")
    for ds_name in soup.index:
        delta = new.loc[ds_name, "f1_at_0_5"] - soup.loc[ds_name, "f1_at_0_5"]
        print(f"  {ds_name}: {delta:+.4f}")


if __name__ == "__main__":
    main()
