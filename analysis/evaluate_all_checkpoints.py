#!/usr/bin/env python3
"""Evaluate ALL existing checkpoints on testlike_dino_myval_v3 sets."""
from __future__ import annotations

import argparse
import json
import os
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), int(row["label"]), row["sample_id"]


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    train_args_path = checkpoint_path.parent / "train_args.json"
    metadata_path = checkpoint_path.parent / "metadata.json"
    train_args = json.loads(train_args_path.read_text(encoding="utf-8")) if train_args_path.is_file() else {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    backbone_name = train_args.get("backbone", DEFAULT_BACKBONE)
    dropout = float(train_args.get("dropout", 0.0))
    image_size = normalize_image_size(metadata.get("image_size", train_args.get("image_size", 224)))
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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    model.to(device)
    model.eval()
    return model, eval_transform


def infer_probs(model, df, transform, device, batch_size, num_workers):
    loader = DataLoader(
        PathDataset(df, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    probs = []
    autocast_enabled = device.type == "cuda"
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
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


def best_threshold(probs, labels):
    best_t, best_f1 = 0.5, -1.0
    for thresh in sorted(set(np.append(probs, [0.0, 0.5, 1.0]))):
        score = f1_at(probs, labels, thresh)
        if score > best_f1:
            best_t, best_f1 = float(thresh), float(score)
    return best_t, best_f1


def discover_checkpoints(base_dir: Path) -> list[tuple[str, Path]]:
    """Find all checkpoints (best.pt, soup*.pt) with train_args.json."""
    results = []
    for dir_path in sorted(base_dir.iterdir()):
        if not dir_path.is_dir():
            continue
        for ckpt_name in sorted(dir_path.iterdir()):
            if ckpt_name.suffix == ".pt" and (ckpt_name.stem.startswith("soup") or ckpt_name.stem == "best"):
                train_args = dir_path / "train_args.json"
                if train_args.is_file():
                    tag = f"{dir_path.name}/{ckpt_name.name}"
                    results.append((tag, ckpt_name))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("analysis/testlike_dino_myval_v3/manifest.csv"))
    parser.add_argument("--cluster-val", type=Path, default=Path("analysis/testlike_dino_myval_v3/test_like_val_cluster.csv"))
    parser.add_argument("--top-val", type=Path, default=Path("analysis/testlike_dino_myval_v3/test_like_val_top.csv"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("train/outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/all_checkpoints_eval"))
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = pd.read_csv(args.manifest)
    myval = manifest[(manifest["source"] == "myval") & (manifest["has_image"])].copy()
    myval = myval[["sample_id", "source", "image_id", "label", "path"]]
    cluster_val = pd.read_csv(args.cluster_val)
    top_val = pd.read_csv(args.top_val)
    datasets = {
        "myval_masked": myval,
        "testlike_cluster_dino_v3": cluster_val,
        "testlike_top_dino_v3": top_val,
    }

    checkpoints = discover_checkpoints(args.outputs_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    rows = []
    for tag, ckpt_path in checkpoints:
        print(f"Evaluating {tag}...")
        try:
            model, transform = load_model(ckpt_path, device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        for ds_name, ds_df in datasets.items():
            probs = infer_probs(model, ds_df, transform, device, args.batch_size, args.num_workers)
            labels = ds_df["label"].astype(int).to_numpy()
            best_t, best_f1 = best_threshold(probs, labels)
            preds = (probs >= 0.5).astype(int)
            rows.append({
                "run_tag": tag,
                "checkpoint": str(ckpt_path),
                "dataset": ds_name,
                "f1_at_0_5": f1_at(probs, labels, 0.5),
                "best_f1": best_f1,
                "best_threshold": best_t,
                "n": len(ds_df),
                "pos_true": int(labels.sum()),
                "pos_pred_at_0_5": int(preds.sum()),
                "prob_mean": float(probs.mean()),
                "prob_median": float(np.median(probs)),
                "prob_std": float(probs.std()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "all_eval_long.csv", index=False)

    # Pivot to wide format
    index_cols = ["run_tag", "checkpoint"]
    base = df[index_cols].drop_duplicates("run_tag").set_index("run_tag")
    metric_cols = ["f1_at_0_5", "best_f1", "best_threshold", "pos_pred_at_0_5", "prob_mean"]
    parts = []
    for ds_name in datasets:
        sub = df[df["dataset"] == ds_name].set_index("run_tag")
        part = sub[metric_cols].rename(columns={c: f"{c}__{ds_name}" for c in metric_cols})
        parts.append(part)
    pivot = base.join(parts, how="left").reset_index()
    pivot.to_csv(args.out_dir / "all_eval_summary.csv", index=False)

    # Print top checkpoints by each metric
    for ds_name, display_name in [
        ("testlike_cluster_dino_v3", "DINO Cluster"),
        ("testlike_top_dino_v3", "DINO Top"),
        ("myval_masked", "Myval Masked"),
    ]:
        col = f"f1_at_0_5__{ds_name}"
        if col in pivot.columns:
            top = pivot.nlargest(10, col)[["run_tag", col, f"best_f1__{ds_name}"]]
            print(f"\n=== Top 10 by {display_name} F1@0.5 ===")
            print(top.to_string(index=False))

    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
