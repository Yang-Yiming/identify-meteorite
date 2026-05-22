#!/usr/bin/env python3
"""Temperature scaling and ensemble optimization for testlike_dino_myval_v3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import minimize
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
    train_args = json.loads(train_args_path.read_text(encoding="utf-8")) if train_args_path.is_file() else {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
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


def get_logits(model, df, transform, device, batch_size, num_workers):
    loader = DataLoader(
        PathDataset(df, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    all_logits = []
    autocast_enabled = device.type == "cuda"
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(images)
            all_logits.append(logits.detach().cpu().numpy())
    return np.concatenate(all_logits)


def get_probs(model, df, transform, device, batch_size, num_workers):
    logits = get_logits(model, df, transform, device, batch_size, num_workers)
    return torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()


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


def temperature_scale(logits, labels, temperature_init=1.0):
    """Fit temperature scaling to minimize NLL."""
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels).long()
    temperature = torch.nn.Parameter(torch.tensor([temperature_init]))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def eval_fn():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    return temperature.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-val", type=Path, default=Path("analysis/testlike_dino_myval_v3/test_like_val_cluster.csv"))
    parser.add_argument("--top-val", type=Path, default=Path("analysis/testlike_dino_myval_v3/test_like_val_top.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("train/outputs/myval_v13_hi288_seed42_soup/soup.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/calibration_tune"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cluster_val = pd.read_csv(args.cluster_val)
    top_val = pd.read_csv(args.top_val)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, transform = load_model(args.checkpoint, device)

    results = []
    for ds_name, ds_df in [("cluster", cluster_val), ("top", top_val)]:
        labels = ds_df["label"].astype(int).to_numpy()

        # Baseline
        probs = get_probs(model, ds_df, transform, device, args.batch_size, args.num_workers)
        base_f1 = f1_at(probs, labels, 0.5)
        base_best_t, base_best_f1 = best_threshold(probs, labels)
        print(f"  {ds_name}: baseline F1@0.5={base_f1:.4f}, best_thr={base_best_t:.4f}, best_f1={base_best_f1:.4f}")

        # Temperature scaling on this dataset
        logits = get_logits(model, ds_df, transform, device, args.batch_size, args.num_workers)
        temp = temperature_scale(logits, labels)
        scaled_probs = torch.softmax(torch.from_numpy(logits) / temp, dim=1)[:, 1].numpy()
        temp_f1 = f1_at(scaled_probs, labels, 0.5)
        temp_best_t, temp_best_f1 = best_threshold(scaled_probs, labels)
        print(f"  {ds_name}: temp={temp:.4f}, scaled F1@0.5={temp_f1:.4f}, best_thr={temp_best_t:.4f}, best_f1={temp_best_f1:.4f}")

        # Threshold sweep
        thr_sweep = []
        for thr in np.arange(0.25, 0.76, 0.01):
            f = f1_at(probs, labels, thr)
            thr_sweep.append({"threshold": thr, "f1": f})
        thr_df = pd.DataFrame(thr_sweep)
        best_thr_row = thr_df.loc[thr_df["f1"].idxmax()]
        print(f"  {ds_name}: threshold sweep best thr={best_thr_row['threshold']:.2f}, f1={best_thr_row['f1']:.4f}")

        results.append({
            "dataset": ds_name,
            "baseline_f1_05": base_f1,
            "baseline_best_f1": base_best_f1,
            "baseline_best_thr": base_best_t,
            "temperature": temp,
            "temp_scaled_f1_05": temp_f1,
            "temp_scaled_best_f1": temp_best_f1,
            "temp_scaled_best_thr": temp_best_t,
            "thr_sweep_best_thr": best_thr_row["threshold"],
            "thr_sweep_best_f1": best_thr_row["f1"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out_dir / "calibration_results.csv", index=False)
    print(f"\nFull results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
