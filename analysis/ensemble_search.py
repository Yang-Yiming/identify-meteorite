#!/usr/bin/env python3
"""Ensemble search: try combinations of top non-mytest checkpoints on testlike sets."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
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
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Bad checkpoint: {checkpoint_path}")
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


def infer_probs_multi(models, transforms, df, device, batch_size, num_workers):
    """Ensemble inference: average probabilities across models."""
    all_probs = np.zeros(len(df))
    for model, transform in zip(models, transforms):
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
        all_probs += np.concatenate(probs)
    return all_probs / len(models)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-val", type=Path, default=Path("analysis/testlike_dino_myval_v3/test_like_val_cluster.csv"))
    parser.add_argument("--top-val", type=Path, default=Path("analysis/testlike_dino_myval_v3/test_like_val_top.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/ensemble_search"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cluster_val = pd.read_csv(args.cluster_val)
    top_val = pd.read_csv(args.top_val)
    datasets = {
        "testlike_cluster_dino_v3": cluster_val,
        "testlike_top_dino_v3": top_val,
    }

    # Non-mytest checkpoints to try
    candidates = [
        Path("train/outputs/myval_v13_hi288_seed42_soup/soup.pt"),
        Path("train/outputs/myval_v13_hi288_seed42_soup/best.pt"),
        Path("train/outputs/myval_v11_hi288_seed42_thr/best.pt"),
        Path("train/outputs/myval_v9_hi288_seed42/best.pt"),
        Path("train/outputs/myval_v6_hi288/best.pt"),
        Path("train/outputs/myval_v4_cosine/best.pt"),
        Path("train/outputs/myval_v3_trsearch/best.pt"),
        Path("train/outputs/myval_v1_gc/best.pt"),
        Path("train/outputs/myval_v3_trsearch/soup_top5.pt"),
        Path("train/outputs/myval_v13_hi288_seed42_soup/soup_top2_uniform.pt"),
        Path("train/outputs/myval_v13_hi288_seed42_soup/soup_weighted.pt"),
        Path("train/outputs/myval_v13_hi288_seed42_soup/soup_top3_sq.pt"),
    ]
    # Filter to existing files
    candidates = [p for p in candidates if p.is_file()]
    print(f"Evaluating {len(candidates)} candidate checkpoints")

    # First, get individual scores for each candidate on testlike sets
    print("=== Individual model scores ===")
    model_cache = {}
    individual_scores = {}
    for cp in candidates:
        tag = f"{cp.parent.name}/{cp.name}"
        model, transform = load_model(cp, device)
        model_cache[tag] = (model, transform)
        scores = {}
        for ds_name, ds_df in datasets.items():
            loader = DataLoader(
                PathDataset(ds_df, transform), batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
            )
            probs = []
            autocast_enabled = device.type == "cuda"
            with torch.no_grad():
                for images, _, _ in loader:
                    images = images.to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                        logits = model(images)
                    probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            probs = np.concatenate(probs)
            labels = ds_df["label"].astype(int).to_numpy()
            scores[f"{ds_name}_f1_05"] = f1_at(probs, labels, 0.5)
            scores[f"{ds_name}_best_f1"] = best_threshold(probs, labels)[1]
        individual_scores[tag] = scores
        print(f"  {tag}: cluster={scores['testlike_cluster_dino_v3_f1_05']:.4f} top={scores['testlike_top_dino_v3_f1_05']:.4f}")

    # Now try ensembles
    print("\n=== Ensemble search ===")
    results = []
    for r in range(2, 6):  # try 2 to 5 model ensembles
        for combo in combinations(list(model_cache.keys()), r):
            models = [model_cache[name][0] for name in combo]
            transforms = [model_cache[name][1] for name in combo]
            combo_name = " + ".join(name.split("/")[-1].replace(".pt", "") for name in combo)
            for ds_name, ds_df in datasets.items():
                all_probs = np.zeros(len(ds_df))
                for model, transform in zip(models, transforms):
                    loader = DataLoader(
                        PathDataset(ds_df, transform), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
                    )
                    probs = []
                    autocast_enabled = device.type == "cuda"
                    with torch.no_grad():
                        for images, _, _ in loader:
                            images = images.to(device)
                            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                                logits = model(images)
                            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
                    all_probs += np.concatenate(probs)
                all_probs /= len(models)
                labels = ds_df["label"].astype(int).to_numpy()
                best_t, best_f = best_threshold(all_probs, labels)
                f1_05 = f1_at(all_probs, labels, 0.5)
                results.append({
                    "n_models": len(combo),
                    "ensemble": combo_name,
                    "checkpoints": " || ".join(str(cp) for cp in combo),
                    "dataset": ds_name,
                    "f1_at_0_5": f1_05,
                    "best_f1": best_f,
                    "best_threshold": best_t,
                    "prob_mean": float(all_probs.mean()),
                })
                print(f"  [{len(combo)}] {combo_name}: {ds_name} F1@0.5={f1_05:.4f} best={best_f:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out_dir / "ensemble_results.csv", index=False)

    # Show best ensembles
    for ds_name in ["testlike_cluster_dino_v3", "testlike_top_dino_v3"]:
        sub = results_df[results_df["dataset"] == ds_name]
        print(f"\n=== Best ensembles for {ds_name} ===")
        for metric in ["f1_at_0_5", "best_f1"]:
            top = sub.nlargest(5, metric)[["n_models", "ensemble", metric, "prob_mean"]]
            print(f"\n  By {metric}:")
            print(top.to_string(index=False))

    print(f"\nWrote to {args.out_dir}")


if __name__ == "__main__":
    main()
