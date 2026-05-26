#!/usr/bin/env python3
"""Evaluate a checkpoint on the myval dataset (primary offline proxy)."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import (
    MeteoriteDataset,
    build_mask_image_index,
)
from modeling import ConvNeXtClassifier, build_transforms
from calibration import compute_binary_f1, search_best_threshold
from utils import DEFAULT_BACKBONE, DEFAULT_MEAN, DEFAULT_STD, normalize_image_size, normalize_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on myval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--myval-root", type=Path, default=Path("../../data/myval"))
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=Path("../../mask"))
    parser.add_argument("--mask-split", type=str, default="myval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve()
    checkpoint_dir = checkpoint_path.parent

    metadata = {}
    md_path = checkpoint_dir / "metadata.json"
    if md_path.is_file():
        with md_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    train_args = {}
    ta_path = checkpoint_dir / "train_args.json"
    if ta_path.is_file():
        with ta_path.open("r", encoding="utf-8") as f:
            train_args = json.load(f)

    pretrain_args = {}
    pa_path = checkpoint_dir / "pretrain_args.json"
    if pa_path.is_file():
        with pa_path.open("r", encoding="utf-8") as f:
            pretrain_args = json.load(f)

    backbone = args.backbone or train_args.get("backbone") or pretrain_args.get("backbone") or metadata.get("backbone_name") or DEFAULT_BACKBONE
    dropout = args.dropout if args.dropout is not None else train_args.get("dropout", pretrain_args.get("dropout", 0.1))
    image_size_val = args.image_size or metadata.get("image_size") or pretrain_args.get("image_size") or 224
    image_size = normalize_image_size(image_size_val)
    image_mean = normalize_stats(None, DEFAULT_MEAN)
    image_std = normalize_stats(None, DEFAULT_STD)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt)
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    model = ConvNeXtClassifier(
        backbone_name=backbone,
        backbone_checkpoint=None,
        num_classes=2,
        dropout=float(dropout),
        pretrained_backbone=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Model load | missing_keys={len(missing)} | unexpected_keys={len(unexpected)}")

    myval_root = args.myval_root.resolve()
    labels_csv = (args.labels_csv or (myval_root / "labels.csv")).resolve()
    df = pd.read_csv(labels_csv)
    df["label_idx"] = df["label"].astype(int)

    val_mask_dir = (args.mask_dir.resolve() / args.mask_split)
    val_mask_index, masked_ids, skipped_ids = build_mask_image_index(
        val_mask_dir, df["id"].astype(str).tolist()
    )
    df = df[df["id"].astype(str).isin(set(masked_ids))].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No myval images found with valid masks in {val_mask_dir}")

    label_counts = df["label_idx"].value_counts().to_dict()
    print(f"myval | total={len(df)} | pos={label_counts.get(1,0)} | neg={label_counts.get(0,0)} | skipped={len(skipped_ids)}")

    _, eval_transform = build_transforms(
        image_size, image_mean, image_std, hflip_prob=0.0, rotate_degrees=0.0,
    )

    dataset = MeteoriteDataset(df, val_mask_index, eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    autocast_enabled = device.type == "cuda"

    with torch.no_grad():
        for pixel_values, labels_t, _ in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels_t = labels_t.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels_t.detach().cpu())

    prob_pos = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    f1_at_05 = compute_binary_f1(prob_pos, labels, threshold=0.5)
    thr_result = search_best_threshold(prob_pos, labels, metric="f1")

    positive_preds = int((prob_pos >= 0.5).sum().item())
    positive_labels = int((labels == 1).sum().item())

    print(f"\n{'='*50}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"myval F1@0.5:  {f1_at_05:.4f}  (pos_pred={positive_preds}, pos_label={positive_labels})")
    print(f"myval best F1: {thr_result['metric_value']:.4f} @ threshold={thr_result['threshold']:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
