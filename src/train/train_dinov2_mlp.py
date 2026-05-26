#!/usr/bin/env python3
"""DINOv2-frozen features + MLP classifier baseline.
Extracts DINOv2 features once, then trains a light classifier.
A completely different paradigm from ConvNeXt finetuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from utils import set_seed


class PathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return self.transform(img), self.labels[index], index


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden=512, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def extract_features(model, paths, transform, device, batch_size, num_workers):
    loader = DataLoader(
        PathDataset(paths, [0] * len(paths), transform),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    features = []
    autocast_enabled = device.type == "cuda"
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                feats = model(images)
            features.append(feats.detach().float().cpu().numpy())
    return np.concatenate(features)


def f1_at(preds, labels):
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-model", type=str, default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--train-labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--train-crop-dir", type=Path, default=Path("preprocess/bbox_crop/train"))
    parser.add_argument("--myval-crop-dir", type=Path, default=Path("preprocess/bbox_crop/myval"))
    parser.add_argument("--myval-labels-csv", type=Path, default=Path("data/myval/labels.csv"))
    parser.add_argument("--test-crop-dir", type=Path, default=Path("preprocess/bbox_crop/test"))
    parser.add_argument("--test-raw-dir", type=Path, default=Path("data/test_images/test_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/outputs/dinov2_mlp"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-cache", type=Path, default=Path("train/outputs/dinov2_mlp/features.npz"))
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load DINOv2 model
    print(f"Loading DINOv2: {args.dino_model}")
    dino = timm.create_model(args.dino_model, pretrained=True, num_classes=0)
    dino.eval().to(device)
    data_config = resolve_model_data_config(dino)
    transform = create_transform(**data_config, is_training=False)

    # Get actual input size from model config
    input_size = data_config.get("input_size", (3, 224, 224))
    feature_dim = dino(torch.randn(1, *input_size).to(device)).shape[1]
    print(f"Input size: {input_size}, Feature dimension: {feature_dim}")

    # Build image paths
    train_labels = pd.read_csv(args.train_labels_csv)
    myval_labels = pd.read_csv(args.myval_labels_csv)

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    train_paths = []
    train_y = []
    for _, row in train_labels.iterrows():
        image_id = str(row["id"])
        stem = Path(image_id).stem
        found = False
        for try_name in [f"{stem}_mask_000.png", f"{stem}.png", f"{stem}.jpg"]:
            p = args.train_crop_dir / try_name
            if p.is_file():
                train_paths.append(str(p))
                train_y.append(int(row["label"]))
                found = True
                break
        if not found:
            for suffix in IMAGE_SUFFIXES:
                p = args.train_crop_dir / f"{image_id}{suffix}"
                if p.is_file():
                    train_paths.append(str(p))
                    train_y.append(int(row["label"]))
                    found = True
                    break

    myval_paths = []
    myval_y = []
    for _, row in myval_labels.iterrows():
        image_id = str(row["id"])
        stem = Path(image_id).stem
        found = False
        for try_name in [f"h-{stem}_mask_000.png", f"{stem}_mask_000.png", f"h-{stem}.png", f"{stem}.png", f"h-{stem}.jpg", f"{stem}.jpg"]:
            p = args.myval_crop_dir / try_name
            if p.is_file():
                myval_paths.append(str(p))
                myval_y.append(int(row["label"]))
                found = True
                break
        if not found:
            for try_name in [f"h-{image_id}_mask_000.png", f"{image_id}_mask_000.png"]:
                p = args.myval_crop_dir / try_name
                if p.is_file():
                    myval_paths.append(str(p))
                    myval_y.append(int(row["label"]))
                    found = True
                    break

    # Test images - get all 194, prefer bbox_crop, fallback to raw
    # Read expected test IDs from soup submission
    soup_sub = pd.read_csv("train/outputs/myval_v13_hi288_seed42_soup/submission_raw.csv")
    expected_ids = soup_sub["id"].tolist()
    test_paths = []
    test_ids = []
    for image_id in expected_ids:
        stem = Path(image_id).stem  # e.g. "000001"
        # Try bbox_crop first
        crop_path = args.test_crop_dir / f"{stem}_mask_000.png"
        if crop_path.is_file():
            test_paths.append(str(crop_path))
            test_ids.append(image_id)
        else:
            # Fallback to raw image
            raw_path = args.test_raw_dir / image_id
            if raw_path.is_file():
                test_paths.append(str(raw_path))
                test_ids.append(image_id)

    print(f"Train: {len(train_paths)} | Myval: {len(myval_paths)} | Test: {len(test_paths)}")

    # Extract features (or load from cache)
    if args.feature_cache.is_file():
        print(f"Loading cached features from {args.feature_cache}")
        cached = np.load(args.feature_cache, allow_pickle=True)
        train_feats = cached["train_feats"]
        myval_feats = cached["myval_feats"]
        test_feats = cached["test_feats"]
    else:
        print("Extracting training features...")
        train_feats = extract_features(dino, train_paths, transform, device, args.batch_size, args.num_workers)
        print("Extracting myval features...")
        myval_feats = extract_features(dino, myval_paths, transform, device, args.batch_size, args.num_workers)
        print("Extracting test features...")
        test_feats = extract_features(dino, test_paths, transform, device, args.batch_size, args.num_workers)
        print(f"Saving features to {args.feature_cache}")
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.feature_cache, train_feats=train_feats, myval_feats=myval_feats, test_feats=test_feats)

    # Normalize features
    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    myval_feats = scaler.transform(myval_feats)
    test_feats = scaler.transform(test_feats)

    # Create classifier
    model = MLPClassifier(feature_dim, hidden=args.hidden, num_classes=2, dropout=args.dropout)
    model.to(device)

    train_x = torch.from_numpy(train_feats).float()
    train_y_t = torch.tensor(train_y, dtype=torch.long)
    myval_x = torch.from_numpy(myval_feats).float()
    myval_y_t = torch.tensor(myval_y, dtype=torch.long)
    test_x = torch.from_numpy(test_feats).float()

    # Class balance
    neg_count = (train_y_t == 0).sum().item()
    pos_count = (train_y_t == 1).sum().item()
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))

    best_val_f1 = -1
    best_state = None
    best_epoch = 0
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Mini-batch training
        indices = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 512):
            idx = indices[i:i + 512]
            x_batch = train_x[idx].to(device)
            y_batch = train_y_t[idx].to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(myval_x.to(device))
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_at(val_preds, np.array(myval_y))
        val_acc = (val_preds == np.array(myval_y)).mean()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, args.output_dir / "mlp_best.pt")

        log.append(f"Epoch {epoch:3d} | val_f1={val_f1:.4f} val_acc={val_acc:.4f} best={best_val_f1:.4f}")
        if epoch % 5 == 0:
            print(log[-1])

    print("\n".join(log[-5:]))

    # Final evaluation
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, args.output_dir / "mlp_best.pt")
    print(f"Best epoch: {best_epoch}, Best val F1: {best_val_f1:.4f}")
    with torch.no_grad():
        val_logits = model(myval_x.to(device))
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        test_logits = model(test_x.to(device))
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()

    val_preds = (val_probs >= 0.5).astype(int)
    print(f"\n=== Final ===")
    print(f"Myval F1@0.5 = {f1_at(val_preds, np.array(myval_y)):.4f}")
    print(f"Myval pos_pred = {val_preds.sum()}/{len(val_preds)}")
    print(f"Test  pos_pred = {(test_probs >= 0.5).sum()}/{len(test_probs)}")

    # Save results
    results = {
        "dino_model": args.dino_model,
        "feature_dim": feature_dim,
        "train_n": len(train_paths),
        "myval_n": len(myval_paths),
        "test_n": len(test_paths),
        "best_val_f1": float(best_val_f1),
        "val_pos_pred": int(val_preds.sum()),
        "test_pos_pred": int((test_probs >= 0.5).sum()),
        "lr": args.lr,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "epochs": args.epochs,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Save test predictions
    submission = pd.DataFrame({"id": test_ids, "label": (test_probs >= 0.5).astype(int)})
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
