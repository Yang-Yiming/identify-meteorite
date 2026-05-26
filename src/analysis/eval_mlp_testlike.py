#!/usr/bin/env python3
"""Evaluate DINOv2 MLP model on testlike v3 sets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from timm.data import create_transform, resolve_model_data_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_dinov2_mlp import PathDataset


def f1_at(preds, labels):
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    device = torch.device("cuda")

    # Load DINOv2
    print("Loading DINOv2...")
    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
    dino.eval().to(device)
    data_config = resolve_model_data_config(dino)
    dino_transform = create_transform(**data_config, is_training=False)

    # Load scaler and MLP
    scaler = StandardScaler()
    features = np.load("../train/outputs/dinov2_mlp/features.npz")
    scaler.fit(features["train_feats"])

    # MLPClassifier
    class MLPClassifier(nn.Module):
        def __init__(self, in_dim, hidden=512, num_classes=2, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden // 2, num_classes),
            )
        def forward(self, x): return self.net(x)

    mlp = MLPClassifier(768).to(device)
    mlp.load_state_dict(torch.load("../train/outputs/dinov2_mlp_v2/mlp_best.pt", map_location="cpu"))
    mlp.eval()

    def infer(paths):
        loader = torch.utils.data.DataLoader(
            PathDataset(paths, [0]*len(paths), dino_transform),
            batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
        )
        feats = []
        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    f = dino(images)
                feats.append(f.detach().float().cpu().numpy())
        feats = scaler.transform(np.concatenate(feats))
        with torch.no_grad():
            logits = mlp(torch.from_numpy(feats).to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs

    results = []
    for ds_name, ds_path in [
        ("testlike_cluster", "evaluation/testlike_dino_myval_v3/test_like_val_cluster.csv"),
        ("testlike_top", "evaluation/testlike_dino_myval_v3/test_like_val_top.csv"),
    ]:
        df = pd.read_csv(ds_path)
        labels = df["label"].astype(int).to_numpy()
        probs = infer(df["path"].tolist())
        f1 = f1_at((probs >= 0.5).astype(int), labels)
        n_pos = int((probs >= 0.5).sum())
        results.append({"dataset": ds_name, "f1_at_0_5": f1, "n_pos": n_pos, "prob_mean": float(probs.mean())})

    # Also evaluate on myval_masked for comparison
    manifest = pd.read_csv("evaluation/testlike_dino_myval_v3/manifest.csv")
    myval_masked = manifest[(manifest["source"] == "myval") & (manifest["has_image"])].copy()
    myval_probs = infer(myval_masked["path"].tolist())
    myval_labels = myval_masked["label"].astype(int).to_numpy()
    myval_f1 = f1_at((myval_probs >= 0.5).astype(int), myval_labels)
    results.append({"dataset": "myval_masked", "f1_at_0_5": myval_f1, "n_pos": int((myval_probs >= 0.5).sum()), "prob_mean": float(myval_probs.mean())})

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv("../train/outputs/dinov2_mlp_v2/testlike_v3_eval.csv", index=False)


if __name__ == "__main__":
    main()
