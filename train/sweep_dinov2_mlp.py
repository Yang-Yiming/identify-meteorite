#!/usr/bin/env python3
"""Quick hyperparameter sweep for DINOv2 MLP using cached features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from utils import set_seed


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden=512, num_classes=2, dropout=0.3, n_layers=2):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            cur_hidden = hidden // (2 ** i) if i > 0 else hidden
            layers.extend([
                nn.Linear(prev_dim, cur_hidden),
                nn.BatchNorm1d(cur_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = cur_hidden
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def f1_at(preds, labels):
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def train_eval(train_x, train_y, val_x, val_y, device, config):
    set_seed(42)
    model = MLPClassifier(
        in_dim=train_x.shape[1],
        hidden=config["hidden"],
        num_classes=2,
        dropout=config["dropout"],
        n_layers=config.get("n_layers", 2),
    ).to(device)

    train_neg = (train_y == 0).sum().item()
    train_pos = (train_y == 1).sum().item()
    pos_weight = train_neg / max(train_pos, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("wd", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]).to(device))

    best_val_f1 = -1
    best_epoch = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        indices = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 512):
            idx = indices[i:i + 512]
            x_batch = train_x[idx].to(device)
            y_batch = train_y[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device))
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_at(val_preds, val_y.cpu().numpy())
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch

    return best_val_f1, best_epoch


def main():
    features_path = Path("train/outputs/dinov2_mlp/features.npz")
    cached = np.load(features_path)

    train_feats = cached["train_feats"]
    myval_feats = cached["myval_feats"]

    # Load labels
    import pandas as pd
    train_labels = pd.read_csv("data/train_labels.csv")
    myval_labels = pd.read_csv("data/myval/labels.csv")

    # Filter to matched features
    train_y_t = torch.tensor(train_labels["label"].values[:len(train_feats)], dtype=torch.long)
    myval_y_t = torch.tensor(myval_labels["label"].values[:len(myval_feats)], dtype=torch.long)

    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    myval_feats = scaler.transform(myval_feats)

    train_x = torch.from_numpy(train_feats).float()
    val_x = torch.from_numpy(myval_feats).float()

    device = torch.device("cuda")

    configs = [
        {"lr": 1e-3, "hidden": 512, "dropout": 0.3, "epochs": 60, "wd": 1e-4},
        {"lr": 5e-4, "hidden": 512, "dropout": 0.3, "epochs": 60, "wd": 1e-4},
        {"lr": 1e-3, "hidden": 256, "dropout": 0.3, "epochs": 60, "wd": 1e-4},
        {"lr": 1e-3, "hidden": 512, "dropout": 0.5, "epochs": 60, "wd": 1e-4},
        {"lr": 1e-3, "hidden": 512, "dropout": 0.3, "epochs": 60, "wd": 1e-3},
        {"lr": 2e-3, "hidden": 512, "dropout": 0.3, "epochs": 60, "wd": 1e-4},
        {"lr": 1e-3, "hidden": 384, "dropout": 0.2, "epochs": 60, "wd": 1e-4},
        {"lr": 1e-3, "hidden": 768, "dropout": 0.4, "epochs": 60, "wd": 1e-4},
        {"lr": 1e-3, "hidden": 512, "dropout": 0.3, "epochs": 120, "wd": 1e-4, "n_layers": 3},
    ]

    results = []
    for config in configs:
        f1, epoch = train_eval(train_x, train_y_t, val_x, myval_y_t, device, config)
        tag = f"lr={config['lr']:.0e} h={config['hidden']} do={config['dropout']} ep={config['epochs']} wd={config['wd']:.0e} nl={config.get('n_layers', 2)}"
        results.append({**config, "val_f1": f1, "best_epoch": epoch, "tag": tag})
        print(f"  {tag}: F1={f1:.4f} @ epoch={epoch}")

    # Show best
    results.sort(key=lambda x: x["val_f1"], reverse=True)
    print("\n=== Best configs ===")
    for r in results[:5]:
        print(f"  {r['tag']}: {r['val_f1']:.4f}")

    Path("train/outputs/dinov2_mlp/hparam_sweep.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
