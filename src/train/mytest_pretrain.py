#!/usr/bin/env python3
"""Pretrain ConvNeXt Tiny backbone on mytest dataset, then export for finetuning.

Usage:
  python mytest_pretrain.py --mytest-root ../../mytest --output-dir ./outputs/mytest_pretrain [train_args...]
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import timm

# Reuse existing modules
from data import IMAGE_SUFFIXES, POSITIVE_LABEL, MeteoriteDataset, build_image_index, stratified_split
from modeling import (
    ConvNeXtClassifier,
    create_optimizer,
    freeze_backbone_for_head_only,
    load_backbone,
    resolve_backbone_data_settings,
    unfreeze_backbone_all,
    build_transforms,
)
from augmentations import apply_cutmix, apply_mixup, build_soft_targets, soft_target_cross_entropy
from calibration import compute_binary_f1, compute_class_priors, search_best_threshold
from utils import DEFAULT_MEAN, DEFAULT_STD, normalize_image_size, normalize_stats, set_seed, save_json


def compute_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().norm(2).item())
        total += grad_norm * grad_norm
    return math.sqrt(total)


def build_mytest_dataframe(mytest_root: Path) -> pd.DataFrame:
    meteorite_dir = mytest_root / "meteorite"
    rock_dir = mytest_root / "rock"
    if not meteorite_dir.is_dir() or not rock_dir.is_dir():
        raise FileNotFoundError(f"mytest missing meteorite/ or rock/ under {mytest_root}")

    rows = []
    for class_dir, label in [(meteorite_dir, 1), (rock_dir, 0)]:
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append({"id": path.name, "label": label, "label_idx": label, "sample_weight": 1.0})

    if not rows:
        raise RuntimeError(f"No images found under {mytest_root}")
    df = pd.DataFrame(rows)
    print(f"mytest dataset | total={len(df)} | pos={(df['label']==1).sum()} | neg={(df['label']==0).sum()}")
    return df


def run_epoch(
    model, loader, device, num_classes, class_weights,
    optimizer=None, cutmix_alpha=0.0, cutmix_prob=0.0,
    mixup_alpha=0.0, mixup_prob=0.0, label_smoothing=0.0,
    max_grad_norm=0.0,
):
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_samples = 0
    cutmix_batches = 0
    mixup_batches = 0
    collected_labels = []
    collected_probabilities = []

    autocast_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled) if is_train else None

    for pixel_values, labels, sample_weights in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)

        soft_targets = build_soft_targets(labels, num_classes=num_classes, smoothing=label_smoothing)
        if is_train:
            mixed = apply_cutmix(pixel_values, labels, sample_weights, num_classes, cutmix_alpha, cutmix_prob, label_smoothing)
            pixel_values = mixed.pixel_values
            soft_targets = mixed.mixed_labels
            sample_weights = mixed.sample_weights
            if mixed.applied:
                cutmix_batches += 1

            mixed2 = apply_mixup(pixel_values, labels, sample_weights, num_classes, mixup_alpha, mixup_prob, label_smoothing)
            pixel_values = mixed2.pixel_values
            soft_targets = mixed2.mixed_labels
            sample_weights = mixed2.sample_weights
            if mixed2.applied:
                mixup_batches += 1

            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(pixel_values)
                loss = soft_target_cross_entropy(logits, soft_targets, class_weights=class_weights, sample_weights=sample_weights)

            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        if not is_train:
            collected_labels.append(labels.detach().cpu())
            collected_probabilities.append(torch.softmax(logits.detach(), dim=1)[:, POSITIVE_LABEL].cpu())

    metrics = {"loss": total_loss / max(1, total_samples)}
    if is_train:
        metrics["cutmix_batches"] = float(cutmix_batches)
        metrics["mixup_batches"] = float(mixup_batches)
    else:
        metrics["labels"] = torch.cat(collected_labels) if collected_labels else torch.empty(0, dtype=torch.long)
        metrics["prob_pos"] = torch.cat(collected_probabilities) if collected_probabilities else torch.empty(0, dtype=torch.float32)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="mytest pretraining")
    parser.add_argument("--mytest-root", type=Path, default=Path("../../mytest"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/mytest_pretrain"))
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--backbone-checkpoint", type=Path, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--val-split-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head-only-epochs", type=int, default=5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cutmix-alpha", type=float, default=0.7)
    parser.add_argument("--cutmix-prob", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=("constant", "cosine"))
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stop", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--rotate-degrees", type=float, default=15.0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    mytest_root = args.mytest_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataframe from mytest
    df = build_mytest_dataframe(mytest_root)
    unique_labels = sorted(df["label"].unique().tolist())

    # Stratified train/val split
    if args.val_split_ratio > 0.0:
        train_df, val_df = stratified_split(df, label_column="label_idx", val_ratio=args.val_split_ratio, seed=args.seed)
    else:
        train_df = df.copy()
        val_df = df.copy()

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Build image index from mytest raw images
    image_index = build_image_index(mytest_root)
    print(f"mytest split | train={len(train_df)} val={len(val_df)}")

    # Model
    model = ConvNeXtClassifier(
        backbone_name=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        num_classes=len(unique_labels),
        dropout=args.dropout,
        pretrained_backbone=not args.no_pretrained,
        drop_path_rate=args.drop_path_rate,
    )

    head_only_epochs = args.head_only_epochs
    finetune_epochs = args.epochs
    total_epochs = head_only_epochs + finetune_epochs

    freeze_backbone_for_head_only(model.backbone)
    if finetune_epochs > 0:
        print(f"Stage head_only | epochs={head_only_epochs} | will unfreeze after")

    backbone_image_size, backbone_image_mean, backbone_image_std = resolve_backbone_data_settings(model.backbone)
    image_size = args.image_size or normalize_image_size(backbone_image_size)
    image_mean = normalize_stats(None, normalize_stats(backbone_image_mean, DEFAULT_MEAN))
    image_std = normalize_stats(None, normalize_stats(backbone_image_std, DEFAULT_STD))

    train_transform, eval_transform = build_transforms(
        image_size, image_mean, image_std,
        hflip_prob=args.hflip_prob, rotate_degrees=args.rotate_degrees,
    )

    train_dataset = MeteoriteDataset(train_df, image_index, train_transform)
    val_dataset = MeteoriteDataset(val_df, image_index, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = model.to(device)

    train_priors = compute_class_priors(train_df, label_column="label_idx")
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    optimizer = create_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay, llrd_decay=0.8)
    current_stage = "head_only"

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=args.lr_min)

    save_json(vars(args), output_dir / "pretrain_args.json")

    history = []
    best_val_f1 = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, total_epochs + 1):
        if current_stage == "head_only" and epoch > head_only_epochs:
            unfreeze_backbone_all(model.backbone)
            optimizer = create_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay, llrd_decay=0.8)
            if args.lr_scheduler == "cosine":
                remaining = total_epochs - head_only_epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=args.lr_min)
            current_stage = "finetune"
            print(f"Stage finetune | epoch={epoch:02d} | unfreezing backbone")

        train_metrics = run_epoch(
            model, train_loader, device, len(unique_labels), class_weights,
            optimizer=optimizer,
            cutmix_alpha=args.cutmix_alpha, cutmix_prob=args.cutmix_prob,
            mixup_alpha=args.mixup_alpha, mixup_prob=args.mixup_prob,
            label_smoothing=args.label_smoothing, max_grad_norm=args.max_grad_norm,
        )

        if scheduler is not None:
            scheduler.step()

        val_metrics = run_epoch(model, val_loader, device, len(unique_labels), class_weights)

        val_prob_pos = val_metrics["prob_pos"]
        val_labels = val_metrics["labels"]
        val_f1 = compute_binary_f1(val_prob_pos, val_labels, threshold=0.5)

        thr_result = search_best_threshold(val_prob_pos, val_labels, metric="f1")
        best_thr_f1 = thr_result["metric_value"]
        best_thr = thr_result["threshold"]

        epoch_metrics = {
            "epoch": epoch, "stage": current_stage,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "val_f1_0.5": float(val_f1),
            "val_f1_best": float(best_thr_f1),
            "best_threshold": float(best_thr),
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d} | stage={current_stage} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1@0.5={val_f1:.4f} | best_val_f1={best_thr_f1:.4f}@{best_thr:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "stage": current_stage,
            "metrics": epoch_metrics,
        }
        if args.save_every_epoch:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:02d}.pt")
        torch.save(checkpoint, output_dir / "last.pt")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if args.early_stop is not None and epochs_without_improvement >= args.early_stop:
            print(f"Early stop | patience={args.early_stop} | best_val_f1@0.5={best_val_f1:.4f}")
            break

    save_json({"history": history}, output_dir / "history.json")
    best_entry = max(history, key=lambda r: r["val_f1_0.5"])
    print(f"\nPretrain complete | best_epoch={best_entry['epoch']} | val_f1@0.5={best_entry['val_f1_0.5']:.4f} | best_f1={best_entry['val_f1_best']:.4f}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
