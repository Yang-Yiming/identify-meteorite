#!/usr/bin/env python3
"""Evaluate historical checkpoints on myval and test-like validation sets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from modeling import ConvNeXtClassifier, build_transforms  # noqa: E402
from utils import DEFAULT_BACKBONE, DEFAULT_MEAN, DEFAULT_STD, normalize_image_size, normalize_stats  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    name: str
    checkpoint: Path | None
    known_test_f1: float | None
    known_myval_f1: float | None
    note: str


RUNS = [
    RunSpec(
        name="soup_reduced_notstone",
        checkpoint=Path("train/outputs/myval_v13_hi288_seed42_soup/soup.pt"),
        known_test_f1=0.71962,
        known_myval_f1=0.7251,
        note="current SOTA: soup + reduced not-stone post-process",
    ),
    RunSpec(
        name="soup_old_notstone",
        checkpoint=Path("train/outputs/myval_v13_hi288_seed42_soup/soup.pt"),
        known_test_f1=0.69856,
        known_myval_f1=0.7251,
        note="same checkpoint, old aggressive not-stone post-process",
    ),
    RunSpec(
        name="mytest_split_protocol",
        checkpoint=Path("train/outputs/mytest_v1_s42/best.pt"),
        known_test_f1=0.65979,
        known_myval_f1=0.7321,
        note="mytest train+val protocol",
    ),
    RunSpec(
        name="mytest_pretrain_finetune",
        checkpoint=Path("train/outputs/mytest_pretrain_finetune_v2/best.pt"),
        known_test_f1=0.55214,
        known_myval_f1=0.7358,
        note="supervised mytest pretrain then original-data finetune",
    ),
    RunSpec(
        name="mytest_augment_soup",
        checkpoint=Path("train/outputs/mytest_augment_v2/soup_top3.pt"),
        known_test_f1=0.67021,
        known_myval_f1=0.7688,
        note="mytest merged into train, top-3 soup",
    ),
    RunSpec(
        name="splitval_augment_soup",
        checkpoint=Path("train/outputs/splitval_augment_v1/soup_top3.pt"),
        known_test_f1=0.63212,
        known_myval_f1=0.7446,
        note="mytest merged, internal split validation, top-3 soup",
    ),
]


class PathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
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
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        hflip_prob=0.0,
        rotate_degrees=0.0,
    )
    model = ConvNeXtClassifier(
        backbone_name=backbone_name,
        backbone_checkpoint=None,
        num_classes=num_classes,
        dropout=dropout,
        pretrained_backbone=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"{checkpoint_path}: missing_keys={len(missing)} unexpected_keys={len(unexpected)}",
            file=sys.stderr,
        )
    model.to(device)
    model.eval()
    return model, eval_transform


def infer_probs(model, df: pd.DataFrame, transform, device: torch.device, batch_size: int, num_workers: int) -> np.ndarray:
    loader = DataLoader(
        PathDataset(df, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
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


def f1_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for threshold in sorted(set([0.0, 0.5, 1.0] + probs.tolist())):
        score = f1_at(probs, labels, threshold)
        if score > best_f1:
            best_t = float(threshold)
            best_f1 = float(score)
    return best_t, best_f1


def summarize_eval(run: RunSpec, dataset_name: str, df: pd.DataFrame, probs: np.ndarray) -> dict[str, object]:
    labels = df["label"].astype(int).to_numpy()
    threshold, best_f1 = best_threshold(probs, labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "run": run.name,
        "dataset": dataset_name,
        "f1_at_0_5": f1_at(probs, labels, 0.5),
        "best_f1": best_f1,
        "best_threshold": threshold,
        "n": int(len(df)),
        "pos_true": int(labels.sum()),
        "pos_pred_at_0_5": int(preds.sum()),
        "prob_mean": float(probs.mean()),
        "prob_median": float(np.median(probs)),
    }


def rank_corr(df: pd.DataFrame, metric_col: str) -> float | None:
    sub = df[["run", metric_col, "known_test_f1"]].dropna().drop_duplicates("run")
    if len(sub) < 3:
        return None
    return float(sub[metric_col].rank().corr(sub["known_test_f1"].rank(), method="pearson"))


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return ""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt) if np.isfinite(value) else "")
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("analysis/testlike/manifest.csv"))
    parser.add_argument("--cluster-val", type=Path, default=Path("analysis/testlike/test_like_val_cluster.csv"))
    parser.add_argument("--top-val", type=Path, default=Path("analysis/testlike/test_like_val_top.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/testlike_eval"))
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
        "testlike_cluster_v1": cluster_val,
        "testlike_top_v1": top_val,
    }

    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    model_cache: dict[Path, tuple[torch.nn.Module, object]] = {}

    for run in RUNS:
        if run.checkpoint is None or not run.checkpoint.is_file():
            print(f"Skipping missing checkpoint: {run.name} {run.checkpoint}")
            continue
        print(f"Evaluating {run.name}: {run.checkpoint}")
        if run.checkpoint not in model_cache:
            model_cache[run.checkpoint] = load_model(run.checkpoint, device)
        model, transform = model_cache[run.checkpoint]
        for dataset_name, dataset_df in datasets.items():
            probs = infer_probs(model, dataset_df, transform, device, args.batch_size, args.num_workers)
            row = summarize_eval(run, dataset_name, dataset_df, probs)
            row.update(
                {
                    "checkpoint": str(run.checkpoint),
                    "known_test_f1": run.known_test_f1,
                    "known_myval_f1_from_docs": run.known_myval_f1,
                    "note": run.note,
                }
            )
            rows.append(row)
            pred_df = dataset_df[["sample_id", "source", "image_id", "label", "path"]].copy()
            pred_df["run"] = run.name
            pred_df["dataset"] = dataset_name
            pred_df["prob_pos"] = probs
            prediction_rows.append(pred_df)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(args.out_dir / "proxy_eval_long.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, axis=0).to_csv(args.out_dir / "proxy_eval_predictions.csv", index=False)

    pivot = long_df.pivot_table(
        index=["run", "checkpoint", "known_test_f1", "known_myval_f1_from_docs", "note"],
        columns="dataset",
        values=["f1_at_0_5", "best_f1", "best_threshold", "pos_pred_at_0_5", "prob_mean"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{dataset}" for metric, dataset in pivot.columns]
    pivot = pivot.reset_index()
    sort_col = "known_test_f1"
    pivot = pivot.sort_values(sort_col, ascending=False)
    pivot.to_csv(args.out_dir / "proxy_eval_summary.csv", index=False)

    markdown = []
    markdown.append("# Test-Like Proxy Evaluation\n")
    markdown.append("Historical checkpoints evaluated on myval and first-pass test-like validation sets.\n")
    markdown.append("## Summary\n")
    display_cols = [
        "run",
        "known_test_f1",
        "known_myval_f1_from_docs",
        "f1_at_0_5__myval_masked",
        "f1_at_0_5__testlike_cluster_v1",
        "f1_at_0_5__testlike_top_v1",
        "best_f1__testlike_cluster_v1",
        "note",
    ]
    markdown.append(markdown_table(pivot[display_cols], floatfmt=".4f"))
    markdown.append("\n\n## Rank Correlation With Known Test F1\n")
    corr_rows = []
    for dataset_name in datasets:
        sub = long_df[long_df["dataset"] == dataset_name].copy()
        for metric in ["f1_at_0_5", "best_f1", "prob_mean"]:
            sub_metric = sub.rename(columns={metric: "metric"})
            corr_rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "spearman_like_rank_corr": rank_corr(sub_metric, "metric"),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(args.out_dir / "proxy_eval_rank_correlations.csv", index=False)
    markdown.append(markdown_table(corr_df, floatfmt=".4f"))
    markdown.append(
        "\n\nNote: `soup_reduced_notstone` and `soup_old_notstone` share the same checkpoint, "
        "so checkpoint-only proxy metrics cannot distinguish their different post-process policies."
    )
    (args.out_dir / "proxy_eval_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(pivot[display_cols].to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
