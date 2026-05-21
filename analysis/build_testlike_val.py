#!/usr/bin/env python3
"""Build first-pass test-like validation candidates.

This script intentionally avoids network-dependent DINO/CLIP features. It uses
lightweight image statistics to create a reproducible baseline proxy:

- RGB/HSV histograms
- grayscale thumbnail shape/layout
- edge and texture-grid summaries

Outputs are written under analysis/testlike/ by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_not_stone_numbers(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    numbers: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            numbers.add(int(line))
        except ValueError:
            numbers.add(int(Path(line).stem))
    return numbers


def crop_path_for_id(crop_root: Path, image_id: str, *, myval: bool = False) -> tuple[Path | None, str]:
    stem = Path(str(image_id)).stem
    candidates = [crop_root / f"{stem}_mask_000.png"]
    if myval:
        candidates.append(crop_root / f"h-{stem}_mask_000.png")
    for path in candidates:
        if path.is_file():
            return path, "mask"
    done_candidates = [crop_root / f"{stem}_nomask.done"]
    if myval:
        done_candidates.append(crop_root / f"h-{stem}_nomask.done")
    for path in done_candidates:
        if path.is_file():
            return None, "nomask"
    return None, "missing"


def iter_mytest(root: Path) -> Iterable[dict[str, object]]:
    for label_name, label in [("meteorite", 1), ("rock", 0)]:
        label_root = root / label_name
        if not label_root.is_dir():
            continue
        for path in sorted(label_root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield {
                    "sample_id": f"mytest/{label_name}/{path.name}",
                    "source": "mytest",
                    "image_id": path.name,
                    "label": label,
                    "path": str(path),
                    "mask_status": "raw",
                }


def build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    train_labels = pd.read_csv(args.train_labels_csv)
    for row in train_labels.itertuples(index=False):
        image_id = str(row.id)
        path, status = crop_path_for_id(args.train_crop_dir, image_id)
        rows.append(
            {
                "sample_id": f"train/{image_id}",
                "source": "train",
                "image_id": image_id,
                "label": int(row.label),
                "path": str(path) if path is not None else "",
                "mask_status": status,
            }
        )

    myval_labels = pd.read_csv(args.myval_labels_csv)
    for row in myval_labels.itertuples(index=False):
        image_id = str(row.id)
        path, status = crop_path_for_id(args.myval_crop_dir, image_id, myval=True)
        rows.append(
            {
                "sample_id": f"myval/{image_id}",
                "source": "myval",
                "image_id": image_id,
                "label": int(row.label),
                "path": str(path) if path is not None else "",
                "mask_status": status,
            }
        )

    not_stone_numbers = read_not_stone_numbers(args.not_stone_txt)
    for path in sorted(args.test_crop_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.name.replace("_mask_000", "")
        try:
            number = int(stem)
        except ValueError:
            number = int(Path(stem).stem)
        image_id = f"{number:06d}.jpg"
        rows.append(
            {
                "sample_id": f"test/{image_id}",
                "source": "test",
                "image_id": image_id,
                "label": np.nan,
                "path": str(path),
                "mask_status": "mask",
                "is_not_stone": number in not_stone_numbers,
            }
        )

    rows.extend(iter_mytest(args.mytest_root))
    df = pd.DataFrame(rows)
    if "is_not_stone" not in df.columns:
        df["is_not_stone"] = False
    df["is_not_stone"] = df["is_not_stone"].fillna(False).astype(bool)
    df["has_image"] = df["path"].astype(str).str.len() > 0
    return df


def image_feature(path: Path, size: int = 128) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0

    feats: list[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=24, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    value = maxc
    for plane in (saturation, value):
        hist, _ = np.histogram(plane, bins=16, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))

    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    thumb = np.asarray(Image.fromarray((gray * 255).astype("uint8")).resize((28, 28))).astype(np.float32) / 255.0
    feats.append(thumb.reshape(-1))

    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx * gx + gy * gy)
    grid_feats: list[float] = []
    for plane in (gray, edge, saturation):
        for y in range(0, size, 16):
            for x in range(0, size, 16):
                patch = plane[y : y + 16, x : x + 16]
                grid_feats.append(float(patch.mean()))
                grid_feats.append(float(patch.std()))
    feats.append(np.asarray(grid_feats, dtype=np.float32))

    return np.concatenate(feats).astype(np.float32)


def stratified_take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    parts = []
    labels = sorted(df["label"].dropna().astype(int).unique().tolist())
    per_label = max(1, n // max(1, len(labels)))
    for label in labels:
        label_df = df[df["label"].astype(int) == label].sort_values("test_like_score", ascending=False)
        parts.append(label_df.head(per_label))
    out = pd.concat(parts, axis=0)
    if len(out) < n:
        rest = df.drop(index=out.index).sort_values("test_like_score", ascending=False).head(n - len(out))
        out = pd.concat([out, rest], axis=0)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def cluster_stratified_take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    cluster_order = (
        df.groupby("cluster")["cluster_test_fraction"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    parts = []
    per_cluster = max(4, n // max(1, min(len(cluster_order), 12)))
    for cluster in cluster_order:
        cluster_df = df[df["cluster"] == cluster].sort_values("test_like_score", ascending=False)
        parts.append(stratified_take(cluster_df, per_cluster, seed + int(cluster)))
        if sum(len(p) for p in parts) >= n:
            break
    out = pd.concat(parts, axis=0).drop_duplicates("sample_id")
    if len(out) < n:
        rest = (
            df.loc[~df["sample_id"].isin(set(out["sample_id"]))]
            .sort_values("test_like_score", ascending=False)
            .head(n - len(out))
        )
        out = pd.concat([out, rest], axis=0)
    return out.head(n).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--myval-labels-csv", type=Path, default=Path("data/myval/labels.csv"))
    parser.add_argument("--train-crop-dir", type=Path, default=Path("preprocess/bbox_crop/train"))
    parser.add_argument("--myval-crop-dir", type=Path, default=Path("preprocess/bbox_crop/myval"))
    parser.add_argument("--test-crop-dir", type=Path, default=Path("preprocess/bbox_crop/test"))
    parser.add_argument("--mytest-root", type=Path, default=Path("mytest"))
    parser.add_argument("--not-stone-txt", type=Path, default=Path("post_process/not-stone_best_071962.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/testlike"))
    parser.add_argument("--clusters", type=int, default=24)
    parser.add_argument("--top-n", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fit-sources",
        type=str,
        default="train,myval,test,mytest",
        help="Comma-separated sources used to fit scaler/PCA/clusters and test-likeness space.",
    )
    parser.add_argument(
        "--candidate-sources",
        type=str,
        default="train,myval",
        help="Comma-separated labeled sources eligible for hard/weighted val outputs.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)

    feature_df = manifest[manifest["has_image"]].copy().reset_index(drop=True)
    features = np.stack([image_feature(Path(path)) for path in feature_df["path"]])
    fit_sources = {item.strip() for item in args.fit_sources.split(",") if item.strip()}
    candidate_sources = {item.strip() for item in args.candidate_sources.split(",") if item.strip()}
    fit_mask = feature_df["source"].isin(fit_sources).to_numpy()
    if not fit_mask.any():
        raise RuntimeError(f"No rows matched --fit-sources={args.fit_sources}")
    scaler = StandardScaler().fit(features[fit_mask])
    features_scaled = scaler.transform(features)
    pca_components = min(64, int(fit_mask.sum()), features_scaled.shape[1])
    pca = PCA(n_components=pca_components, random_state=args.seed).fit(features_scaled[fit_mask])
    features_pca = pca.transform(features_scaled)
    features_norm = features_pca / np.maximum(np.linalg.norm(features_pca, axis=1, keepdims=True), 1e-8)

    test_mask = (feature_df["source"] == "test") & (~feature_df["is_not_stone"])
    test_features = features_norm[test_mask.to_numpy()]
    if len(test_features) == 0:
        raise RuntimeError("No test anchor images found after not-stone filtering.")

    sims = cosine_similarity(features_norm, test_features)
    top_k = min(args.top_k, sims.shape[1])
    topk = np.sort(sims, axis=1)[:, -top_k:]
    feature_df["test_like_topk_mean"] = topk.mean(axis=1)
    feature_df["test_like_topk_max"] = topk.max(axis=1)
    test_centroid = test_features.mean(axis=0, keepdims=True)
    test_centroid = test_centroid / np.maximum(np.linalg.norm(test_centroid), 1e-8)
    feature_df["test_like_centroid_cos"] = cosine_similarity(features_norm, test_centroid).reshape(-1)

    n_clusters = min(args.clusters, int(fit_mask.sum()))
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    kmeans.fit(features_norm[fit_mask])
    feature_df["cluster"] = kmeans.predict(features_norm)
    cluster_summary = (
        feature_df.groupby("cluster")
        .agg(
            total=("sample_id", "count"),
            test_count=("source", lambda s: int((s == "test").sum())),
            train_count=("source", lambda s: int((s == "train").sum())),
            myval_count=("source", lambda s: int((s == "myval").sum())),
            mytest_count=("source", lambda s: int((s == "mytest").sum())),
            pos_count=("label", lambda s: int((s == 1).sum())),
            neg_count=("label", lambda s: int((s == 0).sum())),
            mean_topk=("test_like_topk_mean", "mean"),
        )
        .reset_index()
    )
    cluster_summary["test_fraction"] = cluster_summary["test_count"] / cluster_summary["total"]
    cluster_summary = cluster_summary.sort_values(["test_fraction", "test_count", "mean_topk"], ascending=False)
    cluster_summary.to_csv(args.out_dir / "cluster_summary.csv", index=False)

    cluster_to_fraction = dict(zip(cluster_summary["cluster"], cluster_summary["test_fraction"]))
    feature_df["cluster_test_fraction"] = feature_df["cluster"].map(cluster_to_fraction).astype(float)
    feature_df["test_like_score"] = (
        0.55 * feature_df["test_like_topk_mean"]
        + 0.25 * feature_df["test_like_centroid_cos"]
        + 0.20 * feature_df["cluster_test_fraction"]
    )

    score_cols = [
        "sample_id",
        "source",
        "image_id",
        "label",
        "path",
        "mask_status",
        "is_not_stone",
        "cluster",
        "cluster_test_fraction",
        "test_like_score",
        "test_like_topk_mean",
        "test_like_topk_max",
        "test_like_centroid_cos",
    ]
    scores = feature_df[score_cols].sort_values("test_like_score", ascending=False)
    scores.to_csv(args.out_dir / "test_like_scores.csv", index=False)

    labeled = scores[(scores["source"].isin(candidate_sources)) & scores["label"].notna()].copy()
    top_val = stratified_take(labeled, args.top_n, args.seed)
    cluster_val = cluster_stratified_take(labeled, args.top_n, args.seed)
    weighted = labeled.copy()
    weighted["sample_weight"] = (
        weighted["test_like_score"] - weighted["test_like_score"].min()
    ) / max(float(weighted["test_like_score"].max() - weighted["test_like_score"].min()), 1e-8)
    weighted["sample_weight"] = 0.25 + 0.75 * weighted["sample_weight"]

    top_val.to_csv(args.out_dir / "test_like_val_top.csv", index=False)
    cluster_val.to_csv(args.out_dir / "test_like_val_cluster.csv", index=False)
    weighted.to_csv(args.out_dir / "test_like_val_weighted.csv", index=False)

    summary = {
        "manifest_rows": int(len(manifest)),
        "feature_rows": int(len(feature_df)),
        "test_anchor_count": int(test_mask.sum()),
        "not_stone_txt": str(args.not_stone_txt),
        "clusters": int(args.clusters),
        "actual_clusters": int(n_clusters),
        "top_n": int(args.top_n),
        "fit_sources": sorted(fit_sources),
        "candidate_sources": sorted(candidate_sources),
        "outputs": {
            "manifest": str(args.out_dir / "manifest.csv"),
            "scores": str(args.out_dir / "test_like_scores.csv"),
            "cluster_summary": str(args.out_dir / "cluster_summary.csv"),
            "top_val": str(args.out_dir / "test_like_val_top.csv"),
            "cluster_val": str(args.out_dir / "test_like_val_cluster.csv"),
            "weighted_val": str(args.out_dir / "test_like_val_weighted.csv"),
        },
        "by_source": feature_df["source"].value_counts().to_dict(),
        "top_val_by_source": top_val["source"].value_counts().to_dict(),
        "cluster_val_by_source": cluster_val["source"].value_counts().to_dict(),
        "top_val_by_label": top_val["label"].astype(int).value_counts().to_dict(),
        "cluster_val_by_label": cluster_val["label"].astype(int).value_counts().to_dict(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
