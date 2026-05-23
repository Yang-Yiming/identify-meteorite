#!/usr/bin/env python3
"""Audit verifier candidates with CLIP/SigLIP-style timm embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.build_testlike_val import build_manifest, extract_dino_timm_features  # noqa: E402


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/vlm_neighbor_audit"))
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_siglip_224")
    parser.add_argument("--verifier-features", type=Path, default=Path("analysis/verifier_features/current_positive_verifier_features.csv"))
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--myval-labels-csv", type=Path, default=Path("data/myval/labels.csv"))
    parser.add_argument("--train-crop-dir", type=Path, default=Path("preprocess/bbox_crop/train"))
    parser.add_argument("--myval-crop-dir", type=Path, default=Path("preprocess/bbox_crop/myval"))
    parser.add_argument("--test-crop-dir", type=Path, default=Path("preprocess/bbox_crop/test"))
    parser.add_argument("--mytest-root", type=Path, default=Path("mytest"))
    parser.add_argument("--not-stone-txt", type=Path, default=Path("post_process/not-stone_best_071962.txt"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    verifier = pd.read_csv(args.verifier_features).head(args.top_n).copy()
    query_ids = set(verifier["id"].astype(str))

    manifest = build_manifest(
        argparse.Namespace(
            train_labels_csv=args.train_labels_csv,
            myval_labels_csv=args.myval_labels_csv,
            train_crop_dir=args.train_crop_dir,
            myval_crop_dir=args.myval_crop_dir,
            test_crop_dir=args.test_crop_dir,
            mytest_root=args.mytest_root,
            not_stone_txt=args.not_stone_txt,
        )
    )
    refs = manifest[
        manifest["has_image"]
        & manifest["source"].isin(["train", "myval"])
        & manifest["label"].notna()
    ].copy().reset_index(drop=True)
    queries = manifest[
        manifest["has_image"]
        & (manifest["source"] == "test")
        & manifest["image_id"].astype(str).isin(query_ids)
    ].copy().reset_index(drop=True)
    if len(queries) != len(query_ids):
        found = set(queries["image_id"].astype(str))
        missing = sorted(query_ids - found)
        raise RuntimeError(f"Missing query images for: {missing}")

    all_paths = refs["path"].astype(str).tolist() + queries["path"].astype(str).tolist()
    features = extract_dino_timm_features(
        paths=all_paths,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    ref_features = norm(features[: len(refs)])
    query_features = norm(features[len(refs) :])

    sims = cosine_similarity(query_features, ref_features)
    top_k = min(args.top_k, sims.shape[1])
    neighbor_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for qi, qrow in queries.iterrows():
        order = np.argsort(-sims[qi])[:top_k]
        top_refs = refs.iloc[order].reset_index(drop=True)
        top_sims = sims[qi, order]
        for rank, (ref_row, sim) in enumerate(zip(top_refs.itertuples(index=False), top_sims), start=1):
            neighbor_rows.append(
                {
                    "query_id": qrow["image_id"],
                    "rank": rank,
                    "cosine_sim": float(sim),
                    "ref_sample_id": ref_row.sample_id,
                    "ref_source": ref_row.source,
                    "ref_image_id": ref_row.image_id,
                    "ref_label": int(ref_row.label),
                    "ref_path": ref_row.path,
                }
            )
        labels = top_refs["label"].astype(int)
        summary_rows.append(
            {
                "id": qrow["image_id"],
                "vlm_top1_sim": float(top_sims[0]),
                "vlm_top1_source": top_refs.iloc[0]["source"],
                "vlm_top1_label": int(top_refs.iloc[0]["label"]),
                "vlm_top5_pos_frac": float(labels.head(5).mean()),
                "vlm_top10_pos_frac": float(labels.head(10).mean()),
                "vlm_top20_pos_frac": float(labels.head(min(20, top_k)).mean()),
                "vlm_mean_top5_sim": float(top_sims[:5].mean()),
                "vlm_mean_top10_sim": float(top_sims[:10].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).merge(verifier, on="id", how="left")
    summary["vlm_disagrees_with_dino"] = (
        (summary["vlm_top5_pos_frac"] >= 0.6) & (summary["top5_pos_frac"] <= 0.2)
    ) | ((summary["vlm_top5_pos_frac"] <= 0.2) & (summary["top5_pos_frac"] >= 0.6))
    summary = summary.sort_values(["verifier_rank"]).reset_index(drop=True)
    neighbors = pd.DataFrame(neighbor_rows)

    summary.to_csv(args.out_dir / "vlm_neighbor_summary.csv", index=False)
    neighbors.to_csv(args.out_dir / "vlm_neighbors_topk.csv", index=False)
    report_cols = [
        "id",
        "weak_label",
        "verifier_rank",
        "soup_prob_pos",
        "top5_pos_frac",
        "vlm_top5_pos_frac",
        "vlm_top10_pos_frac",
        "vlm_top1_label",
        "vlm_top1_sim",
        "alt_model_pos_frac",
        "vlm_disagrees_with_dino",
    ]
    report_cols = [c for c in report_cols if c in summary.columns]
    report = [
        "# VLM Neighbor Audit",
        "",
        f"Model: {args.model_name}",
        f"Queries: top {args.top_n} verifier candidates",
        "",
        "## Summary",
        "",
        summary[report_cols].to_csv(index=False),
    ]
    (args.out_dir / "vlm_neighbor_report.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "model_name": args.model_name,
        "top_n": args.top_n,
        "top_k": top_k,
        "query_count": int(len(queries)),
        "ref_count": int(len(refs)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    print(summary[report_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
