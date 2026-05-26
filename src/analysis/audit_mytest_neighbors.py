#!/usr/bin/env python3
"""Audit mytest samples with DINO nearest neighbors over train/myval/test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.build_testlike_val import build_manifest, extract_dino_timm_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation/mytest_audit"))
    parser.add_argument("--dino-model", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(argparse.Namespace(
        train_labels_csv=Path("data/train_labels.csv"),
        myval_labels_csv=Path("data/myval/labels.csv"),
        train_crop_dir=Path("preprocess/bbox_crop/train"),
        myval_crop_dir=Path("preprocess/bbox_crop/myval"),
        test_crop_dir=Path("preprocess/bbox_crop/test"),
        mytest_root=Path("mytest"),
        not_stone_txt=Path("post_process/force_zero_lists/not-stone_best_071962.txt"),
    ))
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)

    ref_sources = {"train", "myval", "test"}
    query_source = "mytest"
    refs = manifest[(manifest["has_image"]) & (manifest["source"].isin(ref_sources))].copy().reset_index(drop=True)
    queries = manifest[(manifest["has_image"]) & (manifest["source"] == query_source)].copy().reset_index(drop=True)

    ref_features = extract_dino_timm_features(
        refs["path"].astype(str).tolist(),
        model_name=args.dino_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    query_features = extract_dino_timm_features(
        queries["path"].astype(str).tolist(),
        model_name=args.dino_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    ref_features = ref_features / np.maximum(np.linalg.norm(ref_features, axis=1, keepdims=True), 1e-8)
    query_features = query_features / np.maximum(np.linalg.norm(query_features, axis=1, keepdims=True), 1e-8)

    sims = cosine_similarity(query_features, ref_features)
    top_k = min(args.top_k, sims.shape[1])
    ref_cols = ["sample_id", "source", "image_id", "label", "path", "mask_status", "is_not_stone"]

    neighbor_rows = []
    query_rows = []
    for qi, qrow in queries.reset_index(drop=True).iterrows():
        row_sims = sims[qi]
        order = np.argsort(-row_sims)[:top_k]
        top_refs = refs.iloc[order].reset_index(drop=True)
        top_sims = row_sims[order]
        query_label = int(qrow["label"])
        source_counts = top_refs["source"].value_counts().to_dict()
        known_top_refs = top_refs[top_refs["label"].notna()].copy()
        label_counts = known_top_refs["label"].value_counts().to_dict()
        same_label_frac = float((known_top_refs["label"].astype(int) == query_label).mean()) if len(known_top_refs) else 0.0
        test_frac = float((top_refs["source"] == "test").mean())
        conflict_score = float((1.0 - same_label_frac) * float(top_sims.mean()) * 10.0)
        query_rows.append(
            {
                "sample_id": qrow["sample_id"],
                "image_id": qrow["image_id"],
                "path": qrow["path"],
                "query_label": query_label,
                "top1_sim": float(top_sims[0]),
                "top1_source": top_refs.iloc[0]["source"],
                "top1_label": int(top_refs.iloc[0]["label"]) if pd.notna(top_refs.iloc[0]["label"]) else -1,
                "top3_same_label_frac": float(
                    (known_top_refs.head(min(3, len(known_top_refs)))["label"].astype(int) == query_label).mean()
                ) if len(known_top_refs) else 0.0,
                "top5_same_label_frac": float(
                    (known_top_refs.head(min(5, len(known_top_refs)))["label"].astype(int) == query_label).mean()
                ) if len(known_top_refs) else 0.0,
                "top5_test_frac": float((top_refs.head(min(5, len(top_refs)))["source"] == "test").mean()),
                "top10_same_label_frac": same_label_frac,
                "top10_test_frac": test_frac,
                "top10_source_train": int(source_counts.get("train", 0)),
                "top10_source_myval": int(source_counts.get("myval", 0)),
                "top10_source_test": int(source_counts.get("test", 0)),
                "top10_source_pos": int(label_counts.get(1, 0)),
                "top10_source_neg": int(label_counts.get(0, 0)),
                "conflict_score": conflict_score,
            }
        )
        for rank, (ref_idx, sim) in enumerate(zip(order, top_sims), start=1):
            ref_row = refs.iloc[ref_idx]
            neighbor_rows.append(
                {
                    "query_sample_id": qrow["sample_id"],
                    "query_image_id": qrow["image_id"],
                    "query_label": query_label,
                    "rank": rank,
                    "ref_sample_id": ref_row["sample_id"],
                    "ref_source": ref_row["source"],
                    "ref_image_id": ref_row["image_id"],
                    "ref_label": int(ref_row["label"]) if pd.notna(ref_row["label"]) else -1,
                    "cosine_sim": float(sim),
                    "ref_path": ref_row["path"],
                    "ref_mask_status": ref_row["mask_status"],
                    "ref_is_not_stone": bool(ref_row["is_not_stone"]),
                }
            )

    query_df = pd.DataFrame(query_rows).sort_values(["conflict_score", "top1_sim"], ascending=False)
    neighbor_df = pd.DataFrame(neighbor_rows).sort_values(["query_sample_id", "rank"])

    query_df.to_csv(args.out_dir / "mytest_neighbor_summary.csv", index=False)
    neighbor_df.to_csv(args.out_dir / "mytest_neighbors_topk.csv", index=False)

    summary = {
        "query_count": int(len(queries)),
        "ref_count": int(len(refs)),
        "dino_model": args.dino_model,
        "top_k": int(top_k),
        "top_conflict_examples": query_df.head(20)[
            ["sample_id", "query_label", "top1_sim", "top1_source", "top1_label", "top5_same_label_frac", "top10_test_frac", "conflict_score"]
        ].to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(query_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
