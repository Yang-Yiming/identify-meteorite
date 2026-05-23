#!/usr/bin/env python3
"""Rank likely false positives in the current test submission.

This is an audit tool, not a trainer.  It builds nearest-neighbor evidence for
each Kaggle test image and combines it with existing submission/model signals.
The main use case is finding high-risk positives to manually review or ablate
with force-zero post-processing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.build_testlike_val import (  # noqa: E402
    build_manifest,
    extract_dino_timm_features,
    extract_stats_features,
)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load_submission(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path} must contain an id column")
    keep = ["id"]
    rename: dict[str, str] = {}
    for col in df.columns:
        if col == "id":
            continue
        keep.append(col)
        rename[col] = f"{prefix}_{col}"
    return df[keep].rename(columns=rename)


def image_number(image_id: str) -> int:
    return int(Path(str(image_id)).stem)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/test_fp_risk_audit"))
    parser.add_argument("--feature-backend", choices=("dino_timm", "stats"), default="dino_timm")
    parser.add_argument("--dino-model", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--ref-sources", type=str, default="train,myval,mytest")
    parser.add_argument("--fit-sources", type=str, default="train,myval,test,mytest")
    parser.add_argument("--candidate-top-n", type=str, default="5,10,15,20,25")
    parser.add_argument("--train-labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--myval-labels-csv", type=Path, default=Path("data/myval/labels.csv"))
    parser.add_argument("--train-crop-dir", type=Path, default=Path("preprocess/bbox_crop/train"))
    parser.add_argument("--myval-crop-dir", type=Path, default=Path("preprocess/bbox_crop/myval"))
    parser.add_argument("--test-crop-dir", type=Path, default=Path("preprocess/bbox_crop/test"))
    parser.add_argument("--mytest-root", type=Path, default=Path("mytest"))
    parser.add_argument("--not-stone-txt", type=Path, default=Path("post_process/not-stone_best_071962.txt"))
    parser.add_argument(
        "--soup-probs",
        type=Path,
        default=Path("train/outputs/myval_v13_hi288_seed42_soup/submission_probs.csv"),
    )
    parser.add_argument(
        "--soup-processed",
        type=Path,
        default=Path("train/outputs/myval_v13_hi288_seed42_soup/submission_processed_best_notstone.csv"),
    )
    parser.add_argument("--dino-mlp-submission", type=Path, default=Path("train/outputs/dinov2_mlp_full/submission.csv"))
    parser.add_argument("--testlike-scores", type=Path, default=Path("analysis/testlike_dino_train_v4/test_like_scores.csv"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_args = argparse.Namespace(
        train_labels_csv=args.train_labels_csv,
        myval_labels_csv=args.myval_labels_csv,
        train_crop_dir=args.train_crop_dir,
        myval_crop_dir=args.myval_crop_dir,
        test_crop_dir=args.test_crop_dir,
        mytest_root=args.mytest_root,
        not_stone_txt=args.not_stone_txt,
    )
    manifest = build_manifest(manifest_args)
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)

    ref_sources = {s.strip() for s in args.ref_sources.split(",") if s.strip()}
    fit_sources = {s.strip() for s in args.fit_sources.split(",") if s.strip()}
    feature_sources = ref_sources | fit_sources | {"test"}
    feature_df = manifest[(manifest["has_image"]) & (manifest["source"].isin(feature_sources))].copy().reset_index(drop=True)
    paths = feature_df["path"].astype(str).tolist()

    if args.feature_backend == "stats":
        features = extract_stats_features(paths)
    else:
        features = extract_dino_timm_features(
            paths=paths,
            model_name=args.dino_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )

    fit_mask = feature_df["source"].isin(fit_sources).to_numpy()
    scaler = StandardScaler().fit(features[fit_mask])
    features_scaled = scaler.transform(features)
    pca_n = min(args.pca_components, int(fit_mask.sum()), features_scaled.shape[1])
    if pca_n < features_scaled.shape[1]:
        pca = PCA(n_components=pca_n, random_state=42).fit(features_scaled[fit_mask])
        features_work = pca.transform(features_scaled)
    else:
        features_work = features_scaled
    features_norm = normalize_rows(features_work)

    refs = feature_df[feature_df["source"].isin(ref_sources) & feature_df["label"].notna()].copy()
    tests = feature_df[feature_df["source"] == "test"].copy()
    ref_idx = refs.index.to_numpy()
    test_idx = tests.index.to_numpy()
    ref_features = features_norm[ref_idx]
    test_features = features_norm[test_idx]
    sims = cosine_similarity(test_features, ref_features)
    top_k = min(args.top_k, sims.shape[1])

    neighbor_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for local_i, (_, qrow) in enumerate(tests.iterrows()):
        row_sims = sims[local_i]
        order = np.argsort(-row_sims)[:top_k]
        top_refs = refs.iloc[order].reset_index(drop=True)
        top_sims = row_sims[order]

        def frac_pos(source: str | None, n: int) -> float:
            sub = top_refs.head(n)
            if source is not None:
                sub = sub[sub["source"] == source]
            if len(sub) == 0:
                return np.nan
            return float((sub["label"].astype(int) == 1).mean())

        def count_label(source: str | None, n: int, label: int) -> int:
            sub = top_refs.head(n)
            if source is not None:
                sub = sub[sub["source"] == source]
            return int((sub["label"].astype(int) == label).sum())

        known_top5_pos_frac = frac_pos(None, min(5, top_k))
        known_top10_pos_frac = frac_pos(None, min(10, top_k))
        train_top10_pos_frac = frac_pos("train", min(10, top_k))
        myval_top10_pos_frac = frac_pos("myval", min(10, top_k))
        mytest_top10_pos_frac = frac_pos("mytest", min(10, top_k))

        summary_rows.append(
            {
                "id": qrow["image_id"],
                "sample_id": qrow["sample_id"],
                "path": qrow["path"],
                "is_not_stone": bool(qrow["is_not_stone"]),
                "mask_status": qrow["mask_status"],
                "top1_sim": float(top_sims[0]),
                "top1_source": top_refs.iloc[0]["source"],
                "top1_label": int(top_refs.iloc[0]["label"]),
                "top5_pos_frac": known_top5_pos_frac,
                "top10_pos_frac": known_top10_pos_frac,
                "train_top10_pos_frac": train_top10_pos_frac,
                "myval_top10_pos_frac": myval_top10_pos_frac,
                "mytest_top10_pos_frac": mytest_top10_pos_frac,
                "top10_train_neg": count_label("train", min(10, top_k), 0),
                "top10_train_pos": count_label("train", min(10, top_k), 1),
                "top10_myval_neg": count_label("myval", min(10, top_k), 0),
                "top10_myval_pos": count_label("myval", min(10, top_k), 1),
                "top10_mytest_neg": count_label("mytest", min(10, top_k), 0),
                "top10_mytest_pos": count_label("mytest", min(10, top_k), 1),
                "mean_top5_sim": float(top_sims[: min(5, top_k)].mean()),
                "mean_top10_sim": float(top_sims[: min(10, top_k)].mean()),
            }
        )

        for rank, (ref_local_idx, sim) in enumerate(zip(order, top_sims), start=1):
            ref_row = refs.iloc[ref_local_idx]
            neighbor_rows.append(
                {
                    "query_id": qrow["image_id"],
                    "rank": rank,
                    "cosine_sim": float(sim),
                    "ref_sample_id": ref_row["sample_id"],
                    "ref_source": ref_row["source"],
                    "ref_image_id": ref_row["image_id"],
                    "ref_label": int(ref_row["label"]),
                    "ref_path": ref_row["path"],
                }
            )

    summary = pd.DataFrame(summary_rows)
    neighbors = pd.DataFrame(neighbor_rows)

    if args.soup_probs.is_file():
        summary = summary.merge(load_submission(args.soup_probs, "soup"), on="id", how="left")
    if args.soup_processed.is_file():
        summary = summary.merge(load_submission(args.soup_processed, "processed"), on="id", how="left")
    if args.dino_mlp_submission.is_file():
        summary = summary.merge(load_submission(args.dino_mlp_submission, "dinomlp"), on="id", how="left")
    if args.testlike_scores.is_file():
        scores = pd.read_csv(args.testlike_scores)
        test_scores = scores[scores["source"] == "test"][["image_id", "cluster", "cluster_test_fraction", "test_like_score"]]
        summary = summary.merge(test_scores.rename(columns={"image_id": "id", "cluster": "v4_cluster"}), on="id", how="left")

    soup_prob = summary.get("soup_prob_pos", pd.Series(np.nan, index=summary.index)).astype(float)
    processed_label = summary.get("processed_label", summary.get("soup_label", pd.Series(0, index=summary.index))).fillna(0).astype(int)
    dinomlp_label = summary.get("dinomlp_label", pd.Series(0, index=summary.index)).fillna(0).astype(int)
    top5_pos = summary["top5_pos_frac"].fillna(summary["top10_pos_frac"]).fillna(0.5)
    train_pos = summary["train_top10_pos_frac"].fillna(summary["top10_pos_frac"]).fillna(0.5)
    myval_pos = summary["myval_top10_pos_frac"].fillna(summary["top10_pos_frac"]).fillna(0.5)
    mytest_pos = summary["mytest_top10_pos_frac"].fillna(0.5)
    low_conf = 1.0 - (soup_prob.fillna(0.5) - 0.5).abs().clip(0, 0.5) * 2.0
    mlp_disagree = (dinomlp_label == 0).astype(float)

    summary["fp_risk_score"] = (
        0.30 * (1.0 - top5_pos)
        + 0.25 * (1.0 - train_pos)
        + 0.15 * (1.0 - myval_pos)
        + 0.10 * (1.0 - mytest_pos)
        + 0.10 * low_conf
        + 0.10 * mlp_disagree
    )
    summary.loc[processed_label == 0, "fp_risk_score"] *= 0.25
    summary["fp_risk_rank"] = summary["fp_risk_score"].rank(method="first", ascending=False).astype(int)
    summary["image_number"] = summary["id"].map(image_number)
    summary = summary.sort_values(["fp_risk_score", "soup_prob_pos"], ascending=[False, False])

    neighbors.to_csv(args.out_dir / "test_neighbors_topk.csv", index=False)
    summary.to_csv(args.out_dir / "test_fp_risk_summary.csv", index=False)

    current_not_stone = set()
    if args.not_stone_txt.is_file():
        for line in args.not_stone_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                current_not_stone.add(int(line))

    positive_candidates = summary[processed_label.loc[summary.index].astype(int) == 1].copy()
    top_ns = [int(x) for x in args.candidate_top_n.split(",") if x.strip()]
    candidate_files = {}
    for n in top_ns:
        force_numbers = sorted(current_not_stone | set(positive_candidates.head(n)["image_number"].astype(int).tolist()))
        out_path = args.out_dir / f"not-stone_plus_top{n}_fp_risk.txt"
        out_path.write_text("\n".join(str(x) for x in force_numbers) + "\n", encoding="utf-8")
        candidate_files[f"top{n}"] = str(out_path)

    report_cols = [
        "id",
        "fp_risk_score",
        "soup_prob_pos",
        "soup_label",
        "processed_label",
        "dinomlp_label",
        "top1_source",
        "top1_label",
        "top1_sim",
        "top5_pos_frac",
        "train_top10_pos_frac",
        "myval_top10_pos_frac",
        "mytest_top10_pos_frac",
        "v4_cluster",
        "test_like_score",
        "path",
    ]
    report_cols = [c for c in report_cols if c in summary.columns]
    top_report = positive_candidates.head(30)[report_cols]
    report_csv = top_report.to_csv(index=False)
    (args.out_dir / "top_fp_risk_positives.md").write_text(
        "# Top FP-risk current positives\n\n```csv\n" + report_csv + "```\n",
        encoding="utf-8",
    )
    meta = {
        "feature_backend": args.feature_backend,
        "dino_model": args.dino_model if args.feature_backend == "dino_timm" else None,
        "feature_rows": int(len(feature_df)),
        "test_rows_with_images": int(len(tests)),
        "ref_rows": int(len(refs)),
        "top_k": int(top_k),
        "candidate_files": candidate_files,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(top_report.head(20).to_string(index=False))
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
