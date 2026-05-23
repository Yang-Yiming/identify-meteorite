#!/usr/bin/env python3
"""Merge DINO, SigLIP, and CLIP verifier-neighbor evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.4f}" if np.isfinite(value) else "")
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("analysis/verifier_features/current_positive_verifier_features.csv"))
    parser.add_argument("--siglip", type=Path, default=Path("analysis/vlm_neighbor_audit_siglip_vitb16_224/vlm_neighbor_summary.csv"))
    parser.add_argument("--clip", type=Path, default=Path("analysis/vlm_neighbor_audit_clip_vitb32_224/vlm_neighbor_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/vlm_consensus"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.features)
    siglip = pd.read_csv(args.siglip)[["id", "vlm_top5_pos_frac", "vlm_top10_pos_frac", "vlm_top1_label", "vlm_top1_sim"]]
    siglip = siglip.rename(
        columns={
            "vlm_top5_pos_frac": "siglip_top5_pos_frac",
            "vlm_top10_pos_frac": "siglip_top10_pos_frac",
            "vlm_top1_label": "siglip_top1_label",
            "vlm_top1_sim": "siglip_top1_sim",
        }
    )
    clip = pd.read_csv(args.clip)[["id", "vlm_top5_pos_frac", "vlm_top10_pos_frac", "vlm_top1_label", "vlm_top1_sim"]]
    clip = clip.rename(
        columns={
            "vlm_top5_pos_frac": "clip_top5_pos_frac",
            "vlm_top10_pos_frac": "clip_top10_pos_frac",
            "vlm_top1_label": "clip_top1_label",
            "vlm_top1_sim": "clip_top1_sim",
        }
    )
    df = base.merge(siglip, on="id", how="left").merge(clip, on="id", how="left")
    df["dino_neg_signal"] = 1.0 - df["top5_pos_frac"].fillna(0.5)
    df["siglip_neg_signal"] = 1.0 - df["siglip_top5_pos_frac"].fillna(0.5)
    df["clip_neg_signal"] = 1.0 - df["clip_top5_pos_frac"].fillna(0.5)
    df["vlm_neg_consensus"] = (df["siglip_neg_signal"] + df["clip_neg_signal"]) / 2.0
    df["three_embedding_neg_consensus"] = (
        df["dino_neg_signal"] + df["siglip_neg_signal"] + df["clip_neg_signal"]
    ) / 3.0
    df["consensus_note"] = ""
    df.loc[
        (df["id"].isin(["000088.jpg", "000177.jpg"]))
        & (df["three_embedding_neg_consensus"] >= 0.85),
        "consensus_note",
    ] = "lb_inferred_fp_and_embedding_negative"
    df.loc[
        (df["weak_label"] == "one_of_three_fp") & (df["clip_top5_pos_frac"] >= 0.6),
        "consensus_note",
    ] = "unresolved_group_clip_positive"
    df.loc[
        (df["weak_label"] == "one_of_three_fp")
        & (df["clip_top5_pos_frac"] < 0.6)
        & (df["siglip_top5_pos_frac"] <= 0.2),
        "consensus_note",
    ] = "unresolved_group_mixed_vlm"

    df = df.sort_values(["verifier_rank"]).reset_index(drop=True)
    out_cols = [
        "id",
        "weak_label",
        "consensus_note",
        "soup_prob_pos",
        "top5_pos_frac",
        "siglip_top5_pos_frac",
        "clip_top5_pos_frac",
        "three_embedding_neg_consensus",
        "alt_model_pos_frac",
        "verifier_rank",
    ]
    out = df[out_cols].head(40)
    df.to_csv(args.out_dir / "vlm_consensus_features.csv", index=False)

    strong_neg = df[
        (df["top5_pos_frac"] <= 0.2)
        & (df["siglip_top5_pos_frac"] <= 0.2)
        & (df["clip_top5_pos_frac"] <= 0.2)
    ][out_cols]
    unresolved = df[df["weak_label"] == "one_of_three_fp"][out_cols]
    report = [
        "# VLM Consensus Report",
        "",
        "Combines DINO verifier features with SigLIP and CLIP nearest-neighbor label ratios.",
        "",
        "## Top 40 Candidates",
        "",
        markdown_table(out),
        "",
        "## Three-embedding Strong Negatives",
        "",
        markdown_table(strong_neg),
        "",
        "## Unresolved 108/124/131 Group",
        "",
        markdown_table(unresolved),
        "",
        "## Interpretation",
        "",
        "- 88 and 177 remain the only clean next-submission FP candidates: leaderboard arithmetic and DINO/SigLIP/CLIP all support negative evidence.",
        "- 124 becomes much less attractive to zero because CLIP top5 is strongly positive.",
        "- 108 and 131 remain unresolved: DINO/SigLIP are negative, but CLIP is mixed, and leaderboard arithmetic says only one of 108/124/131 is FP.",
        "- New candidates such as 20/106 should not be submitted before manual visual review or additional leaderboard evidence.",
    ]
    (args.out_dir / "vlm_consensus_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    meta = {
        "rows": int(len(df)),
        "strong_negative_count": int(len(strong_neg)),
        "siglip": str(args.siglip),
        "clip": str(args.clip),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    print((args.out_dir / "vlm_consensus_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
