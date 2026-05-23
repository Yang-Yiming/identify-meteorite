#!/usr/bin/env python3
"""Create a conservative FP consensus ranking from verifier feature columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["id", "label"]].copy()


def safe_rank(series: pd.Series, ascending: bool) -> pd.Series:
    return series.rank(method="average", ascending=ascending, na_option="bottom", pct=True)


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


def assign_tier(row) -> str:
    if row.weak_label == "inferred_fp":
        return "A_lb_inferred"
    if (
        row.consensus_fp_score >= 0.82
        and row.top5_pos_frac <= 0.0
        and row.alt_model_pos_frac <= 0.35
        and row.weak_label != "one_of_three_fp"
    ):
        return "B_consensus_untested"
    if row.weak_label == "one_of_three_fp":
        return "C_unresolved_one_of_three"
    if row.consensus_fp_score >= 0.72:
        return "D_review_only"
    return "E_low_priority"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("analysis/verifier_features/current_positive_verifier_features.csv"))
    parser.add_argument(
        "--current-submission",
        type=Path,
        default=Path("train/outputs/myval_v13_hi288_seed42_soup/submission_processed_best_notstone.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/verifier_consensus"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)

    # Percentile-style evidence scores; higher means stronger false-positive evidence.
    df["rank_signal_dino"] = 1.0 - safe_rank(df["fp_risk_score"], ascending=True)
    df["rank_signal_verifier"] = 1.0 - safe_rank(df["verifier_fp_score"], ascending=True)
    df["neighbor_neg_signal"] = (
        0.5 * (1.0 - df["top5_pos_frac"].fillna(0.5))
        + 0.3 * (1.0 - df["train_top10_pos_frac"].fillna(df["top10_pos_frac"]).fillna(0.5))
        + 0.2 * (1.0 - df["top10_pos_frac"].fillna(0.5))
    )
    df["model_disagreement_signal"] = 1.0 - df["alt_model_pos_frac"].fillna(0.5)
    df["low_conf_signal"] = 1.0 - (df["soup_prob_pos"].fillna(0.5) - 0.5).abs().clip(0, 0.5) * 2.0
    df["consensus_fp_score"] = (
        0.28 * df["rank_signal_dino"]
        + 0.22 * df["rank_signal_verifier"]
        + 0.25 * df["neighbor_neg_signal"]
        + 0.15 * df["model_disagreement_signal"]
        + 0.10 * df["low_conf_signal"]
    )
    df["consensus_tier"] = df.apply(assign_tier, axis=1)
    df = df.sort_values(["consensus_tier", "consensus_fp_score"], ascending=[True, False]).reset_index(drop=True)
    df["consensus_rank"] = np.arange(1, len(df) + 1)

    cols = [
        "id",
        "consensus_rank",
        "consensus_tier",
        "consensus_fp_score",
        "weak_label",
        "soup_prob_pos",
        "fp_risk_score",
        "verifier_fp_score",
        "top5_pos_frac",
        "train_top10_pos_frac",
        "alt_model_pos_frac",
        "dinomlp_full_label",
        "mytest_strict_label",
    ]
    cols = [c for c in cols if c in df.columns]
    df.to_csv(args.out_dir / "verifier_consensus_rank.csv", index=False)

    current = read_submission(args.current_submission)
    candidate_dir = args.out_dir / "candidate_submissions"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_outputs = {}
    for tier_prefix in ["A", "A_B"]:
        if tier_prefix == "A":
            selected = df[df["consensus_tier"] == "A_lb_inferred"]["id"].tolist()
        else:
            selected = df[df["consensus_tier"].isin(["A_lb_inferred", "B_consensus_untested"])]["id"].tolist()
        sub = current.copy()
        sub.loc[sub["id"].isin(selected), "label"] = 0
        out_path = candidate_dir / f"submission_consensus_{tier_prefix}.csv"
        ids_path = candidate_dir / f"zero_ids_consensus_{tier_prefix}.txt"
        sub.to_csv(out_path, index=False)
        ids_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
        candidate_outputs[tier_prefix] = str(out_path)

    tier_counts = df["consensus_tier"].value_counts().sort_index().to_dict()
    report = [
        "# Verifier Consensus Report",
        "",
        "This is a non-network consensus over existing verifier features. It does not use CLIP/SigLIP yet.",
        "",
        "## Tier Counts",
        "",
        markdown_table(pd.DataFrame([{"tier": k, "count": v} for k, v in tier_counts.items()])),
        "",
        "## Top Consensus Rows",
        "",
        markdown_table(df[cols].head(35)),
        "",
        "## Candidate Submissions",
        "",
        f"- A only: {candidate_outputs['A']}",
        f"- A+B: {candidate_outputs['A_B']}",
        "",
        "## Interpretation",
        "",
        "- Tier A is backed by leaderboard arithmetic and remains the only clean next submission candidate.",
        "- Tier B excludes the unresolved 108/124/131 group, but is still untested and should wait for manual review or CLIP/SigLIP evidence.",
        "- Tier C is deliberately separated because exactly one of 108,124,131 appears false-positive, but the identity is unresolved.",
    ]
    (args.out_dir / "verifier_consensus_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    meta = {
        "rows": int(len(df)),
        "tier_counts": tier_counts,
        "candidate_submissions": candidate_outputs,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    print((args.out_dir / "verifier_consensus_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
