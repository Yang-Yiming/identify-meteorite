#!/usr/bin/env python3
"""Analyze Testlike V4-saturated runs with submission-behavior tie-breakers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KNOWN_TEST_F1 = {
    "backbone_cs_augment/soup_top3.pt": 0.65263,
    "mytest_augment_v2/soup_top3.pt": 0.67021,
    "mytest_pretrain_finetune_v2/best.pt": 0.55214,
    "mytest_v1_s42/best.pt": 0.65979,
    "splitval_augment_v1/soup_top3.pt": 0.63212,
    "trsearch_bbox02/best.pt": 0.64516,
    "myval_v13_hi288_seed42_soup/soup.pt": 0.71962,
}


SUBMISSION_CANDIDATES = {
    "backbone_cs_augment/soup_top3.pt": "train/outputs/backbone_cs_augment/submission.csv",
    "dinov2_mlp_full": "train/outputs/dinov2_mlp_full/submission.csv",
    "dinov2_mlp_v2": "train/outputs/dinov2_mlp_v2/submission.csv",
    "dinov2_mlp_v3": "train/outputs/dinov2_mlp_v3/submission.csv",
    "mytest_augment_v1/best.pt": "train/outputs/mytest_augment_v1/submission.csv",
    "mytest_augment_v2/best.pt": "train/outputs/mytest_augment_v2/submission.csv",
    "mytest_augment_v2/soup_top3.pt": "train/outputs/mytest_augment_v2/submission.csv",
    "mytest_pretrain_finetune_v2/best.pt": "train/outputs/mytest_pretrain_finetune_v2/submission.csv",
    "mytest_strict_dino_v1/best.pt": "train/outputs/mytest_strict_dino_v1/submission_processed_best_notstone.csv",
    "mytest_v1_s42/best.pt": "train/outputs/mytest_v1_s42/submission_mytest_v1_s42.csv",
    "myval_v13_hi288_seed42_soup/soup.pt": "train/outputs/myval_v13_hi288_seed42_soup/submission_processed_best_notstone.csv",
    "myval_v19_ensemble42_123": "train/outputs/myval_v19_ensemble42_123/submission_ensemble42_123.csv",
    "splitval_augment_v1/soup_top3.pt": "train/outputs/splitval_augment_v1/submission.csv",
    "trsearch_bbox02/best.pt": "train/outputs/trsearch_bbox02/submission_final.csv",
}


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain id,label columns")
    out = df[["id", "label"]].copy()
    out["id"] = out["id"].astype(str)
    out["label"] = out["label"].astype(int)
    return out.sort_values("id").reset_index(drop=True)


def submission_stats(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, object]:
    merged = baseline.rename(columns={"label": "baseline_label"}).merge(
        candidate.rename(columns={"label": "candidate_label"}),
        on="id",
        how="inner",
    )
    if len(merged) != len(baseline):
        raise ValueError(f"Submission ID mismatch: baseline={len(baseline)} merged={len(merged)}")
    base = merged["baseline_label"].to_numpy()
    cand = merged["candidate_label"].to_numpy()
    base_pos = base == 1
    cand_pos = cand == 1
    pos_to_neg = merged.loc[base_pos & ~cand_pos, "id"].tolist()
    neg_to_pos = merged.loc[~base_pos & cand_pos, "id"].tolist()
    return {
        "submission_pos": int(cand_pos.sum()),
        "diff_vs_baseline": int((base != cand).sum()),
        "baseline_pos_to_neg": int(len(pos_to_neg)),
        "baseline_neg_to_pos": int(len(neg_to_pos)),
        "pos_to_neg_ids": ",".join(pos_to_neg[:25]),
        "neg_to_pos_ids": ",".join(neg_to_pos[:25]),
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df[columns].itertuples(index=False):
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
    parser.add_argument("--v4-summary", type=Path, default=Path("analysis/all_checkpoints_v4_eval/all_eval_summary.csv"))
    parser.add_argument(
        "--baseline-submission",
        type=Path,
        default=Path("train/outputs/myval_v13_hi288_seed42_soup/submission_processed_best_notstone.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/v4_tiebreaker"))
    parser.add_argument("--cluster-gate", type=float, default=0.993)
    parser.add_argument("--top-gate", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    v4 = pd.read_csv(args.v4_summary)
    baseline = read_submission(args.baseline_submission)

    rows = []
    for row in v4.itertuples(index=False):
        run_tag = str(row.run_tag)
        out = row._asdict()
        out["known_test_f1"] = KNOWN_TEST_F1.get(run_tag)
        out["uses_mytest_supervision"] = run_tag.startswith("mytest") or "augment" in run_tag or "splitval_augment" in run_tag
        sub_path = SUBMISSION_CANDIDATES.get(run_tag)
        if sub_path and Path(sub_path).is_file():
            out["submission_path"] = sub_path
            out.update(submission_stats(read_submission(Path(sub_path)), baseline))
        rows.append(out)

    df = pd.DataFrame(rows)
    cluster_col = "f1_at_0_5__testlike_cluster_dino_v3"
    top_col = "f1_at_0_5__testlike_top_dino_v3"
    df["passes_v4_gate"] = (df[cluster_col] >= args.cluster_gate) & (df[top_col] >= args.top_gate)
    df["v4_mean_f1"] = (df[cluster_col] + df[top_col]) / 2.0
    df["hidden_risk_flag"] = ""
    df.loc[df["uses_mytest_supervision"], "hidden_risk_flag"] += "mytest_supervised;"
    df.loc[df["diff_vs_baseline"].fillna(0) > 30, "hidden_risk_flag"] += "large_submission_diff;"
    df.loc[(df["submission_pos"].fillna(128) < 110) | (df["submission_pos"].fillna(128) > 145), "hidden_risk_flag"] += "pos_count_far;"

    df.to_csv(args.out_dir / "v4_tiebreaker_summary.csv", index=False)

    gated = df[df["passes_v4_gate"]].copy()
    known = gated[gated["known_test_f1"].notna()].sort_values("known_test_f1", ascending=False)
    submission_cols = [
        "run_tag",
        "v4_mean_f1",
        "known_test_f1",
        "submission_pos",
        "diff_vs_baseline",
        "baseline_pos_to_neg",
        "baseline_neg_to_pos",
        "uses_mytest_supervision",
        "hidden_risk_flag",
    ]
    comparable = gated[gated["submission_path"].notna()].copy()
    comparable = comparable.sort_values(
        ["uses_mytest_supervision", "diff_vs_baseline", "v4_mean_f1"],
        ascending=[True, True, False],
    )
    saturated = df[(df[cluster_col] >= 0.999) & (df[top_col] >= 0.999)]

    report = [
        "# Testlike V4 Tie-breaker Report",
        "",
        f"V4 gate: cluster >= {args.cluster_gate}, top >= {args.top_gate}",
        f"Runs evaluated: {len(df)}",
        f"Runs passing gate: {len(gated)}",
        f"Runs saturated at 1.0/1.0: {len(saturated)}",
        "",
        "## Known Kaggle Outcomes Inside V4 Gate",
        "",
        markdown_table(known, [c for c in submission_cols if c in known.columns]),
        "",
        "## Comparable Submissions Passing V4 Gate",
        "",
        markdown_table(comparable, [c for c in submission_cols if c in comparable.columns]),
        "",
        "## Interpretation",
        "",
        "- V4 is a useful gate but not a sufficient ranking metric: known Kaggle regressions pass or saturate it.",
        "- Prefer V4-passing candidates with small submission diffs from current best unless a diff is backed by leaderboard arithmetic.",
        "- Penalize mytest-supervised runs even when V4-perfect; historical Kaggle evidence shows domain-shift regressions.",
        "- The next submission-side candidate remains current best plus inferred force-zero 88,177 only.",
    ]
    (args.out_dir / "v4_tiebreaker_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    meta = {
        "v4_summary": str(args.v4_summary),
        "baseline_submission": str(args.baseline_submission),
        "cluster_gate": args.cluster_gate,
        "top_gate": args.top_gate,
        "runs_evaluated": int(len(df)),
        "runs_passing_gate": int(len(gated)),
        "runs_saturated": int(len(saturated)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    print((args.out_dir / "v4_tiebreaker_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
