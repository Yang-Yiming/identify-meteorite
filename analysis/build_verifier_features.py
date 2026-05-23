#!/usr/bin/env python3
"""Build second-stage verifier features for current-best positive test samples.

The verifier target is narrow: decide which current-best positives are likely
false positives.  This script does not train a model; it assembles evidence and
weak labels so future rules/stackers can be compared consistently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SUBMISSION_SOURCES = {
    "dinomlp_full": "train/outputs/dinov2_mlp_full/submission.csv",
    "dinomlp_v2": "train/outputs/dinov2_mlp_v2/submission.csv",
    "dinomlp_v3": "train/outputs/dinov2_mlp_v3/submission.csv",
    "mytest_augment": "train/outputs/mytest_augment_v2/submission.csv",
    "mytest_strict": "train/outputs/mytest_strict_dino_v1/submission_processed_best_notstone.csv",
    "mytest_split": "train/outputs/mytest_v1_s42/submission_mytest_v1_s42.csv",
    "ensemble42_123": "train/outputs/myval_v19_ensemble42_123/submission_ensemble42_123.csv",
    "splitval_augment": "train/outputs/splitval_augment_v1/submission.csv",
    "trsearch_bbox02": "train/outputs/trsearch_bbox02/submission_final.csv",
}


def read_submission(path: Path, label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or label_col not in df.columns:
        raise ValueError(f"{path} must contain id,{label_col}")
    out = df[["id", label_col]].copy()
    out["id"] = out["id"].astype(str)
    out[label_col] = out[label_col].astype(int)
    return out.rename(columns={label_col: "label"})


def normalize_id(value: str) -> str:
    return f"{int(Path(str(value)).stem):06d}.jpg"


def id_number(value: str) -> int:
    return int(Path(str(value)).stem)


def load_known_sets() -> tuple[set[str], set[str], dict[str, str]]:
    weak_fp = {"000088.jpg", "000177.jpg"}
    # From top5 and manual ablations: among these three, exactly one is likely FP,
    # but the individual identity is not resolved.
    one_of_three = {"000108.jpg", "000124.jpg", "000131.jpg"}
    notes = {sid: "inferred_fp_from_pairwise_lb" for sid in weak_fp}
    for sid in one_of_three:
        notes[sid] = "one_of_108_124_131_is_fp"
    return weak_fp, one_of_three, notes


def add_optional_submission_features(features: pd.DataFrame, sources: dict[str, str]) -> pd.DataFrame:
    out = features
    label_cols = []
    for name, raw_path in sources.items():
        path = Path(raw_path)
        if not path.is_file():
            continue
        sub = read_submission(path).rename(columns={"label": f"{name}_label"})
        out = out.merge(sub, on="id", how="left")
        label_cols.append(f"{name}_label")
    for col in label_cols:
        out[col] = out[col].fillna(-1).astype(int)
    if label_cols:
        labels = out[label_cols].replace(-1, np.nan)
        out["alt_model_pos_votes"] = labels.sum(axis=1, skipna=True)
        out["alt_model_count"] = labels.notna().sum(axis=1)
        out["alt_model_pos_frac"] = out["alt_model_pos_votes"] / out["alt_model_count"].replace(0, np.nan)
        out["alt_model_neg_votes"] = out["alt_model_count"] - out["alt_model_pos_votes"]
    return out


def score_rules(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = {
        "inferred_88_177": features["id"].isin(["000088.jpg", "000177.jpg"]),
        "dino_risk_top2": features["fp_risk_rank"] <= 2,
        "dino_risk_top3": features["fp_risk_rank"] <= 3,
        "dino_risk_top5": features["fp_risk_rank"] <= 5,
        "top5_pos_zero_and_dinomlp_neg": (features["top5_pos_frac"] <= 0.0) & (features.get("dinomlp_full_label", 1) == 0),
        "strict_consensus_neg": (
            (features["top5_pos_frac"] <= 0.0)
            & (features["train_top10_pos_frac"].fillna(1.0) <= 0.1)
            & (features.get("dinomlp_full_label", 1) == 0)
            & (features["soup_prob_pos"] < 0.9)
        ),
        "alt_majority_negative_high_risk": (
            (features["alt_model_pos_frac"].fillna(1.0) <= 0.5)
            & (features["fp_risk_score"] >= 0.75)
        ),
    }
    weak_fp, one_of_three, _ = load_known_sets()
    for name, mask in rules.items():
        selected = features[mask].copy()
        ids = set(selected["id"])
        rows.append(
            {
                "rule": name,
                "n_selected": int(len(selected)),
                "selected_ids": ",".join(selected["id"].tolist()),
                "known_inferred_fp_hits": int(len(ids & weak_fp)),
                "one_of_three_selected": int(len(ids & one_of_three)),
                "mean_fp_risk": float(selected["fp_risk_score"].mean()) if len(selected) else np.nan,
                "mean_soup_prob": float(selected["soup_prob_pos"].mean()) if len(selected) else np.nan,
                "mean_alt_pos_frac": float(selected["alt_model_pos_frac"].mean()) if len(selected) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["known_inferred_fp_hits", "one_of_three_selected", "n_selected", "mean_fp_risk"],
        ascending=[False, True, True, False],
    )


def write_rule_submissions(
    rules: pd.DataFrame,
    current_submission: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    candidate_dir = out_dir / "candidate_submissions"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for row in rules.itertuples(index=False):
        rule = str(row.rule)
        selected_ids = [item for item in str(row.selected_ids).split(",") if item]
        submission = current_submission.rename(columns={"current_label": "label"}).copy()
        submission.loc[submission["id"].isin(selected_ids), "label"] = 0
        out_path = candidate_dir / f"submission_{rule}.csv"
        submission[["id", "label"]].to_csv(out_path, index=False)
        ids_path = candidate_dir / f"zero_ids_{rule}.txt"
        ids_path.write_text("\n".join(selected_ids) + ("\n" if selected_ids else ""), encoding="utf-8")
        outputs[rule] = str(out_path)
    return outputs


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
    parser.add_argument(
        "--current-submission",
        type=Path,
        default=Path("train/outputs/myval_v13_hi288_seed42_soup/submission_processed_best_notstone.csv"),
    )
    parser.add_argument(
        "--soup-probs",
        type=Path,
        default=Path("train/outputs/myval_v13_hi288_seed42_soup/submission_probs.csv"),
    )
    parser.add_argument(
        "--fp-risk-summary",
        type=Path,
        default=Path("analysis/test_fp_risk_audit_dino_nomtest/test_fp_risk_summary.csv"),
    )
    parser.add_argument("--v4-scores", type=Path, default=Path("analysis/testlike_dino_train_v4/test_like_scores.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/verifier_features"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    current = read_submission(args.current_submission).rename(columns={"label": "current_label"})
    current_pos = current[current["current_label"] == 1].copy()
    current_pos["image_number"] = current_pos["id"].map(id_number)

    probs = pd.read_csv(args.soup_probs)
    prob_cols = [c for c in ["id", "prob_pos", "prob_pos_corrected", "label"] if c in probs.columns]
    probs = probs[prob_cols].rename(columns={"prob_pos": "soup_prob_pos", "prob_pos_corrected": "soup_prob_pos_corrected", "label": "soup_raw_label"})
    features = current_pos.merge(probs, on="id", how="left")

    fp = pd.read_csv(args.fp_risk_summary)
    keep_fp_cols = [
        "id",
        "fp_risk_score",
        "fp_risk_rank",
        "top1_source",
        "top1_label",
        "top1_sim",
        "top5_pos_frac",
        "top10_pos_frac",
        "train_top10_pos_frac",
        "myval_top10_pos_frac",
        "mean_top5_sim",
        "mean_top10_sim",
    ]
    features = features.merge(fp[[c for c in keep_fp_cols if c in fp.columns]], on="id", how="left")

    v4 = pd.read_csv(args.v4_scores)
    v4_test = v4[v4["source"] == "test"].copy()
    v4_test["id"] = v4_test["image_id"].map(normalize_id)
    v4_cols = ["id", "cluster", "cluster_test_fraction", "test_like_score", "test_like_topk_mean", "test_like_centroid_cos"]
    features = features.merge(v4_test[[c for c in v4_cols if c in v4_test.columns]], on="id", how="left")
    features = features.rename(columns={"cluster": "v4_cluster"})

    features = add_optional_submission_features(features, SUBMISSION_SOURCES)

    weak_fp, one_of_three, notes = load_known_sets()
    features["weak_label"] = ""
    features.loc[features["id"].isin(weak_fp), "weak_label"] = "inferred_fp"
    features.loc[features["id"].isin(one_of_three), "weak_label"] = "one_of_three_fp"
    features["weak_label_note"] = features["id"].map(notes).fillna("")

    # Composite is intentionally simple and inspectable; it ranks FP-review priority,
    # not a calibrated probability.
    features["verifier_fp_score"] = (
        0.35 * features["fp_risk_score"].fillna(0.0)
        + 0.25 * (1.0 - features["top5_pos_frac"].fillna(0.5))
        + 0.20 * (1.0 - features["train_top10_pos_frac"].fillna(features["top10_pos_frac"]).fillna(0.5))
        + 0.10 * (1.0 - features["alt_model_pos_frac"].fillna(0.5))
        + 0.10 * (1.0 - (features["soup_prob_pos"].fillna(0.5) - 0.5).abs() * 2.0)
    )
    features = features.sort_values(["verifier_fp_score", "fp_risk_score"], ascending=False).reset_index(drop=True)
    features["verifier_rank"] = np.arange(1, len(features) + 1)

    rules = score_rules(features)
    candidate_outputs = write_rule_submissions(rules, current.rename(columns={"label": "current_label"}), args.out_dir)

    features.to_csv(args.out_dir / "current_positive_verifier_features.csv", index=False)
    rules.to_csv(args.out_dir / "verifier_rule_candidates.csv", index=False)

    top_cols = [
        "id",
        "verifier_rank",
        "verifier_fp_score",
        "weak_label",
        "soup_prob_pos",
        "fp_risk_score",
        "fp_risk_rank",
        "top5_pos_frac",
        "train_top10_pos_frac",
        "alt_model_pos_frac",
        "dinomlp_full_label",
        "mytest_strict_label",
        "v4_cluster",
        "test_like_score",
    ]
    top_cols = [c for c in top_cols if c in features.columns]
    report = [
        "# Current-positive Verifier Features",
        "",
        f"Current-best positives: {len(features)}",
        "",
        "## Top FP-review Candidates",
        "",
        markdown_table(features[top_cols].head(30)),
        "",
        "## Rule Candidates",
        "",
        markdown_table(rules),
        "",
        "## Notes",
        "",
        "- `weak_label=inferred_fp` currently means IDs 88 and 177, inferred from leaderboard arithmetic.",
        "- `weak_label=one_of_three_fp` means exactly one of 108,124,131 is likely FP, but identity is unresolved.",
        "- The composite `verifier_fp_score` is for review/rule ranking only; it is not calibrated.",
        "- Candidate submissions for each rule are written under `candidate_submissions/`.",
    ]
    (args.out_dir / "verifier_feature_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    meta = {
        "current_positive_count": int(len(features)),
        "feature_output": str(args.out_dir / "current_positive_verifier_features.csv"),
        "rules_output": str(args.out_dir / "verifier_rule_candidates.csv"),
        "report": str(args.out_dir / "verifier_feature_report.md"),
        "candidate_submissions": candidate_outputs,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    print((args.out_dir / "verifier_feature_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
