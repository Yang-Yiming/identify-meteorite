#!/usr/bin/env python3
"""Error diagnostic: analyze soup model predictions on labeled data.

Reads proxy_eval_predictions.csv and produces:
1. Error summary per dataset (FP, FN, F1, precision, recall)
2. Hard case identification (prob_pos near 0.5)
3. Optimal threshold search per dataset
4. Myval error cross-reference with not-stone audit
5. Per-class probability distribution stats
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_CSV = PROJECT_ROOT / "evaluation" / "testlike_v4_eval" / "proxy_eval_predictions.csv"
NOT_STONE_CSV = PROJECT_ROOT / "evaluation" / "not_stone_audit" / "not_stone_model_audit.csv"
SUBMISSION_CSV = PROJECT_ROOT / "submissions" / "submission_soup_notstone_keep_44_100_145_162_187.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RUN = "soup_reduced_notstone"


def load_predictions(csv_path: Path) -> dict:
    """Load predictions, group by run -> dataset -> list of dicts."""
    data: dict = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row["run"]
            dataset = row["dataset"]
            data.setdefault(run, {}).setdefault(dataset, []).append(row)
    return data


def load_not_stone_audit(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(prob_pos: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (prob_pos >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_pos_pred": int(preds.sum()),
        "n_neg_pred": int((preds == 0).sum()),
    }


def search_best_threshold(prob_pos: np.ndarray, labels: np.ndarray) -> dict:
    candidates = sorted(set(np.concatenate([[0.0, 0.5, 1.0], prob_pos])))
    best = {"threshold": 0.5, "f1": 0.0}
    for t in candidates:
        m = compute_metrics(prob_pos, labels, t)
        if m["f1"] > best["f1"]:
            best = {"threshold": t, "f1": m["f1"], **m}
    return best


def analyze_dataset(rows: list[dict]) -> dict:
    prob_pos = np.array([float(r["prob_pos"]) for r in rows], dtype=np.float64)
    labels = np.array([int(float(r["label"])) for r in rows], dtype=np.int32)
    image_ids = [r["image_id"] for r in rows]

    # Stats at threshold 0.5
    at_05 = compute_metrics(prob_pos, labels, 0.5)
    best = search_best_threshold(prob_pos, labels)

    # Probability distribution
    pos_probs = prob_pos[labels == 1]
    neg_probs = prob_pos[labels == 0]

    dist = {
        "n_total": len(prob_pos),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
        "neg_pos_ratio": round(int((labels == 0).sum()) / max(1, int(labels.sum())), 4),
        "prob_pos_mean": round(float(prob_pos.mean()), 6),
        "prob_pos_median": round(float(np.median(prob_pos)), 6),
        "prob_pos_std": round(float(prob_pos.std()), 6),
        "pos_prob_mean": round(float(pos_probs.mean()), 6) if len(pos_probs) > 0 else None,
        "neg_prob_mean": round(float(neg_probs.mean()), 6) if len(neg_probs) > 0 else None,
        "pos_prob_median": round(float(np.median(pos_probs)), 6) if len(pos_probs) > 0 else None,
        "neg_prob_median": round(float(np.median(neg_probs)), 6) if len(neg_probs) > 0 else None,
    }

    # Identify hard cases (prob_pos near 0.5)
    margin = 0.15
    hard_cases = [
        {"image_id": image_ids[i], "prob_pos": round(float(prob_pos[i]), 6), "label": int(labels[i])}
        for i in range(len(prob_pos))
        if abs(prob_pos[i] - 0.5) < margin
    ]
    hard_cases.sort(key=lambda x: abs(x["prob_pos"] - 0.5))

    # False positives and false negatives at threshold 0.5
    preds_05 = (prob_pos >= 0.5).astype(int)
    fp_ids = [image_ids[i] for i in range(len(prob_pos)) if preds_05[i] == 1 and labels[i] == 0]
    fn_ids = [image_ids[i] for i in range(len(prob_pos)) if preds_05[i] == 0 and labels[i] == 1]

    return {
        "distribution": dist,
        "at_threshold_0.5": at_05,
        "best_threshold": best,
        "hard_cases_near_0.5": {
            "count": len(hard_cases),
            "margin": margin,
            "examples": hard_cases[:30],
        },
        "errors": {
            "false_positives": {"count": len(fp_ids), "ids": fp_ids[:50]},
            "false_negatives": {"count": len(fn_ids), "ids": fn_ids[:50]},
        },
    }


def cross_reference_myval_errors(analysis: dict, not_stone: list[dict]) -> dict:
    """Cross-reference myval errors with not-stone audit list."""
    fp_ids = set(analysis["errors"]["false_positives"]["ids"])
    fn_ids = set(analysis["errors"]["false_negatives"]["ids"])

    not_stone_ids = {r["id"] for r in not_stone}
    overlap_fp = fp_ids & not_stone_ids
    overlap_fn = fn_ids & not_stone_ids

    not_stone_details = {}
    for r in not_stone:
        if r["id"] in fp_ids or r["id"] in fn_ids:
            not_stone_details[r["id"]] = {
                "soup_prob": float(r["soup_prob"]),
                "prob_mean": float(r["prob_mean"]),
                "label_votes": f"{r['label_votes_pos']}/{r['label_votes_total']}",
                "error_type": "FP" if r["id"] in fp_ids else "FN",
            }

    return {
        "not_stone_in_myval_errors": {
            "false_positives": sorted(overlap_fp),
            "false_negatives": sorted(overlap_fn),
            "details": not_stone_details,
        }
    }


def analyze_test_submission(not_stone: list[dict]) -> dict:
    """Analyze test submission pattern."""
    if not not_stone:
        return {}

    submission_ids = set()
    with open(SUBMISSION_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            submission_ids.add(row["id"].replace(".jpg", ""))

    not_stone_ids = {r["id"].replace(".jpg", "").replace(".jpg", "") for r in not_stone}
    overlap = not_stone_ids & submission_ids

    return {
        "not_stone_test_ids": sorted(not_stone_ids),
        "still_in_submission": sorted(overlap),
    }


def main():
    print("=" * 60)
    print("Error Diagnostic: soup_reduced_notstone")
    print("=" * 60)

    # Load data
    all_data = load_predictions(PREDICTIONS_CSV)
    if TARGET_RUN not in all_data:
        print(f"ERROR: Run '{TARGET_RUN}' not found in predictions CSV")
        sys.exit(1)

    not_stone = load_not_stone_audit(NOT_STONE_CSV)
    run_data = all_data[TARGET_RUN]

    results = {
        "target_run": TARGET_RUN,
        "datasets": {},
        "summary": {},
    }

    # Analyze each dataset
    print(f"\nAnalyzing {TARGET_RUN}...")
    for dataset_name in sorted(run_data.keys()):
        rows = run_data[dataset_name]
        print(f"\n  Dataset: {dataset_name} ({len(rows)} samples)")
        analysis = analyze_dataset(rows)
        results["datasets"][dataset_name] = analysis

        d = analysis["distribution"]
        a = analysis["at_threshold_0.5"]
        b = analysis["best_threshold"]
        print(f"    Distribution: pos={d['n_pos']} neg={d['n_neg']} ratio={d['neg_pos_ratio']}:1")
        print(f"    @thr=0.5: F1={a['f1']} P={a['precision']} R={a['recall']} "
              f"TP={a['tp']} FP={a['fp']} FN={a['fn']}")
        print(f"    Best thr={b['threshold']:.4f}: F1={b['f1']}")
        print(f"    Hard cases (margin={analysis['hard_cases_near_0.5']['margin']}): "
              f"{analysis['hard_cases_near_0.5']['count']}")
        print(f"    FP={analysis['errors']['false_positives']['count']} "
              f"FN={analysis['errors']['false_negatives']['count']}")

    # Cross-reference myval with not-stone
    if "myval_masked" in run_data:
        myval_analysis = analyze_dataset(run_data["myval_masked"])
        cross_ref = cross_reference_myval_errors(myval_analysis, not_stone)
        results["cross_reference_myval_not_stone"] = cross_ref

        print(f"\n  Cross-reference with not-stone audit:")
        for err_type, ids in cross_ref["not_stone_in_myval_errors"].items():
            if isinstance(ids, dict):
                for k, v in ids.items():
                    if isinstance(v, dict):
                        for item_id, detail in v.items():
                            print(f"    {item_id}: {detail['error_type']} "
                                  f"soup_prob={detail['soup_prob']:.4f} "
                                  f"votes={detail['label_votes']}")

    # Summary ranking
    print(f"\n{'=' * 60}")
    print("SUMMARY: Weakest Areas")
    print(f"{'=' * 60}")

    for dataset_name in sorted(run_data.keys()):
        analysis = results["datasets"][dataset_name]
        a = analysis["at_threshold_0.5"]
        fp_rate = a["fp"] / max(1, a["fp"] + a["tn"])
        fn_rate = a["fn"] / max(1, a["fn"] + a["tp"])
        print(f"\n  {dataset_name}:")
        print(f"    F1@0.5={a['f1']:.4f}  FP_rate={fp_rate:.4f}  FN_rate={fn_rate:.4f}")
        if fp_rate > fn_rate:
            print(f"    >>> DOMINANT ERROR: FALSE POSITIVES (FP={a['fp']})")
        elif fn_rate > fp_rate:
            print(f"    >>> DOMINANT ERROR: FALSE NEGATIVES (FN={a['fn']})")
        else:
            print(f"    Balanced errors")

    # Big picture
    print(f"\n{'=' * 60}")
    print("BIG PICTURE ANALYSIS")
    print(f"{'=' * 60}")
    print("""
The soup model is trained on ~1:1 balanced data but evaluated on ~4:1 (test).
Core tension: the model is calibrated for equal classes but must perform on
an imbalanced test set where false positives are much more costly.

Key questions from this diagnostic:
1. Do false positives dominate on myval? -> Suggests calibration issue
2. Are there systematic mislabeled training samples?
3. Can threshold optimization significantly improve F1?
4. Which test IDs are most uncertain? -> Expansion targets for not-stone list
""")

    # Save results
    out_path = OUTPUT_DIR / "diagnostic_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Generate markdown report
    generate_report(results, not_stone)


def generate_report(results: dict, not_stone: list[dict]):
    lines = []
    lines.append("# Error Diagnostic Report")
    lines.append(f"\nTarget: `{results['target_run']}`")
    lines.append(f"\n## Per-Dataset Metrics\n")
    lines.append("| Dataset | N | Pos | Neg | Ratio | F1@0.5 | Precision | Recall | FP | FN | Best Thr | Best F1 |")
    lines.append("|---------|---|-----|-----|-------|--------|-----------|--------|----|----|----------|---------|")

    for ds_name in sorted(results["datasets"].keys()):
        a = results["datasets"][ds_name]["at_threshold_0.5"]
        d = results["datasets"][ds_name]["distribution"]
        b = results["datasets"][ds_name]["best_threshold"]
        lines.append(
            f"| {ds_name} | {d['n_total']} | {d['n_pos']} | {d['n_neg']} | "
            f"{d['neg_pos_ratio']}:1 | {a['f1']} | {a['precision']} | {a['recall']} | "
            f"{a['fp']} | {a['fn']} | {b['threshold']} | {b['f1']} |"
        )

    lines.append("\n## Probability Distribution\n")
    lines.append("| Dataset | Prob Mean | Prob Median | Pos Mean | Neg Mean |")
    lines.append("|---------|-----------|-------------|----------|----------|")
    for ds_name in sorted(results["datasets"].keys()):
        d = results["datasets"][ds_name]["distribution"]
        lines.append(
            f"| {ds_name} | {d['prob_pos_mean']} | {d['prob_pos_median']} | "
            f"{d['pos_prob_mean']} | {d['neg_prob_mean']} |"
        )

    lines.append("\n## Hard Cases (prob_pos within 0.35-0.65)\n")
    for ds_name in sorted(results["datasets"].keys()):
        hc = results["datasets"][ds_name]["hard_cases_near_0.5"]
        lines.append(f"\n### {ds_name}: {hc['count']} hard cases\n")
        lines.append("| Image ID | Prob Pos | Label |")
        lines.append("|----------|----------|-------|")
        for case in hc["examples"]:
            lines.append(f"| {case['image_id']} | {case['prob_pos']} | {case['label']} |")

    # Error analysis
    lines.append("\n## Error Analysis\n")
    for ds_name in sorted(results["datasets"].keys()):
        err = results["datasets"][ds_name]["errors"]
        a = results["datasets"][ds_name]["at_threshold_0.5"]
        lines.append(f"\n### {ds_name}\n")
        if a["fp"] > a["fn"]:
            lines.append(f"> **FALSE POSITIVES dominate** (FP={a['fp']} vs FN={a['fn']}). "
                         f"Model over-predicts meteorite class.")
        elif a["fn"] > a["fp"]:
            lines.append(f"> **FALSE NEGATIVES dominate** (FN={a['fn']} vs FP={a['fp']}). "
                         f"Model misses meteorites.")
        else:
            lines.append(f"> Balanced errors (FP={a['fp']}, FN={a['fn']}).")
        if err["false_positives"]["count"] > 0:
            lines.append(f"\nFalse Positives ({err['false_positives']['count']}): "
                         f"{', '.join(err['false_positives']['ids'][:20])}")
        if err["false_negatives"]["count"] > 0:
            lines.append(f"\nFalse Negatives ({err['false_negatives']['count']}): "
                         f"{', '.join(err['false_negatives']['ids'][:20])}")

    # Cross-reference
    if "cross_reference_myval_not_stone" in results:
        cr = results["cross_reference_myval_not_stone"]
        lines.append("\n## Cross-Reference: Myval Errors ∩ Not-Stone List\n")
        ns = cr["not_stone_in_myval_errors"]
        for item_id, detail in ns.get("details", {}).items():
            lines.append(
                f"- **{item_id}**: {detail['error_type']} | "
                f"soup_prob={detail['soup_prob']} | "
                f"mean_prob={detail['prob_mean']} | votes={detail['label_votes']}"
            )

    lines.append("\n## Recommendations\n")
    lines.append("""
Based on the diagnostic:

1. **If FP dominate**: Focus on calibration (threshold search) or not-stone list expansion
2. **If FN dominate**: Focus on hard positive mining or model capacity
3. **High-uncertainty samples**: Review manually for mislabeling
4. **Consider threshold tuning**: Even +0.01 F1 from optimal threshold is worthwhile
""")

    lines.append(f"\n_Generated by {Path(__file__).name}_\n")

    report_path = OUTPUT_DIR / "diagnostic_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
