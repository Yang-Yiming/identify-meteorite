#!/usr/bin/env python3
"""Train frozen-feature probes with MLP classifiers and threshold calibration.

Extends train_frozen_feature_probe.py with:
- Shallow MLP classifier (sklearn MLPClassifier)
- Concatenated features from multiple backbones
- Threshold calibration toward target positive count
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.build_testlike_val import build_manifest, extract_dino_timm_features  # noqa: E402


def f1_at(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    return float(f1_score(labels.astype(int), (probs >= threshold).astype(int), zero_division=0))


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in sorted(set(np.concatenate([probs, np.array([0.0, 0.5, 1.0])]).tolist())):
        score = f1_at(probs, labels, float(t))
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def normalize_id(value: str) -> str:
    return f"{int(Path(str(value)).stem):06d}.jpg"


def dataset_from_manifest(manifest: pd.DataFrame, source: str) -> pd.DataFrame:
    return manifest[(manifest["source"] == source) & (manifest["has_image"]) & manifest["label"].notna()].copy()


def eval_dataset(model, x: np.ndarray, y: np.ndarray, name: str) -> dict[str, object]:
    probs = model.predict_proba(x)[:, 1]
    best_t, best_f1 = best_threshold(probs, y)
    return {
        "dataset": name,
        "f1_at_0_5": f1_at(probs, y, 0.5),
        "best_f1": best_f1,
        "best_threshold": best_t,
        "n": int(len(y)),
        "pos_true": int(y.sum()),
        "pos_pred_at_0_5": int((probs >= 0.5).sum()),
        "prob_mean": float(probs.mean()),
        "prob_median": float(np.median(probs)),
    }


def calibrate_threshold(
    probs: np.ndarray,
    target_positives: int,
    min_threshold: float = 1e-4,
    max_threshold: float = 0.95,
    steps: int = 500,
) -> tuple[float, int]:
    best_t = 0.5
    best_diff = int(1e9)
    best_pos = 0
    for t in np.linspace(min_threshold, max_threshold, steps):
        pos = int((probs >= t).sum())
        diff = abs(pos - target_positives)
        if diff < best_diff:
            best_t, best_diff, best_pos = float(t), diff, pos
    return best_t, best_pos


def extract_multi_features(
    paths: list[str],
    model_names: list[str],
    batch_size: int,
    num_workers: int,
    device: str,
    out_dir: Path,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for model_name in model_names:
        safe_name = model_name.replace("/", "_").replace(".", "_")
        cache_path = out_dir / f"{safe_name}_features.npz"
        if cache_path.is_file():
            feats = np.load(cache_path)["features"]
        else:
            feats = extract_dino_timm_features(
                paths=paths,
                model_name=model_name,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            )
            np.savez_compressed(cache_path, features=feats)
        parts.append(feats)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-names", type=str, default="vit_base_patch16_siglip_224")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/frozen_mlp_probe"))
    parser.add_argument("--classifier", choices=("logistic", "mlp"), default="mlp")
    parser.add_argument("--hidden-sizes", type=str, default="256")
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--c-values", type=str, default="0.01,0.03,0.1,0.3,1.0,3.0,10.0")
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--target-positives", type=str, default="115,120,125,130")
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
    parser.add_argument("--v4-cluster", type=Path, default=Path("analysis/testlike_dino_train_v4/test_like_val_cluster.csv"))
    parser.add_argument("--v4-top", type=Path, default=Path("analysis/testlike_dino_train_v4/test_like_val_top.csv"))
    parser.add_argument(
        "--baseline-submission",
        type=Path,
        default=Path("train/outputs/myval_v13_hi288_seed42_soup/submission_processed_best_notstone.csv"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_names = [m.strip() for m in args.model_names.split(",") if m.strip()]
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(",") if x.strip())

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
    train_df = dataset_from_manifest(manifest, "train").reset_index(drop=True)
    myval_df = dataset_from_manifest(manifest, "myval").reset_index(drop=True)
    test_df = manifest[(manifest["source"] == "test") & manifest["has_image"]].copy().reset_index(drop=True)
    cluster_df = pd.read_csv(args.v4_cluster).copy().reset_index(drop=True)
    top_df = pd.read_csv(args.v4_top).copy().reset_index(drop=True)

    all_df = pd.concat(
        [
            train_df.assign(split="train"),
            myval_df.assign(split="myval"),
            test_df.assign(split="test"),
            cluster_df.assign(split="v4_cluster"),
            top_df.assign(split="v4_top"),
        ],
        ignore_index=True,
    )
    paths = all_df["path"].astype(str).tolist()

    features = extract_multi_features(
        paths=paths,
        model_names=model_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        out_dir=args.out_dir,
    )

    offsets = {}
    start = 0
    for split, df in [
        ("train", train_df),
        ("myval", myval_df),
        ("test", test_df),
        ("v4_cluster", cluster_df),
        ("v4_top", top_df),
    ]:
        offsets[split] = slice(start, start + len(df))
        start += len(df)

    x_train = features[offsets["train"]]
    y_train = train_df["label"].astype(int).to_numpy()
    datasets = {
        "myval": (features[offsets["myval"]], myval_df["label"].astype(int).to_numpy()),
        "v4_cluster": (features[offsets["v4_cluster"]], cluster_df["label"].astype(int).to_numpy()),
        "v4_top": (features[offsets["v4_top"]], top_df["label"].astype(int).to_numpy()),
    }

    baseline = pd.read_csv(args.baseline_submission)
    baseline["id"] = baseline["id"].astype(str)
    baseline_labels = dict(zip(baseline["id"], baseline["label"].astype(int)))

    model_tag = args.model_names.replace(",", "+")
    safe_tag = model_tag.replace("/", "_").replace(".", "_")
    class_weight = "balanced" if args.class_weight == "balanced" else None

    rows = []
    submissions = []

    if args.classifier == "logistic":
        c_values = [float(x) for x in args.c_values.split(",") if x.strip()]
        configs = [("logistic", c, None) for c in c_values]
    else:
        configs = [("mlp", None, hidden_sizes)]

    for config_type, c_val, hidden in configs:
        if config_type == "logistic":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_val,
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            )
        else:
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=min(128, len(x_train)),
                    learning_rate_init=args.mlp_lr,
                    max_iter=args.mlp_epochs,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                ),
            )

        clf.fit(x_train, y_train)
        evals = [eval_dataset(clf, x, y, name) for name, (x, y) in datasets.items()]

        display_model = model_names[0] if len(model_names) == 1 else safe_tag
        row = {
            "model_names": model_tag,
            "classifier": config_type,
            "C": c_val if c_val else "",
            "hidden_sizes": str(hidden) if hidden else "",
            "class_weight": args.class_weight,
        }
        for item in evals:
            prefix = item.pop("dataset")
            for k, v in item.items():
                row[f"{prefix}_{k}"] = v

        test_probs = clf.predict_proba(features[offsets["test"]])[:, 1]
        test_ids = [normalize_id(x) for x in test_df["image_id"].astype(str).tolist()]
        prob_by_id = {tid: float(prob) for tid, prob in zip(test_ids, test_probs)}
        full_ids = baseline["id"].astype(str).tolist()
        base_labels = np.array([int(baseline_labels[tid]) for tid in full_ids], dtype=int)
        missing_test_ids = [tid for tid in full_ids if tid not in prob_by_id]

        target_positives_list = [int(x) for x in args.target_positives.split(",") if x.strip()]
        for target_pos in target_positives_list:
            calib_t, calib_pos = calibrate_threshold(test_probs, target_pos)
            preds = (test_probs >= calib_t).astype(int)
            pred_by_id = {tid: int(pred) for tid, pred in zip(test_ids, preds)}
            full_labels = np.array(
                [pred_by_id.get(tid, int(baseline_labels[tid])) for tid in full_ids], dtype=int
            )

            calib_row = dict(row)
            calib_row["calib_target_positives"] = int(target_pos)
            calib_row["calib_threshold"] = float(calib_t)
            calib_row["test_available_n"] = int(len(test_ids))
            calib_row["test_missing_filled_from_baseline"] = int(len(missing_test_ids))
            calib_row["test_pos_pred_at_0_5"] = int(full_labels.sum())
            calib_row["test_prob_mean"] = float(test_probs.mean())
            calib_row["diff_vs_baseline"] = int((full_labels != base_labels).sum())
            calib_row["baseline_pos_to_neg"] = int(((base_labels == 1) & (full_labels == 0)).sum())
            calib_row["baseline_neg_to_pos"] = int(((base_labels == 0) & (full_labels == 1)).sum())
            rows.append(calib_row)

            suffix = f"_{config_type}_{safe_tag}_pos{target_pos}"
            sub = pd.DataFrame({"id": full_ids, "label": full_labels})
            sub_path = args.out_dir / f"submission{suffix}.csv"
            sub.to_csv(sub_path, index=False)

            sub_probs = pd.DataFrame({
                "id": full_ids,
                "label": full_labels,
                "prob_pos": [prob_by_id.get(tid, np.nan) for tid in full_ids],
                "calib_threshold": calib_t,
                "filled_from_baseline": [tid in missing_test_ids for tid in full_ids],
            })
            sub_probs.to_csv(args.out_dir / f"submission_probs{suffix}.csv", index=False)
            submissions.append(str(sub_path))

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "probe_results.csv", index=False)

    report_cols = [
        "model_names",
        "classifier",
        "C",
        "hidden_sizes",
        "class_weight",
        "v4_cluster_f1_at_0_5",
        "v4_top_f1_at_0_5",
        "myval_f1_at_0_5",
        "calib_target_positives",
        "calib_threshold",
        "test_pos_pred_at_0_5",
        "diff_vs_baseline",
        "baseline_pos_to_neg",
        "baseline_neg_to_pos",
    ]
    available_cols = [c for c in report_cols if c in results.columns]

    report_lines = [
        "# Frozen Feature MLP Probe",
        "",
        f"Backbones: {model_tag}",
        f"Classifier: {args.classifier} ({args.hidden_sizes})",
        f"Class weight: {args.class_weight}",
        f"MLP lr: {args.mlp_lr}, epochs: {args.mlp_epochs}",
        "",
        "## Results",
        "",
        results[available_cols].to_csv(index=False),
    ]
    (args.out_dir / "probe_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    meta = {
        "model_names": model_names,
        "model_tag": model_tag,
        "classifier": args.classifier,
        "hidden_sizes": list(hidden_sizes),
        "class_weight": args.class_weight,
        "mlp_lr": args.mlp_lr,
        "mlp_epochs": args.mlp_epochs,
        "train_n": int(len(train_df)),
        "myval_n": int(len(myval_df)),
        "test_n": int(len(test_df)),
        "v4_cluster_n": int(len(cluster_df)),
        "v4_top_n": int(len(top_df)),
        "submissions": submissions,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    print(results[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
