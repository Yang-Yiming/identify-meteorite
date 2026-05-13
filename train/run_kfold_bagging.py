from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import StratifiedKFold


DEFAULT_DATA_DIR = Path("../data")
DEFAULT_LABELS_CSV = DEFAULT_DATA_DIR / "train_labels.csv"
DEFAULT_TEST_IMAGES_DIR = DEFAULT_DATA_DIR / "test_images"
DEFAULT_SAMPLE_SUBMISSION = DEFAULT_DATA_DIR / "sample_submission.csv"
DEFAULT_OUTPUT_ROOT = Path("./outputs/kfold_bagging")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate K-fold training, per-fold inference, and final bagging by reusing "
            "train_finetune.py / infer_submission.py / bagging-helper.py"
        )
    )
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--test-images-dir", type=Path, default=DEFAULT_TEST_IMAGES_DIR)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)

    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--folds",
        type=str,
        default="",
        help="Comma-separated fold ids to run, e.g. 1,3,5. Empty means all folds.",
    )

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-script", type=Path, default=Path("./train_finetune.py"))
    parser.add_argument("--infer-script", type=Path, default=Path("./infer_submission.py"))
    parser.add_argument("--bagging-script", type=Path, default=Path("./bagging-helper.py"))
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument(
        "--train-extra-args",
        type=str,
        default="",
        help="Extra CLI args appended to each train_finetune.py call.",
    )
    parser.add_argument(
        "--infer-extra-args",
        type=str,
        default="",
        help="Extra CLI args appended to each infer_submission.py call.",
    )
    parser.add_argument("--bagging-threshold", type=float, default=0.5)

    parser.add_argument("--skip-existing-train", action="store_true")
    parser.add_argument("--skip-existing-infer", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-infer", action="store_true")
    parser.add_argument("--no-bagging", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_selected_folds(raw_folds: str, n_splits: int) -> List[int]:
    if not raw_folds.strip():
        return list(range(1, n_splits + 1))

    selected = []
    for part in raw_folds.split(","):
        part = part.strip()
        if not part:
            continue
        fold_id = int(part)
        if fold_id < 1 or fold_id > n_splits:
            raise ValueError(f"Fold id out of range: {fold_id}. Valid range: 1..{n_splits}")
        selected.append(fold_id)

    unique_selected = sorted(set(selected))
    if not unique_selected:
        raise ValueError("No valid fold ids parsed from --folds")
    return unique_selected


def run_command(command: List[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(token) for token in command)
    print(f"[cmd] {printable}")
    if dry_run:
        return
    subprocess.run(command, check=True)


def ensure_required_columns(frame: pd.DataFrame, id_column: str, label_column: str, labels_csv: Path) -> None:
    required = {id_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{labels_csv} is missing required columns: {sorted(missing)}")


def write_fold_splits(
    labels_csv: Path,
    id_column: str,
    label_column: str,
    n_splits: int,
    seed: int,
    splits_dir: Path,
) -> List[dict]:
    frame = pd.read_csv(labels_csv)
    ensure_required_columns(frame, id_column, label_column, labels_csv)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []

    y = frame[label_column].astype(str)
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(frame[id_column], y), start=1):
        fold_train = frame.iloc[train_idx].copy().reset_index(drop=True)
        fold_val = frame.iloc[val_idx].copy().reset_index(drop=True)

        train_csv = splits_dir / f"fold{fold_id}_train.csv"
        val_csv = splits_dir / f"fold{fold_id}_val.csv"
        fold_train.to_csv(train_csv, index=False)
        fold_val.to_csv(val_csv, index=False)

        split_info = {
            "fold": fold_id,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "train_size": int(len(fold_train)),
            "val_size": int(len(fold_val)),
            "train_label_dist": fold_train[label_column].value_counts().to_dict(),
            "val_label_dist": fold_val[label_column].value_counts().to_dict(),
        }
        splits.append(split_info)

    return splits


def main() -> None:
    args = parse_args()

    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")

    labels_csv = args.labels_csv.resolve()
    test_images_dir = args.test_images_dir.resolve()
    sample_submission = args.sample_submission.resolve()

    script_dir = Path(__file__).resolve().parent
    output_root = args.output_root.resolve()
    splits_dir = output_root / "splits"
    fold_outputs_dir = output_root / "folds"
    prob_dir = output_root / "prob"

    output_root.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    fold_outputs_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

    split_records = write_fold_splits(
        labels_csv=labels_csv,
        id_column=args.id_column,
        label_column=args.label_column,
        n_splits=args.n_splits,
        seed=args.seed,
        splits_dir=splits_dir,
    )
    split_meta_path = output_root / "split_summary.json"
    split_meta_path.write_text(json.dumps(split_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved split summary: {split_meta_path}")

    selected_folds = parse_selected_folds(args.folds, args.n_splits)
    print(f"Selected folds: {selected_folds}")

    train_script = (script_dir / args.train_script).resolve() if not args.train_script.is_absolute() else args.train_script
    infer_script = (script_dir / args.infer_script).resolve() if not args.infer_script.is_absolute() else args.infer_script
    bagging_script = (
        (script_dir / args.bagging_script).resolve() if not args.bagging_script.is_absolute() else args.bagging_script
    )

    train_extra = shlex.split(args.train_extra_args)
    infer_extra = shlex.split(args.infer_extra_args)

    for fold_id in selected_folds:
        fold_name = f"fold{fold_id}"
        fold_dir = fold_outputs_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_csv = splits_dir / f"fold{fold_id}_train.csv"
        val_csv = splits_dir / f"fold{fold_id}_val.csv"
        best_ckpt = fold_dir / "best.pt"
        fold_prob_csv = prob_dir / f"{fold_name}_prob.csv"
        fold_submission_csv = fold_dir / "submission.csv"

        if not args.no_train:
            should_skip_train = args.skip_existing_train and best_ckpt.is_file()
            if should_skip_train:
                print(f"Skip train {fold_name}: existing checkpoint {best_ckpt}")
            else:
                train_cmd = [
                    args.python,
                    str(train_script),
                    "--labels-csv",
                    str(train_csv),
                    "--val-labels-csv",
                    str(val_csv),
                    "--val-mask-split",
                    "train",
                    "--output-dir",
                    str(fold_dir),
                ]
                train_cmd.extend(train_extra)
                run_command(train_cmd, dry_run=args.dry_run)

        if not args.no_infer:
            if not best_ckpt.is_file() and not args.dry_run:
                raise FileNotFoundError(f"Best checkpoint not found for {fold_name}: {best_ckpt}")

            should_skip_infer = args.skip_existing_infer and fold_prob_csv.is_file()
            if should_skip_infer:
                print(f"Skip infer {fold_name}: existing probability CSV {fold_prob_csv}")
            else:
                infer_cmd = [
                    args.python,
                    str(infer_script),
                    "--checkpoint",
                    str(best_ckpt),
                    "--test-images-dir",
                    str(test_images_dir),
                    "--sample-submission",
                    str(sample_submission),
                    "--output-csv",
                    str(fold_submission_csv),
                    "--output-prob-csv",
                    str(fold_prob_csv),
                ]
                infer_cmd.extend(infer_extra)
                run_command(infer_cmd, dry_run=args.dry_run)

    if not args.no_bagging:
        bagged_submission = output_root / "bagged_submission.csv"
        bagged_prob = output_root / "bagged_prob.csv"
        bag_cmd = [
            args.python,
            str(bagging_script),
            "--input-dir",
            str(prob_dir),
            "--output-csv",
            str(bagged_submission),
            "--output-prob-csv",
            str(bagged_prob),
            "--threshold",
            str(float(args.bagging_threshold)),
        ]
        run_command(bag_cmd, dry_run=args.dry_run)

    print("Done. K-fold orchestration completed.")


if __name__ == "__main__":
    main()
