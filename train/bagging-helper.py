from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_OUTPUT_CSV = Path("./bagged_submission.csv")
DEFAULT_OUTPUT_PROB_CSV = Path("./bagged_prob.csv")
DEFAULT_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average prob_pos_corrected CSVs in a folder and write a Kaggle submission CSV"
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing per-run probability CSVs")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--output-prob-csv",
        type=Path,
        default=DEFAULT_OUTPUT_PROB_CSV,
        help="CSV with averaged prob_pos_corrected and label",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold used to convert averaged probabilities into labels",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for CSV files recursively under input-dir",
    )
    return parser.parse_args()


def discover_csv_files(input_dir: Path, recursive: bool, output_paths: List[Path]) -> List[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = []
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file() and path.resolve() not in output_paths:
            csv_files.append(path)
    return csv_files


def load_probability_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str})
    required_columns = {"id", "prob_pos_corrected"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")
    if frame["id"].duplicated().any():
        duplicate_ids = frame.loc[frame["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"{path} contains duplicate ids; examples: {duplicate_ids}")
    return frame[["id", "prob_pos_corrected"]].copy()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_csv = args.output_csv.resolve()
    output_paths = [output_csv]
    if args.output_prob_csv is not None:
        output_paths.append(args.output_prob_csv.resolve())

    csv_files = discover_csv_files(input_dir, recursive=args.recursive, output_paths=output_paths)
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {input_dir}")

    frames = [load_probability_frame(path) for path in csv_files]
    base_ids = frames[0]["id"].tolist()
    base_id_set = set(base_ids)

    aligned_probabilities = []
    for path, frame in zip(csv_files, frames):
        frame_id_set = set(frame["id"].tolist())
        if frame_id_set != base_id_set:
            missing_ids = sorted(base_id_set - frame_id_set)
            extra_ids = sorted(frame_id_set - base_id_set)
            raise ValueError(
                f"{path} does not match the first CSV id set | missing={missing_ids[:5]} | extra={extra_ids[:5]}"
            )
        aligned = frame.set_index("id").reindex(base_ids)
        if aligned["prob_pos_corrected"].isna().any():
            raise ValueError(f"{path} has missing prob_pos_corrected values after alignment")
        aligned_probabilities.append(aligned["prob_pos_corrected"].astype(float))

    prob_matrix = pd.concat(aligned_probabilities, axis=1)
    averaged_prob = prob_matrix.mean(axis=1)
    labels = (averaged_prob >= float(args.threshold)).astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": base_ids, "label": labels.to_numpy()})
    submission.to_csv(output_csv, index=False)

    output_prob_csv = args.output_prob_csv.resolve()
    output_prob_csv.parent.mkdir(parents=True, exist_ok=True)
    prob_output = pd.DataFrame(
        {
            "id": base_ids,
            "prob_pos_corrected": averaged_prob.to_numpy(),
            "label": labels.to_numpy(),
        }
    )
    prob_output.to_csv(output_prob_csv, index=False)

    positive_predictions = int(labels.sum())
    print(
        f"Bagged {len(csv_files)} CSVs from {input_dir} | "
        f"rows={len(base_ids)} | threshold={args.threshold:.6f} | positive_predictions={positive_predictions}"
    )
    print(f"Wrote submission CSV to {output_csv}")
    print(f"Wrote averaged probability CSV to {output_prob_csv}")


if __name__ == "__main__":
    main()