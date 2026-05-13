from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_CSV = Path("./label.csv")
DEFAULT_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a probability CSV by confidence and write an id,label CSV"
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to a CSV such as output-prob.csv or bagged_prob.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output label CSV path",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Keep samples whose pseudo-label confidence is at least this value",
    )
    return parser.parse_args()


def load_probability_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str})
    if "id" not in frame.columns:
        raise ValueError(f"{path} is missing required column: id")

    probability_column = None
    for candidate in ("prob_pos_corrected", "prob_pos"):
        if candidate in frame.columns:
            probability_column = candidate
            break

    if probability_column is None:
        raise ValueError(f"{path} must contain either prob_pos_corrected or prob_pos")

    if frame["id"].duplicated().any():
        duplicate_ids = frame.loc[frame["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"{path} contains duplicate ids; examples: {duplicate_ids}")

    frame = frame.copy()
    frame[probability_column] = frame[probability_column].astype(float)
    if probability_column != "prob_pos_corrected":
        frame["prob_pos_corrected"] = frame[probability_column]
    return frame[["id", "prob_pos_corrected"]].copy()


def build_label_frame(frame: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    if not (0.5 <= confidence_threshold <= 1.0):
        raise ValueError("confidence threshold must be within [0.5, 1.0]")

    prepared = frame.copy()
    prepared["pseudo_confidence"] = prepared["prob_pos_corrected"].where(
        prepared["prob_pos_corrected"] >= 0.5,
        1.0 - prepared["prob_pos_corrected"],
    )
    selected = prepared[prepared["pseudo_confidence"] >= float(confidence_threshold)].copy()
    selected["label"] = selected["prob_pos_corrected"].ge(0.5).astype(int)
    return selected[["id", "label"]].reset_index(drop=True)


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv.resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    frame = load_probability_frame(input_csv)
    label_frame = build_label_frame(frame, confidence_threshold=float(args.confidence_threshold))

    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    label_frame.to_csv(output_csv, index=False)

    selected_count = len(label_frame)
    total_count = len(frame)
    print(
        f"Wrote {selected_count}/{total_count} pseudo labels to {output_csv} "
        f"with confidence_threshold={args.confidence_threshold:.6f}"
    )


if __name__ == "__main__":
    main()