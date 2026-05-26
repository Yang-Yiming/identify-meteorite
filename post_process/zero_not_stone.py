#!/usr/bin/env python3
"""Post-process a submission CSV: force predictions to 0 for IDs listed in not-stone.txt."""

import argparse
import csv
from pathlib import Path


def load_not_stone_ids(txt_path: Path) -> set:
    ids = set()
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def pad_number(num_str: str) -> str:
    return num_str.zfill(6)


def extract_number(image_id: str) -> str:
    stem = Path(image_id).stem
    digits = "".join(c for c in stem if c.isdigit())
    return str(int(digits)) if digits else ""


def main():
    parser = argparse.ArgumentParser(description="Zero out predictions for not-stone images")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input submission CSV")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output submission CSV")
    parser.add_argument("--not-stone-txt", type=Path, default=Path(__file__).parent / "force_zero_lists/not-stone.txt")
    args = parser.parse_args()

    not_stone_set = load_not_stone_ids(args.not_stone_txt.resolve())
    print(f"Loaded {len(not_stone_set)} not-stone IDs from {args.not_stone_txt}")

    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    zeroed = 0
    total = 0
    with open(input_csv) as infile, open(output_csv, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["id", "label"])
        for row in reader:
            image_id = row["id"]
            number = extract_number(image_id)
            label = 0 if number in not_stone_set else int(row["label"])
            if label == 0 and number in not_stone_set:
                zeroed += 1
            writer.writerow([image_id, label])
            total += 1

    print(f"Processed {total} rows | zeroed {zeroed} | output: {output_csv}")


if __name__ == "__main__":
    main()
