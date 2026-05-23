#!/usr/bin/env python3
"""Create contact sheets for manual review of verifier FP candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_font(size: int = 16):
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_cell(
    path: Path,
    title: str,
    subtitle: str,
    size: int,
    label_h: int,
    border: tuple[int, int, int],
    font,
    small_font,
) -> Image.Image:
    cell = Image.new("RGB", (size, size + label_h), "white")
    img = fit_image(path, size)
    draw = ImageDraw.Draw(cell)
    cell.paste(img, (0, 0))
    draw.rectangle([0, 0, size - 1, size - 1], outline=border, width=4)
    draw.text((6, size + 4), title[:36], fill=(0, 0, 0), font=font)
    draw.text((6, size + 24), subtitle[:44], fill=(50, 50, 50), font=small_font)
    return cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("analysis/verifier_features/current_positive_verifier_features.csv"))
    parser.add_argument("--neighbors", type=Path, default=Path("analysis/test_fp_risk_audit_dino_nomtest/test_neighbors_topk.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/verifier_contact_sheets"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--neighbors-k", type=int, default=6)
    parser.add_argument("--cell-size", type=int, default=150)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features).head(args.top_n).copy()
    neighbors = pd.read_csv(args.neighbors)
    font = load_font(14)
    small_font = load_font(11)

    label_h = 44
    gap = 10
    cols = args.neighbors_k + 1
    rows = len(features)
    width = cols * args.cell_size + (cols + 1) * gap
    height = rows * (args.cell_size + label_h) + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    index_rows = []

    for row_idx, row in enumerate(features.itertuples(index=False)):
        y = gap + row_idx * (args.cell_size + label_h + gap)
        query_path = Path(row.path) if hasattr(row, "path") and isinstance(row.path, str) else None
        # current_positive_verifier_features does not keep path; recover from fp audit if needed.
        if query_path is None or not query_path.is_file():
            num = int(Path(str(row.id)).stem)
            query_path = Path(f"preprocess/bbox_crop/test/{num:06d}_mask_000.png")
        title = f"Q {row.id} rank={int(row.verifier_rank)}"
        subtitle = f"score={row.verifier_fp_score:.3f} weak={getattr(row, 'weak_label', '')}"
        cell = draw_cell(query_path, title, subtitle, args.cell_size, label_h, (30, 80, 220), font, small_font)
        sheet.paste(cell, (gap, y))

        q_neighbors = neighbors[neighbors["query_id"] == row.id].head(args.neighbors_k).copy()
        for col_idx, nrow in enumerate(q_neighbors.itertuples(index=False), start=1):
            x = gap + col_idx * (args.cell_size + gap)
            ref_label = int(nrow.ref_label)
            border = (35, 150, 65) if ref_label == 1 else (200, 60, 45)
            ntitle = f"#{int(nrow.rank)} {nrow.ref_source} y={ref_label}"
            nsubtitle = f"sim={nrow.cosine_sim:.3f} {Path(str(nrow.ref_image_id)).stem}"
            try:
                cell = draw_cell(Path(nrow.ref_path), ntitle, nsubtitle, args.cell_size, label_h, border, font, small_font)
            except FileNotFoundError:
                cell = Image.new("RGB", (args.cell_size, args.cell_size + label_h), "white")
                ImageDraw.Draw(cell).text((6, 6), "missing", fill=(0, 0, 0), font=font)
            sheet.paste(cell, (x, y))
        index_rows.append(
            {
                "id": row.id,
                "verifier_rank": int(row.verifier_rank),
                "verifier_fp_score": float(row.verifier_fp_score),
                "weak_label": getattr(row, "weak_label", ""),
                "soup_prob_pos": float(row.soup_prob_pos),
                "top_neighbor_labels": ",".join(q_neighbors["ref_label"].astype(str).tolist()),
                "top_neighbor_sources": ",".join(q_neighbors["ref_source"].astype(str).tolist()),
            }
        )

    out_image = args.out_dir / f"verifier_top{args.top_n}_neighbors.jpg"
    sheet.save(out_image, quality=92)
    pd.DataFrame(index_rows).to_csv(args.out_dir / f"verifier_top{args.top_n}_index.csv", index=False)
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Verifier Contact Sheets",
                "",
                "Blue border: query test image. Red border: negative neighbor. Green border: positive neighbor.",
                "Rows are sorted by verifier_fp_score from current_positive_verifier_features.csv.",
                "",
                f"- Sheet: {out_image.name}",
                f"- Index: verifier_top{args.top_n}_index.csv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_image}")


if __name__ == "__main__":
    main()
