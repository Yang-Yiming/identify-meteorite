import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from modeling import ConvNeXtClassifier, build_transforms
from data import stratified_split, build_image_index, build_mask_image_index
from utils import DEFAULT_BACKBONE, DEFAULT_MEAN, DEFAULT_STD, normalize_image_size, normalize_stats
from tta import TTA_MODE_TO_VIEWS, predict_probabilities_with_tta, resolve_tta_views


POSITIVE_LABEL = 1


class ImageListDataset(Dataset):
    def __init__(self, image_ids, image_index, transform):
        self.image_ids = list(image_ids)
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = self.image_index.get(image_id)
        if path is None:
            raise FileNotFoundError(f"Image not found: {image_id}")
        image = Image.open(path).convert("RGB")
        return self.transform(image), image_id


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model")
    train_args_path = checkpoint_path.parent / "train_args.json"
    metadata_path = checkpoint_path.parent / "metadata.json"
    train_args = json.loads(train_args_path.read_text()) if train_args_path.is_file() else {}
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    backbone_name = train_args.get("backbone", DEFAULT_BACKBONE)
    dropout = train_args.get("dropout", 0.0)
    image_size = normalize_image_size(metadata.get("image_size", 224))
    image_mean = normalize_stats(metadata.get("image_mean"), DEFAULT_MEAN)
    image_std = normalize_stats(metadata.get("image_std"), DEFAULT_STD)
    return state_dict, train_args, metadata, backbone_name, dropout, image_size, image_mean, image_std


def infer(model, loader, device, tta_views=None):
    model.eval()
    all_probs = []
    all_ids = []
    autocast_enabled = device.type == "cuda"
    with torch.no_grad():
        for pixel_values, batch_ids in loader:
            pixel_values = pixel_values.to(device)
            if tta_views:
                prob_pos = predict_probabilities_with_tta(
                    model, pixel_values, POSITIVE_LABEL,
                    views=tta_views, device_type=device.type, autocast_enabled=autocast_enabled,
                ).cpu()
            else:
                with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                    logits = model(pixel_values)
                prob_pos = torch.softmax(logits, dim=1)[:, POSITIVE_LABEL].cpu()
            all_probs.append(prob_pos)
            all_ids.extend(batch_ids)
    return torch.cat(all_probs), all_ids


def main():
    parser = argparse.ArgumentParser(description="Analyze probability distribution on val vs test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, default=Path("preprocess/bbox_crop"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--test-images-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--val-split-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123, help="Training seed (split uses seed+3)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-source", type=str, choices=("train_split", "myval"), default="train_split",
                        help="train_split: 20%% train split; myval: external myval set")
    parser.add_argument("--tta", type=str, default=None, choices=list(TTA_MODE_TO_VIEWS) + [None],
                        help="TTA mode: 4way or 8way (default: no TTA)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting, only print stats")
    args = parser.parse_args()

    device = torch.device(args.device)
    mask_dir = args.mask_dir.resolve()
    test_mask_dir = mask_dir / "test"
    tta_views = resolve_tta_views(args.tta) if args.tta else None
    if tta_views:
        print(f"TTA enabled: mode={args.tta}, views={tta_views}")

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict, train_args, metadata, backbone_name, dropout, image_size, image_mean, image_std = load_checkpoint(args.checkpoint, device)

    _, eval_transform = build_transforms(
        image_size, image_mean, image_std,
        hflip_prob=0.0, rotate_degrees=0.0,
    )

    num_classes = len(metadata.get("idx_to_label", {})) if metadata.get("idx_to_label") else 2
    model = ConvNeXtClassifier(
        backbone_name=backbone_name, backbone_checkpoint=None,
        num_classes=num_classes, dropout=dropout, pretrained_backbone=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded | missing_keys={len(missing)} | unexpected_keys={len(unexpected)}")
    model = model.to(device)

    # --- Build val set ---
    val_source_label = args.val_source
    if args.val_source == "train_split":
        train_mask_dir = mask_dir / "train"
        df = pd.read_csv(args.labels_csv)
        label_to_idx = {lab: i for i, lab in enumerate(sorted(df["label"].unique()))}
        df["label_idx"] = df["label"].map(label_to_idx).astype(int)
        val_mask_index, val_masked_ids, val_skipped_ids = build_mask_image_index(
            train_mask_dir, df["id"].astype(str).tolist()
        )
        df_masked = df[df["id"].astype(str).isin(set(val_masked_ids))].reset_index(drop=True)
        print(f"Val source=train_split | mask_images={len(val_mask_index)} | matched_labels={len(df_masked)} | skipped={len(val_skipped_ids)}")
        split_seed = args.seed + 3
        _, val_df = stratified_split(df_masked, label_column="label_idx", val_ratio=args.val_split_ratio, seed=split_seed)
        val_ids = val_df["id"].astype(str).tolist()
        val_labels = val_df["label_idx"].astype(int).tolist()
        val_source_label = f"train_split (ratio={args.val_split_ratio}, seed={args.seed})"
    else:
        myval_labels_csv = Path("data/myval/labels.csv")
        myval_mask_dir = mask_dir / "myval"
        df = pd.read_csv(myval_labels_csv)
        label_to_idx = {lab: i for i, lab in enumerate(sorted(df["label"].unique()))}
        df["label_idx"] = df["label"].map(label_to_idx).astype(int)
        val_mask_index, val_masked_ids, val_skipped_ids = build_mask_image_index(
            myval_mask_dir, df["id"].astype(str).tolist()
        )
        df = df[df["id"].astype(str).isin(set(val_masked_ids))].reset_index(drop=True)
        print(f"Val source=myval | mask_images={len(val_mask_index)} | matched_labels={len(df)} | skipped={len(val_skipped_ids)}")
        val_ids = df["id"].astype(str).tolist()
        val_labels = df["label_idx"].astype(int).tolist()
        val_source_label = f"myval (n={len(df)})"

    val_labels_tensor = torch.tensor(val_labels)
    print(f"Val set | n={len(val_ids)} | pos={sum(val_labels)}/{len(val_labels)} | ratio={len(val_labels)-sum(val_labels)}/{sum(val_labels)}")

    # --- Infer on val ---
    val_dataset = ImageListDataset(val_ids, val_mask_index, eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    val_probs, _ = infer(model, val_loader, device, tta_views=tta_views)
    val_probs_np = val_probs.numpy()

    # --- Infer on test ---
    if args.sample_submission.is_file():
        sub_ids = pd.read_csv(args.sample_submission)["id"].astype(str).tolist()
    else:
        sub_ids = sorted([p.stem.replace("_mask_000", "") for p in test_mask_dir.iterdir() if p.is_file() and p.suffix == ".png"])
    test_mask_index, test_masked_ids, test_skipped_ids = build_mask_image_index(test_mask_dir, sub_ids)
    test_ids = test_masked_ids
    test_dataset = ImageListDataset(test_ids, test_mask_index, eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    test_probs, _ = infer(model, test_loader, device, tta_views=tta_views)
    test_probs_np = test_probs.numpy()
    print(f"Test inference | ids={len(test_ids)}")

    # --- Stats ---
    val_pos_probs = val_probs_np[val_labels_tensor.numpy() == 1]
    val_neg_probs = val_probs_np[val_labels_tensor.numpy() == 0]
    print("\n=== Probability Distribution Analysis ===")
    print(f"Val  neg: mean={val_neg_probs.mean():.4f}  median={np.median(val_neg_probs):.4f}  std={val_neg_probs.std():.4f}")
    print(f"Val  pos: mean={val_pos_probs.mean():.4f}  median={np.median(val_pos_probs):.4f}  std={val_pos_probs.std():.4f}")
    print(f"Val  all: mean={val_probs_np.mean():.4f}  median={np.median(val_probs_np):.4f}")
    print(f"Test all: mean={test_probs_np.mean():.4f}  median={np.median(test_probs_np):.4f}")

    for pct in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  Test p{pct*100:.0f}={np.percentile(test_probs_np, pct*100):.4f}")

    at_thresh = (test_probs_np >= 0.5).sum()
    print(f"Test @thresh=0.5: {at_thresh}/{len(test_probs_np)} positive ({at_thresh/len(test_probs_np)*100:.1f}%)")

    val_f1_05 = compute_f1(val_probs_np, val_labels_tensor.numpy(), 0.5)
    print(f"Val  @thresh=0.5: F1={val_f1_05:.4f}  pos={(val_probs_np>=0.5).sum()}/{len(val_probs_np)}")

    # Find best threshold on val, apply to test proxy
    best_thr, best_f1 = search_threshold(val_probs_np, val_labels_tensor.numpy())
    test_at_best = (test_probs_np >= best_thr).sum()
    print(f"Val  best thr={best_thr:.4f} F1={best_f1:.4f}")
    print(f"Test @best_thr={best_thr:.4f}: {test_at_best}/{len(test_probs_np)} positive")

    print(f"\nVal  class balance (true):  neg={len(val_neg_probs)}  pos={len(val_pos_probs)}  ratio={len(val_neg_probs)/max(1,len(val_pos_probs)):.2f}:1")
    print(f"Test inferred ratio:        {(len(test_probs_np)-at_thresh)} neg / {at_thresh} pos  ratio={(len(test_probs_np)-at_thresh)/max(1,at_thresh):.2f}:1")

    # --- Plot ---
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            bins = np.linspace(0, 1, 51)
            plt.figure(figsize=(10, 6))
            plt.hist(val_neg_probs, bins=bins, alpha=0.6, label=f"Val neg (n={len(val_neg_probs)})", color="tab:blue", density=True)
            plt.hist(val_pos_probs, bins=bins, alpha=0.6, label=f"Val pos (n={len(val_pos_probs)})", color="tab:orange", density=True)
            plt.hist(test_probs_np, bins=bins, alpha=0.4, label=f"Test (n={len(test_probs_np)})", color="tab:red", density=True, histtype="step", linewidth=2)
            plt.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="thresh=0.5")
            plt.xlabel("Positive probability")
            plt.ylabel("Density")
            plt.title(f"Probability distribution: val vs test\n{args.checkpoint.parent.name}  [{args.val_source}]")
            plt.legend()
            out_name = f"prob_dist_{args.val_source}.png"
            out_path = Path(__file__).resolve().parent / "outputs" / args.checkpoint.parent.name / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nPlot saved to {out_path}")
        except ImportError:
            print("\nmatplotlib not available, skipping plot")


def compute_f1(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def search_threshold(probs, labels):
    candidates = sorted(set([0.0, 0.5, 1.0] + probs.tolist()))
    best_t, best_f = 0.5, 0.0
    for t in candidates:
        f = compute_f1(probs, labels, t)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


if __name__ == "__main__":
    main()
