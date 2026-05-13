#!/usr/bin/env python3
"""
Cluster analysis with ConvNeXt-Tiny embeddings.
1. Full images -> HDBSCAN + UMAP
2. Stone-masked (SAM masks applied, mean embedding across all masks) -> HDBSCAN + UMAP
3. Background-masked (inverse SAM masks) -> HDBSCAN + UMAP
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import timm
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import hdbscan
import umap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────
DATA_DIR = Path("/root/project/data")
TRAIN_DIR = DATA_DIR / "train_images" / "train_images"
TEST_DIR = DATA_DIR / "test_images" / "test_images"
TRAIN_MASK_DIR = Path("/root/project/preprocess/sam/output/train")
TEST_MASK_DIR = Path("/root/project/preprocess/sam/output/test")
OUTPUT_DIR = Path("/root/project/preprocess/data-analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_SIZE = 224
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# HDBSCAN params
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5

# UMAP params
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


# ── Helpers ────────────────────────────────────────────

def load_image_paths(root: Path):
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    )


def load_model():
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
    model = model.to(DEVICE)
    model.eval()
    return model


def extract_embeddings(model, dataset, desc="Extracting"):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    all_feats = []
    all_ids = []
    with torch.no_grad():
        for imgs, ids in tqdm(loader, desc=desc):
            feats = model(imgs.to(DEVICE)).cpu().numpy()
            all_feats.append(feats)
            all_ids.extend(ids)
    return np.concatenate(all_feats, axis=0), all_ids


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
        return TRANSFORM(img), p.stem


def apply_mask_to_image(img_path, mask_path, invert=False):
    with Image.open(img_path) as img:
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32)
    with Image.open(mask_path) as m:
        m = m.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        mask_arr = np.array(m, dtype=np.float32) / 255.0
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[:, :, None]
    if invert:
        mask_arr = 1.0 - mask_arr
    masked = np.clip(img_arr * mask_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(masked)


class MaskedImageDataset(Dataset):
    def __init__(self, img_path_map, mask_index, invert=False):
        self.samples = []
        for stem, mask_paths in mask_index.items():
            if stem not in img_path_map:
                continue
            for mp in mask_paths:
                self.samples.append((img_path_map[stem], mp, stem))
        self.invert = invert

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, stem = self.samples[idx]
        masked = apply_mask_to_image(img_path, mask_path, invert=self.invert)
        return TRANSFORM(masked), stem


def aggregate_by_id(embeddings, ids):
    """Mean-pool embeddings that share the same image id."""
    groups = defaultdict(list)
    for emb, id_ in zip(embeddings, ids):
        groups[id_].append(emb)
    agg_ids = []
    agg_embs = []
    for id_ in sorted(groups.keys()):
        agg_ids.append(id_)
        agg_embs.append(np.mean(groups[id_], axis=0))
    return np.array(agg_embs), agg_ids


def build_mask_index(mask_dir, id_set):
    idx = defaultdict(list)
    for stem in id_set:
        for mf in mask_dir.glob(f"{stem}_mask_*.png"):
            idx[stem].append(mf)
        idx[stem].sort()
    return idx


def run_hdbscan(embeddings):
    scaled = StandardScaler().fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    cluster_sizes = {}
    for c in range(n_clusters):
        cluster_sizes[int(c)] = int((labels == c).sum())
    return labels, {
        "n_samples": len(labels),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / len(labels)) if len(labels) > 0 else 0.0,
        "cluster_sizes": cluster_sizes,
    }


# ── UMAP plotting ──────────────────────────────────────

def _umap_reduce(embeddings):
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    )
    return reducer.fit_transform(embeddings)


def plot_umap_train_vs_test(train_emb, test_emb, title, save_path):
    all_emb = np.concatenate([train_emb, test_emb], axis=0)
    reduced = _umap_reduce(all_emb)

    fig, ax = plt.subplots(figsize=(12, 10))
    n_train = len(train_emb)
    ax.scatter(reduced[:n_train, 0], reduced[:n_train, 1],
               c="steelblue", label=f"Train (n={n_train})", alpha=0.4, s=5)
    ax.scatter(reduced[n_train:, 0], reduced[n_train:, 1],
               c="crimson", label=f"Test (n={len(test_emb)})", alpha=0.5, s=12, marker="^", edgecolors="black", linewidths=0.3)
    ax.set_title(title, fontsize=14)
    ax.legend(markerscale=5, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_clusters(train_emb, test_emb, train_labels, title, save_path):
    all_emb = np.concatenate([train_emb, test_emb], axis=0)
    reduced = _umap_reduce(all_emb)

    fig, ax = plt.subplots(figsize=(14, 11))
    n_train = len(train_emb)

    clusters = sorted(set(train_labels) - {-1})
    n_clusters = len(clusters)
    cmap = plt.cm.tab20

    for i, c in enumerate(clusters):
        mask = train_labels == c
        color = cmap(i % 20)
        ax.scatter(reduced[:n_train][mask, 0], reduced[:n_train][mask, 1],
                   c=[color], alpha=0.4, s=5, label=f"C{c}")
    noise_mask = train_labels == -1
    if noise_mask.any():
        ax.scatter(reduced[:n_train][noise_mask, 0], reduced[:n_train][noise_mask, 1],
                   c="grey", alpha=0.15, s=3, label="Train noise")
    ax.scatter(reduced[n_train:, 0], reduced[n_train:, 1],
               c="red", alpha=0.5, s=14, marker="^", edgecolors="black", linewidths=0.3, label="Test")

    ax.set_title(title, fontsize=14)
    ncol = max(1, min(10, n_clusters // 3 + 2))
    ax.legend(markerscale=4, fontsize=7, ncol=ncol, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    model = load_model()
    print("ConvNeXt-Tiny loaded.")

    train_paths = load_image_paths(TRAIN_DIR)
    test_paths = load_image_paths(TEST_DIR)
    print(f"Train images: {len(train_paths)}")
    print(f"Test images:  {len(test_paths)}")

    train_ids = [p.stem for p in train_paths]
    test_ids = [p.stem for p in test_paths]

    all_stats = {}

    # ═══════════════════════════════════════════════════
    # PART 1: Full Images
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 1: Full Images")
    print("=" * 60)

    train_emb, _ = extract_embeddings(model, ImageDataset(train_paths), desc="Train full")
    test_emb, _ = extract_embeddings(model, ImageDataset(test_paths), desc="Test full")
    np.save(OUTPUT_DIR / "embeddings_train_full.npy", train_emb)
    np.save(OUTPUT_DIR / "embeddings_test_full.npy", test_emb)
    print(f"Train embeddings: {train_emb.shape}  Test embeddings: {test_emb.shape}")

    train_labels, train_hdb = run_hdbscan(train_emb)
    test_labels, test_hdb = run_hdbscan(test_emb)
    all_stats["full"] = {"train": train_hdb, "test": test_hdb}
    print(f"Train HDBSCAN: {train_hdb['n_clusters']} clusters, {train_hdb['n_noise']} noise ({train_hdb['noise_ratio']:.1%})")
    print(f"Test  HDBSCAN: {test_hdb['n_clusters']} clusters, {test_hdb['n_noise']} noise ({test_hdb['noise_ratio']:.1%})")

    print("Plotting UMAP ...")
    plot_umap_train_vs_test(train_emb, test_emb,
                            "UMAP: Full Images — Train vs Test",
                            OUTPUT_DIR / "umap_full_train_vs_test.png")
    plot_umap_clusters(train_emb, test_emb, train_labels,
                       "UMAP: Full Images — HDBSCAN (Train) + Test",
                       OUTPUT_DIR / "umap_full_clusters.png")

    # ═══════════════════════════════════════════════════
    # PART 2: Stone-Masked Images
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 2: Stone-Masked Images")
    print("=" * 60)

    train_mask_idx = build_mask_index(TRAIN_MASK_DIR, set(train_ids))
    test_mask_idx = build_mask_index(TEST_MASK_DIR, set(test_ids))

    train_masked_ids = sorted(k for k, v in train_mask_idx.items() if len(v) > 0)
    test_masked_ids = sorted(k for k, v in test_mask_idx.items() if len(v) > 0)
    print(f"Train w/ masks: {len(train_masked_ids)}/{len(train_ids)}")
    print(f"Test  w/ masks: {len(test_masked_ids)}/{len(test_ids)}")

    train_path_map = {p.stem: p for p in train_paths}
    test_path_map = {p.stem: p for p in test_paths}

    # Stone
    train_stone_ds = MaskedImageDataset(train_path_map, train_mask_idx, invert=False)
    test_stone_ds = MaskedImageDataset(test_path_map, test_mask_idx, invert=False)
    print(f"Train stone samples (raw): {len(train_stone_ds)}")
    print(f"Test  stone samples (raw): {len(test_stone_ds)}")

    train_stone_raw, train_stone_ids = extract_embeddings(model, train_stone_ds, desc="Train stone")
    test_stone_raw, test_stone_ids = extract_embeddings(model, test_stone_ds, desc="Test stone")

    train_stone_emb, _ = aggregate_by_id(train_stone_raw, train_stone_ids)
    test_stone_emb, _ = aggregate_by_id(test_stone_raw, test_stone_ids)
    np.save(OUTPUT_DIR / "embeddings_train_stone.npy", train_stone_emb)
    np.save(OUTPUT_DIR / "embeddings_test_stone.npy", test_stone_emb)
    print(f"Train stone (aggregated): {train_stone_emb.shape}")
    print(f"Test  stone (aggregated): {test_stone_emb.shape}")

    train_stone_labels, train_stone_hdb = run_hdbscan(train_stone_emb)
    test_stone_labels, test_stone_hdb = run_hdbscan(test_stone_emb)
    all_stats["stone"] = {"train": train_stone_hdb, "test": test_stone_hdb}
    print(f"Train stone HDBSCAN: {train_stone_hdb['n_clusters']} clusters, {train_stone_hdb['n_noise']} noise ({train_stone_hdb['noise_ratio']:.1%})")
    print(f"Test  stone HDBSCAN: {test_stone_hdb['n_clusters']} clusters, {test_stone_hdb['n_noise']} noise ({test_stone_hdb['noise_ratio']:.1%})")

    print("Plotting UMAP (stone) ...")
    plot_umap_train_vs_test(train_stone_emb, test_stone_emb,
                            "UMAP: Stone-Masked — Train vs Test",
                            OUTPUT_DIR / "umap_stone_train_vs_test.png")
    plot_umap_clusters(train_stone_emb, test_stone_emb, train_stone_labels,
                       "UMAP: Stone-Masked — HDBSCAN (Train) + Test",
                       OUTPUT_DIR / "umap_stone_clusters.png")

    # ═══════════════════════════════════════════════════
    # PART 3: Background-Masked Images
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 3: Background-Masked Images")
    print("=" * 60)

    train_bg_ds = MaskedImageDataset(train_path_map, train_mask_idx, invert=True)
    test_bg_ds = MaskedImageDataset(test_path_map, test_mask_idx, invert=True)

    train_bg_raw, train_bg_ids = extract_embeddings(model, train_bg_ds, desc="Train bg")
    test_bg_raw, test_bg_ids = extract_embeddings(model, test_bg_ds, desc="Test bg")

    train_bg_emb, _ = aggregate_by_id(train_bg_raw, train_bg_ids)
    test_bg_emb, _ = aggregate_by_id(test_bg_raw, test_bg_ids)
    np.save(OUTPUT_DIR / "embeddings_train_bg.npy", train_bg_emb)
    np.save(OUTPUT_DIR / "embeddings_test_bg.npy", test_bg_emb)
    print(f"Train bg (aggregated): {train_bg_emb.shape}")
    print(f"Test  bg (aggregated): {test_bg_emb.shape}")

    train_bg_labels, train_bg_hdb = run_hdbscan(train_bg_emb)
    test_bg_labels, test_bg_hdb = run_hdbscan(test_bg_emb)
    all_stats["background"] = {"train": train_bg_hdb, "test": test_bg_hdb}
    print(f"Train bg HDBSCAN: {train_bg_hdb['n_clusters']} clusters, {train_bg_hdb['n_noise']} noise ({train_bg_hdb['noise_ratio']:.1%})")
    print(f"Test  bg HDBSCAN: {test_bg_hdb['n_clusters']} clusters, {test_bg_hdb['n_noise']} noise ({test_bg_hdb['noise_ratio']:.1%})")

    print("Plotting UMAP (background) ...")
    plot_umap_train_vs_test(train_bg_emb, test_bg_emb,
                            "UMAP: Background-Masked — Train vs Test",
                            OUTPUT_DIR / "umap_bg_train_vs_test.png")
    plot_umap_clusters(train_bg_emb, test_bg_emb, train_bg_labels,
                       "UMAP: Background-Masked — HDBSCAN (Train) + Test",
                       OUTPUT_DIR / "umap_bg_clusters.png")

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    with open(OUTPUT_DIR / "cluster_results.json", "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'cluster_results.json'}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
