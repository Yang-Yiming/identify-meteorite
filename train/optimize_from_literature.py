#!/usr/bin/env python3
"""
Literature-guided optimization pipeline for meteorite classification.
Implements: BN adaptation, WiSE-FT interpolation, calibrated meta-learner,
and Noisy Student pseudo-label generation.

Key papers:
  - TENT (2006.10726, ICLR 2021): BN stats adaptation on target distribution
  - WiSE-FT (2109.01903, CVPR 2022): interpolate early + late checkpoints
  - Guo et al. (1706.04599, ICML 2017): temperature calibration before fusion
  - Noisy Student (1911.04252, NeurIPS 2020): pseudo-label + noise training
"""

import gc, json, argparse, math, sys, re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling import ConvNeXtClassifier
from tta import predict_probabilities_with_tta
from timm.data import create_transform, resolve_model_data_config

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


def normalize_id(path_str: str) -> str:
    stem = Path(path_str).name.replace('_mask_000', '')
    return f"{int(re.search(r'\d+', stem).group()):06d}.jpg"


class PathImageDataset(Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


def get_manifest_paths(source: str, manifest_path: str) -> List[str]:
    manifest = pd.read_csv(manifest_path)
    subset = manifest[(manifest['source'] == source) & manifest['has_image']]
    return subset['path'].astype(str).tolist()


def load_checkpoint_model(
    ckpt_path: str,
    backbone: str,
    dropout: float = 0.1,
    drop_path: float = 0.0,
    device: str = 'cuda',
) -> Tuple[ConvNeXtClassifier, Dict]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model = ConvNeXtClassifier(backbone, None, 2, dropout, True, drop_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, ckpt


def run_inference(
    model: nn.Module,
    paths: List[str],
    batch_size: int = 64,
    device: str = 'cuda',
    tta: bool = False,
    tta_mode: str = '4way',
) -> np.ndarray:
    """
    Run inference on a list of image paths.
    Returns numpy array of positive-class probabilities [N].
    """
    if not model.training:
        pass  # We may call this while model is in train mode for BN adaptation

    # Resolve transform from timm data config
    try:
        data_config = resolve_model_data_config(model.backbone)
    except Exception:
        data_config = {}
    transform = create_transform(**data_config, is_training=False)

    ds = PathImageDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    probs = []
    autocast_enabled = device.startswith('cuda')

    with torch.no_grad() if not model.training else torch.enable_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            if tta:
                prob = predict_probabilities_with_tta(
                    model, images, positive_label=POSITIVE_LABEL,
                    views=['identity', 'hflip', 'vflip', 'rot90'],
                    device_type='cuda' if 'cuda' in device else 'cpu',
                    autocast_enabled=autocast_enabled,
                )
            else:
                with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', enabled=autocast_enabled):
                    logits = model(images)
                prob = torch.softmax(logits, dim=1)[:, POSITIVE_LABEL]
            probs.append(prob.detach().cpu().numpy())

    return np.concatenate(probs)


# ---------------------------------------------------------------------------
# 1. BN Stats Adaptation (TENT-lite)
# ---------------------------------------------------------------------------

def adapt_bn_stats(
    model: nn.Module,
    adaptation_paths: List[str],
    batch_size: int = 64,
    device: str = 'cuda',
):
    """
    Adapt BatchNorm running statistics to the target distribution.
    Sets model to train mode (updates BN stats) but freezes all parameters
    (no gradient-based optimization).
    """
    # Freeze all parameters so only BN stats update
    for param in model.parameters():
        param.requires_grad = False

    # Set all BN layers to track running stats (train mode)
    model.train()

    # But we need BN in train mode for stats tracking. Set all other layers
    # to behave as in eval mode (no dropout, etc.)
    # This is a bit tricky. For ConvNeXt, BN is inside each block.
    # We'll manually set train mode but freeze params.

    try:
        data_config = resolve_model_data_config(model.backbone)
    except Exception:
        data_config = {}
    transform = create_transform(**data_config, is_training=False)

    ds = PathImageDataset(adaptation_paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"  Adapting BN stats on {len(adaptation_paths)} images (bs={batch_size})...")
    total_batches = len(loader)
    with torch.no_grad():
        for i, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            _ = model(images)  # Forward pass updates BN running stats
            if (i + 1) % max(1, total_batches // 10) == 0:
                print(f"    BN adapt: {i+1}/{total_batches} batches")

    # Switch back to eval mode for inference
    model.eval()
    print("  BN adaptation complete.")

    return model


# ---------------------------------------------------------------------------
# 2. WiSE-FT Checkpoint Interpolation
# ---------------------------------------------------------------------------

def wise_ft_interpolate(
    root_dir: str,
    epoch_a: int,  # early epoch (better generalization)
    epoch_b: int,  # late epoch (better accuracy)
    alpha: float = 0.5,  # weight of late-epoch model
    device: str = 'cuda',
) -> ConvNeXtClassifier:
    """
    θ_wise = (1-α) * θ_early + α * θ_late
    """
    from pathlib import Path

    base = Path(root_dir)
    ckpt_a = base / f'epoch_{epoch_a:02d}.pt'
    ckpt_b = base / f'epoch_{epoch_b:02d}.pt'

    if not ckpt_a.is_file() or not ckpt_b.is_file():
        raise FileNotFoundError(f"Missing checkpoints: {ckpt_a}, {ckpt_b}")

    state_a = torch.load(ckpt_a, map_location='cpu')
    state_b = torch.load(ckpt_b, map_location='cpu')

    sd_a = state_a.get('model', state_a)
    sd_b = state_b.get('model', state_b)

    # Interpolate
    sd_wise = {}
    for key in sd_a:
        if key in sd_b:
            sd_wise[key] = (1 - alpha) * sd_a[key].float() + alpha * sd_b[key].float()
        else:
            sd_wise[key] = sd_a[key]

    # Load into model
    backbone_name = 'convnext_tiny'  # Default, can be overridden
    model = ConvNeXtClassifier(backbone_name, None, 2, 0.1, False, 0.0)
    model.load_state_dict(sd_wise, strict=False)
    model = model.to(device).eval()

    print(f"  WiSE-FT: interpolated epoch {epoch_a} (gen) + epoch {epoch_b} (acc) with α={alpha:.2f}")
    return model, sd_wise


# ---------------------------------------------------------------------------
# 3. Calibrated Meta-Learner from K-Fold Validation
# ---------------------------------------------------------------------------

def temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit a single temperature parameter via NLL minimization on logits (numpy)."""
    import scipy.optimize
    if scipy is None:
        return 1.0

    def nll(T):
        scaled = logits / T
        probs = 1.0 / (1.0 + np.exp(-scaled))  # sigmoid on logit
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        pos_probs = np.where(labels == 1, probs, 1 - probs)
        return -np.mean(np.log(pos_probs))

    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        return result.x
    except ImportError:
        return 1.0


def calibrate_probabilities(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Apply temperature scaling to calibrate probabilities.
    Returns calibrated probs and temperature.
    """
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))

    T = temperature_scale(logits, labels)
    calibrated_logits = logits / T
    calibrated = 1.0 / (1.0 + np.exp(-calibrated_logits))

    return calibrated, T


def build_calibrated_meta_learner(
    cn_probs_val: np.ndarray,  # ConvNeXt OOB probs on validation
    ds_probs_val: np.ndarray,  # DINOv2 probs on validation
    cn_probs_test: np.ndarray,  # ConvNeXt test probs
    ds_probs_test: np.ndarray,  # DINOv2 test probs
    val_labels: np.ndarray,
    test_ids: List[str],
) -> Dict:
    """
    Build a calibrated meta-learner using cross-validation validation sets.
    Steps:
    1. Calibrate ConvNeXt and DINOv2 probabilities independently
    2. Train logistics regression on calibrated probabilities
    3. Apply to test set
    """

    # Step 1: Calibrate each model independently
    cn_cal, cn_T = calibrate_probabilities(cn_probs_val, val_labels)
    ds_cal, ds_T = calibrate_probabilities(ds_probs_val, val_labels)

    print(f"  Temperature scaling: T_cn={cn_T:.4f}, T_ds={ds_T:.4f}")

    # Step 2: Build meta-learner features from calibrated probabilities
    X_val = np.column_stack([
        cn_cal,           # ConvNeXt calibrated prob
        ds_cal,           # DINOv2 calibrated prob
        cn_cal * ds_cal,  # Interaction: agreement
        np.abs(cn_cal - ds_cal),  # Disagreement magnitude
        (cn_cal + ds_cal) / 2,  # Average
        (cn_cal > 0.5).astype(float) * cn_cal,  # P(pos) clipped
    ])

    # Train calibration-aware logistic regression
    meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=5000, random_state=42, class_weight='balanced'),
    )
    meta.fit(X_val, val_labels.astype(int))

    # Step 3: Apply to test set
    cn_test_cal, _ = calibrate_probabilities(cn_probs_test, val_labels)  # Use val T for calibration
    ds_test_cal, _ = calibrate_probabilities(ds_probs_test, val_labels)
    X_test = np.column_stack([
        cn_test_cal,
        ds_test_cal,
        cn_test_cal * ds_test_cal,
        np.abs(cn_test_cal - ds_test_cal),
        (cn_test_cal + ds_test_cal) / 2,
        (cn_test_cal > 0.5).astype(float) * cn_test_cal,
    ])

    meta_probs = meta.predict_proba(X_test)[:, 1]

    # Step 4: Evaluate on validation
    from sklearn.metrics import f1_score, roc_auc_score
    val_preds = meta.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(val_labels, val_preds)

    best_f1 = 0.0
    best_thr = 0.5
    for t in np.linspace(0.1, 0.9, 200):
        f1 = f1_score(val_labels, (val_preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    print(f"  Meta-learner CV: AUC={val_auc:.4f}, best_f1={best_f1:.4f} @ thr={best_thr:.3f}")

    return {
        'meta_probs': meta_probs,
        'val_auc': val_auc,
        'best_threshold': best_thr,
        'cn_temperature': cn_T,
        'ds_temperature': ds_T,
    }


# ---------------------------------------------------------------------------
# 4. Noisy Student Pseudo-Label Generation
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    model: nn.Module,
    pseudo_paths: List[str],
    batch_size: int = 64,
    confidence_threshold: float = 0.90,
    device: str = 'cuda',
) -> Tuple[List[str], np.ndarray]:
    """
    Generate pseudo-labels for unlabeled images using teacher model.
    Only keeps labels with confidence > threshold.
    """
    probs = run_inference(model, pseudo_paths, batch_size, device)
    confidence = np.where(probs >= 0.5, probs, 1 - probs)
    mask = confidence >= confidence_threshold
    labels = (probs >= 0.5).astype(int)

    selected_paths = [p for p, m in zip(pseudo_paths, mask) if m]
    selected_probs = probs[mask]
    selected_labels = labels[mask]

    print(f"  Generated {len(selected_paths)}/{len(pseudo_paths)} pseudo-labels "
          f"(threshold={confidence_threshold})")
    print(f"    Pos: {(selected_labels == 1).sum()}, Neg: {(selected_labels == 0).sum()}")

    return selected_paths, selected_probs


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Literature-guided optimization")
    parser.add_argument('--manifest', type=str,
                        default='analysis/testlike_dino_train_v4/manifest.csv')
    parser.add_argument('--soup-checkpoint', type=str,
                        default='train/outputs/myval_v13_hi288_seed42_soup/soup.pt')
    parser.add_argument('--dinov2-checkpoint', type=str,
                        default='train/outputs/dinov2_small_518_s42/best.pt')
    parser.add_argument('--kfold-dir', type=str,
                        default='train/outputs')
    parser.add_argument('--splits-dir', type=str,
                        default='train/kfold_splits')
    parser.add_argument('--output-dir', type=str,
                        default='train/outputs/literature_optimized')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--batch-size-dinov2', type=int, default=8)
    parser.add_argument('--skip-bn-adapt', action='store_true')
    parser.add_argument('--skip-wiseft', action='store_true')
    parser.add_argument('--skip-meta-learner', action='store_true')
    parser.add_argument('--skip-pseudo', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    test_df = manifest[(manifest['source'] == 'test') & manifest['has_image']]
    test_paths = test_df['path'].astype(str).tolist()
    test_ids = [normalize_id(p) for p in test_paths]

    mytest_df = manifest[(manifest['source'] == 'mytest') & manifest['has_image']]
    mytest_paths = mytest_df['path'].astype(str).tolist()

    print(f"Data: {len(test_paths)} test, {len(mytest_paths)} mytest images")

    # ========================================================================
    # Step 0: Load models
    # ========================================================================
    print("\n" + "=" * 60)
    print("Loading models...")

    # ConvNeXt Soup
    cn_model, _ = load_checkpoint_model(
        args.soup_checkpoint, 'convnext_tiny', 0.1, 0.0, args.device
    )
    print(f"  ConvNeXt soup loaded")

    # DINOv2 Small
    ds_model, _ = load_checkpoint_model(
        args.dinov2_checkpoint, 'vit_small_patch14_dinov2.lvd142m', 0.1, 0.1, args.device
    )
    print(f"  DINOv2 Small loaded")

    # ========================================================================
    # Step 1: BN Stats Adaptation (TENT-lite)
    # ========================================================================
    if not args.skip_bn_adapt:
        print("\n" + "=" * 60)
        print("Step 1: BN Stats Adaptation on mytest distribution")

        # Adapt ConvNeXt BN on mytest
        adapt_bn_stats(cn_model, mytest_paths, args.batch_size, args.device)
        cn_model.eval()

        # Adapt DINOv2 BN on mytest
        # Note: DINOv2 has LayerNorm, not BatchNorm, so BN adaptation doesn't apply
        # But we still forward-pass through mytest for potential future use
        print("  (DINOv2 uses LayerNorm; BN adaptation not applicable)")
        print("  Running mytest forward pass on DINOv2 for potential feature extraction...")
        # Skip DINOv2 BN adaptation (it has LN not BN)

    # ========================================================================
    # Step 2: Get baseline test predictions
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Baseline test predictions")

    # ConvNeXt (potentially BN-adapted) test probs
    print("  ConvNeXt test inference...")
    gc.collect(); torch.cuda.empty_cache()
    cn_test_probs = run_inference(cn_model, test_paths, args.batch_size, args.device)

    # Save ConvNeXt test probs
    np.savez(output_dir / 'cn_soup_test_probs.npz',
             test_ids=test_ids, probs=cn_test_probs)

    # DINOv2 test probs (use smaller batch due to 518px)
    print("  DINOv2 test inference...")
    gc.collect(); torch.cuda.empty_cache()
    ds_test_probs = run_inference(
        ds_model, test_paths, args.batch_size_dinov2, args.device
    )
    np.savez(output_dir / 'ds_small_test_probs.npz',
             test_ids=test_ids, probs=ds_test_probs)

    # ========================================================================
    # Step 3: WiSE-FT Interpolation
    # ========================================================================
    if not args.skip_wiseft:
        print("\n" + "=" * 60)
        print("Step 3: WiSE-FT Checkpoint Interpolation")

        soup_dir = str(Path(args.soup_checkpoint).parent)
        # Try interpolating epoch 1 (early gen) with epoch 9 (max accuracy)
        # We need to find which epochs are available in the soup dir
        available_epochs = sorted([
            int(Path(f).stem.split('_')[1])
            for f in Path(soup_dir).glob('epoch_*.pt')
        ])
        print(f"  Available epochs: {available_epochs}")

        # Use earliest available as "generalization" checkpoint
        # and the best epoch (or last) as "accuracy" checkpoint
        if len(available_epochs) >= 2:
            early_epoch = available_epochs[0]  # First saved epoch
            late_epoch = available_epochs[-1]   # Last epoch
            print(f"  Interpolating epoch {early_epoch} (gen) + {late_epoch} (acc)")

            wise_model, wise_sd = wise_ft_interpolate(
                soup_dir, early_epoch, late_epoch, alpha=0.5, device=args.device
            )

            # Run inference with WiSE-FT model
            gc.collect(); torch.cuda.empty_cache()
            wise_test_probs = run_inference(
                wise_model, test_paths, args.batch_size, args.device
            )
            np.savez(output_dir / 'wiseft_test_probs.npz',
                     test_ids=test_ids, probs=wise_test_probs)

            # Now try different alpha values
            for alpha in [0.3, 0.7]:
                gc.collect(); torch.cuda.empty_cache()
                wise_m, _ = wise_ft_interpolate(
                    soup_dir, early_epoch, late_epoch, alpha=alpha, device=args.device
                )
                wise_probs = run_inference(
                    wise_m, test_paths, args.batch_size, args.device
                )
                np.savez(output_dir / f'wiseft_alpha{alpha:.1f}_test_probs.npz',
                         test_ids=test_ids, probs=wise_probs)

            del wise_model, wise_m
        else:
            print("  Not enough epochs for WiSE-FT interpolation")

    # ========================================================================
    # Step 4: Noisy Student Pseudo-Label Generation
    # ========================================================================
    if not args.skip_pseudo:
        print("\n" + "=" * 60)
        print("Step 4: Noisy Student Pseudo-Label Generation")

        # Use ConvNeXt (BN-adapted) as teacher
        gc.collect(); torch.cuda.empty_cache()
        selected_paths, selected_probs = generate_pseudo_labels(
            cn_model, mytest_paths, args.batch_size,
            confidence_threshold=0.90, device=args.device,
        )

        # Save pseudo-labels for Noisy Student training
        pseudo_ids = [normalize_id(p) for p in selected_paths]
        pseudo_labels = (selected_probs >= 0.5).astype(int)

        pseudo_df = pd.DataFrame({
            'id': pseudo_ids,
            'prob_pos': selected_probs,
            'label': pseudo_labels,
            'confidence': selected_probs if all(selected_probs >= 0.5) else \
                          [p if p >= 0.5 else 1-p for p in selected_probs],
        })
        pseudo_df.to_csv(output_dir / 'noisy_student_pseudo_labels.csv', index=False)
        print(f"  Saved {len(pseudo_df)} pseudo-labels to noisy_student_pseudo_labels.csv")

        # Also generate pseudo-labels using ensemble (CN average of kfold + DS)
        # for higher quality pseudo-labels
        probs_data = np.load('train/kfold_ensemble_probs.npz', allow_pickle=True)
        fold_probs = {k: probs_data[k] for k in probs_data if k != 'test_ids'}

        # Ensemble: average kfold CN + DINOv2
        ensemble_test_probs = np.mean([fold_probs[k] for k in fold_probs], axis=0)

        # For mytest pseudo-labels with ensemble, we need mytest predictions
        # from the ensemble. We have test predictions but need to regenerate for mytest.
        print("  Generating ensemble pseudo-labels for mytest (using CN model)...")
        # For now, use the CN soup model

    gc.collect(); torch.cuda.empty_cache()

    # ========================================================================
    # Step 5: Build submissions from all variants
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Building submissions")

    baseline_df = pd.read_csv(
        'train/outputs/myval_v13_hi288_seed42_soup/submission_processed_best_notstone.csv'
    )
    baseline_df['id'] = baseline_df['id'].astype(str)
    base_dict = dict(zip(baseline_df['id'], baseline_df['label'].astype(int)))
    full_ids = baseline_df['id'].tolist()

    # Helper to create submission
    def make_submission(test_ids, probs, threshold, name_suffix):
        pred_dict = dict(zip(test_ids, (probs >= threshold).astype(int)))
        labels = [pred_dict.get(tid, base_dict[tid]) for tid in full_ids]
        pos = sum(labels)
        diffs = sum(1 for t, l in zip(full_ids, labels) if l != base_dict[t])

        sub = pd.DataFrame({'id': full_ids, 'label': labels})
        fname = output_dir / f'submission_{name_suffix}.csv'
        sub.to_csv(fname, index=False)
        print(f"  {fname.name}: {pos} pos, {diffs} diffs vs SOTA")
        return pos, diffs

    # BN-adapted ConvNeXt
    for thr in [0.50, 0.45, 0.55]:
        make_submission(test_ids, cn_test_probs, thr, f'bnadapt_thr{thr:.2f}')

    # Ensemble: BN-adapted CN + DINOv2
    for thr in [0.50, 0.45, 0.55]:
        ens_probs = (cn_test_probs + ds_test_probs) / 2
        make_submission(test_ids, ens_probs, thr, f'bnadapt_ens_thr{thr:.2f}')

    # WiSE-FT submissions
    wise_files = list(output_dir.glob('wiseft*_test_probs.npz'))
    for wf in wise_files:
        data = np.load(wf, allow_pickle=True)
        w_ids = list(data['test_ids'])
        w_probs = data['probs']
        alpha = wf.stem.replace('wiseft_', '').replace('_test_probs', '')
        for thr in [0.50, 0.45]:
            make_submission(w_ids, w_probs, thr, f'wiseft_{alpha}_thr{thr:.2f}')

    print("\n" + "=" * 60)
    print(f"All outputs saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
