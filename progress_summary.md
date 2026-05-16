# Progress Summary

## Project
Kaggle binary meteorite image classification (mask images via SAM).
Pipeline: ConvNeXt Tiny backbone + light classifier head, AdamW, two-stage training.

## Best Result
| Metric | Value |
|--------|-------|
| **val_f1** | **0.9708** (baseline: 0.5833, gain: +0.3875) |
| Run | `train/outputs/trsearch_bbox02` |
| Best epoch | 13 |
| Threshold | 0.5 (no Bayes correction) |
| Branch | `autoresearch/different-backbone` |

### Best Config
```
--backbone convnext_tiny --head-lr 1e-4 --backbone-lr 1e-5
--dropout 0.1 --label-smoothing 0.1 --cutmix-prob 0.3
--weight-decay 0.05 --seed 123 --disable-bayes-correction
--early-stop 6 --batch-size 96 --head-only-epochs 3 --epochs 25
--val-split-ratio 0.2
```
Plus: BBox-crop preprocessing (margin=0.1, output_size=224) via `preprocess/bbox_crop.py`.

### Previous Bests
| Run | val_f1 | Notes |
|-----|--------|-------|
| trsearch_bbox01 | 0.9444 | BBox-crop + Bayes correction (threshold=0.227) |
| trsearch_hlr04 | 0.7664 | Best before bbox-crop (head_lr=1e-4) |

### Kept (cumulative improvements)
| Step | Change | val_f1 |
|------|--------|--------|
| Baseline | Imported split02 | 0.5833 |
| v1 | +threshold_search | 0.5954 |
| v3 | +cutmix=0.3 | 0.6504 |
| v5 | +seed=123 | 0.7154 |
| v7 | +dropout=0.1 | 0.7347 |
| ls01 | +label_smoothing=0.1 | 0.7368 |
| hlr04 | +head_lr=1e-4 | 0.7664 |
| bbox01 | +BBox-crop (Bayes on) | 0.9444 |
| **bbox02** | **+No Bayes + thresh=0.5** | **0.9708** |

### Tested & Discarded
- Cutmix=0.5, weight decay sweep, higher label smoothing (0.2)
- Dropout at seed=42 (seed interaction)
- Higher head_lr (1e-2 training collapse), lower head_lr (3e-5 too low), lower backbone_lr (3e-6)
- Stochastic depth (0.1, 0.05) — over-regularized
- Stronger augs (RandAugment+ColorJitter+MixUp) — over-regularized
- Pseudo-labeling (0.95 conf) — val_f1=0.7317, harms
- Multi-seed ensemble — doesn't beat best singleton
- Alternative backbones: convnext_small (0.6275), efficientnet_b0 (0.5263), swin_tiny (0.6667)
- BBox-crop early attempt (0.5567 without best hyperparams)

## Current State
- **Best model**: val_f1=0.9708, threshold=0.5, no Bayes correction
- BBox crop preprocessing at `preprocess/bbox_crop/` (train: 4780 images, test: 176 images)
- Best checkpoint at `train/outputs/trsearch_bbox02/best.pt`
- Final submission at `train/outputs/trsearch_bbox02/submission_final.csv` (100 positive / 94 negative, post-processed with not-stone.txt)
- Bayes correction fully removable via `--disable-bayes-correction`
- `post_process/zero_not_stone.py` for applying not-stone zero-out

## Potential Future Directions
1. TTA during validation/training
2. Model soup / weight averaging across epochs
3. Different SAM checkpoints or mask strategies
4. BBox-crop margin sweep (try different padding values)
5. Multi-seed ensemble with current best config

## Key Files
- `train/train_finetune.py` — main training entrypoint
- `train/modeling.py` — ConvNeXt classifier
- `train/data.py` — dataset/splits
- `train/augmentations.py` — CutMix, MixUp, label smoothing
- `train/calibration.py` — threshold search, F1, Bayes correction
- `preprocess/bbox_crop.py` — BBox-crop preprocessing script
- `my-autoresearch/handoff.md` — detailed handoff
- `my-autoresearch/experiment_journal.md` — full journal
- `my-autoresearch/results.tsv` — tabular results
- `my-autoresearch/plan.md` — strategy & roadmap
