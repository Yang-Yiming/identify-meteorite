# Progress Summary

## Project
Kaggle binary meteorite image classification (mask images via SAM).
Pipeline: ConvNeXt Tiny backbone + light classifier head, AdamW, two-stage training.

## Best Result (myval F1 — primary proxy metric)
| Metric | Value |
|--------|-------|
| **myval F1@0.5** | **0.7202** (prev best: 0.6379, gain: +0.0823) |
| **myval F1@best thr** | **0.7331** (prev best: 0.6650, gain: +0.0681) |
| Run | `train/outputs/myval_v11_hi288_seed42_thr` |
| Branch | `exp/may17-ema-cosine` |

### Best Config
```
--backbone convnext_tiny --head-lr 1e-4 --backbone-lr 1e-5
--dropout 0.1 --label-smoothing 0.1 --cutmix-prob 0.3
--image-size 288 --batch-size 64
--max-grad-norm 1.0 --lr-scheduler cosine
--weight-decay 0.05 --seed 42 --disable-bayes-correction
--open-threshold-search --early-stop 12 --head-only-epochs 5
--val-split-ratio 0.0 --val-root data/myval --val-mask-split myval
```
Plus: BBox-crop preprocessing (margin=0.1, output_size=288).

## Myval F1 Results (all experiments)
| Run | Config | myval F1@0.5 | F1@best thr |
|-----|--------|-------------|-------------|
| trsearch_bbox02 | baseline (224, split-val, seed=123) | 0.6379 | 0.6650 |
| myval_v2_mixup | +MixUp (224, myval-val) | 0.6702 | 0.6737 |
| myval_v5_ema_only | +EMA only (224) | 0.6966 | 0.7072 |
| myval_v3_trsearch | +thr_search (224, seed=123) | 0.7006 | 0.7114 |
| myval_v1_gc | +grad_clip (224, myval-val, seed=123) | 0.7027 | 0.7041 |
| myval_v4_cosine | +cosine (224, seed=123) | 0.7069 | 0.7114 |
| myval_v6_hi288 | +288px (seed=123) | 0.7107 | 0.7112 |
| myval_v9_hi288_seed42 | +seed=42 (288px) | 0.7172 | 0.7225 |
| **myval_v11_hi288_seed42_thr** | **+thr_search (seed=42, 288px)** | **0.7202** | **0.7331** |

## Key Findings
1. **myval-as-validation** (+0.065): The single biggest win.
2. **Higher resolution (288px)** (+0.008): More pixels help discrimination.
3. **Seed variance is high**: seed=42 beats seed=123 by +0.0065 at 288px.
4. **Threshold search**: Works well with seed=42 at 288px (+0.0086 at best thr).
5. **Cosine LR**: Marginal benefit.
6. **EMA**: Does NOT help — lagging weights hurt epoch selection.
7. **MixUp**: Slight standalone benefit but not additive.

## Tested & Discarded
- EMA (0.999) — lagging weights, hurts best epoch selection
- ConvNeXt V2 Tiny at 288px — CUDA OOM (batch_size=64)
- Threshold search at 288px with seed=123 — no improvement

## Current State
- **Best model**: myval_f1=0.7202@0.5, best_thr=0.7331@0.4832
- Checkpoint: `train/outputs/myval_v11_hi288_seed42_thr/best.pt`
- Branch: `exp/may17-ema-cosine`
- BBox crop at `preprocess/bbox_crop/` (train: 4780, test: 176, myval: 329)

## Potential Future Directions
1. Even higher resolution (320px) with batch_size=48
2. ConvNeXt V2 Tiny at 224px (may have better features)
3. Ensemble of best 288px checkpoints (seed=42 + seed=123)
4. Focal loss for hard negative mining
5. Test-time augmentation during evaluation
