# Plan

## Objective

Maximize **test** F1 (Kaggle). BBox-crop preprocessing is the **default
pipeline**. Bayes correction is disabled (test distribution unknown).
**Primary proxy metric: myval F1@0.5**.

**CRITICAL UPDATE (2026-05-20): myval is an unreliable proxy when mytest
data is involved.** Adding mytest as training data inflates myval F1
(+0.02~+0.04) but degrades test F1 (-0.03~-0.15). See "mytest Generalization
Failure" below.

## Current State

**Test SOTA: test_f1=0.69856** (soup checkpoint: top-3 epochs 20/39/26,
288px, seed=42, cosine, thr=0.5, 4780 original bbox-crop images).

| Run | myval F1@0.5 | test F1 | Description |
|-----|-------------|---------|-------------|
| soup (prev best) | 0.7251 | **0.69856** | no mytest |
| mytest split protocol | 0.7321 | 0.65979 | mytest as train+val |
| mytest pretrain→finetune | 0.7358 | 0.55214 | two-stage |
| mytest aug + myval val | 0.7688 | 0.67021 | mytest merged, myval selects epoch |
| split-val aug soup | 0.7446 | 0.63212 | no myval leak, still degraded |

### mytest Generalization Failure

Every approach that adds mytest data to training hurts test F1, regardless
of whether myval leaks into training or not. The root cause is domain shift:
mytest images come from Encyclopedia of Meteorites and Kaggle rock datasets,
which have different visual characteristics from the competition test set.

The myval→test gap widens with mytest involvement:
- no mytest: gap ~0.027
- mytest aug: gap ~0.099
- mytest pretrain: gap ~0.184

**Decision: Abandon mytest as training data.** Focus on methods that
improve generalization without external data.

### Key Improvements Achieved

| Change | split_val | myval | test |
|--------|-----------|-------|------|
| Baseline (hlr04) | 0.7664 | — | 0.42 |
| + BBox-crop (bayes on) | 0.9444 | — | — |
| + No Bayes + thresh=0.5 | **0.9708** | 0.6379 | 0.64516 |
| + myval-as-validation + 288px + seed=42 + cosine | — | 0.7202 | — |
| **+ top-3 model soup** | — | **0.7251** | **0.69856** |

### Discarded Directions

- Multi-seed ensemble, cutmix=0.5, weight decay sweep, dropout at seed=42
- Label smoothing >0.1, higher/lower head_lr, lower backbone_lr
- Stochastic depth, stronger augs, pseudo-labeling
- Alternative backbones (convnext_small, efficientnet_b0, swin_tiny)
- 320px resolution — myval F1 regressed to 0.6957
- TTA (post-hoc) — did not improve F1@0.5
- Focal loss — no myval gain over CE
- **ALL mytest-based approaches** — myval up, test down
  - mytest split protocol (0.65979)
  - mytest pretrain→finetune (0.55214)
  - mytest augmentation + myval val (0.67021)
  - split-val + mytest augmentation (0.63212)

## Next Directions

1. K-fold bagging on original 4780 training data (no mytest)
2. Data cleaning / hard negative mining within original train set
3. Architecture exploration (ConvNeXt V2 at 224px, newer backbones)
4. Multi-seed retry with split-val (not myval) to avoid proxy overfit
