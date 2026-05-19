# Plan

## Objective

Maximize **test** F1 (Kaggle). BBox-crop preprocessing is the **default
pipeline**. Bayes correction is disabled (test distribution unknown).
**Primary proxy metric: myval F1@0.5** — the 20% train split is saturated.

## Current State

**Current best: myval_f1=0.7251@0.5, test_f1=0.69856** (soup checkpoint:
top-3 epochs 20/39/26, 288px, seed=42, cosine, thr_search).
Previous test best was 0.64516 (+0.0534).

Latest test: `mytest_v1_s42` used high-quality `mytest` as extra
training/validation data. It reached `mytest_val_f1=0.8969` and sealed
`myval_f1=0.7321@0.5`, but Kaggle test regressed to **0.65979**. DISCARD.
This confirms that `myval` remains useful, but small myval gains are mostly
noise; future KEEP decisions need a large sealed-myval improvement.

### Key Improvements Achieved

| Change | split_val | myval | test |
|--------|-----------|-------|------|
| Baseline (hlr04) | 0.7664 | — | 0.42 |
| + BBox-crop (bayes on) | 0.9444 | — | — |
| + No Bayes + thresh=0.5 | **0.9708** | 0.6379 | 0.64516 |
| + myval-as-validation + 288px + seed=42 + cosine | — | 0.7202 | — |
| **+ top-3 model soup** | — | **0.7251** | **0.69856** |
| + mytest train/val split | 0.8969 (mytest_val) | 0.7321 | 0.65979 |

### Discarded Directions

- Multi-seed ensemble, cutmix=0.5, weight decay sweep, dropout at seed=42.
- Label smoothing >0.1, higher/lower head_lr, lower backbone_lr.
- Stochastic depth, stronger augs, pseudo-labeling.
- Alternative backbones (convnext_small, efficientnet_b0, swin_tiny).
- 320px resolution — myval F1 regressed to 0.6957.
- TTA (post-hoc) — did not improve F1@0.5.
- mytest split protocol — myval rose slightly to 0.7321, but test regressed to 0.65979.

## Next Directions

1. ~~**BBox-crop (bayes on)** — trsearch_bbox01: val_f1=0.9444. KEEP.~~
2. ~~**BBox-crop (no bayes, thresh=0.5)** — trsearch_bbox02: val_f1=0.9708, test_f1=0.64516. KEEP — new SOTA.~~
3. ~~TTA during validation/training — DISCARD (post-hoc TTA did not improve myval F1@0.5).~~
4. ~~Model soup / weight averaging across epochs — KEEP (top-3: 0.7251 myval, 0.69856 test, +0.0049/+0.0534).~~
5. ~~mytest split protocol — DISCARD (0.7321 myval but 0.65979 test).~~
6. TTA during training — integrate into validation loop for better epoch selection.
7. Multi-seed ensemble with current best config.
8. Weighted model soup.
9. BBox-crop margin sweep.
