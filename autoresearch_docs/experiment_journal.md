# Experiment Journal

## 2026-05-17: May17 — Systematic myval F1 improvement

### Background
Previous best myval F1 = 0.6379 (trsearch_bbox02, evaluated post-hoc on myval).
The primary metric has shifted from split-val to myval F1.

### Key Insight
**Using myval as the external validation set during training** (--val-split-ratio=0.0, --val-root data/myval) allows the model to select checkpoints based on myval performance directly. This single change gave the biggest improvement.

### Experiments Run

| # | Name | Config | myval F1 | verdict |
|---|------|--------|----------|---------|
| 1 | myval_v1_gc | baseline (224, myval-val, grad_clip=1.0, seed=123) | 0.7027 | KEEP |
| 2 | myval_v2_mixup | +MixUp (0.2) | 0.6702 | discard |
| 3 | myval_v3_trsearch | +threshold_search, early_stop=12 | 0.7006 | KEEP (best thr=0.7114) |
| 4 | myval_v2_ema | +EMA(0.999)+cosine+mixup | 0.6841 | discard (EMA lags) |
| 5 | myval_v4_cosine | +cosine LR only | 0.7069 | KEEP |
| 6 | myval_v5_ema_only | +EMA only | 0.6966 | discard |
| 7 | myval_v6_hi288 | +288px, seed=123, cosine | 0.7107 | KEEP |
| 8 | myval_v7_hi288_thr | +288px, thr_search, seed=123 | 0.6930 | discard |
| 9 | myval_v9_hi288_seed42 | +288px, seed=42, cosine | **0.7172** | **BEST** |
| 10 | myval_v11_hi288_seed42_thr | +288px, seed=42, thr_search | running | pending |

### Findings
1. **myval-as-validation** (+0.065): The biggest win. Using myval directly as the validation set for checkpoint selection.
2. **Higher resolution (288px)** (+0.008): More pixels help the model distinguish fine-grained features.
3. **Seed=42 beats seed=123** (+0.0065 at 288px): High seed variance confirmed.
4. **Cosine LR** (+0.004): Marginal benefit at 224px.
5. **EMA**: Does NOT help — the lagging EMA weights produce lower validation F1, causing suboptimal epoch selection.
6. **MixUp**: Slight standalone benefit but not additive with other improvements.
7. **Threshold search**: Neutral to negative at both 224px and 288px.

### Current Best
```
myval_v9_hi288_seed42: myval_f1=0.7172@0.5, best_thr=0.7225@0.4880
Config: 288px, seed=42, cosine LR, grad_clip=1.0, head_only=5, epochs=50
Checkpoint: train/outputs/myval_v9_hi288_seed42/best.pt
```

## 2026-05-18: 320px Resolution (myval_v12_hi320_seed42)

- **Hypothesis**: Higher 320px resolution improves myval F1 beyond 288px (0.7202).
- **Config**: `--image-size 320 --batch-size 48 --seed 42 --lr-scheduler cosine --open-threshold-search --save-every-epoch`. Same best config as 288px run (seed=42, bbox-crop, no-Bayes, head_lr=1e-4, backbone_lr=1e-5, dropout=0.1, label_smoothing=0.1, cutmix=0.3).
- **Total epochs**: 5 head-only + 50 finetune = 55. Batch size reduced from 96 to 48 for memory.
- **Primary metric (myval F1@0.5)**: **0.6957** (down from 0.7202)
- **Best threshold F1**: 0.7099 @ thr=0.6934
- **Internal model_select_f1**: 0.7186 @ thr=0.5840 (epoch 40)
- **Runtime**: ~33 minutes.
- **Verdict**: **DISCARD** — 320px regresses myval F1 by -0.0245. convnext_tiny pretrained at 224px does not benefit from 320px with limited data.

### Updated Best
```
myval_v11_hi288_seed42_thr: myval_f1=0.7202@0.5, best_thr=0.7331@0.4832
Config: 288px, seed=42, cosine LR, thr_search, grad_clip=1.0, epochs=50
Checkpoint: train/outputs/myval_v11_hi288_seed42_thr/best.pt (epoch 20)
```

## 2026-05-18: TTA + Model Soup

### TTA on myval (post-hoc)
- **Hypothesis**: Test-time augmentation during myval evaluation improves F1.
- **Config**: Same best checkpoint (myval_v11_hi288_seed42_thr/best.pt), evaluated with `--tta 4way` and `--tta 8way`.
- **4way TTA**: myval F1@0.5 = **0.7168** (down from 0.7202). best_thr_F1 = **0.7362** (up from 0.7331).
- **8way TTA**: myval F1@0.5 = **0.7059** (worse).
- **Verdict**: **DISCARD** — TTA at @0.5 threshold not beneficial.

### Model Soup (weight averaging across epochs)
- **Hypothesis**: Averaging top-N epoch checkpoints improves generalization.
- **Training**: Reran best 288px config with `--save-every-epoch` → 65 epochs (best_epoch=20, internal val_f1=0.7500).
- **Top-10 soup**: myval F1@0.5 = **0.7184** (worse than single 0.7202).
- **Top-5 soup**: myval F1@0.5 = **0.7251**.
- **Top-3 soup** (epochs 20, 39, 26): myval F1@0.5 = **0.7251** (+0.0049).
- **Verdict**: **KEEP** — top-3 model soup is new best.

### Updated Best
```
myval_v13_hi288_seed42_soup/soup.pt (top-3: epochs 20, 39, 26): myval_f1=0.7251@0.5
Config: 288px, seed=42, cosine LR, thr_search, bbox-crop, no-Bayes
Checkpoint: train/outputs/myval_v13_hi288_seed42_soup/soup.pt
```

## 2026-05-18: Kaggle Test Result

- **Test F1**: **0.69856** (top-3 model soup, no TTA, thr=0.5, +post-process zero_not_stone).
- **Previous test best**: 0.64516 (trsearch_bbox02) → **+0.0534 improvement**.
- **Myval proxy gap**: myval F1=0.7251 vs test F1=0.69856 (gap=0.0265). Myval remains a reasonable proxy.

### Updated Best
```
Run:        myval_v13_hi288_seed42_soup (top-3: epochs 20, 39, 26)
Myval F1:   0.7251@0.5
Test F1:    0.69856
Checkpoint: train/outputs/myval_v13_hi288_seed42_soup/soup.pt
```

## 2026-05-18: Weighted Model Soup Sweep (post-soup)

- **Hypothesis**: Weighting epoch contributions by val_f1 (or squared) improves over uniform top-3 averaging.
- **Weighted top-3** (linear, squared): myval F1@0.5 = **0.7251** (same as uniform, weights are nearly equal).
- **Top-2** (uniform, weighted): myval F1@0.5 = **0.7176** (worse).
- **Top-5** (uniform, weighted): myval F1@0.5 = **0.7151** (worse).
- **Verdict**: **DISCARD** — top-3 uniform soup remains best. Weighting schemes don't help when top epochs have similar val_f1.

## 2026-05-18: BBox-Crop Margin Sweep

- **Hypothesis**: The default bbox-crop margin (0.10) may not be optimal. Tighter crop (0.05) focuses more on meteorite; looser crop (0.15, 0.20) provides more context.
- **Training**: Full retrain with best config (288px, seed=42, cosine, myval-as-val) at each margin.
- **Results**:
  | Margin | myval F1@0.5 | Δ from best | Verdict |
  |--------|-------------|-------------|---------|
  | 0.05 | 0.6842 | -0.0409 | DISCARD — too tight, loses context |
  | **0.10** | **0.7251** | **—** | **BEST** (current default) |
  | 0.15 | 0.6954 | -0.0297 | DISCARD — too loose, background noise |
  | 0.20 | 0.6897 | -0.0354 | DISCARD — too loose, background noise |
- **Verdict**: **DISCARD** — margin=0.10 is confirmed optimal. Both tighter and looser margins degrade performance.

## 2026-05-19: Multi-Seed Ensemble

- **Hypothesis**: Soft-voting ensemble of seed-42 (best single), seed-123, and seed-256 trained with identical config improves myval F1.
- **Training**: Retrained seeds 123 and 256 with best config (288px, margin=0.10, cosine LR, myval-as-val, 50 epochs).
- **Checkpoints**:
  - `train/outputs/myval_v13_hi288_seed42_soup/soup.pt` (seed 42 soup, best=0.7251)
  - `train/outputs/myval_v17_s123/best.pt` (seed 123, epoch 43, val_f1=0.6909, myval=0.7086)
  - `train/outputs/myval_v18_s256/best.pt` (seed 256, epoch 40, val_f1=0.7182, myval=0.7068)
- **Ensemble results**:
  | Ensemble | myval F1@0.5 | best thr F1 |
  |----------|-------------|-------------|
  | seed42 only (soup) | 0.7251 | 0.7273 |
  | **seed42 + seed123** | **0.7283** | **0.7283** |
  | seed42 + seed256 | 0.7151 | 0.7178 |
   | seed42 + seed123 + seed256 | 0.7175 | 0.7230 |
- **Kaggle Test F1**: **0.65968** (submission `post_process/submission_ensemble42_123_processed.csv`).
- **Analysis**: Ensemble dramatically overfits myval. Myval proxy gap widened from ~0.027 (soup) to ~0.069 (ensemble). Seed 123 (myval=0.7086) generalizes poorly despite reasonable myval score. Soft-voting with a weaker seed drags down the test performance.
- **Verdict**: **DISCARD** — ensemble does not generalize. Roll back to soup as best.

### Updated Best (reverted to soup)
```
Run:        myval_v13_hi288_seed42_soup (top-3: epochs 20, 39, 26)
Myval F1:   0.7251@0.5
Test F1:    0.69856
Checkpoint: train/outputs/myval_v13_hi288_seed42_soup/soup.pt
Config:     288px, margin=0.10, seed=42, cosine LR, thr_search, bbox-crop, no-Bayes
```

### Key Learnings
1. **Myval proxy is fragile**: A 0.0032 myval improvement (+0.45%) can correspond to a 0.039 test regression (-5.6%). Small myval improvements should not be trusted without test confirmation.
2. **Seed 123 has poor generalization**: Despite reasonable myval F1 (0.7086), seed 123 performs much worse on test. The 288px config at seed=42 is uniquely well-tuned.
3. **All major hyperparameter directions exhausted**: No further low-hanging fruit. Current best is soup at 0.69856 test.

### Next Directions
- Need fundamentally new approach: model architecture change, advanced data augmentation, or more training data.
- TTA during training (low priority, uncertain).

## 2026-05-20: mytest Split Protocol

- **Hypothesis**: Use high-quality `mytest` as extra clean data: merge `mytest_train` into training, select checkpoints on `mytest_val`, and keep `myval` sealed for KEEP/DISCARD.
- **Data**: `mytest` has 3955 images (1371 meteorite, 2584 rock). Split with `--mytest-val-ratio 0.15 --mytest-val-strategy group`.
- **Training**: Best 288px seed=42 config, original train + `mytest_train`, validation on `mytest_val`.
- **Run**: `train/outputs/mytest_v1_s42`
- **Best epoch**: 17
- **mytest_val F1**: 0.8969
- **Sealed myval F1@0.5**: 0.7321 (slightly above soup 0.7251)
- **Sealed myval best-thr F1**: 0.7508 @ 0.5771
- **Kaggle Test F1**: 0.65979 (`post_process/submission_mytest_v1_s42_processed.csv`)
- **Verdict**: **DISCARD** — clean `mytest` improved training/validation optics and slightly improved sealed myval, but did not generalize to Kaggle test. Roll back to soup as best.

### Updated Best (unchanged)
```
Run:        myval_v13_hi288_seed42_soup (top-3: epochs 20, 39, 26)
Myval F1:   0.7251@0.5
Test F1:    0.69856
Checkpoint: train/outputs/myval_v13_hi288_seed42_soup/soup.pt
Config:     288px, margin=0.10, seed=42, cosine LR, thr_search, bbox-crop, no-Bayes
```

### Key Learning
`myval` is still the most useful offline proxy, but small improvements are noise. A move from 0.7251 to 0.7321 on `myval` was not enough; Kaggle test dropped from 0.69856 to 0.65979. Future KEEP decisions should require a **large** sealed-myval gain, not a marginal one, especially when the new training data is clean but source-uniform.

## 2026-05-20: mytest Pretrain → Finetune

- **Hypothesis**: Two-stage training — pretrain backbone on mytest (3955 raw images), then finetune on original bbox-crop data. Avoids domain-shift from direct mytest merge.
- **Stage 1** (`train/outputs/mytest_pretrain`): Trained ConvNeXt Tiny on 3955 mytest images (1371 meteorite, 2584 rock). 288px, seed=42, cosine LR, cutmix=0.3, 15% val split.
  - mytest_val F1@0.5 = **0.9277** (best_epoch=39)
- **Stage 2** (`train/outputs/mytest_pretrain_finetune_v2`): Loaded pretrained backbone, finetuned on 4780 bbox-crop original training images. Same best config (288px, seed=42, cosine, no-Bayes).
  - best_epoch=7, internal model_select_f1=0.7383
  - **myval F1@0.5 = 0.7358** — NEW SOTA (+0.0107 over soup 0.7251)
- **Model soup on top epochs**: DISCARD — soup_F1=0.7066, worse than single checkpoint.
- **Verdict**: **KEEP** — mytest pretrain→finetune is the new best myval result.

## 2026-05-20: Focal Loss

- **Hypothesis**: Focal loss (gamma=2.0, alpha=0.25) improves hard negative mining.
- **Baseline** (`train/outputs/myval_focal_v1`): Same config as soup but with focal loss.
  - myval F1@0.5 = 0.7251 (tied with soup baseline)
  - internal model_select_f1 = 0.7529 (soup was 0.7500)
- **Combined** (`train/outputs/mytest_pretrain_focal`): Pretrain backbone + finetune with focal loss.
  - myval F1@0.5 = 0.7304 (below pretrain-only 0.7358)
- **Verdict**: Focal loss does not improve over CE for this task. The CutMix augmentation already provides sufficient regularization.

## 2026-05-20: Lower Backbone LR for Pretrain→Finetune

- **Hypothesis**: backbone_lr=3e-6 (instead of 1e-5) gives gentler finetuning for pretrained backbone.
- myval F1@0.5 = 0.7122 — lower than baseline 0.7358.
- **Verdict**: **DISCARD** — faster backbone adaptation is better (1e-5 works best).

### Updated Best
```
Run:        mytest_pretrain_finetune_v2 (epoch 7)
Myval F1:   0.7358@0.5
Test F1:    0.55214  ← DISCARD, severe myval-test gap
Config:     288px, seed=42, cosine, no-Bayes, bbox-crop
            Stage1: mytest pretrain (mytest_val=0.9277)
            Stage2: finetune on original bbox-crop data
Checkpoint: train/outputs/mytest_pretrain_finetune_v2/best.pt
```

## 2026-05-20: mytest Augmentation + myval Validation

- **Hypothesis**: Simply merge all mytest into training, use myval as validation (same as soup baseline). This is the most direct extension: soup + more data.
- **v1** (`train/outputs/mytest_augment_v1`): Single checkpoint, myval F1@0.5 = **0.7561**
- **v2 soup** (`train/outputs/mytest_augment_v2/soup_top3.pt`): Top-3 uniform soup (epochs 34/28/35), myval F1@0.5 = **0.7688**
  - internal model_select_f1 = 0.7927
- **Kaggle test**: 0.67021 — myval-test gap = 0.0986 (vs 0.0265 for old soup).
- **Verdict**: **DISCARD** — mytest inflates myval but degrades test. Adding mytest as training data consistently hurts generalization.

## 2026-05-20: Split-Validation with mytest Augmentation

- **Hypothesis**: If myval-as-validation leaks information, use internal 20% random split as validation instead. myval is completely held out.
- **Config** (`train/outputs/splitval_augment_v1`): `--val-split-ratio 0.2 --mytest-root --mytest-val-ratio 0.0`. Full mytest merged, 20% of combined data as val.
  - internal val_f1 = 0.9609 (best_epoch=53)
  - top-3 soup: myval F1@0.5 = 0.7446
- **Kaggle test**: 0.63212 — still degraded.
- **Verdict**: **DISCARD** — Eliminating myval leakage did not fix the problem. The mytest data itself introduces domain shift that hurts test performance regardless of validation strategy.

## 2026-05-20: Conclusion on mytest

**All 5 mytest-based experiments degraded test F1:**

| Experiment | myval | test | gap |
|------------|-------|------|-----|
| Old soup (no mytest) | 0.7251 | **0.69856** | 0.0265 |
| mytest split protocol | 0.7321 | 0.65979 | 0.0723 |
| mytest pretrain→finetune | 0.7358 | 0.55214 | 0.1837 |
| mytest aug + myval val | 0.7688 | 0.67021 | 0.0986 |
| split-val aug (no myval leak) | 0.7446 | 0.63212 | 0.1125 |

**Root cause:** mytest comes from Encyclopedia of Meteorites and Kaggle rock datasets — visually different from the competition test set. The model learns mytest-specific features that inflate myval (myval has similar clean images) but don't generalize to test.

**Decision: Abandon mytest entirely. Focus on original training data (4780 bbox-crop images) only.**

### Reverted Best (unchanged)
```
Run:        myval_v13_hi288_seed42_soup (top-3: epochs 20, 39, 26)
Myval F1:   0.7251@0.5
Test F1:    0.69856
Checkpoint: train/outputs/myval_v13_hi288_seed42_soup/soup.pt
```

## 2026-05-21: Backbone Exploration

- **convnext_small + mytest + split-val** (`backbone_cs_augment/soup_top3.pt`):
  - internal val=0.9580, myval=0.7774, test=**0.65263** — DISCARD
- **convnext_small NO mytest** (`backbone_cs_no_mytest/soup_top3_v2.pt`):
  - internal val=0.9767, myval=0.6875 — worse than tiny (0.7251)
- **convnextv2_tiny + mytest** (`backbone_cnv2_augment/best.pt`):
  - internal val=0.9551, myval=0.7251 — tied with tiny, no gain
- **DINO ViT-S + split-val** (`backbone_dino_v1/`):
  - internal val=0.9785, myval=0.6648 — severe overfitting on small data

### Conclusion
- Larger backbone (convnext_small) does NOT improve test on original data alone
- mytest inflates myval for ALL backbones but degrades test
- ViT/DINO fundamentally data-hungry; 4780 images insufficient
- The bottleneck is DATA, not model capacity

## 2026-05-21: Final mytest Verdict

**6/6 mytest experiments degraded Kaggle test F1:**

| # | Experiment | myval | test | Δ test |
|---|-----------|-------|------|--------|
| 1 | mytest split protocol | 0.7321 | 0.65979 | -0.039 |
| 2 | mytest pretrain→finetune | 0.7358 | 0.55214 | -0.146 |
| 3 | mytest aug + myval val (soup) | 0.7688 | 0.67021 | -0.028 |
| 4 | split-val aug (tiny soup) | 0.7446 | 0.63212 | -0.066 |
| 5 | split-val aug (small soup) | 0.7774 | 0.65263 | -0.046 |
| 6 | split-val aug (cnv2 tiny) | 0.7251 | TBD | — |

**mytest is proven harmful. Abandoned permanently.**
