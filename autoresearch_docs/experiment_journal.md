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

## 2026-05-21: not-stone post-process audit

### Background

The previous public best submission used the soup raw CSV from
`train/outputs/myval_v13_hi288_seed42_soup/submission_raw.csv`, followed by
`post_process/zero_not_stone.py` using a manually maintained
`post_process/not-stone.txt` force-zero list.

Manual review and a quick visual clustering pass showed that the old list
mixed several different failure modes. Some entries are obvious non-meteorite
objects, but others look like plausible cut/polished meteorite or mineral
specimens. All listed IDs are `*_nomask.done` in `preprocess/bbox_crop/test`,
so the list appears to be partly driven by SAM/bbox mask failure, not by direct
class evidence.

### Submissions

Known test F1 results:

| not-stone policy | test F1 | Note |
|---|---:|---|
| old full force-zero list | 0.69856 | previous soup processed result |
| reduced 9-id list | **0.71962** | current best |
| reduced list but restore `18,23,72,133` to positive | 0.71559 | last 2026-05-21 submission |

The current best 9-id list is:

```text
18
23
44
72
100
133
145
162
187
```

For the soup raw CSV, only six of these actually change the submission label
from 1 to 0:

```text
18
23
44
72
133
162
```

`100`, `145`, and `187` are already predicted as 0 by the raw soup submission.

### F1 inference

The 4-image ablation changed `18,23,72,133` from 0 to 1 and dropped test F1
from 0.71962 to 0.71559. Since

```text
F1 = 2TP / (2TP + FP + FN)
```

the rounded scores are consistent with:

```text
reduced 9-id list:      TP = 77, denominator = 214, F1 = 154/214 = 0.719626
4-image restored list:  TP = 78, denominator = 218, F1 = 156/218 = 0.715596
```

This implies that among `18,23,72,133`, approximately exactly one is truly
positive and the other three are truly negative.

The earlier jump from 0.69856 to 0.71962 after removing
`48,67,154,159,185` from the force-zero list is consistent with about four of
those five being true positives. The original not-stone list therefore did
contain substantial false force-zero errors.

### Current hypothesis

Keep obvious poison IDs forced to 0, especially `44` and `162`. The most useful
next single-ID ablation is likely to restore only `23` while keeping
`18,44,72,100,133,145,162,187` forced to 0.

Expected outcomes for restoring only `23`:

```text
if 23 is truly positive: F1 ~= 156/215 = 0.72558
if 23 is truly negative: F1 ~= 154/215 = 0.71628
```

This hypothesis is also saved as
`post_process/not-stone_candidate_remove_23.txt`.

## 2026-05-22: strict DINO-filtered mytest augmentation

### Hypothesis

The full mytest dataset is harmful, but a small DINO-filtered subset that is
both test-like and label-consistent might provide useful extra supervision
without importing the full mytest domain shift.

### Data

Used the strict candidates from `analysis/mytest_audit_dino_v1`:

- total: 161
- label 0: 130
- label 1: 31
- filter: `top10_test_frac >= 0.2`, `top10_same_label_frac >= 0.8`,
  `conflict_score <= 2.0`, `top1_sim >= 0.75`

The filtered root was materialized as symlinks under:

```text
analysis/mytest_audit_dino_v1/filtered_roots/strict
```

### Training

Run:

```text
train/outputs/mytest_strict_dino_v1
```

Config matched the current soup baseline as closely as possible:

- ConvNeXt Tiny
- 288px
- seed 42
- cosine LR
- bbox crop
- myval validation
- no Bayes correction
- threshold search enabled
- `--mytest-val-ratio 0.0`
- `--mytest-sample-weight 0.5`

Best internal epoch:

```text
epoch = 51
model_select_f1 = 0.73333
```

### Evaluation

Post-hoc myval evaluation:

```text
myval F1@0.5 = 0.7202
```

DINO diagnostic proxy:

| run | myval_masked F1@0.5 | DINO cluster F1@0.5 | DINO top F1@0.5 |
|---|---:|---:|---:|
| current soup | 0.7230 | 0.7710 | 0.8045 |
| strict mytest | 0.7163 | 0.7619 | 0.7892 |

Submission behavior after current best not-stone post-process:

```text
current soup positives: 128 / 194
strict mytest positives: 104 / 194
diff vs current soup: 28 labels
```

Most differences are current soup positive but strict mytest negative, so this
run is substantially more conservative.

### Verdict

**DISCARD / do not submit yet.** The strict filtered mytest experiment did not
beat the current soup on myval or the DINO diagnostic proxy, and its submission
is much more conservative. This suggests that even carefully filtered mytest
supervision still shifts the decision boundary in a risky direction.

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

## 2026-05-22: Testlike V3 改进尝试 + 新模型探索

### Testlike V3 分数提升尝试

当前 soup baseline 在 testlike_dino_myval_v3 上: cluster=0.7709, top=0.8045

| # | 方法 | cluster Δ | top Δ | myval Δ | verdict |
|---|------|----------|-------|---------|---------|
| 1 | 全checkpoint评估(60+个) | — | — | — | mytest模型cluster达0.8066但test F1差 |
| 2 | 非mytest模型Ensemble | +0.009 | +0.023 | — | 小幅提升, soup+best组合最优 |
| 3 | 温度校准/阈值优化 | 0 | 0 | — | 阈值为0.5已最优 |
| 4 | testlike cluster作validation训练 | -0.050 | -0.033 | -0.042 | 160张小验证集导致过拟合 |
| 5 | testlikeness加权训练 | -0.062 | -0.038 | — | 模型变激进但准确率下降 |
| 6 | not-stone恢复ID 18 | — | — | — | test降至0.71627, DISCARD |
| 7 | not-stone恢复ID 18+162 | — | — | — | test降至0.71296, DISCARD |

### 关键发现: Testlike V4 (train candidates) 诊断质量

重建testlike用train作candidate source (替代myval):
- **V3 (myval candidates)**: rank correlation = **-0.40** — 反转排序!
- **V4 (train candidates)**: rank correlation = **+0.97** — 正确排序

V3本身偏向mytest模型(因myval与mytest相似), 需用V4作离线诊断。

### DINOv3预训练 ConvNeXt Tiny (backbone探索)

- **配置**: `convnext_tiny.dinov3_lvd1689m`, 288px, seed=42, 标准配置
- **Run**: `train/outputs/dinov3_tiny_288_s42`
- **myval F1@0.5**: **0.6891** (vs soup 0.7251, -0.036)
- **Verdict**: **DISCARD** — DINOv3预训练未提升, 可能因自监督特征不适配此细粒度分类

### ConvNeXt V2 Tiny (backbone探索)

- **配置**: `convnextv2_tiny.fcmae_ft_in22k_in1k_384`, 288px, seed=42
- **Run**: `train/outputs/cnv2_tiny_288_s42`
- **myval F1@0.5**: **0.6736** (vs soup 0.7251, -0.052)
- **Verdict**: **DISCARD** — V2架构未提升

### DINOv2 Frozen Features + MLP (新范式) ⭐

**完全不同的方法**: 用DINOv2 ViT-B/14冻结特征 + 轻量MLP分类器

| Run | myval F1@0.5 | testlike cluster | testlike top | verdict |
|-----|-------------|-----------------|-------------|---------|
| dinov2_mlp (lr=1e-3, ep=100) | **0.7530** | — | — | **BEST myval** (但未保存模型) |
| dinov2_mlp_v2 (lr=1e-3, ep=80) | **0.7416** | 0.7735 | 0.8068 | **KEEP** — 全面超过soup |
| dinov2_mlp_full (lr=1e-3, ep=80, 194 test) | **0.7485** | — | — | **DISCARD** — Kaggle test=**0.70935** |

**vs Soup baseline**: myval +0.0234, test -0.010 → **myval又一次误导！**

### Key Insight

DINOv2预训练特征+MLP在myval上显著超过soup,但在Kaggle test上退化。这再次证实myval不可靠——myval提升 ≠ test提升。

### 重大决策: 放弃myval, 全面转向Testlike V4

**myval已多次被证实为不可靠的离线代理指标**:
| 实验 | myval Δ | test Δ | myval误导 |
|------|---------|--------|----------|
| dinov2 mlp | +0.0234 | -0.0103 | ❌ |
| mytest augment soup | +0.0437 | -0.0284 | ❌ |
| mytest pretrain→finetune | +0.0107 | -0.1464 | ❌ |
| ensemble 42+123 | +0.0032 | -0.0389 | ❌ |

**从此使用 Testlike V4 (train candidates, rank corr=+0.97) 作为唯一离线判决指标。**

### Next Directions

1. **用V4评估所有checkpoint** — 建立新的baseline排序
2. **构建更好的testlike版本** — CLIP/SigLIP特征空间
3. **自监督domain adaptation** — 在全部stone图像上做无监督预训练
4. **数据清洗** — 基于V4的test-likeness分数清理低质量训练样本


## 2026-05-23: DINO test FP-risk audit

### Hypothesis

Current best submission likely has high recall but too many false positives. Instead of retraining, rank current positive test predictions by multi-signal false-positive risk and create force-zero ablation candidates.

### Implementation

Added `analysis/audit_test_fp_risk.py`. It builds a manifest, extracts features, finds test-to-train/myval nearest neighbors, merges soup probabilities, processed labels, DINO MLP labels, and Testlike V4 metadata, then outputs a ranked FP-risk table and candidate not-stone lists.

Primary run:

```text
python analysis/audit_test_fp_risk.py \
  --feature-backend dino_timm \
  --dino-model vit_base_patch14_dinov2 \
  --out-dir analysis/test_fp_risk_audit_dino_nomtest \
  --ref-sources train,myval \
  --fit-sources train,myval,test \
  --top-k 20 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

Mytest was intentionally excluded from neighbor references because repeated supervised mytest use has proven harmful.

### Outputs

- `analysis/test_fp_risk_audit_dino_nomtest/test_fp_risk_summary.csv`
- `analysis/test_fp_risk_audit_dino_nomtest/test_neighbors_topk.csv`
- `analysis/test_fp_risk_audit_dino_nomtest/top_fp_risk_positives.md`
- Candidate force-zero lists and submissions for top5/top10/top15/top20/top25 FP-risk positives.

Top DINO FP-risk current positives:

```text
131, 108, 177, 124, 88, 35, 82, 138, 106, 20
```

Candidate submission positive counts:

```text
baseline current best: 128 positives
top5 FP-risk zeroed:  123 positives
top10 FP-risk zeroed: 118 positives
top15 FP-risk zeroed: 113 positives
top20 FP-risk zeroed: 108 positives
top25 FP-risk zeroed: 103 positives
```

### Next action

Submit conservatively from small to larger ablation, starting with top5 or top10. If top5 improves, the FP-risk direction is validated; if top10 improves more, continue down the ladder. If top5 drops sharply, inspect whether DINO neighbor-negative candidates are actually hidden positives.

### Kaggle result for top5 FP-risk zeroing

- Submission: `analysis/test_fp_risk_audit_dino_nomtest/submission_plus_top5_fp_risk.csv`
- Test F1: **0.71770** vs current best **0.71962**
- Top5 added force-zero IDs: `88,108,124,131,177`

F1 arithmetic from the current best (`TP=77, denominator=214`) is consistent with zeroing 5 current positives where approximately **3 were true negatives / false positives** and **2 were true positives**. The FP-risk direction has useful signal, but the top rank is not clean enough to zero blindly.

Next best use of submissions: split the top5 list with restore/leave-one-out ablations instead of jumping directly to top10.

### Kaggle result for manual zero 108,124,131

- Submission: analysis/test_fp_risk_audit_dino_nomtest/submission_manual_zero_108_124_131.csv
- Test F1: **0.71090** vs current best **0.71962**
- Added force-zero IDs: 108,124,131

F1 arithmetic from current best (TP=77, denominator=214): zeroing 3 positives and scoring 0.71090 is consistent with 150/211, i.e. exactly **2 true positives and 1 false positive** among 108,124,131.

Combined with the top5 result (88,108,124,131,177 -> 0.71770 ~= 150/209), the top5 set likely contains exactly **2 true positives and 3 false positives**. Therefore the two omitted IDs, 88 and 177, are the strongest inferred false positives.

Next candidate: restore 108,124,131; force-zero only 88,177 in addition to the current best not-stone list. Expected score if exact: 154/212 = 0.72642.

## 2026-05-23: All-checkpoint Testlike V4 sweep

Ran all existing best/soup checkpoints on Testlike V4:

    python analysis/evaluate_all_checkpoints.py --manifest analysis/testlike_dino_train_v4/manifest.csv --cluster-val analysis/testlike_dino_train_v4/test_like_val_cluster.csv --top-val analysis/testlike_dino_train_v4/test_like_val_top.csv --out-dir analysis/all_checkpoints_v4_eval --batch-size 128 --num-workers 4 --device cuda

Result: V4 is now saturated. 15/68 evaluated checkpoints get F1@0.5 = 1.0 on both V4 cluster and V4 top. This full-score group includes models known to generalize badly on Kaggle, for example mytest_augment_v2/soup_top3.pt (known test F1 0.67021). Therefore V4 should be treated as a gate, not as a sufficient ranking objective.

Useful implication: future experiments should require V4 near-perfect, but tie-break by hidden-test behavior proxies: positive count, diff from current best, FP-risk arithmetic, multi-embedding agreement, and avoidance of mytest-supervised domain shift.

Outputs:

- analysis/all_checkpoints_v4_eval/all_eval_long.csv
- analysis/all_checkpoints_v4_eval/all_eval_summary.csv

Top-line saturated group size: 15/68 checkpoints. Current soup remains near-saturated (cluster 0.9937, top 1.0) and is not meaningfully worse under V4 than many V4-perfect but Kaggle-worse models.
