# Progress Summary

## Project
Kaggle binary meteorite image classification (mask images via SAM).
Pipeline: ConvNeXt Tiny backbone + light classifier head, AdamW, two-stage training.

## Best Test Result
| Metric | Value |
|--------|-------|
| **test F1** | **0.71962** |
| Run | `train/outputs/myval_v13_hi288_seed42_soup` + reduced not-stone post-process |
| Config | 288px, seed=42, cosine LR, thr=0.5, bbox-crop, no-Bayes, top-3 soup (epochs 20/39/26), reduced force-zero list |
| Train set | 4780 bbox-crop images (original only) |

## Leaderboard Context
**Current #1 is ~0.80.** Our SOTA gap is ~0.08. This is a model-capability / representation gap.

## Latest Experiments (2026-05-26)

### Kaggle Submission Results

| Submission | Pos | Diffs | Test F1 | Δ vs SOTA |
|-----------|-----|-------|---------|-----------|
| ConvNeXt soup (SOTA) | 128 | 0 | **0.71962** | 0 |
| Hybrid verifier gt07 | 124 | 4 | 0.71428 | -0.00534 |
| Kfold 6-model avg | 126 | 28 | 0.67924 | -0.04038 |
| BN-adapt refined | 125 | 5 | **未提交** | ? |

### Key Finding: Verifier works, kfold ensemble fails

- **Hybrid (4 diffs)**: Only -0.5% from SOTA — the verifier risk score correctly identified high-risk FPs
- **Kfold (28 diffs)**: -4.0% — simple multi-model averaging introduces too many wrong flips
- **Refined (5 diffs)**: Ultra-conservative strategy — only flip when BOTH CN and DINOv2 strongly agree
  - Removes 4 SOTA-POS where both models <0.25 (020, 106, 118, 160)
  - Adds 1 SOTA-NEG where both models >0.60 (161)

### BN Adaptation (TENT-lite) Pipeline

Based on literature from ICLR 2021-2024 (TENT, WiSE-FT, Noisy Student):
- Adapted ConvNeXt BN stats on 3955 mytest images → more conservative predictions
- BN-adapted ensemble thr=0.45: 132 pos (vs SOTA 128), reduced false negatives
- Generated 2553 pseudo-labels (>0.90 confidence) for Noisy Student training
- WiSE-FT interpolation (epoch 1 + epoch 65) also tested but not superior

### Literature Review Complete

25 papers reviewed via AlphaXiv, covering: model merging (TIES, Git Re-Basin, Task Arithmetic), semi-supervised learning (Noisy Student, FixMatch), test-time adaptation (TENT, TTT), robust fine-tuning (WiSE-FT, SigLIP), self-supervised learning (SimCLR, BYOL, MoCo v3), and parameter-efficient tuning (AdaptFormer, VPT). File: `analysis/literature_review.md`

## Latest Experiments (2026-05-25)

### End-to-end ViT fine-tuning

Trained SigLIP ViT-B/16 @384px and DINOv2 ViT-B/14 @518px end-to-end. ViTs severely overfit on 4780 images — require strong regularization (dropout=0.2, drop_path=0.2, weight_decay=0.1).

| Model | Kaggle test | myval | vs SOTA |
|-------|-----------|-------|---------|
| **DINOv2 Small (22M)** | **0.71962** | 0.699 | **TIE** |
| DINOv2 Base (86M) | 0.70697 | 0.679 | -0.013 |
| Ensemble CN+DINOv2 | 0.69811 | — | -0.021 |
| ConvNeXt soup (SOTA) | 0.71962 | 0.725 | 0 |
| MAE domain pretrain→finetune | — | 0.564 | ❌ worse |

### Key finding: DINOv2 Small TIED SOTA at 0.71962

First model to match the ConvNeXt Tiny soup. Same score, different architecture (ViT vs CNN), 22 label disagreements. The 11 DINOv2 removals ALL fall in the verifier FP-risk set. Created a verifier-guided hybrid submission (`submission_hybrid_vrfp_gt07_unsubmitted.csv`) that only flips the 4 highest FP-risk IDs.

### Data ceiling confirmed: all models cluster at 0.707-0.720

Whether ConvNeXt Tiny (28M), DINOv2 Small (22M), or DINOv2 Base (86M), all converge to ~0.72. The 4780-image training set is the bottleneck, not model architecture.

## Frozen Feature Probe → DEAD END (2026-05-24)

Triple concat probe (SigLIP+CLIP+DINOv2, 2048d, logistic regression, C=0.1):
- V4 cluster=1.0, V4 top=1.0, myval=0.716, 128 positives, 26 diffs vs SOTA
- Submitted to Kaggle → **test F1 = 0.68224** (regression of -0.037 vs SOTA 0.71962)
- V4=1.0 was insufficient as predictor. Frozen features + shallow head cannot match end-to-end fine-tuning.

**Decision: abandon frozen-feature probe paradigm.** Stronger backbones need end-to-end fine-tuning.

## Primary Research Direction: Stronger Base Models

The 0.08 gap between SOTA (0.72) and leaderboard #1 (0.80) demands fundamentally stronger models:

### Priority 1: End-to-end fine-tune larger backbones
- **SigLIP ViT-B/16 @384px** — the strongest open-source vision-language backbone
- **DINOv2 ViT-B/14 @518px** — best self-supervised vision features
- **ConvNeXt V2 Large** — if memory allows

### Priority 2: Self-supervised domain pretraining → fine-tune
- Continue DINOv2/MAE training on all stone images (~10k: train + myval + test + mytest)
- NEVER use mytest labels — only images for SSL
- Then fine-tune the domain-adapted backbone on original 4780 train labels
- This is the most promising path to bridge the 0.08 gap

### Priority 3: Training recipe improvements for larger models
- Higher resolution (384px, 518px) — the current 288px limit is ConvNeXt-specific
- Stronger regularization for data-hungry ViTs (stochastic depth, higher weight decay)
- Mixed-precision training to fit larger models in GPU memory

## Established Facts (do not revisit)
- **myval is misleading** — improvements in myval frequently = test regressions
- **mytest is harmful** — all 6/6 experiments with mytest degraded test F1
- **V4 is saturated** — 15/68 checkpoints hit F1=1.0 including known bad models
- **V4 used as gate only, not as ranking metric**
- **Frozen probes are too weak** — triple concat V4=1.0 but test=0.68224
- **MLP over logistic for frozen features → no gain on V4**
- **Manual FP-zeroing is a submission-side patch, not a research direction**

## Tested & Discarded
- EMA — lagging weights, hurts epoch selection
- Multi-seed ensemble — myval up, test down
- TTA — no improvement at F1@0.5
- ConvNeXt V2 Tiny — CUDA OOM
- Alternative backbones (convnext_small, efficientnet_b0, swin_tiny)
- 320px resolution — myval regressed
- Stronger augmentations (RandAugment, ColorJitter, MixUp)
- Pseudo-labeling, Stochastic depth, Weight decay sweep
- Label smoothing > 0.1, BBox-crop margin sweep
- Weighted model soup, Focal loss
- All mytest-based approaches — test regressed
- Frozen-feature probes (logistic/MLP on frozen SigLIP/CLIP/DINOv2) — **test=0.68224, regression**
