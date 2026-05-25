# Plan

## Objective

Maximize **test** F1 (Kaggle). Current SOTA: **0.71962**. Leaderboard #1: **~0.80**. Gap: **~0.08**.

This gap is a **model-capability / representation gap**, not fixable by manual FP-zeroing, post-processing, or shallow frozen-feature probes.

## Current State

**Best test F1: 0.71962** (ConvNeXt Tiny, 288px, seed=42, cosine LR, bbox-crop, top-3 soup, reduced not-stone post-process).

**Frozen-feature probes → DEAD END.** Triple concat (SigLIP+CLIP+DINOv2 + logistic regression) achieved V4=1.0/1.0 but scored test=0.68224 (-0.037 vs SOTA). The V4 proxy overfits to DINOv2 nearest-neighbor selection; models that rely heavily on DINOv2 features can "game" V4 without truly generalizing.

## Primary Research Direction: Stronger End-to-End Base Models

The 0.08 gap demands fundamentally stronger architectures AND end-to-end SGD training, not frozen features + shallow heads.

### Priority 1: End-to-end fine-tune SigLIP ViT-B/16

SigLIP ViT-B/16 is the strongest open-source vision-language backbone. It outperforms CLIP and DINOv2 on most fine-grained classification tasks.

```
Train a standard end-to-end classifier:
- Backbone: vit_base_patch16_siglip_224 (timm)
- Resolution: 384px (native for SigLIP)
- Training: standard 2-stage (head-only warmup → full fine-tune)
- Optimizer: AdamW, head_lr=1e-4, backbone_lr=1e-5
- Augmentation: CutMix=0.3, label_smoothing=0.1, dropout=0.1
- LR schedule: cosine, 50 epochs
- Data: 4780 original bbox-crop train images ONLY
- Validation: internal 20% split (NOT myval)
- Batch size: as large as GPU memory allows (target 64+)
- Mixed precision: bf16 or fp16
```

**Why this first:** SigLIP features already showed strong frozen performance (V4=1.0, lowest diffs as single backbone). End-to-end fine-tuning should unlock significantly more capacity.

### Priority 2: End-to-end fine-tune DINOv2 ViT-B/14

DINOv2 ViT-B/14 is the strongest self-supervised vision backbone. DINOv3 may be even better.

```
- Backbone: vit_base_patch14_dinov2.lvd142m or dinov3 variant
- Resolution: 518px (native for DINOv2)
- Same training recipe as Priority 1
- May need stronger regularization (stochastic depth, higher weight decay)
```

**Alternative:** Try DINOv3 variants if available in timm (convnext_tiny.dinov3_lvd1689m was tested earlier but regressed — may need different hyperparams).

### Priority 3: Self-supervised domain pretraining → fine-tune

This is the highest-potential direction but requires more compute.

**Approach:**
1. Collect all unlabeled stone images: train (4780) + myval (332) + test (194) + mytest (3955) ≈ 9261 images. NEVER use mytest labels.
2. Continue DINOv2 or MAE pretraining on these ~9k images for 50-200 epochs
3. Fine-tune the domain-adapted backbone on original 4780 train labels using the recipe from Priority 1
4. This injects domain knowledge (stone/mineral textures, lighting, backgrounds) without supervised mytest leakage

**Why high-potential:** The Kaggle test images share visual characteristics with mytest (both from Encyclopedia of Meteorites / Kaggle rock datasets). SSL on the combined pool adapts the backbone to this domain without using mytest labels. This is the most principled way to bridge the train→test domain gap.

### Priority 4: ConvNeXt V2 at feasible resolution

ConvNeXt V2 has better inductive biases for small datasets than ViT. V2 Tiny at 224px was tested earlier (discarded for OOM at 288px), but with mixed precision and gradient checkpointing, 288px or 384px may be feasible.

### Evaluation Protocol (for all experiments)

1. **V4 cluster/top F1** — gate check (must be ≥ 0.99). But V4 is NOT sufficient for ranking.
2. **Submission behavior**: positive count, diff vs current best (128 positives). Must be plausible.
3. **No mytest labels in training** — domain pretraining on images OK, supervised on labels NOT OK.
4. **Kaggle submission** is the only true metric. Do not trust offline proxies for final selection.

## Discarded: Will Not Revisit

### Frozen-feature probes
Triple concat V4=1.0 → test=0.68224. Dead end.
- Logistic regression on frozen features cannot match end-to-end SGD + augmentations.
- V4 overfits to DINOv2 proximity; DINOv2-heavy models can game V4.

### Manual FP-zeroing / not-stone expansion
Gap of 0.08 cannot be closed by zeroing a few test IDs. Keep `inferred_88_177` as a submission-side safety patch only (expected +0.007 if arithmetic holds).

### mytest as supervised data
6/6 experiments degraded test. Domain shift is real and harmful. Use mytest images for SSL pretraining only, never for supervised training.

### myval as proxy
Repeatedly misleading — myval↑ with test↓. Internal split validation (20% of train) is safer.

## Implementation Notes for New Training Script

The existing training pipeline (`train/`) uses ConvNeXt Tiny. To support ViT backbones:
- Replace `convnext_tiny` with timm model factory
- Use timm's `resolve_model_data_config` + `create_transform` for proper ViT preprocessing
- Support mixed precision (`torch.cuda.amp`)
- Support gradient checkpointing for memory
- Keep the proven recipe: 2-stage training, CutMix, cosine LR, no mytest

Script to create: `train/train_vit.py` or extend existing `train/train.py` to accept `--backbone` argument.
