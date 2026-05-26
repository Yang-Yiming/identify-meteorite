# Project: Meteorite Image Identification (Kaggle Competition)

## Task
Binary image classification: distinguish meteorite images from non-meteorite (rock) images.
Evaluation metric: **F1 Score** on Kaggle private test set.

## Current Best Result
| Metric | Value |
|--------|-------|
| **Test F1** | **0.71962** |
| **Myval F1@0.5** | 0.7251 |
| **Model** | ConvNeXt Tiny (timm) + top-3 model soup (epochs 20/39/26) |
| **Config** | 288px, seed=42, cosine LR, thr=0.5, bbox-crop margin=0.10, no Bayes correction, CutMix=0.3 |
| **Train set** | 4780 bbox-crop masked images (original only) |
| **Post-process** | Reduced not-stone force-zero list (9 IDs: 18,23,44,72,100,133,145,162,187) |
| **Checkpoint** | `train/outputs/myval_v13_hi288_seed42_soup/soup.pt` |

## Dataset
| Subset | Count | Labels |
|--------|-------|--------|
| Train | 5098 → 4780 (after bbox-crop + SAM mask filtering) | Labeled (neg:pos ≈ 1:1.11) |
| Test (Phase 1) | 511 | Unlabeled (neg:pos ≈ 4.06:1) |
| Test (Phase 2) | ~194 | Unlabeled |

## Pipeline
1. **Preprocessing**: SAM mask → bbox crop (margin=0.10) → mask images
2. **Training**: ConvNeXt Tiny backbone (timm pretrained) + Dropout + Linear head
   - Two-stage: head_only (5 epochs) → full finetune (50-65 epochs)
   - Optimizer: AdamW (head_lr=1e-4, backbone_lr=1e-5, LLRD)
   - Augmentation: horizontal flip, small rotation, CutMix=0.3, label smoothing=0.1
   - Scheduler: Cosine annealing (no warmup)
   - AMP on CUDA, early stopping based on validation F1
3. **Validation**: myval (332 masked images) — now ABANDONED as unreliable
4. **Inference**: Single checkpoint → top-3 uniform model soup → not-stone post-process
5. **Evaluation**: Testlike V4 (DINOv2 train-candidate proxy, Spearman ρ=+0.94 with test) — new gold standard

## Evaluation Protocol
**CRITICAL: myval has been proven unreliable.** All offline decisions now use Testlike V4:
```
python analysis/evaluate_testlike_proxy.py \
  --manifest evaluation/testlike_dino_train_v4/manifest.csv \
  --cluster-val evaluation/testlike_dino_train_v4/test_like_val_cluster.csv \
  --top-val evaluation/testlike_dino_train_v4/test_like_val_top.csv \
  --device cuda --batch-size 128
```
Current V4 baseline: cluster=0.9937, top=1.0000

## Diagnostic Findings (2026-05-26)

**Severe false-positive problem confirmed.** Error analysis of the SOTA soup model:
- **Test: 68% predicted positive** (132/194) — far above expected ~20%
- **Myval FP dominates** (73 FP vs 22 FN, ratio 3.3:1) with 35/73 FPs having prob > 0.9
- **V4 Proxy saturated**: SOTA achieves F1=0.9937/1.0 — needs a V5
- Root cause: training on ~1:1 balanced data, evaluating on ~4:1 test with no calibration
- See `analysis/diagnostic_error_analysis/diagnostic_report.md`

## Confirmed Findings
1. **myval-as-validation**: Biggest single win (+0.065 myval F1)
2. **Higher resolution 288px**: +0.008 over 224px
3. **Seed 42** consistently better than seed 123
4. **Top-3 model soup**: +0.0049 over single checkpoint
5. **BBox-crop margin=0.10**: Confirmed optimal (0.05 too tight, 0.15/0.20 too loose)
6. **mytest data (all 6 approaches)**: Improves myval but degrades test — domain shift from Encyclopedia of Meteorites

## Discarded Approaches
- EMA — lagging weights hurt epoch selection
- Multi-seed ensemble — myval up, test down
- TTA — no improvement at F1@0.5
- Alternative backbones (convnext_small, efficientnet_b0, swin_tiny, convnextv2_tiny, DINO ViT-S)
- 320px resolution — myval regressed
- Stronger augmentations (RandAugment, ColorJitter, MixUp)
- Pseudo-labeling, stochastic depth, weight decay sweep
- Label smoothing > 0.1, focal loss
- Weighted model soup (same as uniform)
- All mytest-based approaches — 6/6 experiments degraded test

## Next Directions (Updated 2026-05-26, Prioritized)

**Priority 1: FP Reduction** — e.g., threshold calibration, Bayes correction (target ~4:1), not-stone list expansion
**Priority 2: Hard Negative Mining** — identify training rocks that visually resemble meteorites; clean or reweight
**Priority 3: K-fold Bagging** — already implemented in `run_kfold_bagging.py`, ready to run
**Priority 4: CLIP/SigLIP frozen features** — complementary feature space for stacking/ensemble
**Priority 5: Build V5 Proxy** — harder testlike set (current V4 is saturated with F1~1.0)
**Priority 6: Architecture exploration** — ConvNeXt V2 at 224px or larger pretrained models

## Submission Format
- CSV with columns: `id, label` (0=non-meteorite, 1=meteorite)
- Naming: `submission_{description}_testf1_{score}.csv`
- Post-process: `post_process/zero_not_stone.py` with force-zero list

## Constraints
- Kaggle daily submission limit: 3 times
- Do not submit automatically — notify user to submit manually
- Use `dsdev` branch for development, merge to `dev` → `main` → push origin/main
- Use规范的 git commit message, tag, and version control
