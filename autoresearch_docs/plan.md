# Plan

## Objective

Maximize **test** F1 (Kaggle). BBox-crop preprocessing is the **default
pipeline**. Bayes correction is disabled (test distribution unknown).
Offline selection must now report both **myval F1@0.5** and the frozen-DINO
test-like diagnostic.

**CRITICAL UPDATE (2026-05-20): myval is an unreliable proxy when mytest
data is involved.** Adding mytest as training data inflates myval F1
(+0.02~+0.04) but degrades test F1 (-0.03~-0.15). See "mytest Generalization
Failure" below.

**CRITICAL UPDATE (2026-05-23): ABANDON myval as offline proxy. Use Testlike V4
(train candidates, DINOv2 embeddings, rank corr=+0.97) for ALL offline
judgment.** myval has been repeatedly shown to mislead — improvements in myval
frequently correspond to test regressions (dinov2 mlp: myval +0.023, test -0.010;
mytest augment: myval +0.044, test -0.028; etc).

The V4 diagnostic dataset is at:
`analysis/testlike_dino_train_v4/`
Evaluation script: `analysis/evaluate_testlike_proxy.py` (now defaults to V4).

## Current State

**Test SOTA: test_f1=0.71962** (soup checkpoint: top-3 epochs 20/39/26,
288px, seed=42, cosine, thr=0.5, 4780 original bbox-crop images, reduced
not-stone post-process).

| Run | myval F1@0.5 | test F1 | Description |
|-----|-------------|---------|-------------|
| soup + reduced not-stone | 0.7251 | **0.71962** | current SOTA |
| soup (old full not-stone) | 0.7251 | 0.69856 | old post-process |
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

**Decision: do not use mytest as trusted supervised data.** Filtered or
low-weight mytest experiments are allowed, but they must be judged by myval and
DINO diagnostics before any submission.

### Key Improvements Achieved

| Change | split_val | myval | test |
|--------|-----------|-------|------|
| Baseline (hlr04) | 0.7664 | — | 0.42 |
| + BBox-crop (bayes on) | 0.9444 | — | — |
| + No Bayes + thresh=0.5 | **0.9708** | 0.6379 | 0.64516 |
| + myval-as-validation + 288px + seed=42 + cosine | — | 0.7202 | — |
| + top-3 model soup | — | **0.7251** | 0.69856 |
| **+ reduced not-stone post-process** | — | **0.7251** | **0.71962** |

### Current Offline Comparison Protocol

For every new run, report at least:

1. post-hoc myval F1@0.5:

   ```bash
   python analysis/prob_dist.py \
     --checkpoint train/outputs/<run_name>/best.pt \
     --mask-dir preprocess/bbox_crop \
     --val-source myval \
     --no-plot \
     --device cuda \
     --batch-size 128
   ```

2. DINO test-like diagnostic:

   ```bash
   python analysis/evaluate_testlike_proxy.py \
     --manifest analysis/testlike_dino_myval_v3/manifest.csv \
     --cluster-val analysis/testlike_dino_myval_v3/test_like_val_cluster.csv \
     --top-val analysis/testlike_dino_myval_v3/test_like_val_top.csv \
     --dataset-prefix <run_tag> \
     --out-dir analysis/testlike_<run_tag>_eval \
     --device cuda \
     --batch-size 128 \
     --num-workers 4
   ```

Current soup baseline in the DINO diagnostic:

| run | myval_masked F1@0.5 | DINO cluster F1@0.5 | DINO top F1@0.5 |
|---|---:|---:|---:|
| `soup_reduced_notstone` | 0.7230 | 0.7709 | 0.8045 |

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

1. **Use DINO test-like diagnostics for every experiment.**
   `analysis/testlike_dino_myval_v3` is the current diagnostic set. It is not a
   perfect final judge, but it is better than hard train-heavy test-like lists
   and should be reported for every new checkpoint.

2. **Audit and optimize `post_process/not-stone.txt`** — tied to the test-like
   validation work. The current list has only 14 forced-zero test IDs and may
   contain mistakes. Treat it as a hypothesis, not ground truth.

   Proposed audit:
   - For each not-stone test ID, retrieve top-k nearest neighbors from
     train, myval, and mytest under DINO/CLIP/ConvNeXt embeddings.
   - Compare neighbor label ratios and source distribution. If an ID's nearest
     neighbors are mostly labeled meteorites, especially from multiple feature
     families, it should not be blindly forced to 0.
   - Aggregate existing model probabilities for each ID from the old soup and
     discarded runs. If many independent models assign high positive
     probability, mark the sample for manual review instead of automatic zeroing.
   - Check cluster membership. IDs in isolated non-stone/OCR/artifact clusters
     are stronger force-zero candidates than IDs embedded in the main test
     distribution.
   - Optionally use reverse image search manually for ambiguous IDs, especially
     if nearest neighbors suggest the same source site may have been crawled
     into mytest.

   Suggested artifacts:
   - `analysis/not_stone_audit/not_stone_neighbors.csv`
   - `analysis/not_stone_audit/not_stone_summary.csv`
   - revised candidate lists such as `not-stone.keep.txt`,
     `not-stone.remove.txt`, and `not-stone.review.txt`

3. K-fold bagging on original 4780 training data (no mytest) — already implemented in `run_kfold_bagging.py`
4. **Pseudo-label / consistency adaptation on test** — lower priority until the
   new proxy exists. Prefer soft labels and consistency loss over hard
   high-confidence pseudo-labels.
5. **Self-supervised DINO-style adaptation** — continue DINOv2/DINOv3 or similar
   on all unlabeled stone images, including train/myval/test/mytest, but avoid
   using mytest labels.
6. **CLIP/SigLIP frozen-feature baselines and stacking** — use large pretrained
   visual embeddings with logistic regression, kNN, SVM, or shallow stacking.
7. Architecture exploration — ConvNeXt V2 at 224px (no mytest, OOM at 288px)


## Long-Horizon / High-Variance Directions for Testlike V4

Current leaderboard submission work is paused because daily submissions are exhausted. For offline exploration, optimize Testlike V4 aggressively, while treating it as a proxy rather than ground truth. Every candidate should report V4 cluster/top F1, current-submission diff count, positive count, and whether it preserves the known Kaggle arithmetic from recent submissions.

Update after all-checkpoint sweep: V4 is already saturated by many historical checkpoints, including some known Kaggle regressions. Use V4 as a gate. Tie-break with positive count, current-best diff count, FP-risk arithmetic, multi-embedding agreement, and absence of mytest-supervised domain shift.

Update after tie-breaker report: V4-gated comparable non-baseline submissions are dominated by mytest-supervised variants with large behavior shifts. Prioritize a second-stage verifier on current-best positives and multi-embedding FP consensus over any further mytest-supervised model selection.

### A. V4-first model search

1. Re-score all saved checkpoints and soups on V4, then build a V4-selected ensemble/stacker rather than myval-selected ensemble. Avoid naive soft-voting; learn a small meta-rule from out-of-fold train predictions plus V4 candidates.
2. Train a second-stage verifier only on current-soup positive candidates. Objective: reduce FP among high-recall positives. Candidate features: soup prob, DINO MLP label, DINO kNN label ratios, CLIP/SigLIP scores, crop/mask metadata, V4 cluster/test-likeness.
3. Optimize threshold and per-cluster thresholds on V4, but constrain total positive count near the inferred hidden-test range. Record positive-count sensitivity.

### B. Multi-embedding FP/TP inference

1. Add SigLIP/CLIP embeddings to the DINO FP-risk audit. Use them mainly as semantic anomaly detectors, not as direct classifiers.
2. Build consensus kNN evidence across DINOv2, DINOv3, SigLIP, ConvNeXt penultimate features, and simple image statistics. Rank only samples where multiple embedding families agree.
3. Use pairwise submission arithmetic to infer labels for small candidate groups. Recent results imply 88 and 177 are strong FP candidates, while exactly one of 108,124,131 is likely FP.

### C. Self-supervised / domain adaptation

1. Continue DINOv2 or MAE-style SSL on all unlabeled stone crops: train + myval + test + mytest images, but never use mytest labels. Then train linear/MLP probes and small adapters on original train labels only.
2. Try parameter-efficient ViT adaptation: freeze backbone, train LoRA/adapters/LayerNorm affine on train labels, optionally with consistency loss on unlabeled test crops.
3. Try TENT-style test-time adaptation with only normalization/adapters updated. Reject candidates that increase positive count or collapse confidence.

### D. Data cleaning guided by V4

1. Identify train samples least similar to test distribution under V4/DINO and downweight or exclude them. Evaluate whether a smaller, more test-like supervised set improves V4.
2. Hard-negative mine from original train only: train negatives that are nearest to current test positives, plus high-probability false positives under OOF prediction.
3. Audit duplicated or contradictory train/myval images in embedding space. Remove or downweight suspicious labels before training verifier models.

### E. Submission policy after daily reset

The next most informative candidate is current best plus force-zero 88,177 only: analysis/test_fp_risk_audit_dino_nomtest/submission_inferred_zero_88_177.csv. If the arithmetic inference is exact, expected F1 is about 0.72642. Do not submit larger DINO FP-risk batches until this is tested.
