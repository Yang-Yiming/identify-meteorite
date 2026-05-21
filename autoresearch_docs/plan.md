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

1. **Build a test-like validation set from embeddings** — highest priority.
   The current myval proxy is fragile, especially when mytest is involved. Use
   frozen embeddings to estimate which labeled samples resemble the Kaggle test
   distribution, then use that subset as a better offline model-selection proxy.
   Candidate pool should include original train, myval, and mytest images, but
   mytest labels should be treated cautiously because supervised mytest training
   has repeatedly hurt test F1.

   Proposed implementation:
   - Extract embeddings for `preprocess/bbox_crop/{train,myval,test}` and
     `mytest/{meteorite,rock}` with several frozen feature families:
     DINOv2/DINOv3, CLIP/SigLIP, and the current ConvNeXt best checkpoint
     penultimate feature.
   - Build the test anchor set from Kaggle test images **excluding**
     IDs in `post_process/not-stone.txt`, because these hand-marked obvious
     poison samples should not define the normal target distribution.
   - Score each labeled candidate by test-likeness, e.g. top-k cosine similarity
     to test anchors, distance to the test centroid, and/or test-neighbor ratio
     in a kNN graph.
   - Cluster candidate + test embeddings, identify clusters with high test
     density, and sample `test_like_val.csv` stratified by label and cluster.
   - Re-evaluate existing runs on this proxy first: old soup, mytest augment,
     mytest pretrain, multi-seed ensemble, focal loss, and pseudo-label runs.
     A useful proxy should rank the known bad mytest-heavy submissions below
     the old soup, matching Kaggle test behavior better than myval does.

   Suggested artifacts:
   - `analysis/testlike/test_like_scores.csv`
   - `analysis/testlike/test_like_val.csv`
   - `analysis/testlike/cluster_summary.csv`

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
