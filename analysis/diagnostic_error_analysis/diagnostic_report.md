# Error Diagnostic Report — soup_reduced_notstone

Generated: 2026-05-26

## Executive Summary

The SOTA model (ConvNeXt Tiny, 288px, top-3 soup) has a **severe false positive problem**:
- **Test prediction: 132/194 positive (68%)** — far above expected ~20% (Phase 1 was 4:1 neg:pos)
- **Myval FP rate: 39.2%** (73/186 negatives classified as meteorite)
- **Myval FN rate: 15.1%** (22/146 positives missed)
- **35/73 FPs have probability > 0.9** — model is confidently wrong, not a calibration issue

The dominant error mode is **False Positives ≈ 3.3× False Negatives**.

## Per-Dataset Metrics

| Dataset | N | Pos | Neg | Ratio | F1@0.5 | Precision | Recall | FP | FN | Best Thr | Best F1 |
|---------|---|-----|-----|-------|--------|-----------|--------|----|----|----------|---------|
| myval_masked | 332 | 146 | 186 | 1.27:1 | 0.7230 | 0.6294 | 0.8493 | 73 | 22 | 0.5366 | 0.7251 |
| testlike_cluster_dino_v4 | 160 | 79 | 81 | 1.03:1 | 0.9937 | 0.9875 | 1.0 | 1 | 0 | 0.5 | 0.9937 |
| testlike_top_dino_v4 | 160 | 80 | 80 | 1.0:1 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0.5 | 1.0 |

## Testlike V4 Proxy Status

**SATURATED** — SOTA achieves F1=0.99-1.0. Cannot discriminate between top models.

Rank correlation with known test F1 (from 7 historical runs):
| Dataset | Metric | Spearman ρ |
|---------|--------|-----------|
| testlike_top_dino_v4 | f1_at_0_5 | **+0.9411** |
| testlike_cluster_dino_v4 | f1_at_0_5 | +0.7945 |
| myval_masked | f1_at_0_5 | **-0.5798** (NEGATIVE!) |
| myval_masked | prob_mean | +0.8117 |

The V4 proxy is discriminative across runs but saturated for SOTA. **A harder proxy is needed** (V5?).

## False Positive Patterns (myval)

### Confident FPs (prob > 0.9, 35 images)
```
r9.jpg(0.949), r19.jpg(0.905), r23.jpg(0.964), h-r1.jpg(0.952), h-r8.jpg(0.936),
h-r17.jpg(0.917), h-r20.jpg(0.917), h-r24.jpg(0.909), img_387c6648b726.jpg(0.971),
img_7aaffe316d71.jpg(0.971), img_4be44f9503bf.jpg(0.968), img_7065809558f6.jpg(0.961),
img_972eeed38df4.jpg(0.961), img_ff48806f69bd.jpg(0.960), img_433bc7e52199.jpg(0.959),
img_ace60e7a5459.jpg(0.952), img_f001b57acc0e.jpg(0.946), img_c28047c4cecb.jpg(0.946),
img_8b389411f514.jpg(0.940), img_5cae17bfc1b6.jpg(0.939), img_542408493187.jpg(0.938),
img_879be686ed6b.jpg(0.935), img_fde66b4a3083.jpg(0.929), img_9051316f36eb.jpg(0.928),
img_b9af7f02298d.jpg(0.927), img_a9f285aca1f4.jpg(0.927), img_fbdd399ca99b.jpg(0.926),
img_0d21b451c31c.jpg(0.924), img_772ca62c6f03.jpg(0.921), img_e7e7a1fd9eaa.jpg(0.920),
img_bc77811e3b43.jpg(0.918), img_5f63d1ce3fe8.jpg(0.914), img_f078c7f64a83.jpg(0.911),
img_06a0b9cf01f0.jpg(0.911), img_2b1b1c1c168a.jpg(0.903)
```

### Confident FNs (prob < 0.1, 7 images)
```
m4.jpg(0.065), img_27f58d8258af.jpg(0.042), img_fcd7781dfeea.jpg(0.044),
img_d000e4b65308.png(0.050), img_70407ec96783.jpg(0.052), img_462e23e2e18a.jpg(0.068),
img_9c7682e76aeb.jpg(0.077)
```

### FP Probability Distribution
| Range | Count |
|-------|-------|
| > 0.9 | 35 |
| 0.7-0.9 | 27 |
| 0.5-0.7 | 11 |
| Total | 73 |

## Not-Stone Post-Process Status

14 original candidate IDs → Reduced to 5 kept as zero (44, 100, 145, 162, 187).

Of the 9 removed from force-zero list:
- **18, 48, 154, 185**: 5/5 models agree positive — correctly removed
- **133**: 2/5 models positive, soup_prob=0.918 — ambiguous, but plausible meteorite
- **67**: 3/5 positive, soup_prob=0.861 — borderline
- **72**: 3/5 positive, soup_prob=0.562 — uncertain, borderline
- **23**: 1/5 positive, soup_prob=0.776 — only 1/5 models says positive, suspicious
- **159**: 1/5 positive, soup_prob=0.771 — only 1/5 says positive, suspicious

**ID 000162** (soup_prob=0.823, 4/5 votes positive) is still force-zeroed despite strong multi-model consensus for positive. May be a mistake.

## Core Weaknesses

1. **Over-prediction of meteorite class** — 68% positive on test vs ~20% expected
2. **39% FP rate on myval** — model lacks discrimination against rocks
3. **V4 proxy saturation** — cannot guide further improvement at SOTA level
4. **Hard negatives in training data** — model learns to over-generalize meteorite-like patterns
5. **Conservative not-stone list** — only 5/14 candidate IDs kept as zero

## Recommended Next Directions

1. **K-fold bagging** — code ready (`run_kfold_bagging.py`), expected +0.003-0.008
2. **Hard negative mining** — identify training rocks that look like meteorites, clean or augment
3. **Not-stone list expansion** — audit more test IDs, especially those near decision boundary
4. **Better threshold / Bayes correction** — with target ratio estimated from Phase 1 (~4:1)
5. **V5 proxy** — build harder testlike set that can discriminate SOTA models
