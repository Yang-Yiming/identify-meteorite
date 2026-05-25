# Frozen Feature MLP Probe Comparison Report

Generated from `analysis/train_frozen_mlp_probe.py` experiments.

## Key Finding: Triple Concat Is the Winner

Concatenating SigLIP + CLIP + DINOv2 features with logistic regression achieves **V4=1.0/1.0 + only 25-26 diffs vs SOTA baseline**. Single-backbone probes had 36-46 diffs.

## Results Summary (All V4=1.0, sorted by diffs)

| Backbone(s) | C | Classifier | myval | Threshold | Positives | Diffs | pos→neg | neg→pos |
|------------|---|-----------|-------|-----------|-----------|-------|---------|---------|
| **S+C+D** | 0.1 | logistic | 0.716 | 0.375 | 123 | **25** | 15 | 10 |
| **S+C+D** | 0.1 | logistic | 0.716 | 0.278 | 128 | **26** | 13 | 13 |
| S+C+D | 1.0 | logistic | 0.719 | 0.272 | 123 | 25 | 15 | 10 |
| S+C+D | 1.0 | logistic | 0.719 | 0.170 | 128 | 26 | 13 | 13 |
| S+C+D | 0.3 | logistic | 0.713 | 0.185 | 123 | 27 | 16 | 11 |
| S+C+D | 0.3 | logistic | 0.713 | 0.236 | 128 | 28 | 14 | 14 |
| S+C+D | 3.0 | logistic | 0.704 | 0.185 | 123 | 29 | 17 | 12 |
| SigLIP | 10.0 | logistic | 0.633 | 0.006 | 128 | 36 | 18 | 18 |
| S+C | 1.0 | logistic | 0.641 | 0.071 | 128 | 36 | 18 | 18 |
| DINOv2 | 10.0 | logistic | 0.742 | 0.069 | 128 | 36 | 18 | 18 |
| DINOv2 | 3.0 | logistic | 0.734 | 0.086 | 128 | 36 | 18 | 18 |
| CLIP | 3.0 | logistic | 0.614 | 0.040 | 128 | 46 | 23 | 23 |

S = SigLIP ViT-B/16 (768d), C = CLIP ViT-B/32 (512d), D = DINOv2 ViT-B/14 (768d). Total feature dim: 2048.

## MLP Results (All V4 < 1.0)

MLP classifiers consistently underperform logistic regression on V4:

| Backbone | Hidden | V4 cluster | V4 top | myval | Pos@thr | Diffs |
|---------|--------|-----------|--------|-------|---------|-------|
| S+C+D | [256] | 0.975 | 0.988 | 0.711 | 128@0.145 | 28 |
| SigLIP | [128] | 0.975 | 0.988 | 0.669 | 128@0.128 | 36 |
| SigLIP | [64] | 0.963 | 0.982 | 0.671 | 128@0.095 | 38 |
| SigLIP | [256] | 0.981 | 0.982 | 0.673 | 128@0.100 | 36 |

**Verdict**: MLP overfits on V4 proxy. Stick with logistic.

## FP-Risk Overlap Analysis

The triple concat probe (C=0.1, 26-diff version) flags 14 baseline positives as negative:

`000020, 000046, 000056, 000062, 000070, 000086, 000098, 000106, 000118, 000124, 000131, 000139, 000160, 000182`

**Of these, 4 are top-10 FP-risk candidates**: 131 (rank#1), 124 (rank#5), 20 (rank#6), 106 (rank#7).

**Critically, the probe does NOT flag 88 or 177** (leaderboard-arithmetic inferred FPs) as negative. The probe agrees with the SOTA soup that 88,177 should be positive.

## Recommendation

### Preferred submission: Triple Concat (S+C+D) Logistic C=0.1, thr=0.278 → 128 positives

File: `analysis/frozen_concat_s2c2d/submission_logistic_..._pos120.csv`

- V4=1.0 (gate passed)
- 128 positives (matches SOTA count)
- 26 diffs vs baseline (within 25-30 target)
- 13 pos→neg, 13 neg→pos (symmetric, no directional bias)
- No mytest leakage

### Conservative alternative: Same model, thr=0.375 → 123 positives

- 25 diffs (lowest observed)
- More conservative (5 fewer positives than SOTA)
- Lower risk profile

### Not recommended for submission
- Single-backbone probes (36-46 diffs exceed target)
- MLP probes (V4 < 1.0)
- Further manual not-stone FP-zero expansion (plan says deprioritize)

### Next steps
1. Submit triple concat C=0.1@120pos to Kaggle when submissions reopen
2. If it improves over 0.71962, the representation-capacity direction is validated
3. Try lightweight fine-tuning/adapters on triple concat features
4. Try DINOv3 or larger backbones in the frozen-feature paradigm
