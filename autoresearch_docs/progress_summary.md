# Progress Summary

## Project
Kaggle binary meteorite image classification (mask images via SAM).
Pipeline: ConvNeXt Tiny backbone + light classifier head, AdamW, two-stage training.

## Best Test Result
| Metric | Value |
|--------|-------|
| **test F1** | **0.71962** |
| **myval F1@0.5** | **0.7251** |
| Run | `train/outputs/myval_v13_hi288_seed42_soup` + reduced not-stone post-process |
| Config | 288px, seed=42, cosine LR, thr=0.5, bbox-crop, no-Bayes, top-3 soup (epochs 20/39/26), reduced force-zero list |
| Train set | 4780 bbox-crop images (original only) |

## not-stone Post-process Update

The previous full `post_process/not-stone.txt` force-zero list was too
aggressive. Reducing the list improved test F1 from 0.69856 to **0.71962**.

Current best force-zero list:

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

A 4-image ablation restored `18,23,72,133` to positive and scored 0.71559.
The F1 arithmetic implies roughly one of those four is truly positive and
three are truly negative. The next best single-ID candidate is likely restoring
only `23`, while keeping `18,44,72,100,133,145,162,187` forced to 0.

## mytest Generalization Failure

ALL attempts to incorporate mytest as training data improved myval F1 but degraded test F1:

| Experiment | myval F1@0.5 | test F1 | myval-test gap | Note |
|------------|-------------|---------|----------------|------|
| Old soup (baseline) | 0.7251 | **0.69856** | 0.0265 | no mytest |
| mytest split protocol | 0.7321 | 0.65979 | 0.0723 | mytest as train+val |
| mytest pretrain→finetune | 0.7358 | 0.55214 | 0.1837 | two-stage pretrain |
| mytest aug + myval val (soup) | 0.7688 | 0.67021 | 0.0986 | mytest merged into train |
| split-val aug (soup) | 0.7446 | 0.63212 | 0.1125 | no myval leak, still failed |

**Key finding:** mytest data causes severe domain shift. The model learns mytest-specific features that inflate myval but don't generalize to the Kaggle test set. This is consistent across pretrain→finetune, direct augmentation, and even split-val approaches with no myval leakage.

## Key Findings (confirmed)
1. **myval-as-validation** (+0.065): The single biggest myval win. However, repeated use for hyperparameter tuning creates leak risk.
2. **Higher resolution (288px)** (+0.008): Helpful.
3. **Seed variance**: seed=42 consistently better than seed=123.
4. **Model soup**: Small but reliable myval gain (+0.0049).
5. **mytest as training data**: Improves myval, degrades test. NOT USABLE.

## Tested & Discarded
- EMA — lagging weights, hurts epoch selection
- Multi-seed ensemble — myval up, test down
- TTA — no improvement at F1@0.5
- ConvNeXt V2 Tiny — CUDA OOM
- Alternative backbones (convnext_small, efficientnet_b0, swin_tiny)
- 320px resolution — myval regressed
- Stronger augmentations (RandAugment, ColorJitter, MixUp)
- Pseudo-labeling
- Stochastic depth
- Weight decay sweep
- Label smoothing > 0.1
- BBox-crop margin sweep (0.10 confirmed optimal)
- Weighted model soup (same as uniform)
- Focal loss — myval matched CE, no gain
- All mytest-based approaches — test regressed

## Future Directions
1. K-fold bagging on original data (no mytest)
2. Fine-grained data cleaning / hard negative mining on original train set
3. Architecture exploration (ConvNeXt V2 at 224px to avoid OOM)
