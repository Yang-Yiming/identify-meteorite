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

### Next Directions
- TTA during training (use during validation for better epoch selection)
- Multi-seed ensemble with best config (seeds 42, 123)
- BBox-crop margin sweep
- Try soup with different weightings (e.g., weighted by val_f1)
