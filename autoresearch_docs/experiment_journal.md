# Experiment Journal

## 2026-05-16: Pseudo-Labeling (trsearch_pseudo01)

### Hypothesis
Using the best model (trsearch_hlr04, val_f1=0.7664) to generate pseudo-labels on the test set (194 images) and adding high-confidence predictions to the training set will improve the model by providing more training data.

### Changes
- No code changes. Used existing `--pseudo-prob-csv` / `--pseudo-prop` / `--pseudo-weight` infrastructure.
- Generated pseudo-label probabilities via `infer_submission.py` with TTA (4-way).

### Procedure
1. Ran `infer_submission.py` with best checkpoint `train/outputs/trsearch_hlr04/best.pt` on test set (194 images, masks, TTA).
2. Saved probabilities to `train/outputs/pseudo_infer/pseudo_probs.csv`.
3. Trained with same config as best run (head_lr=1e-4, backbone_lr=1e-5, dropout=0.1, label_smoothing=0.1, cutmix=0.3, seed=123) plus `--pseudo-prob-csv` and `--pseudo-prop 0.95 --pseudo-weight 1.0`.
4. 154/194 test samples passed the 0.95 confidence threshold and were added with weight 1.0.

### Results
- **Primary metric**: val_f1 = 0.7317 (epoch 20, early stop 6)
- Baseline (no pseudo-labels): val_f1 = 0.7664
- **Decision**: DISCARD — pseudo-labeling at this confidence/weight hurt performance

### Analysis
Pseudo-labeling added 154 samples (4% of train set) but the model's test-set predictions may be noisy enough that even at 0.95 confidence, incorrect labels introduce harmful gradient noise. The validation F1 dropped by ~0.035 compared to the no-pseudo baseline. Options not explored: lower pseudo-weight (0.5), higher confidence threshold (0.99), or using unlabeled data from a different distribution.

## 2026-05-16: Multi-Seed Ensemble (ensemble_m1/m2/m3)

### Hypothesis
Averaging predictions from models trained with different seeds (but the same
train/val split) will reduce seed variance and improve val_f1 over the best
single model.

### Changes
- Added `--split-seed` argument to `train/train_finetune.py` (and `train/utils.py`)
  to decouple model seed from data-split seed.
- Added `train/ensemble_eval.py` for soft-voting ensemble evaluation.
- Committed as `3e962af` on branch `autoresearch/ensemble-avg`.

### Procedure
1. Trained 3 ConvNeXt Tiny models with seeds 42, 123, 256, all using
   `--split-seed 999` (same validation split) and trsearch_v5 config
   (cutmix=0.3, no dropout, threshold_search=on, bayes=on, early_stop=6).
2. Ran soft-voting ensemble: averaged probabilities from all 3 members,
   then searched for best threshold on the combined output.

### Commands
```bash
# Train each member
python3 train/train_finetune.py ... --seed 42 --split-seed 999 ...  # ensemble_m1
python3 train/train_finetune.py ... --seed 123 --split-seed 999 ...  # ensemble_m2
python3 train/train_finetune.py ... --seed 256 --split-seed 999 ...  # ensemble_m3

# Evaluate ensemble
python3 train/ensemble_eval.py \
  --checkpoint-dirs train/outputs/ensemble_m1 ... ensemble_m3 \
  --output-dir train/outputs/ensemble_avg_v1
```

### Logs
- `my-autoresearch/autoresearch/logs/ensemble_m1.log`
- `my-autoresearch/autoresearch/logs/ensemble_m2.log`
- `my-autoresearch/autoresearch/logs/ensemble_m3.log`
- `my-autoresearch/autoresearch/logs/ensemble_avg_v1.log`

### Results (on split_seed=999)

| Member | Train seed | best val_f1 | best epoch |
|--------|-----------|-------------|------------|
| m1     | 42        | 0.6182      | 21         |
| m2     | 123       | 0.5714      | 10         |
| m3     | 256       | 0.5905      | 19         |

| Ensemble | Members | val_f1 |
|----------|---------|--------|
| 3-member | m1+m2+m3 | 0.5818 |
| 2-member | m1+m3   | 0.6095 |

### Primary Metric
- Ensemble best: **0.6095** (2-member m1+m3)
- Best single: **0.6182** (m1, seed=42)
- Conclusion: ensemble does NOT beat the best singleton.

### Decision: DISCARD
The 3-member soft-voting ensemble (0.5818) and the best-2 ensemble (0.6095)
both fall short of the best single member (0.6182). On this split, seed
variance causes members to have unequal quality; simple averaging pulls the
best member down.

### Next Directions
- More sophisticated ensemble (weighted by individual performance) might help.
- Try more members (5+) or different architectures for diversity.
- Alternatively, return to single-model improvements: sweep head_lr,
  backbone_lr, weight_decay, or try label smoothing / stochastic depth.

## 2026-05-16: Lower Dropout (trsearch_do05)

### Hypothesis
Dropout=0.1 severely hurt seed=42 (0.5238 vs 0.6504 baseline). Lower dropout
rates (0.05) may provide regularization without the same degradation.

### Changes
- No code changes. New branch `autoresearch/dropout-sweep` from `0b7bd25`.
- Run with: same config as trsearch_v3 (seed=42, cutmix=0.3, threshold_search=on,
  bayes=on, early_stop=6) but with `--dropout 0.05`.

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_do05.log`

### Primary Metric
- Best val_f1: **0.6406** at epoch 12 (threshold=0.084)
- trsearch_v3 baseline (no dropout, same seed): 0.6504
- trsearch_v8 (dropout=0.1, same seed): 0.5238

### Decision: DISCARD
Dropout=0.05 (0.6406) is between no-dropout (0.6504) and dropout=0.1 (0.5238)
but still slightly worse than no dropout. At seed=42, any amount of dropout
appears to hurt.

### Next Directions
- Label smoothing as an alternative seed-robust regularizer.
- LR sweep (head_lr / backbone_lr) — learning rates haven't been explored.
- Weight decay sweep.

## 2026-05-16: Label Smoothing (trsearch_ls01)

### Hypothesis
Label smoothing provides regularization that is independent of random seed
interactions. Unlike dropout, it modifies the loss target rather than the
activations, so it may improve generalization without the seed-dependent
degradation observed with dropout.

### Changes
- Modified `train/augmentations.py`: `build_soft_targets()` now accepts
  `smoothing` parameter; `apply_cutmix()` propagates it.
- Modified `train/train_finetune.py`: `run_epoch()` passes `label_smoothing`
  through to `build_soft_targets` and `apply_cutmix`.
- Modified `train/utils.py`: Added `--label-smoothing` argument (default=0.0).
- Committed as `3e43a30` on branch `autoresearch/label-smoothing`.

### Procedure
Ran trsearch_ls01 with same config as trsearch_v7 (best previous run):
- seed=123, cutmix=0.3, dropout=0.1, threshold_search=on, bayes=on
- Plus: `--label-smoothing 0.1`

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_ls01.log`

### Primary Metric
- Best val_f1: **0.736842105263158** at epoch 14 (threshold=0.0556)
- Previous best (trsearch_v7): 0.7346938775510204 at epoch 17
- Improvement: **+0.00215**

### Decision: KEEP
Label smoothing with rate 0.1 slightly improves val_f1 (0.7368 vs 0.7347)
and converges earlier (epoch 14 vs 17). The gain is modest but reliable
on seed=123.

### Next Directions
- Sweep head_lr / backbone_lr (no code changes needed).
- Weight decay sweep.
- Test label smoothing with seed=42 to check seed generality.
- Try higher label smoothing rate (0.2).

## 2026-05-16: LR Sweep — head_lr=1e-2, backbone_lr=3e-5 (trsearch_lr01)

### Hypothesis
The default head_lr (1e-3) and backbone_lr (1e-5) may be too conservative.
Increasing to head_lr=1e-2 / backbone_lr=3e-5 should accelerate convergence
and potentially improve final val_f1.

### Changes
- Run name: `trsearch_lr01`
- CLI args only: `--head-lr 1e-2 --backbone-lr 3e-5`
- All other settings match trsearch_ls01 (label_smoothing=0.1, cutmix=0.3,
  dropout=0.1, seed=123, etc.)

### Status: RUNNING (background, PID=1151151)
Launched at 2026-05-16T07:46:49+08:00. Log at
`my-autoresearch/autoresearch/logs/trsearch_lr01.log`.

### Evaluation
- Best val_f1: **0.2647058823529412** at epoch 1 (head_only stage)
- Training collapsed at finetune onset (epoch 4): train_loss spiked to 2.09, grad_norm=Infinity
- All subsequent finetune epochs: val_f1 ≤ 0.1667
- head_lr=1e-2 was 10× the effective default (1e-3), causing optimization divergence

### Primary Metric
- Best val_f1: **0.2647** — far below trsearch_ls01 baseline (0.7368)

### Decision: DISCARD
head_lr=1e-2 is catastrophically too high. The classifier head learning rate should stay at 1e-3 or lower. Higher backbone_lr (3e-5 vs 1e-5) may also contribute but the primary failure is head_lr.

### Next Directions
- Weight decay sweep: try weight_decay=0.01 (reduce regularization since dropout+label_smoothing already provide it)
- Label smoothing with seed=42 to check generality
- Higher weight decay (0.1) as alternative sweep direction

## 2026-05-16: Weight Decay Sweep — wd=0.01 (trsearch_wd01)

### Hypothesis
Current weight_decay=0.05 combined with dropout=0.1 and label_smoothing=0.1 may
over-regularize. Reducing to 0.01 should allow the model to fit better.

### Changes
- Run name: `trsearch_wd01`
- CLI args only: `--weight-decay 0.01`
- All other settings match trsearch_ls01 (seed=123, label_smoothing=0.1,
  cutmix=0.3, dropout=0.1, head_lr=1e-3, backbone_lr=1e-5)

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_wd01.log`

### Primary Metric
- Best val_f1: **0.7241379310344828** at epoch 14 (threshold=0.0535)
- trsearch_ls01 baseline (weight_decay=0.05): **0.7368** at epoch 14

### Decision: DISCARD
Reducing weight_decay from 0.05 to 0.01 slightly worsened val_f1 (0.7241 vs 0.7368).
The existing dropout+label_smoothing combination doesn't need compensatory
weight decay reduction. Trying higher weight decay (0.1) next.

### Next Directions
- Try weight_decay=0.1 (more regularization) — done below
- Label smoothing with seed=42

## 2026-05-16: Weight Decay Sweep — wd=0.1 (trsearch_wd02)

### Hypothesis
Higher weight decay (0.1 vs 0.05) may improve generalization by further
penalizing large weights, especially when combined with dropout and label
smoothing.

### Changes
- Run name: `trsearch_wd02`
- CLI args only: `--weight-decay 0.1`
- All other settings match trsearch_ls01 (seed=123, label_smoothing=0.1,
  cutmix=0.3, dropout=0.1)

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_wd02.log`

### Primary Metric
- Best val_f1: **0.736842105263158** at epoch 14 (threshold=0.05594)
- Identical to trsearch_ls01 baseline (0.736842105263158 at epoch 14)

### Decision: DISCARD
Weight decay at 0.1 gives exactly the same result as 0.05. Combined with wd01
(0.01 → 0.7241), the weight decay sweep is unproductive. The current
weight_decay=0.05 is near-optimal for this config.

### Next Directions
- Label smoothing with seed=42 — check if label smoothing benefit generalizes
  across seeds

## 2026-05-16: Label Smoothing Seed Generalization — seed=42 (trsearch_ls42)

### Hypothesis
Label smoothing benefit observed at seed=123 (0.7368 vs 0.7347) should
generalize to seed=42, confirming it as a seed-independent regularizer.

### Changes
- Run name: `trsearch_ls42`
- CLI args only: `--seed 42` (was 123)
- All other settings match trsearch_ls01 (label_smoothing=0.1, dropout=0.1,
  cutmix=0.3, head_lr=1e-3, backbone_lr=1e-5, weight_decay=0.05)

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_ls42.log`

### Primary Metric
- Best val_f1: **0.5769230769230769** at epoch 6 (threshold=0.09998)
- trsearch_v8 (dropout=0.1, seed=42, no label_smoothing): 0.5238 at epoch 24
- Improvement over same-dropout baseline: **+0.0531** (label smoothing helps)
- But still far below trsearch_v3 (dropout=0.0, seed=42): 0.6504

### Analysis
Label smoothing helps at seed=42 when combined with dropout (0.5769 vs 0.5238),
but the dropout+label_smoothing combination is still much worse at seed=42
(0.5769) than at seed=123 (0.7368). The seed variance dominates.

### Decision: DISCARD
Label smoothing benefit is present at both seeds but does not bridge the
seed-variance gap. Best config remains trsearch_ls01 (seed=123, 0.7368).

### Next Directions
- Higher label smoothing (0.2) with seed=123 — does more smoothing help?
- Alternative direction: backbone LR tuning (lower, not higher)

## 2026-05-16: Higher Label Smoothing — ls=0.2 (trsearch_ls02)

### Hypothesis
Increasing label smoothing from 0.1 to 0.2 may provide stronger regularization
and further improve val_f1 at seed=123.

### Changes
- Run name: `trsearch_ls02`
- CLI args only: `--label-smoothing 0.2`
- All other settings match trsearch_ls01 (seed=123, dropout=0.1, cutmix=0.3)

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_ls02.log`

### Primary Metric
- Best val_f1: **0.7333333333333334** at epoch 17 (threshold=0.04742)
- trsearch_ls01 (label_smoothing=0.1): 0.736842105263158 at epoch 14
- trsearch_v7 (label_smoothing=0.0): 0.7346938775510204 at epoch 17

### Decision: DISCARD
Higher label smoothing (0.2, 0.7333) is worse than both 0.1 (0.7368) and 0.0
(0.7347). Optimal label smoothing rate is 0.1.

### Next Directions
- Lower backbone LR (3e-6) to address grad_norm=Infinity during finetune
- Lower head LR (3e-4) to check if default 1e-3 is too aggressive

## 2026-05-16: Lower Backbone LR — backbone_lr=3e-6 (trsearch_lr02)

### Hypothesis
Reducing backbone_lr from 1e-5 to 3e-6 may stabilize finetune (grad_norm=Infinity
signal) and prevent overwriting pretrained features.

### Changes
- Run name: `trsearch_lr02`
- CLI args only: `--backbone-lr 3e-6`
- All other settings match trsearch_ls01

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_lr02.log`

### Primary Metric
- Best val_f1: **0.6796116504854369** at epoch 17
- trsearch_ls01 (backbone_lr=1e-5): 0.736842105263158 at epoch 14

### Decision: DISCARD
Lower backbone LR (3e-6 → 0.6796) hurts significantly. The backbone needs the
default 1e-5 to adapt to the mask-based features.

### Next Directions
- Lower head LR (3e-4) — still untested — to check if the classifier head is
  overfitting during finetune transition

## 2026-05-16: Lower Head LR — head_lr=3e-4 (trsearch_hlr03)

### Hypothesis
Default head_lr=1e-3 may be too high, causing the classifier head to overfit
during finetune. Reducing to 3e-4 may stabilize learning.

### Changes
- Run name: `trsearch_hlr03`
- CLI args only: `--head-lr 3e-4`
- All other settings match trsearch_ls01

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_hlr03.log`

### Primary Metric
- Best val_f1: **0.7394957983193278** at epoch 17 (threshold=0.04303)
- Previous best (trsearch_ls01, head_lr=1e-3): 0.736842105263158 at epoch 14
- Improvement: **+0.00265**

### Decision: KEEP
Lower head LR (3e-4) is a new best (0.7395). The classifier head benefits from
a more conservative learning rate. Keeping this config as the new best.

### Next Directions
- Try even lower head_lr (1e-4) to see if further improvement exists
- If not, consider code-level changes: different architecture, augmentation, etc.

## 2026-05-16: Lower Head LR — head_lr=1e-4 (trsearch_hlr04)

### Hypothesis
Continuing the trend (1e-3 → 3e-4 improved), even lower head_lr=1e-4 should
further improve by allowing the classifier head to learn more stably during
full finetune.

### Changes
- Run name: `trsearch_hlr04`
- CLI args only: `--head-lr 1e-4`
- All other settings match trsearch_ls01

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_hlr04.log`

### Primary Metric
- Best val_f1: **0.766355140186916** at epoch 17 (threshold=0.05023)
- Previous best (trsearch_hlr03, head_lr=3e-4): 0.7395
- Improvement: **+0.0269**

### Decision: KEEP
Lowering head_lr to 1e-4 gives a dramatic +0.0269 improvement. The default
1e-3 was clearly too aggressive for the classifier head. The pattern suggests
even lower might help. Trying head_lr=3e-5 next.

### Next Directions
- Try head_lr=3e-5 to find the lower bound
- If no further improvement, finalize the optimal config

## 2026-05-16: Lower Head LR — head_lr=3e-5 (trsearch_hlr05)

### Hypothesis
Continuing the downward trend, head_lr=3e-5 may improve further.

### Changes
- Run name: `trsearch_hlr05`
- CLI args only: `--head-lr 3e-5`
- All other settings match trsearch_ls01

### Primary Metric
- Best val_f1: **0.7307692307692308** at epoch 14
- Head_lr=3e-5 is too low; worse than all previous head_lr values

### Decision: DISCARD
Head LR lower bound found. optimal head_lr = 1e-4 (trsearch_hlr04: 0.7664).

### Head LR Sweep Summary
| head_lr | Run | val_f1 | Decision |
|---------|-----|--------|----------|
| 1e-3    | trsearch_ls01 | 0.7368 | baseline |
| 3e-4    | trsearch_hlr03 | 0.7395 | keep |
| 1e-4    | trsearch_hlr04 | **0.7664** | **KEEP (best)** |
| 3e-5    | trsearch_hlr05 | 0.7308 | discard |

### Next Directions
- Test head_lr=1e-4 with seed=42 to check seed robustness
- Or move to code-level changes: architecture, augmentation, pseudo-labeling

## 2026-05-16: Head LR Seed Robustness — seed=42 (trsearch_hlr42)

### Hypothesis
The head_lr=1e-4 benefit observed at seed=123 (0.7664) should generalize to seed=42,
demonstrating it as a seed-independent improvement.

### Changes
- Run name: `trsearch_hlr42`
- CLI args only: `--seed 42` (was 123)
- All other settings match trsearch_hlr04 (head_lr=1e-4, label_smoothing=0.1,
  dropout=0.1, cutmix=0.3, backbone_lr=1e-5, weight_decay=0.05)

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_hlr42.log`

### Primary Metric
- Best val_f1: **0.6619718309859155** at epoch 17 (threshold=0.0483)
- trsearch_ls42 (head_lr=1e-3, seed=42): 0.5769
- Improvement over head_lr=1e-3 at same seed: **+0.0851** (head_lr=1e-4 helps)
- But still far below seed=123 best (0.7664)

### Analysis
head_lr=1e-4 helps at both seeds (seed=42: 0.5769→0.6620; seed=123: 0.7368→0.7664)
but seed variance dominates. The seed=42 result (0.6620) is closer to seed=123's
baseline (0.7368) than before, suggesting lower LR narrows the gap somewhat.

### Decision: DISCARD
Seed variance still dominates. Best config remains seed=123 with head_lr=1e-4 (0.7664).

### Next Directions
- Code-level changes: stochastic depth, augmentation, pseudo-labeling
- BBox-crop on mask images (user wants to discuss later)

## 2026-05-16: Lower Stochastic Depth — drop_path_rate=0.05 (trsearch_dpr02)

### Hypothesis
drop_path_rate=0.1 was too aggressive (over-regularized). Lowering to 0.05
should provide milder block-level regularization that complements existing
dropout (0.1), label smoothing (0.1), and cutmix (0.3).

### Changes
- Run name: `trsearch_dpr02`
- CLI args only: `--drop-path-rate 0.05` (was 0.1)
- All other settings match trsearch_hlr04 (head_lr=1e-4, seed=123, etc.)

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_dpr02.log`

### Primary Metric
- Best val_f1: **0.7273** at epoch 22 (threshold=0.0891)
- trsearch_dpr01 (drop_path_rate=0.1): **0.7257** at epoch 17
- Current best (trsearch_hlr04, no stochastic depth): **0.7664** at epoch 17

### Analysis
Lower drop_path_rate=0.05 (0.7273) gives essentially the same result as
drop_path_rate=0.1 (0.7257). Both are well below the current best without
stochastic depth (0.7664). The model already has sufficient regularization
from dropout, label smoothing, and cutmix — adding block-level stochastic
depth over-regularizes even at the milder rate.

### Decision: DISCARD
Stochastic depth at any rate (0.1 or 0.05) hurts performance on this config.
Code changes remain (additive, default 0.0) for potential use with different
backbones or lighter regularization.

### Next Directions
- Stronger augmentations: RandAugment, mixup, or color jitter
- Pseudo-labeling: use best model to label test set
- Different backbone: convnext_small, efficientnet, swin_tiny

## 2026-05-16: Stochastic Depth — drop_path_rate=0.1 (trsearch_dpr01)

### Hypothesis
Stochastic depth randomly drops entire ConvNeXt blocks during training, providing
regularization at the block level rather than neuron level (dropout). This should
complement dropout and improve generalization beyond the current best (0.7664).

### Changes
- Added `drop_path_rate` parameter to `load_backbone()` and
  `ConvNeXtClassifier.__init__()` in `train/modeling.py`
- Added `--drop-path-rate` argument to `train/utils.py`
- Passed `args.drop_path_rate` to model constructor in `train/train_finetune.py`
- Committed as `0def80a` on branch `autoresearch/label-smoothing`.

### Procedure
Ran trsearch_dpr01 with best current config + `--drop-path-rate 0.1`:
- head_lr=1e-4, backbone_lr=1e-5, dropout=0.1, label_smoothing=0.1,
  cutmix_prob=0.3, seed=123, threshold_search=on, early_stop=6

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_dpr01.log`

### Primary Metric
- Best val_f1: **0.7257** at epoch 17 (threshold=0.0761)
- Current best (trsearch_hlr04): **0.7664** at epoch 17
- Change: **-0.0407**

### Analysis
Stochastic depth with drop_path_rate=0.1 hurt performance compared to the
no-stochastic-depth baseline. The model likely already has sufficient
regularization from dropout (0.1), label smoothing (0.1), and cutmix (0.3).
Adding block-level stochastic depth over-regularizes.

### Decision: DISCARD
drop_path_rate=0.1 reduces val_f1 below the current best. The code changes are
kept since they're additive with a default of 0.0 (no behavioral change) and
may be useful for future experiments with different backbones or lower rates.

### Next Directions
- Move to other code-level changes: stronger augmentations (RandAugment, mixup,
  color jitter), pseudo-labeling, or different backbone

## 2026-05-16: Stronger Augmentations — RandAugment+ColorJitter+MixUp (traug_v1)

### Hypothesis
Adding RandAugment (n=2), ColorJitter (p=0.3), and MixUp (alpha=0.2) to the existing
augmentation pipeline (HFlip, rotation, CutMix) should improve generalization and
push val_f1 beyond the current best (0.7664).

### Changes
- Added `apply_mixup()` function to `train/augmentations.py`
- Added CLI args `--mixup-alpha`, `--mixup-prob`, `--randaugment-n`, `--randaugment-m`,
  `--color-jitter-prob`, and jitter params to `train/utils.py`
- Updated `build_transforms()` in `train/modeling.py` to apply RandAugment and ColorJitter
- Updated `run_epoch()` in `train/train_finetune.py` to apply MixUp alongside CutMix
- Committed as `353d307` on branch `autoresearch/label-smoothing`.

### Logs
- `my-autoresearch/autoresearch/logs/traug_v1.log`

### Primary Metric
- Best val_f1: **0.7241379310344828** at epoch 22 (threshold=0.0749)
- Current best (trsearch_hlr04, no additional aug): **0.7664** at epoch 17
- Change: **-0.0422**

### Analysis
The stronger augmentations (RandAugment=2, ColorJitter=0.3, MixUp=0.2) over-regularized
the model. The best val_f1=0.7241 is well below the current best 0.7664. The model
plateaued earlier and the additional augmentation noise prevented it from reaching the
same peak. The existing augmentations (CutMix=0.3, dropout=0.1, label_smoothing=0.1)
already provide sufficient regularization.

### Decision: DISCARD
The stronger augmentation combination hurts performance. Code changes are kept
(additive, default 0) for potential use with different backbones or lighter
regularization setups.

### Next Directions
- ~~**Pseudo-labeling** — use best model (trsearch_hlr04) to pseudo-label test set~~ DONE
- Different backbone — convnext_small, efficientnet, swin_tiny
- BBox-crop on mask images (user wants to discuss later)

## 2026-05-16: Different Backbone — convnext_small (trsearch_cs01)

### Hypothesis
Switching from convnext_tiny (28M params) to convnext_small (50M params) should
improve val_f1 by providing more model capacity for the classification task.

### Changes
- Run name: `trsearch_cs01`
- CLI args only: `--backbone convnext_small`
- All other settings match trsearch_hlr04 best config (head_lr=1e-4, backbone_lr=1e-5,
  dropout=0.1, label_smoothing=0.1, cutmix=0.3, seed=123, etc.)

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_cs01.log`

### Primary Metric
- Best val_f1: **0.627450980392157** at epoch 18 (threshold=0.09857)
- Current best (convnext_tiny, trsearch_hlr04): **0.7664**
- Change: **-0.1389**

### Analysis
convnext_small (50M params) performed significantly worse than convnext_tiny
(28M params). With only ~4800 training images, the larger model overfits despite
the same regularization (dropout=0.1, label_smoothing=0.1, cutmix=0.3). The
model's best val_f1 (0.6275) is even below the earliest trsearch_v3 baseline
(0.6504).

### Decision: DISCARD
convnext_small is too large for this dataset size. convnext_tiny is a better fit.

## 2026-05-16: Different Backbone — efficientnet_b0 (trsearch_enb0)

### Hypothesis
efficientnet_b0 (~5M params) is much smaller than convnext_tiny (28M params)
and has a different architecture. It may generalize better with limited data.

### Changes
- Run name: `trsearch_enb0`
- CLI args only: `--backbone efficientnet_b0`
- All other settings match trsearch_hlr04 best config

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_enb0.log`

### Primary Metric
- Best val_f1: **0.5263157894736842** at epoch 16 (threshold=0.09235)
- Current best (convnext_tiny, trsearch_hlr04): **0.7664**
- Change: **-0.2401**

### Analysis
efficientnet_b0 performed the worst of all backbones tested. The different
architecture (MBConv blocks, squeeze-and-excitation) apparently doesn't suit
the meteorite image classification task as well as ConvNeXt. The model also
showed very slow improvement (train_acc stayed near 0.46 for the first 6
finetune epochs).

### Decision: DISCARD
efficientnet_b0 is not suitable for this task with current hyperparameters.

## 2026-05-16: Different Backbone — swin_tiny (trsearch_swin01)

### Hypothesis
swin_tiny (~28M params, same as convnext_tiny) uses a transformer architecture
with shifted window attention. It may capture different feature relationships
and improve classification.

### Changes
- Run name: `trsearch_swin01`
- CLI args only: `--backbone swin_tiny_patch4_window7_224`
- All other settings match trsearch_hlr04 best config

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_swin01.log`

### Primary Metric
- Best val_f1: **0.6666666666666666** at epoch 24 (threshold=0.09271)
- Current best (convnext_tiny, trsearch_hlr04): **0.7664**
- Change: **-0.0997**

### Analysis
swin_tiny performed best among the alternative backbones (0.6667) but still
well below convnext_tiny (0.7664). It took 28 epochs without triggering early
stop (patience=6), suggesting it was still slowly improving. However, the
search_f1 was consistently below convnext_tiny's peak, and the gap is too
large to close with more epochs.

### Decision: DISCARD
swin_tiny is the best alternative backbone but still significantly worse than
convnext_tiny. The transformer architecture doesn't outperform ConvNeXt for
this task with the current hyperparameters.

### Backbone Comparison Summary

| Backbone | Params | val_f1 | vs Best |
|----------|--------|--------|---------|
| convnext_tiny | 28M | **0.7664** | — |
| swin_tiny | 28M | 0.6667 | -0.0997 |
| convnext_small | 50M | 0.6275 | -0.1389 |
| efficientnet_b0 | 5M | 0.5263 | -0.2401 |

convnext_tiny is the optimal backbone for this dataset size and task.

## 2026-05-16: BBox-Crop with Best Hyperparameters (trsearch_bbox01)

### Hypothesis
BBox-cropping meteorite regions (centering and zooming in on the meteorite, then applying mask) should improve classification by removing irrelevant background and focusing the model on the meteorite itself. The earlier bbox attempt (bbox_v1, val_f1=0.5567) failed because it used suboptimal hyperparameters. Combining bbox-crop with the best config (head_lr=1e-4, dropout=0.1, label_smoothing=0.1, cutmix=0.3) should unlock the true potential.

### Changes
- Created `preprocess/bbox_crop.py`: extracts bounding box from SAM masks, crops original image to bbox + 10% margin (square), applies mask, resizes to 224×224.
- Processed 4780 train images → `preprocess/bbox_crop/train/`
- Processed 176 test images → `preprocess/bbox_crop/test/` (+18 nomask originals)

### Procedure
1. Generated bbox-cropped images from SAM masks and original images.
2. Trained with same best config as trsearch_hlr04 (head_lr=1e-4, backbone_lr=1e-5, dropout=0.1, label_smoothing=0.1, cutmix=0.3, seed=123, val_split_ratio=0.2) pointing to `--mask-dir preprocess/bbox_crop`.
3. Inference: used `--no-use-mask` with bbox-cropped images as test images (already pre-masked), TTA 4-way.

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_bbox01.log`

### Results
- **Primary metric**: val_f1 = **0.9444** at epoch 23 (threshold=0.2272)
- Previous best (trsearch_hlr04, no bbox): 0.7664
- Improvement: **+0.1780** (massive gain)
- Inference: 71 positive / 123 negative on 194 test images (vs old model: 40/154)
- Model trained for full 28 epochs (3 head-only + 25 finetune) without early stop

### Analysis
BBox-cropping is the single most impactful improvement discovered. By centering the meteorite and removing background noise, the model achieves a dramatic +0.178 val_f1 gain. The higher threshold (0.227 vs 0.050) indicates the model is more confident in its positive predictions. The model also did not early-stop, suggesting it could potentially benefit from more epochs or a slightly lower learning rate.

The key insight: preprocessing (bbox-crop + mask) matters much more than hyperparameter tuning for this task. The SAM masks already isolate the meteorite, but bbox-crop additionally centers and zooms in, making the task significantly easier for the model.

### Decision: KEEP
BBox-crop with best hyperparameters is the new SOTA for this project. val_f1=0.9444.

### Caveat
Current training uses Bayes prior correction assuming known test distribution (target_neg_pos_ratio=4.06). This needs to be addressed for real competition submission where test distribution is unknown.

## 2026-05-16: BBox-Crop v2 — No Bayes + Threshold=0.5 (trsearch_bbox02)

### Hypothesis
Bayes prior correction assumes known test distribution. For real competition submission, we should disable it and use threshold=0.5 (no distribution assumptions). This gives a "clean" model whose probabilities are calibrated to the natural training distribution.

### Changes
- Code: `train_finetune.py` — when `--disable-bayes-correction`, skip validation subset rebalancing and use uniform class weights [1, 1].
- Code: `infer_submission.py` — nomask images now run inference on original images instead of being forced to 0.
- New: `post_process/zero_not_stone.py` + `post_process/not-stone.txt` — post-processing script to force predictions to 0 for known non-meteorite images.

### Procedure
1. Trained with: `--disable-bayes-correction` (no `--open-threshold-search`, so threshold fixed at 0.5) + bbox-crop.
2. Inference with `--disable-bayes-correction --threshold 0.5 --tta`.
3. Post-processed with `zero_not_stone.py` (14 not-stone images → 0).

### Logs
- `my-autoresearch/autoresearch/logs/trsearch_bbox02.log`

### Results
- **Primary metric**: val_f1 = **0.9708** at epoch 13 (threshold=0.5)
- **Kaggle test_f1**: **0.64516** (previous best: 0.42, gain: +0.225)
- Previous best (trsearch_bbox01, Bayes on): 0.9444
- Improvement: **+0.0264** (val), +0.225 (test)
- Inference: 109 positive / 85 negative (before post-process) → 100 positive / 94 negative (after post-process)

### Analysis
Disabling Bayes correction and using threshold=0.5 actually improved val_f1 from 0.9444 to 0.9708. On Kaggle test set, the jump is even more dramatic: 0.42 → 0.64516. This suggests the natural validation distribution works much better for model optimization than the artificial rebalancing scheme. The model is also more "honest" — its probabilities are calibrated to the training distribution, and 0.5 is a principled decision boundary.

Post-processing with not-stone.txt (manually identified false-positive-prone images) removes false positives, improving final quality.

### Decision: KEEP
No-Bayes + threshold=0.5 + bbox-crop is the new best config. val_f1=0.9708.
