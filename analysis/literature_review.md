# Literature Review: Techniques for Fine-Grained Classification with Foundation Models on Limited Data

*Compiled from 25+ papers via AlphaXiv and arxiv, contextualized to our meteorite classification project (4,780 labeled + 8,600 unlabeled images).*

**Project Context:** Binary meteorite classification with bbox-crop images. Fine-tuning ConvNeXt Tiny and DINOv2 Small achieves ~0.72 test F1. The gap to top Kaggle entries (≥0.80 F1) demands smarter use of unlabeled data, better ensemble construction, and principled handling of distribution shift between myval and test.

**Notation:** Sections marked **[NEW]** are additions to the initial review. Papers from ICLR/ICML/NeurIPS/CVPR are flagged with venue.

---

## 1. Model Merging & Weight Interpolation [NEW]

### [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836) — ICLR 2023 (Ainsworth et al., UW)
**Core insight:** Two neural networks trained independently on the same task may appear to be in different loss basins, but this is often an illusion caused by permutation symmetries of hidden units. By matching neurons between the two models (solving a linear assignment problem), we can "re-basin" them into the same basin, after which linear interpolation in weight space works.

**Key contributions:**
- Proves that many independently trained models can be permuted into a single basin
- Three permutation-matching algorithms: activation-based, weight-based, and straight-through estimator
- After permutation, weight averaging between independently trained models works as well as within a single training run
- Explains *why* model soups work: fine-tuning from the same initialization avoids the permutation problem entirely

**Our application:** This explains why our within-run ConvNeXt soup works well — no permutation mismatch to resolve. If we wanted to merge ConvNeXt and DINOv2 weights (different architectures), we'd need Git Re-Basin plus architecture mapping. More practically: if we trained 5 ConvNeXt models from *different* random seeds (not k-fold splits from same init), Git Re-Basin could merge them into one weight set — potentially better than prediction averaging.

### [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) — NeurIPS 2023 (Yadav et al., UNC Chapel Hill + MIT)
**Core insight:** When merging multiple task-specific fine-tuned models, naively averaging their task vectors (fine-tuned minus pretrained weights) causes destructive interference because different models update the same parameters in conflicting directions.

**Key contributions:**
- **T**rim: Keep only top-k% most important parameter changes per model (magnitude-based)
- **E**lect sign: Resolve sign conflicts via majority voting across models
- D**IS**joint merge: Average only parameters where signs agree after election
- Significantly outperforms simple averaging and even model soups on multi-task benchmarks
- Works for both same-architecture and cross-architecture merging

**Our application:** When we average ConvNeXt fold predictions, we lose per-parameter agreement information. TIES-Merging provides a principled alternative: merge the 5 fold models' *parameters* (not predictions), resolving conflicts where folds disagree on weight update direction. This would produce a single model with k-fold knowledge — no inference-time ensemble cost. **This is an untried, high-potential direction.**

### [Task Arithmetic: Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) — ICLR 2024 (Ilharco et al., UW + AI2 + Microsoft)
**Core insight:** The difference between a fine-tuned model's weights and its pretrained weights forms a "task vector." Simple arithmetic on these task vectors can add, subtract, or combine model capabilities — without any additional training.

**Key contributions:**
- **Task vector = θ_ft − θ_pretrained**: compact representation of what the model learned
- **Adding task vectors** improves multi-task performance (θ + α⋅τ_A + β⋅τ_B)
- **Negating task vectors** can "forget" undesired behaviors (θ − γ⋅τ_bad)
- Task vectors are surprisingly composable via simple weighted addition
- Works across CLIP ViT models for diverse vision tasks

**Our application:** If our k-fold ConvNeXt models are seen as "task vectors" from the same pretrained base, task arithmetic provides a principled merge formula: soup + weighted_sum(task_vectors). The α coefficients can be tuned on myval. More speculatively: we could train one model to be "good at recall" (high α) and another to be "good at precision" (low α), then interpolate between them.

### [ZipIt! Merging Models from Different Tasks without Training](https://arxiv.org/abs/2305.03053) — ICLR 2024 (Stoica et al., Georgia Tech)
**Core insight:** Models trained on *different* tasks (not just different hyperparameters) can be merged into a single model that performs all tasks, by "zipping" together corresponding features across models.

**Key contributions:**
- Merges models with different output heads (different classification tasks)
- Uses feature similarity to compute optimal "zip" (merge) assignments
- Survives partial merging: can merge a subset of layers while keeping others task-specific
- Outperforms model soups and weight averaging when tasks differ significantly

**Our application:** Less directly applicable than TIES-merging since our models share the same binary classification task. However, the "partial merge" concept is interesting: we could merge early layers (general features) while keeping later layers (classifier-specific) separate, creating a multi-branch model.

---

## 2. Semi-Supervised Learning with Pretrained Models [NEW]

### [FixMatch: Simplifying Semi-Supervised Learning](https://arxiv.org/abs/2001.07685) — NeurIPS 2020 (Sohn et al., Google Brain)
**Core insight:** Semi-supervised learning can be dramatically simplified to two ingredients: (1) use a weakly-augmented image to generate a pseudo-label, and (2) train the model to predict that pseudo-label from a strongly-augmented version of the same image — but only when the model is confident.

**Key contributions:**
- **Weak augmentation** (flip + shift) → generate pseudo-label → keep if confidence > τ
- **Strong augmentation** (RandAugment + Cutout) → train to match pseudo-label via cross-entropy
- Simple cross-entropy loss, no additional losses or hyperparameters beyond τ
- Achieves SOTA on CIFAR-10 with only 4 labeled examples per class
- Key mechanism: confident pseudo-labels + strong augmentation forces consistent learning

**Our application:** We have 8,600 unlabeled mytest images. FixMatch would:
1. Take our best ConvNeXt model (trained on 4,780 labeled)
2. Generate pseudo-labels on mytest with confidence > 0.95 threshold
3. Retrain with labeled + pseudo-labeled images, using weak/strong augmentation pairs
4. The key risk: mytest distribution differs from train — confident pseudo-labels may be systematically wrong. **FixMatch works best when labeled and unlabeled data share the same distribution.**

### [Self-training with Noisy Student](https://arxiv.org/abs/1911.04252) — NeurIPS 2020 (Xie et al., Google Brain)
**Core insight:** A larger student model, trained with noise (dropout, stochastic depth, RandAugment) on a teacher's pseudo-labels, can *exceed* the teacher's performance — even when the teacher is already SOTA.

**Key contributions:**
- **Equal-or-larger student**: Counter-intuitively, bigger students work better
- **Noise injection during student training** is critical — without it, student merely copies teacher
- Pseudo-labeling uses soft (probabilistic) labels, not hard
- Iterative: student becomes next teacher, process repeats
- On ImageNet, improved EfficientNet from 84.3% → 88.4% using 300M unlabeled images
- Also improves robustness to adversarial and out-of-distribution examples

**Our application:** This is our strongest semi-supervised candidate because:
1. Train ConvNeXt on 4,780 labeled → teacher
2. Generate pseudo-labels on 8,600 mytest
3. Train a *different* model (DINOv2 + linear head) on combined data with aggressive noise
4. The student may learn features that generalize better than the teacher
5. Unlike FixMatch, Noisy Student doesn't assume i.i.d. — noise injection helps with distribution shift
6. **This approach has a proven track record of improving generalization when unlabeled data differs from labeled data.**

### [Rethinking Pre-training and Self-training](https://arxiv.org/abs/2006.06882) — NeurIPS 2020 (Zoph et al., Google Brain)
**Core insight:** The value of pre-training has been overstated for tasks with sufficient labeled data. Self-training on task-specific data often matches or exceeds the benefit of ImageNet pre-training — and the two are *complementary* (using both helps most).

**Key contributions:**
- For COCO detection with enough labeled data, training from scratch + self-training beats ImageNet pre-training
- Pre-training helps most when labeled data is scarce — its benefit diminishes as labeled data grows
- **Self-training is more data-efficient than pre-training** when unlabeled data closely matches the target task
- The combination (pre-training + self-training) always gives the best results, but the marginal gain of pre-training shrinks with more labeled data

**Our application:** We are in the "scarcity" regime (4,780 labeled). Pre-training is essential. But self-training on unlabeled mytest images — which are from the *same domain* as the test set — could provide gains complementary to DINOv2/ImageNet pretraining. The paper suggests that the benefit of self-training is roughly additive to pre-training, making it a safe bet.

---

## 3. Test-Time Adaptation [NEW]

### [TENT: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726) — ICLR 2021 (Wang et al., UC Berkeley + Adobe)
**Core insight:** Rather than training a model to be robust to all possible distribution shifts, adapt the model *at test time* by minimizing the entropy of its predictions on the target (unlabeled) data. Only batch normalization parameters are updated — the rest stays frozen.

**Key contributions:**
- **Test-time adaptation**: No access to source training data, only target (test) data
- Optimizes channel-wise affine parameters (γ, β) of BatchNorm layers via entropy minimization
- Single backward pass on each test batch → efficient
- Outperforms domain adaptation methods on ImageNet-C corruptions without any source data access
- Key insight: entropy minimization implicitly encourages confident, consistent predictions

**Our application:** We have 8,600 unlabeled *mytest* images that share the test set distribution. We could:
1. Take our trained ConvNeXt (BN layers pre-adapted to train distribution)
2. Run TENT on all 8,600 mytest images (batch-wise entropy minimization)
3. Use the adapted model for test set inference
4. This bridges the myval → test distribution gap without using test labels
5. **Critical requirement**: TENT needs target-distribution data — our mytest images provide exactly that.

### [Test-Time Training with Self-Supervision](https://arxiv.org/abs/1909.13231) — ICML 2020 (Sun et al., UC Berkeley)
**Core insight:** Train the model with a shared backbone for both the main task (classification) and a self-supervised auxiliary task (rotation prediction). At test time, continue training the auxiliary task on each test sample, which adapts the shared features to the test distribution.

**Key contributions:**
- **Y-shaped architecture**: Shared backbone, two heads (classification + rotation prediction)
- At test time: update shared features via rotation loss, then classify with updated features
- Works on any single test sample; no batch needed (unlike TENT)
- Robust to diverse corruptions: Gaussian noise, blur, pixelation, weather effects
- More powerful than TENT (updates all layers) but slower (requires auxiliary task training)

**Our application:** The self-supervised auxiliary task could be replaced with something domain-specific (e.g., stone texture discrimination) to make adaptation more relevant. However, the computational cost and architectural changes make this less practical than TENT for our use case.

---

## 4. Robust Fine-Tuning & Distribution Shift [NEW]

### [WiSE-FT: Robust Fine-tuning of Zero-Shot Models](https://arxiv.org/abs/2109.01903) — CVPR 2022 (Wortsman et al., UW + OpenAI + Google)
**Core insight:** Fine-tuning CLIP on downstream tasks improves in-distribution accuracy but destroys out-of-distribution (OOD) robustness. Simply interpolating (weighted average) between the fine-tuned and zero-shot models restores OOD performance with minimal in-distribution accuracy loss.

**Key contributions:**
- **Weight Interpolation**: θ_WiSE = (1−α)·θ_zeroshot + α·θ_finetuned, with α tuned on validation
- When α = 0.5, recovers most OOD robustness while retaining most accuracy gains from fine-tuning
- The zero-shot model has complementary strengths: better on distribution shifts, worse on the specific downstream task
- Works because the zero-shot image encoder and fine-tuned encoder share the same architecture (CLIP ViT)
- Even a *linear interpolation* between two weight checkpoints dramatically improves robustness

**Our application:** This directly applies to our ConvNeXt training. We observe strong overfitting (myval F1 peaks early then drops, while train loss continues to decline). WiSE-FT suggests:
1. Keep a copy of the model at epoch 1-2 (best generalization)
2. At the best myval epoch, interpolate: θ_best = 0.5·θ_early + 0.5·θ_myval_best
3. This should preserve robustness from early training while gaining accuracy from later training
4. We've effectively done this with our SWA-style averaging, but WiSE-FT gives a principled mixing ratio.

### [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343) — ICCV 2023 (Zhai et al., Google DeepMind)
**Core insight:** The contrastive loss (CLIP) requires a large batch size because it operates on pairwise comparisons within the batch. Replacing it with a per-sample sigmoid loss over binary image-text matching removes the batch dependency, enabling better scaling with batch size and model size.

**Key contributions:**
- Sigmoid loss: independently classify each image-text pair as match/non-match (no batch normalization of negatives)
- Better performance at small batch sizes (critical for limited GPU memory)
- Scales better with model size than contrastive loss
- Enables efficient training with mismatched image/text batch sizes
- Achieves SOTA zero-shot performance with smaller batches than CLIP

**Our application:** We used SigLIP as one of the frozen feature extractors in our early (failed) probe experiment. But SigLIP's core insight — per-sample loss removes batch dependency — is relevant to our fine-tuning: we could use SigLIP-pretrained backbones for tasks requiring small batch sizes (e.g., DINOv2 at 518px with batch_size=8).

### [When and Why Vision-Language Models Behave Like Bags-of-Words](https://arxiv.org/abs/2210.01936) — ICLR 2023 (Yuksekgonul et al., Stanford)
**Core insight:** CLIP and similar VLMs have a "bag-of-words" limitation: they struggle with compositional understanding (e.g., distinguishing "the dog chases the cat" from "the cat chases the dog"). This stems from the contrastive training objective's focus on coarse image-text alignment.

**Key contributions:**
- VLMs are systematically bad at attribute binding, relational understanding, and word order
- The issue is fundamental to the contrastive pre-training paradigm
- Fine-tuning on hard negative captions (negated attributes, swapped relations) can partially mitigate
- But the limitation persists even after extensive fine-tuning

**Our application:** This is less about our meteorite task (no text input) but is relevant for two reasons:
1. CLIP-based models have systematic visual biases — certain visual features dominate others regardless of context
2. This might explain why frozen CLIP features underperformed DINOv2 for our task: CLIP is optimized for text-image correspondence, not pure visual discrimination
3. The paper suggests that SSL-only models (like DINOv2) may be better for fine-grained visual-only tasks than VLM models

---

## 5. Modern Self-Supervised Learning Foundations [NEW]

### [SimCLR: A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709) — ICML 2020 (Chen et al., Google Brain)
**Core insight:** Contrastive learning of visual representations requires three key ingredients: (1) a composition of strong data augmentations, (2) a learnable nonlinear projection head between representation and contrastive loss, and (3) large batch sizes.

**Key contributions:**
- **Data augmentation composition is critical**: random crop + color distortion + Gaussian blur gives best representations
- **Projection head**: MLP that transforms representations before contrastive loss; discarded after training
- Larger batch sizes (≥ 256) work better due to more negative samples
- Longer training (≥ 1000 epochs) continues to improve linearly
- Simpler architecture than MoCo (no memory bank) but requires large batch size

**Our application:** SimCLR is a landmark paper that established the SSL paradigm we rely on. The key practical takeaway for our fine-tuning: **strong augmentation composition matters**. Our ConvNeXt training already uses RandAugment + Mixup + CutMix, following SimCLR's lesson that augmentation diversity is the most impactful hyperparameter.

### [BYOL: Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733) — NeurIPS 2020 (Grill et al., DeepMind)
**Core insight:** Contrastive learning doesn't require negative samples. BYOL trains an online network to predict the target network's representation of a different augmented view — and the target network is simply an exponential moving average (EMA) of the online network. No collapse occurs because the EMA acts as an implicit regularizer.

**Key contributions:**
- **No negative pairs needed**: online network predicts target representation via a predictor MLP
- **Target network = EMA of online**: θ_target ← τ·θ_target + (1−τ)·θ_online
- Collapse is avoided because the target network "lags behind" and provides stable, slowly-changing targets
- Outperforms SimCLR while using smaller batch sizes (no need for many negatives)
- Works with both ResNet and ViT architectures

**Our application:** The EMA target network concept is directly relevant: our ConvNeXt soup could be reformulated as an EMA throughout training (not just at the end), potentially stabilizing training and reducing overfitting. EMA typically gives 0.5-1% improvement over best single checkpoint.

### [MoCo v3: Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057) — ICCV 2021 (Chen et al., FAIR)
**Core insight:** Vision Transformers, despite lacking CNN inductive biases, can be effectively trained with self-supervised contrastive learning. However, ViTs exhibit training instability (spikes in loss) that requires careful handling (frozen patch projection, specific optimizer settings).

**Key contributions:**
- First systematic study of SSL for ViTs (MoCo v3 framework)
- ViT-B/16 with MoCo v3 pre-training reaches 76.7% linear probing on ImageNet — competitive with supervised
- **Instability problem**: ViT training with contrastive loss exhibits occasional "dips" in accuracy, caused by gradient spikes in the patch projection layer
- **Solution**: freeze the patch projection layer during SSL pre-training
- Larger ViTs (ViT-L, ViT-H) benefit more from SSL pre-training than smaller ones

**Our application:** When fine-tuning DINOv2 ViT, we observed unstable training (erratic validation F1). MoCo v3's insight suggests that freezing the patch embedding layer during the early epochs of fine-tuning might stabilize training — and we should consider using a smaller learning rate for the patch embedding than the rest of the model.

---

## 6. Parameter-Efficient & Calibrated Fine-Tuning [NEW]

### [AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition](https://arxiv.org/abs/2205.13535) — NeurIPS 2022 (Chen et al., HKU + Tencent)
**Core insight:** Instead of full fine-tuning or adding prompts (VPT), insert lightweight bottleneck modules (AdaptMLP) in parallel to the FFN layers of a frozen ViT. This achieves better accuracy than VPT with comparable parameter efficiency.

**Key contributions:**
- **AdaptMLP**: Down-project → ReLU → Up-project bottleneck placed parallel to ViT's MLP block
- Combines with original features via learnable scaling factor
- Outperforms VPT on video action recognition and image classification benchmarks
- ~1% extra parameters but matches or exceeds full fine-tuning
- The parallel design is key: adapts features that the frozen backbone *missed*, rather than replacing backbone features

**Our application:** AdaptFormer is more powerful than VPT because it operates on intermediate features, not just input tokens. For our meteorite classification:
1. Freeze DINOv2 Small backbone → only train AdaptMLP modules (~0.2M params)
2. This prevents catastrophic forgetting of pretrained features while learning domain-specific patterns
3. **Untried direction with high potential** — combines the regularization benefits of frozen backbone with the expressive power of learned adaptation

### [ConvNeXt V2: Co-designing ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) — CVPR 2023 (Woo et al., KAIST + Meta AI)
**Core insight:** Masked autoencoders (MAE) designed for ViTs don't work well when naively applied to ConvNets due to "feature collapse." ConvNeXt V2 introduces Fully Convolutional MAE (FCMAE) and Global Response Normalization (GRN) to make MAE work for ConvNets.

**Key contributions:**
- **FCMAE**: Sparse convolution-based masking (only processes unmasked pixels, unlike ViT which needs mask tokens)
- **GRN**: New normalization layer that prevents feature collapse in ConvNet MAE training
- ConvNeXt V2 + FCMAE pre-training achieves state-of-the-art among ConvNet architectures
- The co-design principle: architecture and pre-training method must evolve together

**Our application:** ConvNeXt V2 is directly relevant for future iterations:
1. Upgrading from ConvNeXt V1 to V2 backbone would likely give a direct accuracy boost
2. The GRN normalization layer specifically addresses overfitting in small-data regimes by preventing feature collapse
3. **Low-effort upgrade**: simply swapping `convnext_tiny` for `convnextv2_tiny` in our pipeline
4. However, timm may not have ConvNeXt V2 weights readily available — need to check

---

## 7. Updated Recommendations (All Papers Considered)

### Tier 1: High Impact, Low Difficulty
| Technique | Papers | What We'd Do | Expected Gain |
|-----------|--------|-------------|---------------|
| **Semi-supervised with Noisy Student** | Noisy Student (1911.04252), Rethinking PT/ST (2006.06882) | Teacher → pseudo-label 8600 mytest → train student with noise | **+2-4% F1** |
| **WiSE-FT interpolation** | WiSE-FT (2109.01903) | Interpolate early-epoch (best gen) + late-epoch (best acc) weights | **+0.5-1.5% F1** |
| **TIES-Merging k-fold models** | TIES-Merging (2306.01708), Git Re-Basin (2209.04836) | Merge 5 fold checkpoints into 1 model via TIES | **+0.5-1% F1** + no ensemble cost |
| **TENT test-time adaptation** | TENT (2006.10726) | Adapt BN stats on mytest images, then predict on test | **+1-2% F1** (bridges myval→test gap) |

### Tier 2: Medium Impact, Medium Difficulty
| Technique | Papers | What We'd Do | Expected Gain |
|-----------|--------|-------------|---------------|
| **Model calibration + meta-learner** | Guo et al. (1706.04599), Wolpert (1992) | Platt-scale both models, then meta-learn on kfold val | **+1-2% F1** |
| **AdaptFormer on DINOv2** | AdaptFormer (2205.13535) | Freeze DINOv2, train AdaptMLP modules only | **+0.5-1.5% F1** |
| **SAM optimizer** | SAM (2010.01412) | Replace AdamW with SAM for ConvNeXt fine-tuning | **+0.3-0.8% F1** |
| **ConvNeXt V2 upgrade** | ConvNeXt V2 (2301.00808) | Swap backbone to convnextv2_tiny | **+0.3-0.5% F1** |

### Tier 3: High Risk / Speculative
| Technique | Papers | Challenge |
|-----------|--------|-----------|
| **Task arithmetic editing** | Task Arithmetic (2212.04089) | Useful for multi-task but we have binary classification — limited benefit |
| **ZipIt! cross-task merging** | ZipIt! (2305.03053) | Our models share the same task, so simpler merging (TIES) is better |
| **FixMatch semi-supervised** | FixMatch (2001.07685) | Requires i.i.d. assumption between labeled and unlabeled — mytest ≠ train |
| **Test-Time Training** | TTT (1909.13231) | Computationally expensive; TENT is simpler and sufficient |

### Key Insight from Literature Synthesis

The papers collectively suggest a **pipeline** rather than isolated tricks:

1. **Train** with SAM on ConvNeXt V2 backbone → best single model
2. **Apply WiSE-FT** interpolation → balance accuracy vs robustness
3. **Pseudo-label** mytest via Noisy Student self-training → leverage unlabeled data
4. **TENT adaptation** on mytest → reduce distribution shift at inference
5. **TIES-Merge** k-fold checkpoints → single model with ensemble benefits
6. **Conformal calibration** → control FP rate with statistical guarantees

The gap to 0.80 F1 likely requires *all* of these — not any one alone — combined with the ensemble diversity we've already built.

---

## 8. Summary Table: All Papers Reviewed (25 Papers)

| Paper | ID | Venue | Key Contribution | Our Status |
|-------|-----|-------|-----------------|------------|
| **Model Merging** ||||
| Model Soups | 2203.05482 | ICML 2022 | Weight averaging of fine-tuned models | **Used** (soup baseline) |
| SWA | 1803.05407 | UAI 2018 | Averaging along SGD trajectory | **Used** (soup variant) |
| Git Re-Basin | 2209.04836 | ICLR 2023 | Permutation symmetry enables cross-run merging | Untried |
| TIES-Merging | 2306.01708 | NeurIPS 2023 | Resolve interference when merging task vectors | **Untried, high priority** |
| Task Arithmetic | 2212.04089 | ICLR 2024 | Add/subtract task vectors to edit models | Untried (speculative) |
| ZipIt! | 2305.03053 | ICLR 2024 | Merge models from different tasks without training | Not applicable |
| **Semi-Supervised Learning** ||||
| FixMatch | 2001.07685 | NeurIPS 2020 | Simple SSL with confidence thresholding | Untried (i.i.d. concern) |
| Noisy Student | 1911.04252 | NeurIPS 2020 | Self-training with noise beats teacher | **Untried, high priority** |
| Rethinking PT & ST | 2006.06882 | NeurIPS 2020 | Pre-training + self-training are complementary | Contextual insight |
| **Test-Time & Robustness** ||||
| TENT | 2006.10726 | ICLR 2021 | Test-time entropy minimization for BN layers | **Untried, high priority** |
| Test-Time Training | 1909.13231 | ICML 2020 | Self-supervision at test time for adaptation | Untried (complex) |
| WiSE-FT | 2109.01903 | CVPR 2022 | Interpolate zero-shot + fine-tuned weights | **Untried, high priority** |
| Bag-of-Words VLMs | 2210.01936 | ICLR 2023 | CLIP has compositional reasoning failures | Insight (explains CLIP gap) |
| **Self-Supervised Learning** ||||
| SimCLR | 2002.05709 | ICML 2020 | Contrastive learning with strong augmentations | Background |
| BYOL | 2006.07733 | NeurIPS 2020 | SSL without negative samples (EMA target) | EMA concept useful |
| MoCo v3 | 2104.02057 | ICCV 2021 | ViT self-supervised training instability | Explains DINOv2 instability |
| DINOv2 | 2304.07193 | TMLR 2024 | SSL foundation model for vision | **Used** (SOTA model) |
| MAE | 2111.06377 | CVPR 2022 | Masked autoencoding for SSL | **Tried** (failed) |
| SigLIP | 2303.15343 | ICCV 2023 | Sigmoid loss enables small-batch contrastive pretraining | Background |
| **Fine-Tuning Strategies** ||||
| DeiT | 2012.12877 | ICML 2021 | Distillation for data-efficient ViTs | Untried (teacher-student) |
| Scaling ViT | 2106.04560 | ICML 2022 | ViT scaling laws; small model optimal for small data | Explains DINOv2 Small > Base |
| VPT | 2203.12119 | ECCV 2022 | Learnable prompts for frozen ViT adaptation | Untried |
| AdaptFormer | 2205.13535 | NeurIPS 2022 | Bottleneck adapters for frozen ViTs | **Untried, promising** |
| ConvNeXt V2 | 2301.00808 | CVPR 2023 | FCMAE + GRN for ConvNet self-supervised learning | **Untried, easy upgrade** |
| **Evaluation & Theory** ||||
| Deep Ensembles | 1612.01474 | NeurIPS 2017 | Model averaging for accuracy + calibration | **Used** (kfold theory) |
| Model Calibration | 1706.04599 | ICML 2017 | Modern NNs are miscalibrated | **Must fix before fusion** |
| SAM | 2010.01412 | ICLR 2021 | Optimize for flat minima directly | Untried |
| Conformal Prediction | 2107.07511 | Book 2022 | Distribution-free prediction sets | Untried |
| **Architecture** ||||
| ConvNeXt | 2201.03545 | CVPR 2022 | Modernized ConvNet beats ViT | **Used** (SOTA backbone) |
| Swin Transformer | 2103.14030 | ICCV 2021 | Hierarchical ViT with shifted windows | Not tried |
| CLIP | 2103.00020 | ICML 2021 | Contrastive language-image pretraining | **Used** (frozen probes, failed) |

---

## Appendix: Our Experimental Results in Context

| Experiment | Result | Literature Explanation |
|------------|--------|------------------------|
| Frozen probe (SigLIP+CLIP+DINOv2 → MLP) | test 0.682 (failed) | VPT/AdaptFormer papers: frozen *features* ≠ frozen *attention*. Probing only captures what's in the [CLS] token; adaptation inside the Transformer (prompts/adapters) is far more expressive. |
| MAE domain pretraining on 9k images | myval 0.564 (regression) | MAE paper: requires large-scale pretraining. 9k images destroy ImageNet priors without replacing them. |
| DINOv2 Small > DINOv2 Base | test 0.7196 > 0.7070 | Scaling ViT: optimal model size depends on data quantity. We are on the left side of the curve. |
| K-fold ensemble (5 ConvNeXt + DINOv2) | 126 pos, 28 diffs | Deep Ensembles + Krogh-Vedelsby: diversity from data splits creates complementary errors. |
| Verifier-guided hybrid | 124 pos, 4 diffs | Conformal-like: risk-scored flipping is an ad-hoc implementation of prediction set adjustment. |
| Meta-learner on myval | Conservative (104 pos) | Calibration paper: uncalibrated probabilities + distribution shift = systematic bias in meta-learner decisions. |
| Myval ≠ test distribution | ~0.68 myval vs 0.72 test | WiSE-FT: fine-tuning destroys OOD robustness; myval is OOD relative to the fine-tuned model. TENT could help. |
