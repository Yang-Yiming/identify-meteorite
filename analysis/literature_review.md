# Literature Review: Ensemble Techniques for Small-Dataset Fine-Tuning

*Compiled by researching papers via AlphaXiv and arxiv, with findings contextualized to our meteorite classification project.*

**Project Context:** We have 4,780 labeled bbox-crop images (binary classification: is_meteorite) and ~8,600 unlabeled mytest images. Fine-tuning pretrained models (ConvNeXt, DINOv2) achieves ~0.72 test F1. The gap to top entries (≥0.80 F1) lies in smarter ensemble construction and uncertainty-driven prediction.

---

## 1. Model Soups & Weight Averaging

### [Model Soups](https://arxiv.org/abs/2203.05482) (Wortsman et al., 2022)
**Core insight:** Fine-tuned models from the same pretrained backbone, even with different hyperparameters, reside in the same low-error basin. Averaging their weights (rather than their predictions) retains this low error while often exceeding any individual model — with zero inference cost increase.

**Key contributions:**
- **Uniform soup**: Simply average weights of all fine-tuned models
- **Greedy soup**: Sequentially add models to the soup only if they improve validation accuracy
- Fine-tuned CLIP and ViT models attain strong ImageNet accuracy via greedy soup on 60+ fine-tuned variants

**Our application:** We trained ConvNeXt Tiny soups (greedy averaging over epochs) that achieved SOTA (0.71962). Weight-averaging within a single training run (SWA variant) provided 0.3-0.5% improvement over best single checkpoint without extra inference cost.

### [Stochastic Weight Averaging (SWA)](https://arxiv.org/abs/1803.05407) (Izmailov et al., 2018)
**Core insight:** SGD with constant or cyclical learning rates explores the periphery of wide minima; averaging these points finds a solution centered in the wide basin, yielding better generalization.

**Key contributions:**
- Simple: average model weights along the SGD trajectory after convergence
- Wide optima consistently generalize better than sharp ones
- Works with any architecture, no extra hyperparameters needed

**Our application:** Our ConvNeXt "soup" training uses a variant of this — averaging weights from epoch 15-20 with constant LR. This is distinct from the "model soups" approach of averaging across independent fine-tunings, but both exploit the same geometric property of the loss landscape.

### [Sharpness-Aware Minimization (SAM)](https://arxiv.org/abs/2010.01412) (Foret et al., 2021)
**Core insight:** Rather than finding wide minima post-hoc (like SWA), SAM directly optimizes for them by minimizing the maximum loss in a neighborhood around current parameters.

**Key contributions:**
- Simultaneously minimizes loss value and loss sharpness
- Achieves state-of-the-art on CIFAR, ImageNet without extra data
- Compatible with any existing optimizer

**Potential for our project:** Could replace or augment SWA-style averaging. Since SAM directly seeks flat minima, models fine-tuned with SAM might generalize better on our 4,780-image dataset where overfitting is the primary concern. **This is an untried direction.**

---

## 2. Deep Ensembles & Model Fusion

### [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474) (Lakshminarayanan et al., 2017)
**Core insight:** Training multiple models independently (with different random seeds) and averaging their predictive distributions provides both accuracy gains and well-calibrated uncertainty estimates — without the complexity of Bayesian methods.

**Key contributions:**
- Ensembling 5 independently trained models gives state-of-the-art uncertainty calibration
- Disagreement between ensemble members captures epistemic (model) uncertainty
- Proper scoring rules + adversarial training further improves calibration
- Simpler and more scalable than MC-dropout, variational inference, etc.

**Our application:** This is the theoretical foundation for our k-fold ensemble. Rather than training 5 models with different seeds (which would have 80% overlapping training data on 4,780 images), we use 5-fold cross-validation splits to ensure each model sees a different training subset — inducing stronger epistemic diversity.

### [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (Guo et al., 2017)
**Core insight:** Modern deep networks (ResNet, DenseNet, etc.) are **poorly calibrated** — their predicted confidence significantly overestimates the true likelihood of correctness. Temperature scaling (a single scalar parameter) is a simple and effective fix.

**Key contributions:**
- Deep networks are systematically overconfident (expected calibration error rising with depth/capacity)
- Batch normalization and low weight decay are contributing factors
- Temperature scaling on a held-out validation set dramatically improves calibration
- "Modern" networks calibrated worse than older, shallower ones

**Our application:** This is critical for our verifier-guided approach. If ConvNeXt and DINOv2 produce miscalibrated probabilities, their raw disagreement patterns cannot be trusted. Before using model disagreement to decide which predictions to flip, we should calibrate both models (e.g., via Platt scaling or temperature scaling on myval).

### Collaborative Decision Between Ensembles
**Key papers (background):**
- **Wolpert (1992)** — "Stacked Generalization": Train a meta-learner on top of base model predictions. The meta-learner learns to correct systematic errors in each base model.
- **Breiman (1996)** — "Bagging predictors": Bootstrap aggregation creates model diversity through data resampling, reducing variance.
- **Breiman (2001)** — "Random Forests": Combines bagging with random feature selection for further decorrelation.

**Our application:** We attempted a meta-learner (logistic regression on both models' probabilities) but trained on myval, which is distributionally different from the test set. The meta-learner became too conservative (104-108 positives vs. 120 in individual models). The correct approach would be:
1. Calibrate both models first
2. Use k-fold within the training set or a proper validation split to train the meta-learner
3. Use meta-learner confidence estimates, not raw output, to inform ensemble decisions

---

## 3. Small-Dataset Fine-Tuning of Pretrained Models

### [Training Data-Efficient Image Transformers (DeiT)](https://arxiv.org/abs/2012.12877) (Touvron et al., 2021)
**Core insight:** Vision Transformers can be trained on ImageNet-1k (1.3M images) without external data, matching CNN performance, through knowledge distillation from a CNN teacher and aggressive data augmentation.

**Key contributions:**
- **Distillation token**: A special token attends to the CNN teacher's output, enabling effective knowledge transfer
- Hard-label distillation outperforms soft distillation for ViT
- Extensive augmentation (RandAugment, Mixup, CutMix, repeated augmentation) is critical
- ConvNet teachers work better than Transformer teachers for distillation

**Our application:** Our dataset (4,780 images) is 270x smaller than ImageNet. DeiT's finding that ViTs need more data or stronger regularization aligns with our experience: DINOv2 ViT-B (86M params) overfits badly without heavy augmentation, while DINOv2 Small (22M) generalizes better. The distillation token approach could potentially help here by allowing a well-tuned ConvNeXt to teach DINOv2.

### [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560) (Zhai et al., 2021)
**Core insight:** ViT performance follows predictable scaling laws with model size, data size, and compute. But the key practical insight: **with small data, smaller ViTs generalize better.**

**Key contributions:**
- ViT accuracy follows power-law scaling with model size and data
- For fixed data, there is an optimal model size — larger models eventually hurt
- Downstream transfer performance trends follow similar scaling patterns
- Scaling data is more impactful than scaling model size for downstream tasks

**Our application:** This directly explains why DINOv2 Small (22M) outperforms DINOv2 Base (86M) on our dataset (test 0.7196 vs 0.7070). We are on the left side of the scaling curve where model capacity hurts. Further confirms that efficient architectures (ConvNeXt Tiny, 28M) are optimal for our data scale.

### [Visual Prompt Tuning (VPT)](https://arxiv.org/abs/2203.12119) (Jia et al., 2022)
**Core insight:** Instead of full fine-tuning, inject a small number of learnable "prompt" tokens into the input sequence of a frozen pretrained ViT. Achieves near full-fine-tuning performance with <1% of trainable parameters.

**Key contributions:**
- **VPT-Shallow**: Add prompts to input only — simplest, most parameter-efficient
- **VPT-Deep**: Add prompts at every Transformer layer — more expressive
- Competitive with full fine-tuning across 24 downstream tasks
- Stores only the prompt vectors per task (e.g., ~0.1M params vs 86M for ViT-B)

**Our application:** We attempted frozen feature probing (SigLIP+CLIP+DINOv2 features → MLP head) but it underperformed (test 0.682). VPT is fundamentally different: prompts are learned *within* the Transformer, modulating attention patterns, not just at the classification head. VPT should outperform frozen probing on our data. **This is an untried direction** that could reduce overfitting by keeping the backbone frozen.

### Parameter-Efficient Tuning and Regularization for Small Datasets
**Background from multiple papers (2103.00020, 2101.00123, etc.):**
- CLIP zero-shot already impressive, but fine-tuning with strong dropout, stochastic depth, and weight decay is essential on <10k images
- Using larger image resolution (288→518) helps but requires smaller batch sizes that introduce gradient noise instability
- Augmentation strategy (RandAug, Mixup, CutMix) is often more impactful than architecture choice on small datasets

**Our application:** ConvNeXt at 288px with batch_size=32 (aggressive augmentations) consistently beats all other configurations. DINOv2 at 518px with batch_size=8 works but is noisy. The observation that augmentation is more impactful than architecture mirrors the literature.

---

## 4. Self-Supervised Learning & Domain Adaptation

### [Masked Autoencoders Are Scalable Vision Learners (MAE)](https://arxiv.org/abs/2111.06377) (He et al., 2021)
**Core insight:** A simple asymmetric encoder-decoder architecture, where a high-capacity encoder sees only 25% of image patches (visible tokens) and a lightweight decoder reconstructs all patches from encoded latents + mask tokens, enables efficient and scalable self-supervised pretraining.

**Key contributions:**
- **Asymmetric design**: Encoder runs only on visible patches (75% masked), decoder is lightweight
- Masking 75% of patches makes the task non-trivial, forcing semantic understanding
- Linear probing and fine-tuning transfers surpass supervised pretraining
- Scales well: ViT-Huge achieves 87.8% ImageNet top-1 with no labels

**Our application:** We attempted MAE domain pretraining on ~9,000 unlabeled stone images (mytest + subset from web), then fine-tuned on labeled data. Result: myval F1 dropped from 0.679 to 0.564. **The failure is consistent with the paper's insights:**
- MAE pretraining removes the well-curated priors from ImageNet/CLIP pretraining
- Our 9k domain images may be insufficient quality/diversity to learn useful representations
- The paper demonstrates MAE requires large-scale pretraining (ImageNet-1k+) to be effective
- On very small domain data, MAE is likely worse than simply using a good general-purpose pretrained model

### [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (Oquab et al., 2023)
**Core insight:** Self-supervised pretraining on large, carefully curated data produces frozen features that work "out of the box" across diverse tasks and image distributions — approaching NLP-style foundation model behavior.

**Key contributions:**
- **Automatic data curation pipeline**: Clusters images from uncurated web data, selects diverse and high-quality subsets (LVD-142M dataset)
- Combines discriminative (DINO) and reconstructive (iBOT) objectives, plus KoLeo regularization
- Frozen features match or exceed weakly-supervised (SWAG) and text-guided (OpenCLIP) models
- At pixel level: produces sharper, more accurate patch-level features than prior SSL methods

**Our application:** DINOv2 Small fine-tuned on 4,780 images achieved 0.71962 — the first non-ConvNeXt model to match our SOTA. DINOv2's strong pretraining priors survive fine-tuning better than CLIP's or MAE's. The 22M parameter count is near-optimal for our data scale per Scaling ViT findings.

---

## 5. Evaluation Metrics & Model Selection

### Conformal Prediction
**[A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)** (Angelopoulos & Bates, 2021)

**Core insight:** Conformal prediction provides distribution-free, finite-sample guarantees on prediction sets. Given a new test point, it produces a prediction set that contains the true label with a user-specified probability (e.g., 95%), regardless of the underlying model or data distribution.

**Key concepts:**
- **Coverage guarantee**: P(y_test ∈ prediction_set) ≥ 1-α for any α, distribution-free
- Uses a held-out calibration set to compute "conformity scores" (e.g., 1 - softmax probability for true class)
- Applies threshold from calibration set to test predictions
- Can produce "abstention" when no class meets threshold

**Our application:** We implicitly applied conformal-like reasoning when building the verifier-based hybrid submission. For each test image, we computed a "risk score" combining model disagreement magnitude + prediction confidence. Images with risk > threshold were candidates for flipping. The formal conformal framework would give us statistical guarantees on the false positive risk. **Potential improvement**: use myval as a calibration set with proper conformal prediction to produce abstention decisions with guaranteed error rates.

---

## 6. Cross-Validation Ensembles & Generalization Theory

### K-Fold Cross-Validation as an Ensemble Method
**Background:** Wong (2015) "Parametric Methods for Comparing the Performance of Two Classification Algorithms" and related works establish:
- K-fold cross-validation provides out-of-sample predictions for every training point
- Models trained on different folds see different data, creating diversity
- Averaging these models approximates bootstrap aggregation but with guaranteed coverage of the entire training set

**Our application:** Our 5-fold ConvNeXt ensemble uses:
- 5 models, each trained on 3,824 images (80% of 4,780) with different augmentation
- Average probabilities → test predictions
- Key advantage over simple model soup: each model's training set is disjoint in composition, producing greater diversity than different hyperparameter settings on the same data

### Disagreement as a Signal
From the ensemble uncertainty literature:
- **Lakshminarayanan et al. (2017)**: Epistemic uncertainty ≈ variance across ensemble predictions
- **Krogh & Vedelsby (1995)**: Ensemble error = average individual error − average ambiguity. More disagreement theoretically reduces ensemble error IF individual models are accurate.
- **Dietterich (2000)**: Ensemble methods work best when base learners are accurate AND disagree

**Our application:** We observed 22 disagreements between ConvNeXt soup and DINOv2 Small at SOTA-level (both 0.71962). This is the ideal scenario: both models are "accurate enough" and they disagree, creating ambiguity that ensemble averaging can exploit. The challenge is distinguishing productive disagreement (model diversity → better average) from unproductive disagreement (one model is simply wrong).

---

## 7. Recommended Next Steps (Literature-Guided)

| Technique | Paper Support | Expected Impact | Difficulty |
|-----------|--------------|-----------------|------------|
| **Temperature/Platt calibration before meta-learner** | Guo et al. (1706.04599) | Medium — reduces false confidence in disagreements | Low |
| **Visual Prompt Tuning (VPT)** | Jia et al. (2203.12119) | High — potentially less overfitting than full FT | Medium |
| **SAM optimizer** | Foret et al. (2010.01412) | Medium — flatter minima, better generalization | Low |
| **Conformal prediction for abstention** | Angelopoulos & Bates (2107.07511) | High — statistical guarantees on FP control | Medium |
| **DeiT-style distillation** | Touvron et al. (2012.12877) | Medium — ConvNeXt teacher → DINOv2 student | Medium |
| **Cross-validation meta-learner** | Wolpert (1992), Breiman (1996) | High — proper hold-out for fusion model | Low (we can reuse kfold splits) |
| **Additional pretrained backbones** | ConvNeXt (2201.03545) | Low-Medium — data regime limits returns | Low |
| **Scaling up unlabeled domain data** | MAE (2111.06377), Scaling ViT (2106.04560) | LOW — our MAE experiment suggests quality > quantity | High |

### Rationale for Priority Ranking

1. **Calibration + CV meta-learner** (highest priority): The two submitted candidates (hybrid verifier and kfold ensemble) both make binary decisions about model disagreements. A calibrated meta-learner trained on proper cross-validation folds would learn *systematic* bias patterns in each model, not just confidence thresholds. This directly addresses the 22-disagreement bottleneck.

2. **Conformal prediction**: Formal abstention guarantees where models disagree could reduce FPs without sacrificing recall. Especially relevant since top Kaggle entries likely use abstention strategies.

3. **VPT + SAM**: Parameter-efficient tuning reduces overfitting risk (the #1 limiting factor), and SAM adds another regularization dimension. These are orthogonal improvements to ensemble construction.

4. **Scaling unlabeled domain data**: Our MAE failure suggests that more unlabeled data from the same domain won't help unless the data is of sufficient quality and diversity. The DINOv2 pretrained features are already strong.

---

## Summary Table: Papers Reviewed

| Paper | ID | Key Contribution | Relevance to Project |
|-------|-----|-----------------|---------------------|
| Model Soups | 2203.05482 | Weight averaging of fine-tuned models | Our soup baseline; kfold is next step |
| SWA | 1803.05407 | Averaging along SGD trajectory finds wide minima | ConvNeXt soup uses this |
| SAM | 2010.01412 | Optimize directly for flat minima | **Untried — high potential** |
| Deep Ensembles | 1612.01474 | Simple model averaging for accuracy + calibration | Theoretical basis for kfold ensemble |
| Model Calibration | 1706.04599 | Modern NNs are miscalibrated | **Must fix before meta-learner** |
| DINOv2 | 2304.07193 | SSL foundation model for vision | Our second-best model (0.71962) |
| ConvNeXt | 2201.03545 | Modernized ConvNet beats Transformers | Our SOTA backbone |
| Swin Transformer | 2103.14030 | Hierarchical ViT with shifted windows | Not tried; potential third backbone |
| Scaling ViT | 2106.04560 | ViT scaling laws; smaller model = better on small data | Explains DINOv2 Small > Base |
| DeiT | 2012.12877 | Distillation for data-efficient ViTs | Potential teacher-student approach |
| MAE | 2111.06377 | Masked autoencoding for SSL pretraining | Our domain-pretraining attempt failed |
| VPT | 2203.12119 | Parameter-efficient ViT fine-tuning | **Untried — less overfitting** |
| CLIP | 2103.00020 | Contrastive language-image pretraining | One backbone in early frozen probes |
| Conformal Prediction | 2107.07511 | Distribution-free uncertainty quantification | Formal basis for our verifier approach |
