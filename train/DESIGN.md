# Design

当前 `train/` 目录的设计目标已经不只是“把 DINO 替换成 ConvNeXt”，而是维持一条可复现实验、可校准、可 bagging、可自训练的二分类流水线，核心优化目标仍然是离线验证 F1 与线上提交的一致性。

## Design Targets

- 保持主干简单：单 backbone + 轻量分类头 + 标准交叉熵变体。
- 把分布偏移问题显式建模，而不是只靠阈值拍脑袋。
- 让训练、推理、bagging、伪标签尽量复用同一份概率语义：`prob_pos_corrected`。
- 保证每次实验都能从输出目录单独恢复主要配置，不依赖口头记忆。

## Backbone Strategy

- 默认 backbone 是 `convnext_tiny`，来源于 `timm`。
- 实现层面保留 `--backbone` 扩展口，允许换成别的兼容 timm 模型，但当前文档、默认值和实验习惯都以 `convnext_tiny` 为中心。
- 支持三种权重起点：
  - timm 预训练权重
  - 指定 `--backbone-checkpoint`
  - `--no-pretrained` 从头开始
- 设计上已经彻底去掉对 `transformers` / `AutoModel` / ModelScope 的依赖。

## Optimization Strategy

- 训练调度是“可选 warmup + 全量 finetune”，而不是旧版的“只解冻最后几个 block”：
  - `head_only_epochs > 0` 时，先冻结整个 backbone，只训练分类头。
  - 进入 finetune 阶段后，直接解冻整个 backbone。
- 优化器统一用 `AdamW`。
- 分类头与 backbone 使用独立学习率：
  - `--head-lr`
  - `--backbone-lr`
- backbone 的学习率不是单一值，而是通过 LLRD 逐层衰减：
  - 越靠近输入层，学习率越小。
  - `--llrd-decay` 控制相邻层组的缩放比。
- CUDA 上默认启用 AMP，以减少显存压力并保持训练吞吐。

## Data Strategy

- 训练主视图是 mask 图，而不是原图。
  - 这意味着主模型更接近“区域裁切后的分类器”，不是端到端检测器。
- 训练前支持 skip list 清洗，且清洗发生在 split 之前，避免污染先验统计。
- 支持两种验证来源：
  - 外部验证集：默认 `../data/myval + ../mask/myval`
  - 训练集内部分层切分：通过 `--val-split-ratio`
- 支持 `mytest` 作为额外高质量外部数据源：
  - `mytest_train` 并入训练集
  - `mytest_val` 作为主验证集
  - `myval` 封存为最终参考集，只在判别阶段使用
  - 默认按文件名 metadata group 做分层切分，避免同源样本跨 train/val 泄漏
- 支持 `--train-sample-ratio` 做快速小样本实验，不需要手动改 CSV。
- 伪标签直接并入同一个 DataLoader，不额外维护第二条训练支路。
  - 置信度阈值由 `--pseudo-prop` 控制。
  - 样本影响力由 `--pseudo-weight` 控制。

## Calibration And Model Selection Strategy

- 设计重点之一是把“训练分布”和“目标评测分布”拆开处理。
- `target_neg_pos_ratio` 同时影响两件事：
  - 验证子集的重采样比例
  - class weight / Bayes correction 的目标先验
- 验证集被拆成两个职责不同的子集：
  - `threshold-search`
    - 专门负责找阈值。
  - `model-selection`
    - 专门负责选 checkpoint。
- 这种拆分的目的，是降低“同一批样本既调阈值又挑模型”带来的过拟合。
- `prob_pos_corrected` 是设计上的统一概率语义：
  - 先拿模型输出的 `prob_pos`
  - 再按 train prior 和 target prior 做 Bayes prior correction
- 默认情况下，threshold-search 还是关闭的：
  - 不传 `--open-threshold-search` 时，固定阈值为 `0.5`
  - 打开后，才会在 `threshold-search` 子集上搜索最佳 F1 阈值
- checkpoint 选择最终看的是 `model-selection` 子集上的校正后 F1，而不是训练 loss，也不是 threshold-search 本身的最优分数。

## Augmentation Strategy

- 当前增强刻意保持保守，重点在稳定而不是花哨：
  - resize
  - horizontal flip
  - small-angle rotation
  - CutMix
- loss 侧使用 soft-target cross entropy 来兼容 CutMix。
- `sample_weight` 会贯穿整个 loss 计算，因此伪标签可以低权重混入，而不用单独写 loss 分支。

## Inference Strategy

- 推理阶段优先复用训练输出目录中的 metadata，而不是要求用户手工重复输入所有预处理参数。
- 默认输出两层语义：
  - submission label
  - 可选概率明细 CSV
- 推理图像源可以切换：
  - 原图
  - `mask/test`
  - `flip-mask` 背景图
- TTA 只使用 deterministic 几何变换，避免引入不可追溯的随机性。
- 概率后处理顺序固定为：
  - forward probability
  - optional Bayes correction
  - thresholding
  - CSV export

## Ensemble And Self-Training Strategy

- bagging 不聚合 hard label，只聚合 `prob_pos_corrected`。
- 这样做的原因是：
  - 阈值逻辑只保留一份
  - 后续还能继续把平均概率拿去做伪标签
- `run_kfold_bagging.py` 的角色是编排器，不重写训练逻辑。
  - fold 训练仍调用 `train_finetune.py`
  - fold 推理仍调用 `infer_submission.py`
  - 最终聚合仍调用 `bagging-helper.py`
- self-training 也是沿用同样的概率接口：
  - 先产出概率 CSV
  - 再筛高置信度样本
  - 最后把伪标签样本并回训练集

## Tracking And Reproducibility

- 每次训练都会同时写出：
  - `train_args.json`
  - `metadata.json`
  - `history.json`
  - `best.pt`
  - `last.pt`
- checkpoint 里会冗余保存阈值、先验、增强和训练阶段信息，避免“只剩 pt 文件时无法复盘”。
- W&B 是可选项，但输出目录内的 JSON/pt 文件始终是第一追溯源。
- 早停逻辑以验证 F1 为基准，而不是训练指标。

## Test-Time Adaptation & Distribution Shift

- `optimize_from_literature.py` 整合了文献驱动的三项优化（ICLR 2021-2024 论文支撑）：
  - **BN Stats Adaptation (TENT-lite)**：对目标分布 (mytest) 更新 BN running stats，无需标签
    - ConvNeXt 使用 BatchNorm → 有效
    - DINOv2 使用 LayerNorm → BN adaptation 不适用
  - **WiSE-FT Checkpoint Interpolation**：对早期+晚期 epoch checkpoint 做线性插值
    - θ_wise = (1-α)·θ_early + α·θ_late
    - α ∈ {0.3, 0.5, 0.7} 网格搜索
  - **Noisy Student Pseudo-Label Generation**：高置信度 (>0.9) 伪标签生成
    - 预测结果保存到 `train/outputs/lit_opt_v1/noisy_student_pseudo_labels.csv`
    - 可直接作为 train_finetune.py 的 `--pseudo-prob-csv` 输入
- BN 适配后的 ConvNeXt 预测倾向于更保守（更多 NEG），概率更尖锐
- 设计原则：改编自 TENT (2006.10726, ICLR 2021) — 仅在 target distribution 上 update BN stats

## Ensemble & Disagreement Resolution (Updated)

- **关键发现**: 6 模型 kfold 平均集成 (5×ConvNeXt + DINOv2) → test F1=0.67924 (比 SOTA 低 4%)
  - 原因: 简单平均引入过多错误分歧，kfold 模型看到不同数据子集导致系统偏差
  - 教训: 集成需要更智能的融合策略，不是简单多模型平均
- **Verifier-Guided Hybrid**: 仅翻转 verifier risk score >0.7 的 top-4 候选 → test F1=0.71428 (-0.5%)
  - 4 removals 中约 2-3 是 FP，1-2 是误杀的真阳性
  - 验证了 verifier 风险评分的方向正确性
- **Refined Conservative Strategy** (当前最优未提交):
  - 从 SOTA baseline 出发，仅翻转双方模型都高置信度一致否定的预测
  - 仅 5 处修改 (4 removes + 1 add) — 最小风险
  - Criteria: CN_prob < 0.25 AND DS_prob < 0.35 AND ensemble < 0.25 → NEG (remove SOTA-POS)
  - Criteria: CN_prob > 0.65 AND DS_prob > 0.6 AND ensemble > 0.55 → POS (add over SOTA-NEG)

## Known Tradeoffs

- 当前主线高度依赖 mask 质量；如果 mask 漏掉关键区域，分类器没有额外机制补救。
- 验证集重采样和 Bayes correction 都依赖目标负正比估计；这个估计错了，校准也会被一起带偏。
- threshold-search 默认关闭，意味着很多实验实际上是在固定 `0.5` 阈值下完成 model selection。
- 代码虽然支持换 timm backbone，但文档和经验参数主要围绕 `convnext_tiny`，直接换 backbone 仍应视为新实验而不是无缝替换。
- BN adaptation 对 ConvNeXt 有效 (BN 层)，但对 ViT/DINOv2 (LN 层) 无效 — 需要完整 TENT (更新 affine params) 或完全不同的适配策略。
- myval 与 test 分布存在差异：myval 上表现好的模型在 test 上未必最优，反之亦然。这增加了模型选择和超参调优的难度。
- kfold ensemble 虽然理论上有减少方差的优势，但在 4780 张图片数据量下，每折仅 3824 张训练图片，每个 fold 模型都可能欠拟合。
