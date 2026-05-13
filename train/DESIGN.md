# Design

当前设计目标：把原先依赖 DINO/transformers 的 backbone 全面替换为 timm ConvNeXt V1 Tiny，并保持训练-校准-推理链路闭环。

## Backbone Strategy

- 统一 backbone: convnext_tiny（timm）。
- 默认使用 timm 预训练权重；也允许通过 backbone-checkpoint 注入本地权重。
- 不再依赖 ModelScope、transformers、AutoModel。

## Finetuning Strategy

- 两阶段训练：
  - head-only：冻结整个 backbone，仅训练分类头，快速对齐任务分布。
  - finetune：解冻最后 N 个 block（默认 2），做轻量深度适配。
- 优化器采用 AdamW，分类头和 backbone 使用独立学习率。

## Data And Calibration Strategy

- 训练前支持 skip 列表过滤，先清洗再 split，避免先验统计被异常样本污染。
- 验证集拆成 threshold-search / model-selection 双子集：
  - threshold-search 用于找最佳阈值。
  - model-selection 用于 checkpoint 选择，减少阈值搜索过拟合。
- 验证子集按目标负正比重采样（默认约 4.06:1），并结合 Bayes prior correction 缓解训练集和评测分布偏移。
- 验证数据目录不再依赖单一固定位置；默认从 `--val-root` 派生，但可单独覆盖 images/labels 路径，便于快速切换新的 validation set。

## Augmentation And Inference Strategy

- 训练增强：flip + rotate + CutMix + soft-target CE。
- 推理支持 deterministic 几何 TTA（4way/8way）。
- 推理流程固定为：概率输出 -> 可选 Bayes 修正 -> 阈值化 -> CSV 导出。
- bagging 场景下，直接对多次推理导出的 `prob_pos_corrected` 做均值，再统一用固定阈值生成最终提交。
- 伪标签属于标准 self-training 变体：对未标注样本先生成概率，再按高置信度阈值筛选后并入训练集。

## Tracking And Reproducibility

- 训练参数与元数据写入 train_args.json + metadata.json。
- 关键 checkpoint: best.pt / last.pt。
- best.pt 默认按验证集 val F1 选择；可通过 --early-stop N 启用连续 N 个 epoch 无提升即停止。
- 可选 W&B 记录 step/epoch 指标，支持快速回溯与对比实验。
