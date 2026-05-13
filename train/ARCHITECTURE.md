# Architecture

当前代码库是一个围绕 timm ConvNeXt V1 Tiny 的二分类训练与推理流水线，目标任务是陨石图片二分类（正类约定为 label == 1）。

## Core Entrypoints

- train_finetune.py
  - 训练主入口。
  - 负责数据清洗、分层切分、重采样、训练、阈值搜索、checkpoint 产出。
  - 训练与验证图片固定使用 `--mask-dir` 下的去背景图片；验证标签路径可通过 `--val-root` 或 `--val-labels-csv` 切换。
- infer_submission.py
  - 推理主入口。
  - 读取 checkpoint + metadata，恢复图像预处理与阈值配置，导出 submission CSV。
  - 可选导出概率明细 CSV，便于后续 bagging。
- bagging-helper.py
  - 聚合多个推理概率 CSV，对 `prob_pos_corrected` 求均值，再按阈值导出最终 submission CSV。
  - 默认同时导出平均后的概率文件，便于下一轮 bagging 或伪标签。

## Module Responsibilities

- modeling.py
  - 使用 timm.create_model(backbone_name, num_classes=0) 构建 backbone。
  - 分类器结构：ConvNeXt backbone + Dropout + Linear。
  - 支持从本地 checkpoint 加载 backbone 权重（自动兼容 backbone.* 或完整 model 字段前缀）。
  - 提供分阶段冻结策略：
    - head-only 阶段：冻结整个 backbone。
    - finetune 阶段：解冻最后 N 个 block 容器元素（默认按 stages/features/blocks 自动识别）。
  - 提供 transforms 与优化器构建工具。

- data.py
  - 负责图片索引、样本过滤、分层切分、验证子集二次切分与比例重采样。
  - MeteoriteDataset 返回 (pixel_values, label_idx)。

- augmentations.py
  - 训练增强与 soft target 逻辑：CutMix、soft label 构造、soft-target cross entropy。

- calibration.py
  - 处理先验估计、类权重、Bayes prior correction、阈值搜索、F1 指标。

- tta.py
  - 推理期几何 TTA（4way/8way）与概率聚合。

- utils.py
  - 统一管理默认参数（backbone=convnext_tiny）、路径、随机种子、JSON 落盘。

- wandb_utils.py
  - 实验追踪初始化、配置与 summary 记录。

## Training Flow

1. 读取标签 CSV，并可选根据 skip-image-ids-txt 过滤异常样本。
2. 将原始标签映射为 label_idx（二分类）。
3. 从 `mask/train` 读取去背景图片；train/val 分层切分；val 再切分为 threshold-search 与 model-selection 两个子集。
4. 对两个验证子集按目标负正比重采样（默认约 4.06:1）。
5. 构建 ConvNeXtClassifier。
6. Stage 1（head-only）：冻结 backbone，仅训练分类头。
7. Stage 2（finetune）：解冻最后 trainable-blocks 个 block，继续联合训练。
8. 每轮在 threshold-search 子集搜索最佳阈值，在 model-selection 子集评估 val F1 并决定是否更新 best.pt；可选启用 early-stop，连续 N 个 epoch 的 val F1 不再提升就提前结束。
9. 保存 last.pt、best.pt、history.json、metadata.json、train_args.json。

## Inference Flow

1. 读取 checkpoint（通常是 best.pt 或 last.pt）。
2. 读取同目录 metadata.json/train_args.json，恢复 image_size/mean/std、阈值与 backbone 名称。
3. 按 backbone 名重建 ConvNeXtClassifier，然后加载完整 model state_dict。
4. 对测试集输出正类概率；可选启用几何 TTA。
5. 可选执行 Bayes prior correction，再按阈值二值化。
6. 导出 submission CSV（可选额外导出概率明细 CSV）。
7. 如果需要多次随机种子/多 checkpoint bagging，可用 bagging-helper.py 对多份概率 CSV 求均值后再二值化。
8. 训练时可选传入平均后的概率文件，按高置信度阈值筛选伪标签样本并并入训练集。

## Artifacts

默认输出目录：outputs/convnext_tiny_finetune

- train_args.json: 原始训练参数快照。
- metadata.json: 推理恢复所需元信息（label 映射、预处理、先验、阈值策略等）。
- last.pt: 每轮覆盖的最近 checkpoint。
- best.pt: 按 model-selection val F1 选出的最佳 checkpoint。
- history.json: 按 epoch 记录的训练/验证指标。
