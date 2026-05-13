# Architecture

`train/` 目录当前是一套围绕 `timm` backbone 的二分类训练与提交流水线，默认 backbone 是 `convnext_tiny`，目标任务仍然是陨石图片二分类，约定正类为 `label_idx == 1`。

虽然类名仍叫 `ConvNeXtClassifier`，实现已经不再硬编码 ConvNeXt 专属逻辑。只要 backbone 能通过 `timm.create_model(..., num_classes=0)` 构建，并暴露 `num_features`，理论上都可以接入；当前默认和主要实验对象仍然是 `convnext_tiny`。

## Scope And Assumptions

- 当前训练配置只支持二分类。
- 训练阶段默认依赖预先生成好的 mask 图，命名约定为 `<stem>_mask_000.png`。
- 默认数据布局如下：
  - `../data/train_labels.csv`
  - `../data/myval/labels.csv`
  - `../data/test_images/`
  - `../mask/train/`
  - `../mask/myval/`
  - `../mask/test/`
- 推理阶段可以直接读原图，也可以通过 `--use-mask` 切到 `mask/test`。

## Core Entrypoints

- `train_finetune.py`
  - 训练主入口。
  - 覆盖数据清洗、mask 索引、可选 train/val 切分、伪标签并入、两阶段训练、阈值策略、checkpoint 与 metadata 落盘。
- `infer_submission.py`
  - 单 checkpoint 推理入口。
  - 从 checkpoint 目录恢复训练期的预处理、backbone、dropout、阈值和先验修正配置。
  - 支持 mask 推理、反向 mask 推理、TTA、概率 CSV 导出。
- `bagging-helper.py`
  - 聚合多个概率 CSV。
  - 以 `prob_pos_corrected` 为主键列求均值，再统一阈值化。
- `run_kfold_bagging.py`
  - 负责生成 K-fold CSV split，串联训练、逐 fold 推理和最终 bagging。
  - 通过 `--train-extra-args` / `--infer-extra-args` 复用主脚本，不维护第二套训练逻辑。
- `pseudo_helper.py`
  - 从 `output-prob.csv` 或 `bagged_prob.csv` 这类文件中筛选高置信度伪标签。
- `detect_keyword.py`
  - 独立的 OCR 关键词检测工具，不参与主训练图，但可作为额外候选信号源。

## Module Responsibilities

- `modeling.py`
  - 用 `timm.create_model(backbone_name, num_classes=0)` 构建 backbone。
  - 分类头固定为 `Dropout + Linear`。
  - 支持从本地 checkpoint 注入 backbone 权重，并兼容 `backbone.*`、`model.backbone.*`、`encoder.*` 等常见前缀。
  - `freeze_backbone_for_head_only()` 会冻结整个 backbone。
  - `unfreeze_backbone_all()` 会在 finetune 阶段一次性解冻整个 backbone。
  - `build_backbone_llrd_param_groups()` 按 block 自动构造 layer-wise learning rate decay 参数组。
  - `resolve_backbone_data_settings()` 优先读取 timm 的默认输入尺寸与归一化统计。

- `data.py`
  - 负责原图索引和 mask 索引。
  - `build_image_index()` 会为唯一 stem 建立跨扩展名别名，减少 `.jpg/.jpeg/.png` 不一致带来的掉样。
  - `filter_dataframe_by_skip_ids()` 在 split 前清洗异常样本。
  - `stratified_split()` / `stratified_subsplit()` 用于 train/val 和 threshold-search/model-selection 两级分层切分。
  - `rebalance_binary_subset_to_ratio()` 用目标负正比重采样验证子集。
  - `build_pseudo_labeled_dataframe()` 从概率 CSV 生成带 `label_idx` 的伪标签样本。
  - `MeteoriteDataset` 返回 `(pixel_values, label, sample_weight)`，而不是旧版的二元组。

- `augmentations.py`
  - 实现 one-hot soft target、sample-weight aware soft-target cross entropy、CutMix。
  - CutMix 不只混 label，也同步混合 `sample_weight`。

- `calibration.py`
  - 负责 train/val/target prior 统计。
  - 按 target prior 与 train prior 的偏移构建 class weight。
  - 提供 Bayes prior correction、F1 计算、阈值搜索。

- `tta.py`
  - 提供 deterministic 几何 TTA。
  - 当前支持 `4way` 和 `8way` 两组 view。

- `utils.py`
  - 定义训练 CLI 默认值、随机种子、JSON 落盘工具。
  - 现有默认输出目录是 `./outputs/convnextv2_tiny_finetune`，名字仍保留了历史命名。

- `wandb_utils.py`
  - 负责 W&B run identity、初始化、summary 更新与收尾。

## Training Flow

1. 读取 `labels.csv`，并可选通过 `--skip-image-ids-txt` 先做样本过滤。
2. 将原始 `label` 映射为二分类 `label_idx`，正类固定要求映射到 `1`。
3. 根据 `--train-sample-ratio` 可选抽样训练集，用于快速实验。
4. 从 `mask/train` 建立训练图索引，自动丢弃没有对应 mask 的样本。
5. 构造验证集来源，二选一：
   - `--val-split-ratio > 0`：直接从训练集分层切出验证集，验证图像仍来自 `mask/train`。
   - 否则：读取 `--val-labels-csv` 或 `<val-root>/labels.csv`，并从 `mask/<val-mask-split>` 读取验证图像。
6. 若提供 `--pseudo-prob-csv`，则按 `--pseudo-prop` 置信度阈值筛选伪标签，并以 `--pseudo-weight` 加入训练集。
7. 将验证集再切成 `threshold-search` 与 `model-selection` 两个子集，并分别重采样到 `--target-neg-pos-ratio`。
8. 构建模型，优先采用 backbone 自带的输入尺寸、mean、std，除非用户显式覆盖。
9. 训练采用两阶段调度：
   - `head_only`：冻结整个 backbone，只训练分类头。
   - `finetune`：重新构建优化器并解冻整个 backbone，使用 LLRD。
10. 每个 epoch 都会计算：
    - train loss / accuracy / grad norm / CutMix 触发次数
    - threshold-search 子集指标
    - model-selection 子集指标
11. 阈值策略：
    - 默认不开 `--open-threshold-search`，此时固定使用 `0.5`。
    - 打开后，会在校正后的 `threshold-search` 概率上搜索最佳 F1 阈值，再拿这个阈值评估 `model-selection`。
12. checkpoint 选择依据是 `model_select_f1_corrected_search_threshold`，也就是校正后、按搜索阈值计算的验证 F1。
13. 每轮保存 `last.pt`；如果验证 F1 创新高，则覆盖 `best.pt`；传入 `--save-every-epoch` 时还会保留 `epoch_XX.pt`。
14. 若设置 `--early-stop`，连续若干轮验证 F1 无提升就提前停止。

## Inference Flow

1. 读取 `best.pt` 或 `last.pt`，并尝试从同目录加载 `metadata.json` 与 `train_args.json`。
2. 恢复运行期设置：
   - `image_size`
   - `image_mean` / `image_std`
   - `threshold`
   - `dropout`
   - `backbone_name`
3. 构建测试集 id 列表：
   - 若 `sample_submission.csv` 存在，则按其中 `id` 顺序推理。
   - 否则退化为目录内文件名排序。
4. 若开启 `--use-mask`，则把测试图替换成 `mask/test/<stem>_mask_000.png`。
5. 若同时开启 `--flip-mask`，则在推理时用原图减掉 meteorite 区域，只保留背景。
6. 若开启 `--tta`，则在 `4way` 或 `8way` view 上做概率平均；否则单视角前向。
7. 输出原始正类概率 `prob_pos`，并在 metadata 允许且未显式禁用时应用 Bayes prior correction，得到 `prob_pos_corrected`。
8. 用运行期阈值二值化，写出 submission CSV；如果传入 `--output-prob-csv`，再额外写出概率明细。

## Output Artifacts

训练目录默认会生成：

- `train_args.json`
  - 原始训练 CLI 快照。
- `metadata.json`
  - 推理恢复与实验复盘需要的元数据。
  - 包含 label 映射、输入尺寸、mean/std、先验统计、class weight、验证来源、增强配置、伪标签配置、训练阶段配置、W&B identity。
- `history.json`
  - 每个 epoch 的训练/验证指标时间线。
- `last.pt`
  - 最近一个 epoch 的 checkpoint。
- `best.pt`
  - 按 `val_f1` 选出的最佳 checkpoint。
- `epoch_XX.pt`
  - 仅在 `--save-every-epoch` 时生成。

checkpoint 内部除了 `model` / `optimizer`，还会重复保存本轮阈值、先验统计、增强参数和训练阶段信息，保证单文件可追溯。

## Auxiliary Workflows

- `bagging-helper.py`
  - 对多份 `prob_pos_corrected` 做简单平均，输出 `bagged_submission.csv` 和 `bagged_prob.csv`。
- `run_kfold_bagging.py`
  - 用 `StratifiedKFold` 生成 split。
  - 每个 fold 的训练集和验证集都是 CSV 级别划分，图像仍然从 `mask/train` 取。
  - 最终 bagging 仍然复用 `bagging-helper.py`。
- `pseudo_helper.py`
  - 常用于把单模型或 bagging 产生的 `prob_pos_corrected` 转成高置信度伪标签清单，再回灌到 `train_finetune.py`。

## Known Constraints

- 训练、校准、阈值搜索逻辑都假设是二分类。
- mask 文件名必须满足 `<stem>_mask_000.png` 约定。
- threshold-search 虽然有独立数据流，但默认是关闭的，需要显式传 `--open-threshold-search`。
- 当前默认输出目录名仍保留 `convnextv2` 字样，这只是历史路径名，不代表默认 backbone 已切回 ConvNeXt V2。
