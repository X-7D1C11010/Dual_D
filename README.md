# Dual_D 双判别器独立训练算法

该目录是一个独立的双判别器多模态域适应训练工程。它不需要从其他项目目录导入
`DataLoad.py`、`Models.py`、`Tensor.py` 等脚本，数据加载、模型、张量对齐、
双判别器、训练循环、日志和结果保存均在本目录内实现。

## 设计目标

- 在 TAL 张量对齐后的融合特征空间中引入双判别器。
- 使用主判别器约束“目标域 -> 源域稳定特征”的映射。
- 使用辅助判别器约束“源域 -> 目标域天气特征”的映射。
- 通过双向生成、闭环一致性、身份保持、特征级对比损失和分类反馈保持语义稳定。
- 通过配置文件和适配器接口与原有训练流程兼容。

## 训练入口

正式训练脚本为：

```bash
python scripts/train_dual_d.py \
  --source-root /path/to/source_weather \
  --target-root /path/to/target_weather \
  --output-dir runs \
  --epochs 100 \
  --batch-size 32
```

如果 RTX 5090 或其他新显卡与当前 PyTorch CUDA 构建不兼容，可先用 CPU 验证流程：

```bash
python scripts/train_dual_d.py \
  --source-root /path/to/source_weather \
  --target-root /path/to/target_weather \
  --device cpu \
  --epochs 1
```

## 输出文件

每次训练会在 `output_dir/run_name/` 下保存：

- `train.log`：训练日志。
- `metrics.csv`：每轮训练/验证指标。
- `checkpoints/best_model.pt`：验证集准确率最优 checkpoint。
- `checkpoints/last_model.pt`：最后一轮 checkpoint。
- `best_metrics.json`：最优轮详细指标。
- `result_summary.json`：训练汇总。
- `resolved_config.json`：实际使用的参数和标签映射。
- `label_map.json`：类别标签映射。

## 目录结构

```text
D:\Code\Dual_D
  configs\
    dual_d_default_config.json
  docs\
    integration_notes.md
  dual_d\
    __init__.py
    config.py
    gradient_reversal.py
    feature_generators.py
    primary_discriminator.py
    auxiliary_discriminator.py
    losses.py
    collaborative_training.py
    integration_adapter.py
    data\
      multimodal_dataset.py
      paired_sampler.py
    models\
      backbones.py
      tensor_alignment.py
    training\
      checkpointing.py
      logging_utils.py
      metrics.py
      trainer.py
  scripts\
    example_integration_usage.py
    train_dual_d.py
```

## 数据目录

默认支持两种布局，并可通过 `--source-layout`、`--target-layout` 指定。

### modality_first

```text
root/train/可见光/<class_id>/*.jpg
root/train/红外/<class_id>/*.jpg
root/val/可见光/<class_id>/*.jpg
root/val/红外/<class_id>/*.jpg
```

### class_first

```text
root/train/<class_id>/可见光/*.jpg
root/train/<class_id>/红外/*.jpg
root/val/<class_id>/可见光/*.jpg
root/val/<class_id>/红外/*.jpg
```

如果文件夹名不是 `可见光` 和 `红外`，使用：

```bash
--vis-folder VIS --ir-folder IR
```

接口验证示例仍保留：

```bash
python scripts/example_integration_usage.py
```

它只用随机张量验证模块接口，不读取数据、不训练模型。
