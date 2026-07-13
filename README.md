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

推荐先填写优化后的默认配置中的数据路径，再启动：

```bash
python scripts/train_dual_d.py --config configs/train_dual_d_default.json
```

该配置使用 ImageNet 预训练 ResNet-18；离线环境没有缓存权重时，可显式添加
`--no-pretrained-visual --no-freeze-visual-backbone`，代码不会再冻结随机初始化的早期层。

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
- `data_audit.json`：训练/验证路径、内容哈希、标签目录和模态配对审计。

`metrics.csv` 还会记录完整目标训练集准确率、raw/source-like 验证指标、主/判别器
梯度范数、两侧学习率及其比值。主优化器和判别器调度器使用同一监控指标同步衰减，
避免后期判别器相对学习率不断变大。

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

## 稳定性与泛化默认策略

- 前 5 轮只训练分类/TAL/重建目标，随后用 15 轮线性引入对抗损失。
- 主学习率 `3e-4`、判别器学习率 `1e-4`，二者同步调度；判别器每 3 步更新一次。
- 验证集存在类别宏 F1 用于调度、最优模型和早停；最早第 75 轮后才允许早停。
- 判别器使用更小的 MLP、谱归一化和 0.4 dropout；不再把 cycle 重建结果同时标为 fake。
- VIS/IR 共用随机裁剪和翻转参数，并使用温和的模态特定颜色/对比度增强。
- 类平衡采样在复用样本前先遍历类内池，减少小类样本的无意义重复。
