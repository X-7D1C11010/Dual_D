# Dual_D 双判别器独立训练算法

该目录是一个独立的 VIS/IR/AIS 三模态双判别器域适应训练工程。它不需要从其他项目目录导入
`DataLoad.py`、`Models.py`、`Tensor.py` 等脚本，数据加载、模型、张量对齐、
双判别器、训练循环、日志和结果保存均在本目录内实现。

## 设计目标

- 在 TAL 张量对齐后的融合特征空间中引入双判别器。
- 使用复值 1D 网络编码 AIS I/Q 信号，并与可见光、红外共同参与 TAL。
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

一次完成黑天、逆光、雾天、雨天四个目标域，每个目标域独立重复 3 次：

```bash
python scripts/train_dual_d.py \
  --source-root /data/晴天 \
  --target-parent-root /data/目标域 \
  --target-domains 黑天 逆光 雾天 雨天 \
  --iterations 3 \
  --output-dir runs
```

若 AIS 与图像数据分开存放，可增加：

```bash
--source-ais-root /data_ais/晴天 \
--target-ais-parent-root /data_ais/目标域
```

`--iterations` 表示每个目标域从头初始化并训练的独立实验次数，默认值为 1；
单次训练的 epoch 数仍由 `--epochs` 控制。

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

批量运行还会在 `output_dir` 下生成 `batch_summary_<timestamp>.json`，记录四个域每次
实验的随机种子、运行目录、最佳指标及各域均值和标准差。

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
      ais_signal.py
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
root/train/AIS/<class_id>/*.{npy,npz,csv,txt,json}
root/val/可见光/<class_id>/*.jpg
root/val/红外/<class_id>/*.jpg
root/val/AIS/<class_id>/*.{npy,npz,csv,txt,json}
```

### class_first

```text
root/train/<class_id>/可见光/*.jpg
root/train/<class_id>/红外/*.jpg
root/train/<class_id>/AIS/*.{npy,npz,csv,txt,json}
root/val/<class_id>/可见光/*.jpg
root/val/<class_id>/红外/*.jpg
root/val/<class_id>/AIS/*.{npy,npz,csv,txt,json}
```

每个 AIS 文件对应一个 VIS/IR 样本。默认 `--ais-match auto` 优先按同名文件的 stem
匹配，不能全部同名时退回同类别目录内的排序索引匹配。论文式 I/Q 数据可保存为复数
NumPy 数组或形状为 `[2,L]`/`[L,2]` 的数值数组；普通 AIS 属性向量也可加载，建议配合
`--ais-encoder mlp --no-ais-normalize`，并把 `--ais-sequence-length` 设置为属性向量的原始
长度，避免逐样本标准化消除航速、位置等绝对量纲。I/Q 信号默认统一为长度 128。

如果文件夹名不是 `可见光`、`红外` 和 `AIS`，使用：

```bash
--vis-folder VIS --ir-folder IR --ais-folder AIS_SIGNAL
```

接口验证示例仍保留：

```bash
python scripts/example_integration_usage.py
```

它只用随机张量验证模块接口，不读取数据、不训练模型。

## 稳定性与泛化默认策略

- 前 5 轮关闭判别器和生成器对抗项，但分类、TAL、Cycle、Identity、对比及生成分类反馈仍参与训练；随后用 15 轮线性引入对抗损失。
- 主学习率 `3e-4`、判别器学习率 `1e-4`，二者同步调度；判别器每 3 步更新一次。
- 验证集存在类别宏 F1 用于调度、最优模型和早停；最早第 75 轮后才允许早停。
- 判别器使用更小的 MLP、谱归一化和 0.4 dropout；不再把 cycle 重建结果同时标为 fake。
- VIS/IR 共用随机裁剪和翻转参数，并使用温和的模态特定颜色/对比度增强；AIS 不做图像几何增强。
- 类平衡采样在复用样本前先遍历类内池，减少小类样本的无意义重复。
