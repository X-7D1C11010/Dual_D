# Dual_D 代码修改相关对话汇总

- 整理日期：2026-08-20
- 项目目录：`D:\Code\Dual_D`
- 覆盖时间：2026-07-01 至 2026-08-20
- 文档性质：从完整对话中筛选与代码创建、代码修改、参数调整、训练入口、数据加载、服务器运行和错误修复直接相关的内容，并按实际演进过程汇总

## 1. 整理范围与信息来源

本汇总主要依据以下信息：

1. 早期对话归档 `Dual_D_TACL_conversation_archive_20260713.md`；
2. 当前对话中 2026-07-13 之后可见的用户需求、错误日志和修改说明；
3. 当前代码、配置文件、测试文件和 Git 提交记录；
4. `runs/`、根目录训练日志以及 `docs/` 中已经形成的诊断报告。

本文不是逐字复制全部对话，而是保留与代码修改有关的需求、判断、实现、涉及文件、验证结果和遗留限制。Visio 绘图、纯理论公式解释和组会汇报文本原则上不收录；只有当它们直接引发算法代码更新或暴露理论与实现偏差时才会提及。

## 2. 代码演进总览

| 阶段 | 主要问题或需求 | 主要代码结果 |
|---|---|---|
| 2026-07-01 | 原方案缺少独立的双生成器、双判别器实现 | 创建 `dual_d/` 核心模块和独立训练栈 |
| 2026-07-13 | `train_acc=1` 但总损失较高，日志信息不足 | 拆分损失和准确率、加入数据审计、预热、早停和完整训练集评估 |
| 2026-08-11 | 方案升级为张量对齐、双向域翻译、类别感知反馈，并要求三模态 | 加入类别原型对比、AIS 分支、三模态 TAL 和四天气实验矩阵 |
| 2026-08-17 | 服务器 AIS 不是逐图片文件，而是 JMDA-Net 全局 MAT 文件 | 实现 JMDA 兼容 I/Q MAT 加载和复值 AIS 编码器 |
| 2026-08-18 | VIS/IR 实际不配对，AIS 无法映射到图片；RTX 5090 多卡 NCCL 报错 | 改为类别级弱配对、独立增强、AIS 行审计、CUDA 架构检查和单卡回退 |
| 2026-08-19 | 逆光、雨天曾多次达到 1，担心小数据集记忆；不希望保存模型 | 增强正则化、降低目标监督权重、缩小生成器/判别器，并提供可选 checkpoint |
| 2026-08-20 | 上一轮正则化过强，雨天变为严重欠拟合 | 回调到中等正则化，恢复监督和学习率，并修复采样步数向下取整问题 |

## 3. 第一阶段：建立独立的双生成器、双判别器算法

### 3.1 用户提出的修改需求

最初要求是在不修改原 `JMDA-Net` 脚本的前提下，把 TACL 的双判别器思想实现为独立模块，并保存到 `D:\Code\Dual_D`。随后进一步要求 `Dual_D` 成为不依赖其他目录的完整算法，能够独立加载数据、训练、验证、记录日志和保存结果。

### 3.2 实际实现

核心域翻译路径被实现为：

```text
目标域特征 Ft --G_t2s--> 源域风格特征 Fts
源域特征 Fs   --G_s2t--> 目标域风格特征 Fst
```

两个判别器分别约束两个输出域：

```text
D_s：区分真实 Fs 与生成 Fts
D_t：区分真实 Ft 与生成 Fst
```

生成器侧同时加入对抗、循环一致性、身份保持、样本级对比和生成特征分类反馈；后续又加入类别原型对比。训练采用判别器与主网络交替更新。

### 3.3 创建的主要文件

- `dual_d/feature_generators.py`：双向特征生成器；
- `dual_d/primary_discriminator.py`：源域方向判别器；
- `dual_d/auxiliary_discriminator.py`：目标域方向判别器；
- `dual_d/losses.py`：对抗、Cycle、Identity、样本对比和原型对比损失；
- `dual_d/collaborative_training.py`：双生成器、双判别器和模块 C 的统一协调器；
- `dual_d/integration_adapter.py`：训练接口适配；
- `dual_d/config.py`、`configs/dual_d_default_config.json`：结构与损失权重；
- `dual_d/data/`：数据集、配对采样和审计；
- `dual_d/models/`：VIS、IR、AIS 编码器、TAL 和分类器；
- `dual_d/training/`：训练、评估、日志和 checkpoint；
- `scripts/train_dual_d.py`：正式训练入口；
- `configs/train_dual_d_default.json`：默认训练参数。

### 3.4 训练入口变化

早期的 `scripts/example_integration_usage.py` 只用于随机张量接口验证，不能训练。补齐训练栈后，正式入口统一为：

```bash
python scripts/train_dual_d.py --config configs/train_dual_d_default.json ...
```

## 4. 第二阶段：根据训练日志增强稳定性、指标和数据审计

### 4.1 对话中发现的问题

早期黑天日志出现 `train_acc=1`，但总损失仍约为 2 至 3。分析发现：

- `train_acc` 只表示采样批次中目标域原始特征的 argmax 正确率；
- 标签平滑使分类损失存在非零下界；
- 总损失还包含双判别器、Cycle、Identity、对比和 TAL；
- 后期判别器变强时，生成器对抗损失可能回升；
- 目标域缺类且类别不平衡，单一 ACC 容易掩盖问题。

### 4.2 实际代码修改

`dual_d/training/trainer.py` 增加或强化了以下内容：

- 分开记录 `loss_cls_source`、`loss_cls_target`；
- 分开记录 source、target、source-like、target-like 四路训练准确率；
- 记录 `adv_primary`、`adv_auxiliary`、`cycle`、`identity`、`contrastive`、`classification_feedback`；
- 记录主网络和判别器梯度范数；
- 每隔指定轮次用无随机增强的完整目标训练集计算 `train_full_acc`；
- 加入判别器 warm-up 和线性 ramp，避免训练初期对抗信号过强；
- 加入基于配置指标的学习率调度与 early stopping；
- 当 `freeze_visual_backbone=true` 但没有预训练权重时，避免冻结随机初始化骨干；
- 保存 `resolved_config.json`、`best_metrics.json`、`result_summary.json` 和类别分布。

`dual_d/training/metrics.py` 将缺类场景下的指标拆分为：

- `*_macro_present`：只对验证集中实际存在的类别求宏平均；
- `*_macro_all`：固定全部类别空间的宏平均；
- `*_weighted_present`：按存在类别样本量加权；
- per-class precision、recall、F1、support 和混淆矩阵。

`dual_d/data/audit.py` 增加了训练集/验证集路径重叠、文件内容哈希重复、缺失文件、标签目录错误和 VIS/IR stem 不一致统计；严格审计可以在发现真实泄漏时中止训练。

### 4.3 TAL 负损失的代码解释

当前 TAL 定义为负相关性：

```text
L_TAL = - mean(correlation)
```

因此 TAL 为负本身是正常现象，表示源域和目标域的相关性为正。后续对话也明确指出，不能只根据 TAL 为负判断模块有效，还要结合正交误差、消融和下游指标验证。

## 5. 第三阶段：使实现与“张量对齐 + 双向域翻译 + 类别感知反馈”方案接近

### 5.1 用户提出的修改需求

在多轮架构讨论后，要求当前代码与最新方案保持一致，最新方案包含：

1. 多模态张量对齐；
2. 双生成器、双判别器的双向特征翻译；
3. Cycle 和 Identity 约束；
4. 样本级同类对比；
5. 类别原型对比；
6. 生成特征分类反馈。

### 5.2 对模块 C 的代码补充

`dual_d/losses.py` 和 `dual_d/collaborative_training.py` 增加类别原型计算与对比损失。当前实现的原型是当前 batch 内同类别真实特征的均值，不是跨 batch 的 EMA 或全数据原型。

生成特征分类反馈在两个方向使用输入样本的真实类别：

```text
CE(C(G_s2t(Fs)), ys)
CE(C(G_t2s(Ft)), yt)
```

这保证生成器改变领域风格时不改变类别语义。类别原型对比约束生成特征靠近“输出域中的同类别中心”，分类反馈约束其仍能预测为“输入样本原类别”；两者约束的类别身份相同，因此不冲突。

### 5.3 张量对齐实现

`dual_d/models/tensor_alignment.py` 使用每模态源域矩阵 `U_m` 和目标域矩阵 `V_m`，通过因式分解张量收缩计算多模态联合相关性，避免显式创建 `512 x 512 x 512` 张量。最终送入后续模块的是各模态低维投影的拼接，而张量收缩结果主要用于 TAL 相关性损失。

每次优化器更新后使用 QR 分解把 `U_m/V_m` 投回正交基：

```text
U_updated = Q_u R_u，随后 U <- Q_u
V_updated = Q_v R_v，随后 V <- Q_v
```

代码没有使用“协方差 + SVD 闭式更新”；此前含 SVD 的流程描述与当前实现不一致，后续对话已经纠正。

## 6. 第四阶段：增加三模态、四天气和多次迭代训练

### 6.1 用户提出的修改需求

要求源域和目标域都支持 VIS、IR、AIS 三模态，数据路径在服务器运行时通过参数提供；一次命令完成黑天、逆光、雾天、雨天四种目标域；增加可配置独立迭代次数。

### 6.2 三模态代码

- `dual_d/data/ais_signal.py`：加载 `.npy`、`.npz`、`.csv`、`.txt`、`.json` 和 JMDA-Net MAT；
- `dual_d/data/multimodal_dataset.py`：返回可选 AIS 张量及其来源信息；
- `dual_d/models/backbones.py`：加入复值 I/Q 1D 编码器和 MLP 备选编码器；
- `dual_d/models/tensor_alignment.py`：TAL 支持两模态或三模态；
- `dual_d/training/trainer.py`：按 `use_ais` 动态创建 AIS 分支，并将融合维度设为 `proj_dim * num_modalities`；
- `tests/test_three_modal_pipeline.py`：覆盖两模态、三模态、AIS MAT 加载、张量收缩等路径。

### 6.3 四天气实验矩阵

`scripts/train_dual_d.py` 增加：

- `--target-parent-root`：四个目标域共同父目录；
- `--target-domains 黑天 逆光 雾天 雨天`：目标域列表；
- `--iterations N`：每个天气域独立运行 N 次；
- 每次迭代使用递增随机种子；
- 生成 `runs/batch_summary_<timestamp>.json`，统计各域均值、标准差、最小值和最大值。

配置文件当前默认 `iterations=3`。

## 7. 第五阶段：AIS 加载策略的多次修正

### 7.1 首次 AIS 报错

服务器报错：

```text
RuntimeError: No AIS files found for class='1'.
```

原因是初版代码假定每个类别目录下存在逐样本 AIS 文件，而实际数据只有一个全局 MAT 文件。

### 7.2 参考 JMDA-Net 后的实现

根据 `D:\Code\JMDA-Net` / `D:\Code\JMDA-Net` 相关参考项目的 `main.py` 数据路径，代码兼容以下键：

```text
balanced_rcv_I
balanced_rcv_Q
new_balanced_label
```

全局 AIS 被转换为 `[N, 2, L]` I/Q 张量，并由复值 1D 编码器提取特征。

### 7.3 后续得到的真实数据事实

用户进一步明确：

- VIS/IR 不是同地点、同时刻、同角度采集，不能声称逐样本配准；
- AIS 只有一个全局 MAT，无法知道每行信号对应哪张图片；
- 目标域真实标签来自类别目录名称。

因此代码再次修正：

- VIS/IR 改为同类别弱配对，而非逐文件物理配对；
- 两个模态默认独立进行几何增强；
- 类内数量不同不再简单 `zip` 截断，而是循环较短模态池；
- 全局 AIS 被明确视为未与图像配对的先验；
- AIS MAT 行按训练/验证拆分并记录 `ais_index`，审计时阻止重复使用同一行；
- `--use-ais/--no-use-ais` 显式控制第三模态。

考虑到 AIS 无图像级对应关系，当前默认配置为：

```json
"use_ais": false
```

即代码保留 AIS 分支，但默认训练使用 VIS/IR；启用 AIS 应作为明确消融实验，并如实说明它是未配对先验。

## 8. 第六阶段：服务器运行命令、RTX 5090 与 NCCL 修复

### 8.1 nohup 命令问题

错误命令把环境变量当作可执行文件：

```bash
nohup CUDA_VISIBLE_DEVICES=0,1 python ...
```

正确形式应为：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1 python scripts/train_dual_d.py ... &
```

这属于启动方式修正，不是模型算法修改。

### 8.2 RTX 5090 架构兼容

旧 PyTorch 不包含 `sm_120` 时，代码可能直到第一批数据才失败。`dual_d/training/trainer.py` 增加 CUDA 架构检查，在正式训练前确认当前 PyTorch 的编译架构包含 GPU 所需的 `sm_120`，否则给出明确错误。

服务器升级到 `torch 2.11.0+cu128` 后，CUDA smoke test显示两张 RTX 5090 均可见，编译架构包含 `sm_120`。

### 8.3 多卡 NCCL 错误

训练在 `nn.DataParallel` 参数广播阶段出现：

```text
RuntimeError: NCCL Error 2: unhandled system error
```

代码增加 `_probe_data_parallel()`：正式包装 VIS、IR、AIS 编码器和分类器前，先执行一次最小 NCCL 广播测试；测试失败时自动取消 DataParallel 并回退到主 GPU。考虑到单张卡显存足够，默认配置最终设为：

```json
"multi_gpu": false
```

推荐服务器先用 `CUDA_VISIBLE_DEVICES=0` 和 `--no-multi-gpu` 单卡运行。

## 9. 第七阶段：处理逆光、雨天达到 1 的过拟合担忧

### 9.1 用户提出的修改方向

用户希望通过网络参数降低雨天和逆光多次迭代都为 1 的情况，并提出不记录部分 `ACC=1`、隐蔽实现或通过迭代安排使最终均值不为 1。

### 9.2 实施边界

实际实现没有隐藏、删除或改写真实验证指标，也没有筛选随机种子来制造目标均值。所有迭代仍保存真实的 `val_acc`、F1、最佳轮次和 batch 汇总。合理目标被限定为降低小样本记忆、提高跨种子稳定性。

模型 checkpoint 改为可选，而不是删除全部保存能力：

```text
--save-checkpoints
--no-save-checkpoints
```

默认不保存 `.pt`，但继续保存日志、CSV、最佳指标和结果汇总。

### 9.3 2026-08-19 的第一轮强正则化

为抑制旧批次中逆光和雨天三次均为 1，曾进行以下调整：

- `min_steps_per_epoch: 8 -> 1`；
- 数据增强 `0.5 -> 0.7`；
- label smoothing `0.1 -> 0.15`；
- classifier dropout `0.4 -> 0.5`；
- 新增 `target_classification_weight=0.5`；
- 主、视觉、判别器学习率减半；
- weight decay 增大到 `0.001`；
- 双生成特征分类反馈降到 `0.5`；
- 判别器缩小为 `[128, 64]`，生成器 hidden dim 降为 192，残差幅度降为 0.3；
- 默认关闭 checkpoint；
- 默认运行三次迭代。

## 10. 第八阶段：根据最新日志修正过强正则化

### 10.1 最新批次诊断

对 `runs/batch_summary_20260819_111406.json` 的三次迭代分析得到：

| 目标域 | 最佳 ACC 均值 | 最小值 | 最大值 | 判断 |
|---|---:|---:|---:|---|
| 黑天 | 0.9417 | 0.8917 | 0.9667 | 跨种子波动较大 |
| 逆光 | 0.9550 | 0.9189 | 0.9730 | 已不再全部为 1 |
| 雾天 | 0.9747 | 0.9697 | 0.9848 | 相对稳定 |
| 雨天 | 0.5128 | 0.4615 | 0.5385 | 严重欠拟合 |

雨天最佳轮次的完整训练集 ACC 也只有约 `0.49-0.57`，类别 9、11 多次召回为 0。这证明问题不是验证过拟合，而是上一轮同时减少更新步数、监督权重和学习率，并叠加强正则化，导致模型没有学会。

### 10.2 2026-08-20 的中等强度参数

`configs/train_dual_d_default.json` 最终回调为：

```json
{
  "min_steps_per_epoch": 4,
  "augmentation_strength": 0.6,
  "label_smoothing": 0.1,
  "classifier_dropout": 0.45,
  "target_classification_weight": 0.75,
  "lr_main": 0.000075,
  "lr_visual": 0.0000075,
  "lr_discriminator": 0.0000375,
  "weight_decay": 0.00075,
  "early_stopping_patience": 10,
  "early_stopping_min_epochs": 20
}
```

`configs/dual_d_default_config.json` 调整为：

```json
{
  "classification": 0.75,
  "adv_primary": 0.04,
  "adv_auxiliary": 0.04,
  "cycle": 0.3,
  "identity": 0.15,
  "contrastive": 0.08,
  "prototype_contrastive": 0.08
}
```

判别器 `[128, 64]`、生成器 hidden dim 192 和 residual scale 0.3 暂时保持，避免一次同时修改过多变量。

### 10.3 采样器修复

`dual_d/data/paired_sampler.py` 原来使用整除：

```text
natural_batches = min(source_size, target_size) // batch_size
```

不足一个完整 batch 的尾部样本会被忽略。现改为向上取整：

```text
natural_batches = ceil(min(source_size, target_size) / batch_size)
```

再与 `min_steps_per_epoch=4` 取最大值。按当前数据量，雨天每轮至少 4 步，逆光自然为 5 步，黑天和雾天按自然数据量运行；这比旧版所有域固定至少 8 步更少重复采样，也比上一版雨天只有 3 步更有学习能力。

## 11. 当前关键代码文件及职责

| 相对路径 | 主要职责 |
|---|---|
| `scripts/train_dual_d.py` | 命令行入口、四天气实验矩阵、迭代和路径参数 |
| `configs/train_dual_d_default.json` | 数据、训练、正则化、调度、GPU 和保存策略 |
| `configs/dual_d_default_config.json` | 双判别器、生成器和模块 C 损失权重 |
| `dual_d/models/tensor_alignment.py` | 因式分解张量收缩、U/V 投影和 QR 正交化 |
| `dual_d/feature_generators.py` | 源到目标、目标到源双向特征生成 |
| `dual_d/primary_discriminator.py` | 源域方向真伪判别 |
| `dual_d/auxiliary_discriminator.py` | 目标域方向真伪判别 |
| `dual_d/losses.py` | 对抗、Cycle、Identity、样本对比和原型对比 |
| `dual_d/collaborative_training.py` | 双判别器和模块 C 总损失协调 |
| `dual_d/data/multimodal_dataset.py` | VIS/IR 弱配对、可选 AIS 和数据增强 |
| `dual_d/data/ais_signal.py` | JMDA MAT 与逐文件 AIS 加载 |
| `dual_d/data/paired_sampler.py` | 源/目标同类别采样及每轮步数 |
| `dual_d/data/audit.py` | 训练/验证泄漏、AIS 行复用和数据完整性审计 |
| `dual_d/models/backbones.py` | VIS、IR、AIS 编码器、分类器和标签平滑 CE |
| `dual_d/training/trainer.py` | 训练、验证、优化器、调度、日志、GPU 回退和结果保存 |
| `dual_d/training/metrics.py` | ACC、present/all F1、per-class 和混淆矩阵 |
| `tests/test_three_modal_pipeline.py` | 二/三模态、AIS、训练矩阵和 smoke tests |
| `tests/test_training_safety.py` | 数据审计、对抗预热、原型损失和采样器回归测试 |

## 12. 验证记录

对话过程中完成过以下验证：

- 核心模块 `compileall` / `py_compile` 语法检查；
- `scripts/example_integration_usage.py` 随机张量前向与损失计算；
- `scripts/train_dual_d.py --help` 参数入口检查；
- 两模态和三模态单轮训练 smoke test；
- JMDA MAT 加载和复值 AIS 编码器测试；
- 显式张量收缩与因式分解张量收缩等价性测试；
- 四天气、多迭代实验解析测试；
- 数据泄漏审计、类别原型、warm-up 和采样器测试；
- 2026-08-20 最后一次本地验证为 16 项 `unittest` 全部通过；
- `git diff --check` 和命令行解析通过。

本地环境没有安装 pytest，因此最后一次使用 `python -m unittest discover -s tests -v` 完成测试，而不是 pytest。

## 13. 已确认但仍需注意的限制

1. 不能通过参数调整保证四个天气域必然超过指定历史基线，更不能保证恰好提升 0.5%；最终性能必须由服务器重复实验给出。
2. 逆光和雨天验证集很小，单个样本就会引起明显 ACC 跳变；达到 1 不必然证明数据泄漏，也不能仅凭 ACC 判断过拟合。
3. 不允许隐藏、跳过或改写 `val_acc=1`，也不允许通过筛选种子人为控制最终均值；所有真实迭代都应纳入汇总。
4. VIS/IR 只有类别级对应关系，当前 `L_pair` 实际是跨域同类别多正样本对比，不是同一物理目标的实例配对。
5. 全局 AIS 与图片无映射关系，默认关闭是当前最严谨的主实验设置；启用时必须标注为未配对先验。
6. 目标域只有 6 至 9 个实际类别，四个天气实验并非统一完整 14 类任务；报告结果时应同时给出类别覆盖和 `F1-present` / `F1-all`。
7. 类别原型目前为 batch 原型，存在小 batch 和少数类重复采样带来的噪声。
8. TAL 的张量交互主要通过损失影响投影矩阵，后续分类器使用的是各模态投影拼接；理论图不能表述为下游直接使用显式高阶张量。
9. TAL 每次 AdamW 更新后用 QR 覆写 U/V，但优化器动量未同步投影；这一点仍需要正交误差和消融验证。
10. 多卡 DataParallel 已有自动回退，但服务器当前更适合先使用单张 RTX 5090 运行完整实验。

## 14. 当前推荐的服务器验证顺序

先只运行逆光和雨天三次，确认雨天从欠拟合恢复、逆光没有重新出现三次全部记忆：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 python scripts/train_dual_d.py \
  --config configs/train_dual_d_default.json \
  --source-root /home/lixiang/lx/Data/晴天 \
  --target-parent-root /home/lixiang/lx/Data \
  --target-domains 逆光 雨天 \
  --iterations 3 \
  --device cuda \
  --no-multi-gpu \
  --no-use-ais \
  > train_tuned_backlight_rain.log 2>&1 &
```

只有确认雨天训练集和验证集均恢复到合理水平后，再运行黑天、逆光、雾天、雨天完整三次实验，并依据所有真实运行结果计算均值和标准差。

## 15. 总结

与代码修改有关的对话主线，是把一个参考 TACL 的双判别器扩展逐步发展成独立的跨天气多模态训练工程：先补全双向翻译和双判别器，再完善独立训练、指标、数据审计和模块 C，随后加入 AIS、四天气矩阵和服务器兼容，最后根据真实日志在“重复记忆”和“严重欠拟合”之间重新平衡参数。

当前代码已经包含最新方案的主要模块，但仍不能说理论图与实现“完全一致”：最重要的差异是弱配对数据事实、全局 AIS 无图片对应、batch 原型、TAL 下游采用投影拼接以及有监督目标域训练设定。后续实验和论文表述应围绕这些真实实现边界展开。
