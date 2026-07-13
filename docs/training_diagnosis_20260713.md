# Dual_D 黑天训练诊断与优化方案（2026-07-13）

分析对象：`runs/dual_d_黑天_20260707_175617`，共 100 个 epoch。本文同时结合
`Dual_D_TACL_conversation_archive_20260713.md` 中已经确定的算法背景与评价口径。

## 1. 结论摘要

1. 总损失全局最低点在 epoch 72（2.4394），最佳验证结果在 epoch 73
   （accuracy 0.9167、present-class macro F1 0.8983）。epoch 80 后开始出现有正斜率的
   震荡，epoch 90 后上升明显加速。
2. 后 20 轮上升不是分类损失或 TAL 失控，而是主判别器相对生成器越来越强：主优化器
   学习率在 epoch 84、95 再次减半，判别器学习率却始终为 `5e-4`，D/main 学习率比从
   1 依次扩大到 2、4、8、16。
3. epoch 80–100 中，总损失与 `dual_g` 的相关系数为 0.856，与 `adv_primary` 为
   0.856，与主判别器损失为 -0.800。证据明确指向 primary adversarial imbalance。
4. `train_acc=1` 的 argmax 计算本身正确，但它只覆盖类平衡、带放回采样得到的训练批次，
   不能证明完整目标训练集准确率也是 1。旧 run 没有完整训练集的确定性评估。
5. 现有产物可以排除“train 与 val 实际指向同一批记录”这一粗粒度泄露：两者分别为
   465 和 120 条记录，类别计数也不同。但没有保存文件哈希，因而无法仅凭 run 产物排除
   不同路径下的重复图片、相邻视频帧泄露或人工标签语义错误。
6. 目标域只有 8/14 类且高度不平衡。训练最大/最小类为 162/11（14.7 倍），验证为
   41/3（13.7 倍）；应以 present-class macro F1 为主指标，同时保留 accuracy/micro F1。

## 2. 损失震荡的定量定位

### 2.1 关键轮次

| 事件 | epoch | 数值或影响 |
|---|---:|---|
| 总损失全局最低 | 72 | `train_loss=2.4394`、`dual_g=1.4793` |
| 验证最优 | 73 | `val_acc=0.9167`、`val_f1_present=0.8983` |
| 后 20 轮趋势开始可见 | 80/81 | epoch 80–100 总损失线性斜率 `+0.0053/epoch` |
| 主优化器再次降 LR | 84 | main=`6.25e-5`，D/main=8 |
| 上升明显加速 | 90/91 | epoch 90–100 总损失斜率约 `+0.0182/epoch` |
| 主优化器再次降 LR | 95 | main=`3.125e-5`，D/main=16 |
| 主对抗项快速恶化 | 96–100 | `adv_primary` 从 1.5452 升至 2.7818 |

学习率变化如下；旧代码只给 main optimizer 配置了 `ReduceLROnPlateau`：

| epoch 起点 | main LR | discriminator LR | D/main |
|---:|---:|---:|---:|
| 1 | 5.00e-4 | 5.00e-4 | 1 |
| 59 | 2.50e-4 | 5.00e-4 | 2 |
| 70 | 1.25e-4 | 5.00e-4 | 4 |
| 84 | 6.25e-5 | 5.00e-4 | 8 |
| 95 | 3.125e-5 | 5.00e-4 | 16 |

### 2.2 十轮窗口均值

| 窗口 | total loss | dual_g | adv_primary | D_primary loss | val acc | present F1 |
|---|---:|---:|---:|---:|---:|---:|
| 61–70 | 2.5155 | 1.4987 | 0.7366 | 0.6896 | 0.8617 | 0.8451 |
| 71–80 | 2.4780 | 1.4890 | 0.7955 | 0.6778 | 0.8767 | 0.8608 |
| 81–90 | 2.5102 | 1.4935 | 0.8490 | 0.6800 | 0.8658 | 0.8452 |
| 91–100 | 2.5757 | 1.5823 | 1.6383 | 0.6090 | 0.8592 | 0.8392 |

分类损失在最后 20 轮约为 1.12–1.14，基本不变；TAL 与总损失的后期相关系数仅
0.004。经典梯度爆炸通常会出现非有限数、多个损失项同步跳升或准确率崩溃，本 run
均未出现，因此它不是首要解释。但旧日志没有保存裁剪前梯度范数，不能完全排除局部梯度
尖峰；新代码已增加该指标和非有限损失保护。

### 2.3 技术原因排序

1. **判别器与生成器学习率失衡（高置信度）**：main LR 多次衰减、D LR 固定，和
   `D_primary loss` 下降、`adv_primary` 上升严格同向。
2. **对抗目标存在不必要冲突（高置信度）**：旧配置把接近真实源/目标特征的 cycle
   reconstruction 同时标为 fake，而 cycle loss 又要求它接近真实特征。后期判别器会学习
   极小重建差异，放大对抗拉扯。
3. **判别器容量偏大且无谱归一化（中高置信度）**：每个 D 使用 512/256/128 MLP，
   面对 465 个目标样本容易压制生成器。
4. **验证准确率离散且噪声大（中置信度）**：验证仅 120 个样本，每错/对一个样本就变化
   0.00833；直接用 accuracy 驱动 LR 容易在长平台期反复降 LR。
5. **数据重复采样与过拟合（高置信度）**：每轮有 14×32=448 次目标抽样，8 类均匀
   抽取时每类期望 56 次。11、12、16 样本的小类每轮分别被抽约 5.09、4.67、3.50 次，
   旧 sampler 还是逐次 `random.choice`，可能在遍历类内样本前重复同一图片。

## 3. train=1 与 val≈0.9 的真实性和过拟合

### 3.1 指标计算审查

- `train_acc` 的分子是 `argmax(pred_tgt) == target_labels`，分母是实际采样数，公式正确。
- 最新 run 的四条训练分支在最后 20 轮都接近 1：target、source、source-like、
  target-like 均不是单一字段别名或偶然计算错误。
- 但它们来自 `PairedClassSampler` 的 448 次平衡抽样，不是对 465 条目标训练记录各评估
  一次。旧日志因此只能证明“采样训练批次被记住”，不能证明完整训练集 accuracy=1。
- 14 类、`label_smoothing=0.1` 时，单个平滑 CE 理论下界约 0.5473；source+target
  下界约 1.0945。后期 `train_loss_cls≈1.13` 与 accuracy=1 并不矛盾。

新代码每轮对无随机增强的完整目标训练集做一次 eval-mode 推理，新增
`train_full_acc`、`train_full_f1_macro_present` 和 `train_sampled_minus_full_acc`。这才是
下一轮判断训练准确率真实性的主要证据。

### 3.2 泄露、标签与数据质量

从旧产物可确认：

- train/val 记录数不同，且各类分别约为 4:1，符合独立的分层切分特征；
- 两个 split 使用同一个 source-derived label map，类别 ID 一致；
- 训练和验证出现的是同一组 8 个目标类，没有“验证出现未知标签”的证据；
- 指标混淆矩阵总数为 120、正确 110，与 0.9167 完全一致。

旧产物无法确认：

- 不同文件路径是否保存了相同图像；
- 同一视频/连续帧是否被拆到 train 和 val；
- 文件夹标签在语义上是否由人工标错；
- 排序后 `zip(vis_files, ir_files)` 是否始终对应同一场景。

为此新增 `data_audit.json`：检查 train/val 是否解析为同一目录、路径交集、SHA-256 内容
重复、缺失文件、标签目录不一致及 VIS/IR 文件 stem 不一致。默认严格模式发现泄露即停止。
未知目标标签也不再被静默丢弃，而是直接报错。

### 3.3 具体过拟合原因

1. 目标训练仅 465 条、验证仅 120 条，却包含 ResNet-18、独立 IR CNN、TAL、双生成器、
   双判别器和三层分类器，容量明显大于数据规模。
2. 旧配置 `pretrained_visual=false` 且 `freeze_visual_backbone=true`。代码会冻结随机初始化的
   ResNet 早期层，这不是有效迁移学习，并会留下不可控的随机特征瓶颈。
3. 类平衡采样对小类有益，但小类每轮重复 3.5–5.1 次，100 轮后极易记忆。
4. 旧增强只有 crop/flip，而且 VIS 与 IR 分别调用随机 transform，几何参数并不同步，
   会破坏多模态对应关系。
5. 目标域缺 6 类，paired sampler 只训练共同存在的 8 类；当前 0.9167 只代表这 8 类，
   不能外推为完整 14 类性能。
6. 最佳轮到最终轮，验证准确率从 0.9167 降至 0.8417，泛化差距从 8.33 个百分点扩大到
   15.83 个百分点，是继续训练过久的直接证据。

最佳轮的 10 个错误主要集中在：raw label 1 有 8/26 错误（6 个预测成 raw label 2，
2 个预测成 raw label 6），raw label 7 有 2/11 错误（预测成 raw label 12）。这提示后续应
重点检查这些类别的视觉相似性、标签边界和序列切分，而不是只看总 accuracy。

## 4. 方案比较

| 方案 | 核心措施 | 预期效果 | 实施难度 | 状态 |
|---|---|---|---|---|
| A. 平衡 D/G 优化 | D LR 降至 1e-4；main/D 同步调度；D 每 3 步更新；记录 LR 比 | 直接消除后 20 轮主因 | 低 | 已实现 |
| B. 稳定对抗目标 | 5 轮 warmup+15 轮 ramp；谱归一化；缩小 D；重建结果不再标 fake | 降低 `adv_primary` 后期发散 | 中 | 已实现 |
| C. 早停与正确监控 | present macro F1 驱动调度、checkpoint、早停；最早 epoch 75 后停止 | 保留不平衡数据上的真正最佳模型 | 低 | 已实现 |
| D. 泛化正则 | ImageNet 预训练并冻结早期层；classifier/D dropout 0.4；WD 5e-4 | 缩小 train/val gap | 低到中 | 已实现/需权重 |
| E. 多模态增强与采样 | 同步几何增强、温和模态特定抖动；类内样本遍历后再复用 | 减少小类记忆和模态错位 | 中 | 已实现 |
| F. 数据与指标审计 | 全训练集 eval；raw/source-like 双验证；内容哈希；梯度范数 | 排除伪 1.0、泄露和梯度异常 | 低 | 已实现 |
| G. source-only 辅助分类 | 为目标缺失的 6 类增加独立 source classification loader | 若部署要求 14 类，可保留完整标签空间 | 中高 | 后续可选 |

方案 G 不应与“只评价目标实际存在类别”的当前目标混为一谈。若黑天域业务上永远只有这
8 类，增加 6 类 source-only 训练可能分散容量；若部署仍要求 14 类，则它是必需实验。

## 5. 已落地的代码位置

- `dual_d/training/trainer.py`
  - 主/判别器双 `ReduceLROnPlateau`；早停；对抗 warmup/ramp；完整训练集评估；raw 验证；
    梯度范数、LR 比和非有限损失保护；数据审计接入。
- `dual_d/collaborative_training.py`、`dual_d/integration_adapter.py`
  - 支持 `adversarial_scale`，只对两项 adversarial generator loss 做线性引入。
- `dual_d/data/multimodal_dataset.py`
  - VIS/IR 共享 crop/flip；模态特定 photometric jitter；未知标签直接报错；确定性训练集评估。
- `dual_d/data/paired_sampler.py`
  - 类内池遍历完成后才复用，避免旧版 `random.choice` 的过早重复。
- `dual_d/data/audit.py`
  - 路径/内容/标签/配对完整性审计。
- `configs/train_dual_d_default.json`
  - 推荐训练参数。
- `configs/dual_d_default_config.json`
  - 更小、带谱归一化的判别器；关闭 reconstruction-as-fake。
- `tests/test_training_safety.py`
  - 同步增强、内容泄露审计、对抗 ramp 回归测试。

## 6. 推荐参数与启动方式

先填写 `configs/train_dual_d_default.json` 中的 `source_root`、`target_root`，然后：

```bash
python scripts/train_dual_d.py --config configs/train_dual_d_default.json
```

核心推荐值：

```text
pretrained_visual=true
freeze_visual_backbone=true
lr_main=3e-4
lr_visual=1e-5
lr_discriminator=1e-4
discriminator_update_interval=3
weight_decay=5e-4
classifier_dropout=0.4
adversarial_warmup_epochs=5
adversarial_ramp_epochs=15
monitor_metric=val_f1_macro_present
early_stopping_min_epochs=75
early_stopping_patience=15
augmentation_strength=0.5
```

若运行环境无法下载或没有缓存 ImageNet 权重：

```bash
python scripts/train_dual_d.py \
  --config configs/train_dual_d_default.json \
  --no-pretrained-visual \
  --no-freeze-visual-backbone
```

## 7. 预期指标与验收标准

以下是复现实验的合理目标，不是未训练前的性能保证：

- 首要：`val_f1_macro_present >= 0.90`，且 `val_acc >= 0.9167`（至少保持旧最佳）。
- 进取目标：`val_acc=0.925–0.94`、present macro F1 `0.90–0.92`；在 120 个验证样本上
  这相当于比旧最佳多正确约 1–3 个样本。
- `train_full_acc - val_acc <= 0.08`；若仍明显大于 0.08，优先增加数据或按视频/场景重切分，
  不应继续堆正则化掩盖数据问题。
- epoch 80 后 `dual_g` 十轮均值不应持续上升，`adv_primary` 不应出现从约 1 快速升到 2.8
  的趋势；D/main LR 比应始终保持初始值约 1/3。
- `train_grad_norm_main` 和 `train_grad_norm_discriminator` 必须有限；若长期远大于 clip=1，
  应进一步把 TAL/main LR 降 30%–50%，而不是只依赖裁剪。
- 数据审计必须满足 path/content overlap 均为 0；否则验证结果无效。

使用旧指标做反事实早停时，`min_epochs=75, patience=15` 会在 epoch 88 停止，并保留
epoch 73 的最佳结果，避开 epoch 95 后最严重的对抗失衡。新配置改变了训练动力学，仍需用
至少 3 个随机种子报告均值和标准差；单次 120 样本验证集不足以证明稳定提升。
