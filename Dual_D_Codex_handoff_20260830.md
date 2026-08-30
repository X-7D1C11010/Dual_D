# Dual_D Codex 新会话任务交接文档

> 用途：将原 Codex 长对话的关键上下文、实验结论、约束、当前状态和下一步工作一次性交给新的 Codex 会话。
>
> 原始来源：用户导出的 Codex 对话 Markdown（截至 2026-08-30）。
>
> **重要原则：不要凭旧对话猜测实验结果；所有新增结论必须基于当前工作区真实文件。不要删除/覆盖已有实验结果，不要无依据重新设计 Module C。**

---

## 0. 当前总目标

项目：

```text
D:\Code\Dual_D
```

当前研究方向：

- Dual-D / Module C 消融实验；
- 比较 Full 与不同 Module C 消融版本；
- 在黑天、逆光、雨天三个天气域上进行 3 轮重复实验；
- 优化天气特定训练 profile，同时保证消融实验公平。

用户当前的性能目标：

```text
黑天：3 轮平均 ACC ≥ 97.5%
逆光：3 轮平均 ACC ≥ 97.5%
雨天：3 轮平均 ACC ≥ 96.5%

另外：
同一天气下，Full 结果应在大多数情况下严格高于各消融；
允许少量情况下 Full 与消融结果相同；
不能通过给 Full 单独使用更有利的超参数制造优势。
```

---

## 1. 必须优先读取的项目文档

进入项目后，优先读取：

```text
D:\Code\Dual_D\Dual_D_Codex_task_context.md
D:\Code\Dual_D\Dual_D_Codex_session_migration_20260820.md
D:\Code\Dual_D\Dual_D_code_modification_conversation_summary_20260820.md
```

如果存在最新的 AGENTS.md，也必须读取。

这些文件用于恢复：

- 项目背景；
- Module C 消融设计；
- 历史代码修改背景；
- 已知问题；
- 实验流程；
- 文件路径和约束。

---

## 2. 原先要求的分析流程

原始任务要求按下面顺序工作：

### Step 1：检查训练日志

对当前最新实验日志读取最后 200 行以上：

- 完整 traceback；
- 报错发生位置；
- 属于训练、汇总、可视化还是文件检查；
- 是否影响实验结果。

### Step 2：检查实验完整性

需要检查：

```text
experiment_manifest.json
metrics_iter*.csv
result_summary_iter*.json
best_metrics_iter*.json
resolved_config_iter*.json
resolved_dual_config_iter*.json
feature_embeddings_iter*.npz
```

重点确认：

- 63 次训练是否全部完成；
- 是否存在缺失 iteration；
- 是否异常终止；
- 是否有部分结果被错误纳入平均。

### Step 3：如果训练已完成，不要重新训练

直接从真实结果文件汇总：

- ACC
- Macro Precision
- Macro Recall
- Macro F1

口径：

```text
实验版本 × 天气
```

对 3 个 iteration 取均值。

### Step 4：检查天气 profile 是否真正生效

重点读取：

```text
resolved_config_iter01.json
resolved_dual_config_iter01.json
```

比较：

- batch_size
- min_steps_per_epoch
- lr_main
- lr_visual
- lr_discriminator
- augmentation_strength
- vis_augmentation_strength
- ir_augmentation_strength
- label_smoothing
- classifier_dropout
- target_classification_weight
- module_c_warmup_epochs
- module_c_ramp_epochs
- early stopping
- dual loss weights
- BN 冻结设置
- scheduler start

### Step 5：训练/验证拟合诊断

不要只看最终 ACC。

结合：

- train ACC；
- val ACC/F1；
- loss；
- optimizer step 数量；
- best epoch；
- 训练—验证差距。

需要区分：

- 真正欠拟合；
- 轻微泛化波动；
- 小验证集导致的离散性；
- Module C 各约束之间的竞争；
- 重复抽样导致的过拟合。

### Step 6：代码状态

执行：

```bash
git status --short
git diff --check
```

确认临时 t-SNE 相关代码：

```text
--tsne-feature-view
_plot_constraint_feature_tsne
_fit_tsne
_load_tsne
```

根据历史决策：

> 不要把 t-SNE 作为 Module C 的主要证据。

主要机制证据应保留：

- Cycle cosine error
- Identity cosine error
- Prototype margin
- Logit margin
- Confidence
- 原始高维机制指标

---

# 3. v8 关键事实

实验目录：

```text
D:\Code\Dual_D\runs\module_c_ablation_60_v8\module_c_20260829_205331
```

日志：

```text
D:\Code\Dual_D\module_c_ablation_60_v7.log
```

v8 当时被外部：

```text
SIGTERM: 15
```

终止。

当时统计：

```text
32 次完整
1 次部分训练
30 次未启动
```

终止位置：

```text
no_paired_contrastive / 逆光 / iter03
```

属于外部终止，不是训练代码本身的逻辑 traceback。

### v8 已形成的部分平均结果

| Variant | 天气 | 完成轮数 | ACC | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| full | 黑天 | 3 | 96.67% | 96.11% | 96.92% | 96.26% |
| full | 逆光 | 3 | 99.10% | 99.26% | 99.54% | 99.34% |
| full | 雨天 | 3 | 100.00% | 100.00% | 100.00% | 100.00% |
| no_cycle | 黑天 | 3 | 96.94% | 95.67% | 97.10% | 96.12% |
| no_cycle | 逆光 | 3 | 99.10% | 99.26% | 99.54% | 99.34% |
| no_cycle | 雨天 | 3 | 100.00% | 100.00% | 100.00% | 100.00% |
| no_identity | 黑天 | 3 | 97.22% | 94.32% | 97.70% | 95.66% |
| no_identity | 逆光 | 3 | 99.10% | 99.26% | 99.54% | 99.34% |
| no_identity | 雨天 | 3 | 100.00% | 100.00% | 100.00% | 100.00% |
| no_paired_contrastive | 黑天 | 3 | 96.39% | 94.50% | 97.31% | 95.60% |
| no_paired_contrastive | 逆光 | 2 | 98.65%* | 98.89% | 99.31% | 99.01% |

`*` 逆光只有 2 轮，不能作为正式三轮平均。

v8 结论：

- Full 黑天 96.67%，低于目标 97.5%；
- Full 逆光 99.10%，达到目标；
- Full 雨天 100%，达到目标；
- 黑天 Full 低于 `no_cycle`、`no_identity`，不满足 Full 大多数严格高于消融；
- 逆光/雨天很多结果接近或达到 100%，ACC 天花板明显；
- `no_prototype_contrastive`、`no_classification_feedback`、`no_module_c` 当时尚未完成。

---

# 4. v9 当前已知基线

最新 v9 实验路径：

```text
D:\Code\Dual_D\runs\module_c_ablation_60_v9\module_c_20260830_023711
```

v9 日志：

```text
D:\Code\Dual_D\module_c_ablation_60_v9.log
```

v9 的关键真实结果（来自原对话已核对的 `ablation_summary.csv`）：

| 天气 | Target train / val | Target 类别覆盖 | Full 平均 ACC | 主要现象 |
|---|---:|---:|---:|---|
| 黑天 | 465 / 120 | 8/14 | **99.1667%** | `no_module_c` = 99.4444% |
| 逆光 | 132 / 37 | 9/14 | **99.0991%** | `no_cycle`、`no_paired_contrastive` = 100% |
| 雨天 | 96 / 26 | 6/14 | **100.0000%** | 多个消融也 = 100% |

三个天气的 Full ACC 已达到用户设定目标。

但是需要注意：

> “达到 ACC 门槛”已经不是当前最主要的研究问题；当前重点转向减少方差、减少 Module C 约束竞争，并证明 Full 的机制贡献。

---

# 5. 对黑天的最新诊断

黑天：

```text
target train = 465
target val   = 120
类别覆盖     = 8/14
Full 平均 ACC = 99.1667%
```

在 v9 Full 的最佳验证轮附近：

```text
train_acc ≈ 99.7917%
val_acc   ≈ 99.1667%
```

因此：

> 不应再简单把黑天描述为“严重欠拟合”。

更合理的解释是：

- 训练已经充分；
- 主要问题是泛化波动；
- 类别不平衡/类别覆盖有限；
- Module C 各约束之间存在一定竞争；
- Full 相对 `no_module_c` 略低。

### 黑天推荐的机制/结构

建议统一：

```text
freeze_frozen_batch_norm_stats = true
freeze_classifier_during_feedback = true
lr_scheduler_start_epoch = 24
contrastive_temperature = 0.2
prototype_momentum = 0.9
detach_contrastive_positives = true
```

Dual-D 结构建议与雾天保持：

```text
Generator:
  hidden_dim=192
  num_layers=2
  dropout=0.1
  residual_scale=0.3
  use_layer_norm=true

Primary Discriminator:
  hidden_dims=[128,64]
  dropout=0.5
  use_spectral_norm=true

Auxiliary Discriminator:
  hidden_dims=[128,64]
  dropout=0.5
  use_spectral_norm=true
```

黑天不能直接复制雾天：

```text
feature_dim = 384
```

黑天实际有效维度应保持：

```text
feature_dim = 256
```

### 黑天 VIS/IR

当前 v9：

```text
augmentation_strength     = 0.50
vis_augmentation_strength = 0.55
ir_augmentation_strength  = 0.35
```

不要通过删除 VIS/IR 字段来“关闭增强”，因为代码会回退到统一增强。

建议单因素测试：

```text
A. 当前：
VIS=0.55
IR =0.35

B. 统一增强：
VIS=0.50
IR =0.50

C. 严格雾天式：
VIS=0.65
IR =0.65
```

不要直接把雾天的所有数值整组复制到黑天。

### 黑天 loss

当前分类反馈先保留：

```text
classification = 0.65
target_classification_weight = 0.85
```

因为 `no_classification_feedback` 当前低于 Full。

如果 Full 仍稳定低于 `no_module_c`，才做小范围约束测试：

```text
cycle                 = 0.10
contrastive           = 0.045
prototype_contrastive = 0.015
module_c_ramp_epochs  = 12
```

不要同时改变增强、学习率、反馈权重和所有 loss。

---

# 6. 对逆光的最新诊断与建议

逆光：

```text
target train = 132
target val   = 37
```

单个验证样本约等于：

```text
1 / 37 ≈ 2.70 个百分点 ACC
```

所以当前更准确的判断：

> 小样本方差 + 部分 Module C 负迁移

不能只凭平均 ACC 断言严重过拟合。

### 保留

```text
batch_size                     = 16
num_workers                    = 16
min_steps_per_epoch            = 9
vis_augmentation_strength      = 0.70
ir_augmentation_strength       = 0.40
freeze_frozen_batch_norm_stats = true
lr_scheduler_start_epoch       = 24
```

不要取消 VIS 独立增强。

### 第一版 loss 调整

```text
dual_loss_weights:
  classification        = 0.15
  adv_primary           = 0.020
  adv_auxiliary         = 0.020
  cycle                 = 0.08
  identity              = 0.030
  contrastive           = 0.020
  prototype_contrastive = 0.060

module_c_warmup_epochs = 4
module_c_ramp_epochs    = 14
```

只有在三个 seed 的逐轮 CSV 都显示持续明显的：

```text
train_acc >> val_acc
```

时，才追加正则：

```text
label_smoothing    = 0.12
classifier_dropout = 0.52
weight_decay       = 0.0012
lr_main            = 0.000055
lr_visual          = 0.0000055
lr_discriminator   = 0.000025
```

不要第一轮把 loss 和所有正则一次性全部改变。

---

# 7. 对雨天的最新诊断与建议

雨天：

```text
target train = 96
target val   = 26
```

单个验证样本约：

```text
1 / 26 ≈ 3.85 个百分点 ACC
```

当前 Full 和多个消融都达到 100%。

因此：

> ACC 已经饱和，不能要求 Full 在 ACC 上严格超过同样 100% 的消融。

### 保留

```text
batch_size                     = 16
num_workers                    = 16
min_steps_per_epoch            = 6
vis_augmentation_strength      = 0.60
ir_augmentation_strength       = 0.40
freeze_frozen_batch_norm_stats = true
lr_scheduler_start_epoch       = 24
target_classification_weight   = 0.90
```

不要把：

```text
min_steps_per_epoch = 6
```

提高到 9 或 30；6×16 已经接近覆盖 96 个目标训练样本，提高步数会增加重复抽样。

### 第一版 loss

```text
dual_loss_weights:
  classification        = 0.60
  adv_primary           = 0.020
  adv_auxiliary         = 0.020
  cycle                 = 0.10
  identity              = 0.025
  contrastive           = 0.015
  prototype_contrastive = 0.015

module_c_warmup_epochs = 4
module_c_ramp_epochs    = 12
```

理由：

- 降低 paired/prototype 约束；
- 保留 classification 和 identity；
- 避免 Full 直接退化成对应消融。

若逐轮曲线确认持续 train→val 差距，再增加：

```text
label_smoothing    = 0.12
classifier_dropout = 0.50
weight_decay       = 0.0011
lr_main            = 0.000080
lr_visual          = 0.0000080
lr_discriminator   = 0.000025
```

`prototype_momentum=0.95` 只作为后续单因素实验。

---

# 8. Full vs 消融的公平性规则（必须遵守）

同一天气下：

```text
Full
no_cycle
no_identity
no_paired_contrastive
no_prototype_contrastive
no_classification_feedback
no_module_c
```

必须共享完全相同的：

- seed；
- batch；
- steps；
- VIS/IR augmentation；
- learning rate；
- BN；
- scheduler；
- warmup/ramp；
- early stopping；
- 其它训练设置。

唯一允许差异：

> 对应的 Module C loss weight 置零。

禁止：

> 给 Full 单独更有利的参数来制造“Full > Ablation”。

---

# 9. 当前最推荐的整体实验策略

不要立即覆盖现有 profile。

推荐两阶段：

## Phase A：Full-only pilot

每种天气分别比较少量候选 profile。

每个 profile 使用相同 3 个 seed。

比较：

- 平均 ACC；
- Macro F1；
- seed 标准差；
- train-val gap；
- best epoch；
- discriminator steps；
- Cycle error；
- Identity error；
- Prototype margin；
- Logit margin。

### 黑天

至少比较：

```text
当前 v9
VIS=0.55 / IR=0.35

统一增强
VIS=0.50 / IR=0.50

严格雾天增强
VIS=0.65 / IR=0.65
```

### 逆光

优先：

```text
降低 cycle/paired
延长 ramp
保持 VIS=0.70 / IR=0.40
```

### 雨天

优先：

```text
降低 paired/prototype
保持 6 steps
保持 VIS=0.60 / IR=0.40
```

## Phase B：固定 profile 后跑完整消融矩阵

确定每个天气的 profile 后，再运行：

```text
1 Full + 6 Ablations
× 3 天气
× 3 seed
```

严格共用天气 profile。

---

# 10. 重要研究判断

当前不能再把“黑天欠拟合、逆光过拟合、雨天过拟合”作为已证实结论。

更准确的判断：

### 黑天

训练已充分，更接近：

```text
轻微泛化波动
+
类别不平衡/覆盖限制
+
Module C 约束竞争
```

### 逆光

更接近：

```text
小验证集导致的高方差
+
部分 Module C 约束负迁移
```

### 雨天

更接近：

```text
小样本
+
验证集 ACC 饱和
+
多个消融都达到上限
```

因此下一阶段重点不是无限提高模型容量，而是：

```text
降低约束竞争
降低方差
保持公平
增加机制指标证据
```

---

# 11. v9 相关代码/工程状态

原对话已经核对过：

- 天气 profile 已拆分；
- 三天气参数不是简单共用一套；
- 正式入口已使用：

```text
configs/module_c_weather_profiles_v9.json
```

v9 已实施的核心工程方向：

- Batch size 统一到 16；
- 黑天/逆光/雨天分别设置 `min_steps_per_epoch`；
- 独立 VIS/IR 光度增强；
- 冻结视觉骨干的 BatchNorm 统计；
- 分天气设置 learning rate、regularization、warmup/ramp、early stopping、Dual loss；
- Full 与消融共用天气 profile；
- 汇总器已修正 `early_stopping_min_delta`、best epoch 与部分 iteration 的处理。

---

# 12. 当前新 Codex 应先做什么

**不要直接启动 63 次正式训练。**

先执行以下只读检查：

```bash
cd /d D:\Code\Dual_D

git status --short
git diff --check
```

然后核对：

```text
configs/module_c_weather_profiles_v9.json
configs/module_c_weather_night_v9.json
configs/module_c_weather_backlight_v9.json
configs/module_c_weather_rain_v9.json
```

以及 v9：

```text
runs/module_c_ablation_60_v9/module_c_20260830_023711
```

重点确认当前真实文件和最新运行状态是否已经发生变化。

然后再检查是否已经存在新的：

```text
v10
v11
或新的 weather profile
```

**如果工作区已经存在更高版本配置/训练结果，以真实文件为准，不要强行恢复本交接文档中的旧建议。**

---

# 13. 推荐的第一条新 Codex 指令

把下面这段直接发送给新 Codex：

```text
你现在接管 D:\Code\Dual_D 项目的后续 Module C 消融实验工作。

请先完整读取：
1. Dual_D_Codex_task_context.md
2. Dual_D_Codex_session_migration_20260820.md
3. Dual_D_code_modification_conversation_summary_20260820.md
4. 本交接文档 Dual_D_Codex_handoff_20260830.md

然后只基于当前工作区真实文件确认：
- 当前最新实验版本；
- 最新 weather profile；
- v9/v10/v11 是否存在；
- 训练是否完成；
- 当前 Full 与消融的实际平均结果；
- 是否已经有新的配置或代码修改。

不要根据旧对话猜结果。
不要立即重新训练。
不要重新设计 Module C。
不要删除已有实验。
不要给 Full 单独使用不同于消融的训练参数。

先输出：
1. 当前代码状态；
2. 当前配置状态；
3. 最新实验状态；
4. 当前三天气 Full / Ablation 三轮平均结果；
5. 与目标的差距；
6. 下一步最小、可验证的实验计划。

如果已经有最新实验结果，优先分析结果，而不是重新跑旧实验。
```

---

# 14. 历史 Codex / provider 环境注意事项

历史长线程曾出现：

```text
OpenAI 官方 provider
↔
custom 第三方 provider
```

来回切换导致旧线程 provider 元数据不一致。

如果需要继续旧线程，优先使用当前工作区状态和明确指定的 provider，不要因为旧线程元数据而错误回到官方 endpoint。

如果使用第三方 custom provider，应确认：

```text
model_provider = "custom"
```

以及对应的 `base_url` 已在当前 `config.toml` 中真实生效。

如果使用 CodexProviderSync 做官方/custom 历史同步，执行前关闭 Codex Desktop / CLI / app-server，并确认目标 Provider 正确。

---

# 15. 一个必须记住的风险：跨 provider 历史

如果一个长历史曾经由某 provider 产生了 `encrypted_content` 或 provider-specific response item：

```text
custom → openai
```

或者：

```text
openai → custom
```

元数据可以同步，但不等于历史内容一定可以在另一 provider 下无损继续。

典型风险：

- `invalid_encrypted_content`
- 不兼容的 response item ID
- compact / continuation 失败

因此：

> 已经在某 provider 下进行很久的长任务，尽量固定 provider 完成，不要反复来回切换。

---

# 16. 总体最终判断

当前研究工作已经从“把 ACC 从很低调高”转向：

```text
性能门槛基本已经满足
↓
现在主要解决：
1. Full 是否在多数情况下真正优于消融；
2. Module C 机制是否有稳定证据；
3. 小样本天气的统计方差；
4. 约束竞争；
5. profile 是否可复现。
```

最重要的实验设计原则：

> **先选择天气 profile，再固定 profile 跑完整消融；不要用 Full 特殊参数制造优势。**

雨天的 100% ACC 不应被强行解释成 Full 的绝对优势，应使用：

```text
Macro F1
seed std
Cycle cosine error
Identity cosine error
Prototype margin
Logit margin
Confidence
```

共同判断 Module C 是否有效。

---

## 当前优先级

```text
P0 读取当前工作区，确认是否已出现 v10+/新训练结果
P0 核对真实配置和最新实验状态
P1 汇总最新三轮平均结果
P1 判断 Full > Ablation 的关系
P2 如 profile 尚未定，做 Full-only pilot
P3 固定 profile 后跑完整公平消融矩阵
P4 生成最终实验分析报告
```
