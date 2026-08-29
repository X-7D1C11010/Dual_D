# Module C v7/v8 诊断与 v9 优化方案

分析日期：2026-08-30

本报告只使用当前工作区中的日志、JSON、CSV、NPZ 证据和源代码。没有重跑
v6、v7、v8，也没有在本机启动 v9 训练。v9 是已经落地、但尚待服务器三次
迭代验证的候选配置，不能把预期值表述为已实现结果。

## 1. 结论摘要

1. `module_c_ablation_60_v7.log` 的文件名与内容版本不一致；该日志实际启动的
   是 `runs/module_c_ablation_60_v8/module_c_20260829_205331`。
2. v8 在 `no_paired_contrastive/逆光/iter03` 的第 3 epoch 后收到外部
   `SIGTERM: 15`。这是训练阶段外部终止，不是汇总、可视化或文件检查错误。
   已完成的 32 次训练有效；部分 iter03 和未启动的 30 次不能纳入统计。
3. v7 是当前最新的完整 63-run 矩阵。按三次均值比较，Full 的 ACC 相对 18 个
   消融单元为 9 胜、6 平、3 负，尚未达到“大多数严格高于”的要求；Macro F1
   为 12 胜、5 平、1 负。
4. v8 Full 的正式三次均值为：黑天 ACC 96.6667%、逆光 99.0991%、雨天
   100%。黑天尚差 3 个累计正确预测才能达到 97.5%；逆光和雨天已经达到目标。
5. 黑天不是容量性欠拟合：三个 seed 的训练端已接近饱和。逆光和雨天也没有
   持续过拟合曲线证据；主要风险是极小验证集造成的离散跳变和指标天花板。
6. v9 不加深或加宽网络，也不改 Module C 定义。它在所有变体公平共用同一
   天气 profile 的前提下，调整小 batch、有效步数、VIS/IR 独立增强、冻结骨干
   BatchNorm 统计、正则化、warmup/ramp 和早停。

## 2. 证据与指标口径

主要证据路径：

- v8 日志：`D:/Code/Dual_D/module_c_ablation_60_v7.log`
- v8 manifest：`runs/module_c_ablation_60_v8/module_c_20260829_205331/experiment_manifest.json`
- v8 运行目录：`runs/module_c_ablation_60_v8/module_c_20260829_205331`
- v7 完整运行目录：`runs/module_c_ablation_60_v7/module_c_20260829_145907`
- v7 高维机制证据：`runs/module_c_ablation_60_v7/module_c_20260829_145907/constraint_feature_evidence.csv`
- 每次运行的曲线、最终结果、配置和样本分布：各目录的
  `metrics_iter*.csv`、`result_summary_iter*.json`、
  `resolved_config_iter*.json`、`resolved_dual_config_iter*.json`、
  `class_distribution_iter*.json` 和 `data_audit_iter*.json`

正式结果统一取 `result_summary_iter*.json` 中训练器实际保存的
`best_metrics.val`：

- ACC：`accuracy`
- Macro Precision：`precision_macro_present`
- Macro Recall：`recall_macro_present`
- Macro F1：`f1_macro_present`
- 监控指标：`val_f1_macro_present`

不把顶层独立峰值 `best_acc` 与另一 epoch 的 P/R/F1 混在一张结果表中，也不使用
把验证集中缺席类别计为 0 的 `*_macro_all`。本次还修正了汇总器口径：若存在
`result_summary`，必须服从训练器写入的 `best_metrics.epoch`，包括
`early_stopping_min_delta` 的影响。

## 3. v8 运行状态与报错

日志第 2843 行附近启动
`module_c_no_paired_contrastive_逆光/iter03`，第 2846 行记录完 epoch 3；
traceback 从第 2855 行开始。父进程在
`scripts/ablate_module_c.py` 的 `subprocess.run(..., check=True)` 收到：

```text
subprocess.CalledProcessError: ... died with <Signals.SIGTERM: 15>.
```

日志没有记录谁从外部发送了 SIGTERM，因此不能推测是人工停止、调度器时限还是
服务器资源管理。能够确定的是：错误发生在训练子进程，而非训练后的汇总、绘图或
引用目录检查。

v8 manifest 声明 7 个变体 × 3 种天气 × 3 次迭代，共 63 次。实际文件计数为：

| Artifact | 数量 | 解释 |
|---|---:|---|
| `metrics_iter*.csv` | 33 | 32 个完整运行加 1 个仅 3 epoch 的部分运行 |
| `best_metrics_iter*.json` | 33 | 部分运行也写过阶段性 best |
| `resolved_config_iter*.json` | 33 | 已启动迭代均存在 |
| `resolved_dual_config_iter*.json` | 33 | 已启动迭代均存在 |
| `result_summary_iter*.json` | 32 | 只有训练正常结束才存在 |
| `feature_embeddings_iter*.npz` | 32 | 只有 32 个完整结果存在 |

因此结论是 32 次完成、1 次异常中止、30 次尚未启动。已完整的组包括 Full、
`no_cycle`、`no_identity` 的三天气三次迭代，以及
`no_paired_contrastive/黑天` 三次；`no_paired_contrastive/逆光` 只有两次完整。

## 4. 当前结果汇总

### 4.1 v7 完整矩阵

下表是 3 次迭代算术平均，单位均为百分比。

| Variant | 天气 | ACC | Macro P | Macro R | Macro F1 |
|---|---|---:|---:|---:|---:|
| full | 黑天 | 96.3889 | 95.2947 | 97.1549 | 95.9434 |
| full | 逆光 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| full | 雨天 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_cycle | 黑天 | 97.2222 | 95.2739 | 97.1240 | 95.7838 |
| no_cycle | 逆光 | 99.0991 | 99.2593 | 99.5370 | 99.3416 |
| no_cycle | 雨天 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_identity | 黑天 | 96.6667 | 94.9923 | 97.1979 | 95.8787 |
| no_identity | 逆光 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_identity | 雨天 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_paired_contrastive | 黑天 | 96.6667 | 95.1737 | 97.0003 | 95.7916 |
| no_paired_contrastive | 逆光 | 99.0991 | 99.2593 | 99.5370 | 99.3416 |
| no_paired_contrastive | 雨天 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_prototype_contrastive | 黑天 | 96.3889 | 95.9408 | 96.9573 | 96.0734 |
| no_prototype_contrastive | 逆光 | 99.0991 | 99.2593 | 99.5370 | 99.3416 |
| no_prototype_contrastive | 雨天 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_classification_feedback | 黑天 | 94.4444 | 92.2442 | 95.4410 | 93.2295 |
| no_classification_feedback | 逆光 | 99.0991 | 99.2593 | 99.5370 | 99.3416 |
| no_classification_feedback | 雨天 | 98.7179 | 98.8889 | 98.8889 | 98.7654 |
| no_module_c | 黑天 | 93.6111 | 91.1910 | 95.4133 | 92.8892 |
| no_module_c | 逆光 | 99.0991 | 99.2593 | 99.5370 | 99.3416 |
| no_module_c | 雨天 | 98.7179 | 98.8889 | 98.8889 | 98.7654 |

### 4.2 v8 已完成的三次组

| Variant | 天气 | ACC | Macro P | Macro R | Macro F1 |
|---|---|---:|---:|---:|---:|
| full | 黑天 | 96.6667 | 96.1058 | 96.9207 | 96.2590 |
| full | 逆光 | 99.0991 | 99.2593 | 99.5370 | 99.3416 |
| full | 雨天 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_cycle | 黑天 | 96.9444 | 95.6693 | 97.1019 | 96.1162 |
| no_cycle | 逆光 | 99.0991 | 99.2593 | 99.5370 | 99.3416 |
| no_cycle | 雨天 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_identity | 黑天 | 97.2222 | 94.3223 | 97.6992 | 95.6568 |
| no_identity | 逆光 | 99.0991 | 99.2593 | 99.5370 | 99.3416 |
| no_identity | 雨天 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| no_paired_contrastive | 黑天 | 96.3889 | 94.5009 | 97.3148 | 95.6004 |

v8 剩余组不满三次，不能用部分结果推断完整 Module C 排序。已完成的 23 个
同 seed Full-versus-ablation 配对中，ACC 为 1 胜、20 平、2 负，F1 为 4 胜、
18 平、1 负；该统计只说明已跑部分高度饱和，不代表整个 v8 矩阵。

## 5. 数据、曲线与拟合状态

### 5.1 数据规模和预处理

v8 日志和 `class_distribution_iter01.json` 记录源域训练样本 992 个、14 类。
三个目标域为：

| 天气 | target train | target val | 出现类别数 | 单个验证错误造成的 ACC 变化 |
|---|---:|---:|---:|---:|
| 黑天 | 465 | 120 | 8 | 0.8333 个百分点 |
| 逆光 | 132 | 37 | 9 | 2.7027 个百分点 |
| 雨天 | 96 | 26 | 6 | 3.8462 个百分点 |

目标类别明显不平衡。黑天 train 中原始类别 5/1/2/7 分别有
162/100/89/44 个样本；逆光原始类别 7 有 29 个，而类别 14 只有 2 个；雨天
原始类别 5 有 32 个，而类别 7 只有 6 个。当前 `PairedClassSampler` 已按共同
类别均匀抽取并保证 source/target 同类配对，因此没有再叠加第二套 class weight。

三个 Full iter01 的 data audit 都记录 VIS/IR 路径重叠为 0、内容哈希重叠为 0，
没有 train/val 泄漏证据。训练增强包含随机裁剪和翻转；原实现只允许一个光度
增强强度同时控制 VIS/IR，v9 已改为两模态独立强度。

### 5.2 黑天

v8 Full 三个 seed 的正式选模情况：

| iter | 选中 epoch | 完成 epoch | optimizer steps | 末次 full-train ACC | 正式 val ACC | loss 首轮→末轮 |
|---:|---:|---:|---:|---:|---:|---:|
| 01 | 36 | 51 | 306 | 97.6344% | 96.6667% | 5.6796→2.0942 |
| 02 | 60 | 60 | 360 | 98.4946% | 95.0000% | 6.0125→2.0763 |
| 03 | 41 | 56 | 336 | 99.5699% | 98.3333% | 5.9307→2.0702 |

采样训练 ACC 后期接近 100%，完整训练集 ACC 也达到 97.6%–99.6%，loss 持续
下降，所以黑天不是“训练集和验证集都低”的经典欠拟合，也没有模型深度/宽度
不足的证据。问题更接近：类别不平衡下的泛化误差、seed 波动、冻结视觉骨干的
BN 统计漂移，以及 Module C 约束竞争。

三次选模混淆矩阵累计为 348/360 正确。主要错误是 class index 11→0 五次、
9→11 三次；映射回原始标签分别为 7→1、5→7。达到 97.5% 至少需要
351/360 正确，即只需把累计错误从 12 个降到不超过 9 个。

### 5.3 逆光

三个 seed 的选中 epoch 为 32/25/41，optimizer steps 为 200/180/236；末次
full-train ACC 均为 100%。loss 分别从 6.1941/5.8524/6.0087 下降到
1.9341/2.0486/1.9388。正式结果为 100%、97.2973%、100%，三次总计仅错
1/111，唯一错误为 index 11→10（原始标签 7→6）。

这存在一个 seed 的训练-验证间隙和后期波动，但没有三个 seed 一致的验证下降，
不能据此诊断成持续过拟合。验证集只有 37 个样本，单个错误就改变 2.70 个百分点，
是当前波动和消融平局的首要解释。

### 5.4 雨天及与逆光的对比

三个 seed 的选中 epoch 为 41/28/29，optimizer steps 为 240/200/200；末次
full-train ACC 为 98.9583%、98.9583%、100%。loss 从
6.0781/5.5973/6.1647 降至 1.9582/2.0307/2.0429，验证均为 26/26 正确。

雨天没有“训练高、验证持续下降”的现象，验证甚至会等于或高于完整训练集 ACC，
因此经典过拟合不成立。它和逆光都受小验证集量化影响，但雨天更严重：只有 26 个
验证样本，一个错误就是 3.85 个百分点，并且类别只有 6 个。雨天应避免过强正则
重新造成欠优化，同时首先消除每 epoch 近四倍重复抽样。

## 6. Module C 消融与机制贡献

### 6.1 公平性和架构差异

v8 日志对 Full 和每个已启动消融都记录相同参数量：总参数 17,277,714，
可训练参数 17,120,210。生成器、主/辅判别器、视觉/红外特征提取器、TAL 和
分类器结构均不变；消融只把指定 Module C loss weight 置零。天气覆盖逻辑只更新
仍为非零的权重，不会把已消融约束重新启用。

因此当前实验评估的是“约束项贡献”，不是删除网络分支后的参数量贡献。
`no_module_c` 也只移除 Cycle、Identity、paired/prototype contrastive 和
classification feedback，仍保留对抗翻译核心，不能写成删除整个生成器模块。

### 6.2 Full 是否大多数高于消融

v7 按同天气、同 iteration、同 seed 严格配对 54 次，seed 全部匹配：

| 指标 | Full 高于 | 相等 | Full 低于 |
|---|---:|---:|---:|
| ACC | 15 | 32 | 7 |
| Macro Precision | 16 | 30 | 8 |
| Macro Recall | 18 | 30 | 6 |
| Macro F1 | 16 | 30 | 8 |

按每个“天气 × 消融”的三次均值比较 18 次：ACC 为 9/6/3，Macro F1 为
12/5/1（高/平/低）。所以 F1 已表现出多数优势，ACC 仍只有一半严格胜出，且
平局不算“少量”。当前要求没有完全满足。

逆光和雨天大量平局不是 Module C 完全无效的充分证据：37/26 个验证样本使多个
模型同时到达 100% 天花板。若要统计上证明 Full 严格更高，应增加独立验证/测试
样本或使用预注册的重复分层交叉验证，不能故意削弱消融模型，也不能给 Full 使用
不同超参数。

### 6.3 原始高维机制证据

`constraint_feature_evidence.csv` 在原始特征空间给出：

- Cycle 使黑天/逆光/雨天的 target cycle cosine error 平均降低约
  0.0433/0.0301/0.0223。
- Identity 使三天气 identity cosine error 平均降低约
  0.0058/0.0041/0.0022。
- Classification feedback 使 generated-feature correct-class logit margin
  平均提高约 0.6855/0.4553/0.5486。
- paired/prototype margin 在黑天和逆光总体为正；雨天不同 seed 正负混合且均值
  偏负，说明雨天存在负迁移风险，权重应保持温和。

这些结果说明 Cycle、Identity 和 classification feedback 确实改变了其目标机制，
即便离散 ACC 因天花板而相同。主证据继续保留 Cycle/Identity cosine error、
prototype margin、logit margin、confidence 和原始高维指标。t-SNE 仅是可选后处理，
一键入口继续显式传 `--no-tsne-feature-view`，不作为 Module C 主要证据。

## 7. 已实施的 v9 优化

### 7.1 训练和数据参数

profile 索引为 `configs/module_c_weather_profiles_v9.json`，三个天气分别解析到独立
JSON。关键值如下：

| 参数 | 黑天 v9 | 逆光 v9 | 雨天 v9 |
|---|---:|---:|---:|
| batch size | 16 | 16 | 16 |
| min steps/epoch | 30 | 9 | 6 |
| 每 epoch 抽样数 | 480 | 144 | 96 |
| base / VIS / IR augmentation | .50/.55/.35 | .65/.70/.40 | .55/.60/.40 |
| label smoothing | .06 | .10 | .10 |
| classifier dropout | .44 | .48 | .46 |
| target classification weight | .85 | .75 | .90 |
| main / visual LR | 1e-4 / 1e-5 | 7e-5 / 7e-6 | 9e-5 / 9e-6 |
| discriminator LR | 3.2e-5 | 2.75e-5 | 2.75e-5 |
| weight decay | 8e-4 | 1e-3 | 9e-4 |
| adversarial warmup/ramp | 5/15 | 6/14 | 6/15 |
| Module C warmup/ramp | 3/8 | 4/10 | 3/8 |
| stability window | 3 | 3 | 3 |
| early-stop min epoch/patience | 50/15 | 35/12 | 35/12 |
| freeze frozen BN stats | true | true | true |

按目标 train 样本数计算，v8 的 batch 92 实际每 epoch 抽样 552/368/368 个，
重复率约 1.19x/2.79x/3.83x；v9 变为约 1.03x/1.09x/1.00x。黑天获得更多、
更小粒度的 optimizer update，逆光和雨天则显著减少同一 epoch 内重复记忆。

Dual loss weights 为：

| Weight | 黑天 | 逆光 | 雨天 |
|---|---:|---:|---:|
| classification | .65 | .15 | .60 |
| adv primary / auxiliary | .025/.025 | .025/.025 | .02/.02 |
| cycle | .12 | .12 | .12 |
| identity | .025 | .03 | .025 |
| paired contrastive | .055 | .035 | .025 |
| prototype contrastive | .025 | .06 | .03 |

同一 profile 对 Full 和全部消融完全共用；被消融项的 0 权重优先级更高，天气覆盖
不会恢复它。黑天降低 Cycle/Identity 竞争但保留已证明的机制作用；逆光使用 v7
已达到 100% Full 的温和分类反馈组合；雨天保留 v8 达到 100% 的低对比权重区间。

参数方向还有当前工作区 pilot 支持，但必须正确解释为单 seed：

- `runs/local_weather_tuning_20260827/candidate_full_night_seed42`：seed 42，
  monitor-selected ACC 99.1667%。
- `runs/local_pilot/pilot_backlight_revised_seed46`：seed 46，ACC 100%。
- `runs/local_pilot/pilot_rain_candidate_seed51` 的强正则组合只有 96.1538%；
  `runs/local_weather_tuning_20260827/candidate_full_rain_seed51_retry` 恢复中等
  正则后为 100%。

这些 pilot 只能支持参数方向，不能替代三次平均验收。

### 7.2 代码层调整

- `dual_d/data/multimodal_dataset.py`：增加
  `vis_augmentation_strength` / `ir_augmentation_strength`，未设置时完全回退到旧
  `augmentation_strength`，不破坏旧配置。
- `dual_d/training/trainer.py`：当 profile 启用时，每次 `net_vis.train()` 后把
  affine 参数全部冻结的 BatchNorm 层重新设为 eval，防止冻结骨干仍更新 running
  mean/variance；并在日志记录实际三种增强强度和 BN 开关。
- `scripts/train_dual_d.py`：新字段已加入 profile 白名单、类型/范围校验和 CLI。
- `scripts/run_all_experiments.py`：三天气正式入口默认使用 v9 profile，并继续关闭
  PCA/t-SNE 主要视图。
- `scripts/ablate_module_c.py`：汇总优先使用 trainer 最终选中 epoch，并跳过同一
  grouped run 中有 metrics、无 result summary 的中断迭代。
- 定向测试覆盖独立增强回退、冻结 BN、profile 校验、消融架构一致和汇总 epoch
  口径。

没有改变通用 `configs/train_dual_d_default.json`，因为它还服务雾天流程，而 v9
只覆盖当前三种天气。直接调用 `ablate_module_c.py --run` 时必须显式传 v9；推荐
统一通过 `run_all_experiments.py`。

## 8. 下一轮验证方法

### 8.1 预注册验收门槛

三个 seed 的验证样本总数分别为 360/111/78。ACC 门槛应按整数正确数验收：

| 天气 | ACC 门槛 | 等价累计正确数 |
|---|---:|---:|
| 黑天 | >=97.5% | >=351/360 |
| 逆光 | >=97.5% | >=109/111 |
| 雨天 | >=96.5% | >=76/78 |

每个结果必须使用 trainer monitor-selected epoch，报告三次均值、标准差、逐 seed
结果和混淆矩阵，不允许挑 seed。Full-vs-ablation 的主验收预先定义为 18 个
“天气 × 消融”三次均值比较中 ACC 严格胜出至少 10 个，平局不超过 3 个；同时
报告 54 个同 seed 次级配对和 Macro F1，不得在看到结果后改口径。

### 8.2 一次性正式矩阵

在原服务器数据布局上，推荐只执行一次完整 63-run 矩阵：

```bash
python scripts/run_all_experiments.py \
  --source-root /home/lixiang/lx/Data/晴天 \
  --target-parent-root /home/lixiang/lx/Data \
  --target-domains 黑天 逆光 雨天 \
  --iterations 3 --epochs 60 --seed 42 --device cuda \
  --weather-profile-config configs/module_c_weather_profiles_v9.json \
  --output-dir runs/module_c_ablation_60_v9
```

该入口恰好运行 Full 加六个消融，不重复 Full，三个天气使用各自 profile，且同天气
Full/消融使用相同 seed 序列。

### 8.3 可选的无重复分阶段门控

若算力有限，可先只跑 Full。Full 三天气全部通过上表门槛后，再只跑六个消融；
不要随后再用完整入口重复训练 Full。

```bash
python scripts/ablate_module_c.py --run \
  --source-root /home/lixiang/lx/Data/晴天 \
  --target-parent-root /home/lixiang/lx/Data \
  --target-domains 黑天 逆光 雨天 \
  --variants full --iterations 3 --epochs 60 --seed 42 --device cuda \
  --weather-profile-config configs/module_c_weather_profiles_v9.json \
  --output-dir runs/module_c_v9_full_gate
```

通过后，用同样参数把 `--variants` 改为：

```text
no_cycle no_identity no_paired_contrastive no_prototype_contrastive
no_classification_feedback no_module_c
```

最终分析时把 Full 实验目录作为 `--reference-runs-root` 合并，只做汇总，不重新训练。

## 9. 限制与预期效果

- v9 预期通过小 batch 和接近一次/epoch 的目标抽样，降低逆光/雨天重复记忆，同时
  给黑天更细粒度更新；独立模态增强避免对 IR 施加与 VIS 相同强度的光度扰动；
  冻结 BN 预期减少 seed 间统计漂移。
- 黑天只需额外修正三次累计中的 3 个样本即可达标，但这不是保证。逆光和雨天
  已在 v8 Full 达标，v9 的重点是保持阈值并改善训练稳定性。
- 当前验证集太小，特别是雨天，无法可靠区分多个接近 100% 的模型。即使 v9 指标
  达标，也应在更大的独立测试集或预注册重复分层验证上复核 Full 的严格优势。
- 没有证据支持增加网络深度、宽度或注意力模块；那会增加消融混杂并违背不重设计
  Module C 的约束。本轮结构侧优化仅修正冻结骨干 BN 行为，保留 17.28M 参数架构。
- 只有新三 seed 正式结果可以确认目标是否实现。若 Full gate 未通过，应先依据新增
  混淆矩阵和高维机制指标做单变量调整，不应直接启动完整 63-run 矩阵。
