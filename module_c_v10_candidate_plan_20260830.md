# Module C v10 候选验证计划

本计划以 v9 完整 63-run 结果为基线。v10 仅用于 Full gate，未经验证前不替换
`scripts/run_all_experiments.py` 中的正式 v9 默认配置，也不得覆盖任何 v9 产物。

## 1. 允许的性能取舍与公平性

允许 v10 的峰值性能略低于 v9，但 Full 三轮平均必须满足：

- 黑天 ACC >= 97.5%；
- 逆光 ACC >= 97.5%；
- 雨天 ACC >= 96.5%。

性能取舍用于降低方差、约束竞争和小验证集带来的虚假确定性，不用于人为削弱消融。
天气 profile 一旦通过 Full gate，必须在查看该 profile 的消融结果前冻结；同一天气
的 Full 和全部消融共享完全相同的数据划分、seed、增强、优化器、scheduler、早停和
非消融 loss，仅允许被消融项权重为 0。

## 2. 相对 v9 的单因素修改

| 天气 | v10 唯一修改 | 原因 |
|---|---|---|
| 黑天 | VIS/IR augmentation `.55/.35 -> .50/.50` | 五类 Module C 机制增益均为正，先检验模态增强不对称，而不削弱约束 |
| 逆光 | paired contrastive `.035 -> .020` | v9 paired mechanism gain 与 F1 gain 均为负 |
| 雨天 | prototype contrastive `.030 -> .015` | v9 prototype mechanism gain 为负，且 ACC 已饱和 |

除表中字段外，v10 与 v9 完全一致。回归测试会比较两版完整 profile，防止意外联动修改。

## 3. Phase A：只运行 Full gate

服务器示例：

```bash
python scripts/ablate_module_c.py --run \
  --base-train-config configs/train_dual_d_default.json \
  --base-dual-config configs/dual_d_default_config.json \
  --weather-profile-config configs/module_c_weather_profiles_v10.json \
  --source-root /home/lixiang/lx/Data/晴天 \
  --target-parent-root /home/lixiang/lx/Data \
  --target-domains 黑天 逆光 雨天 \
  --variants full \
  --iterations 3 --epochs 60 --seed 42 --device cuda \
  --group-iterations --no-pca-feature-view --no-tsne-feature-view \
  --output-dir runs/module_c_v10_full_gate
```

共 9 次新训练。使用与 v9 相同的 seed 分配：黑天 42--44、逆光 45--47、雨天
48--50。

Full gate 同时检查：

1. 三天气 ACC 绝对门槛；
2. Macro F1、错误样本数和 seed 标准差；
3. 每次验证 ACC 的 Wilson 95% 区间；
4. train/validation gap 与 best epoch；
5. Cycle/Identity cosine error、prototype margin、logit margin 和 confidence；
6. 三个候选 profile 的 resolved config 与本计划完全一致。

雨天 26/26 不能单独作为强证据。即使仍为 100%，也必须报告 `0/26` 错误数及约
`[87.13%, 100%]` 的单次 Wilson 95% 区间；不得把三个 seed 对同一验证集的重复预测
合并成 78 个独立样本。若后续能提供独立 test 或新的分层划分，应以其结果作为主证据。

## 4. Phase B：固定 profile 后运行消融

只有 Phase A 通过后，才运行六个消融；不要再次训练 Full：

```bash
python scripts/ablate_module_c.py --run \
  --base-train-config configs/train_dual_d_default.json \
  --base-dual-config configs/dual_d_default_config.json \
  --weather-profile-config configs/module_c_weather_profiles_v10.json \
  --source-root /home/lixiang/lx/Data/晴天 \
  --target-parent-root /home/lixiang/lx/Data \
  --target-domains 黑天 逆光 雨天 \
  --variants no_cycle no_identity no_paired_contrastive \
             no_prototype_contrastive no_classification_feedback no_module_c \
  --iterations 3 --epochs 60 --seed 42 --device cuda \
  --group-iterations --no-pca-feature-view --no-tsne-feature-view \
  --output-dir runs/module_c_v10_ablations
```

汇总时用 `--reference-runs-root` 合并 Phase A 的 9 个 Full 结果，只做分析，不重复
训练 Full。若 Full gate 未通过，停止 Phase B，并针对失败天气只做下一个单因素候选。

