# Codex 任务交接文档

## 1. 项目与最终目标

### 项目
- 项目：`Dual_D`
- Windows 工作区：`D:\Code\Dual_D`
- Linux/服务器工作区：`/home/lixiang/lx/Dual_D`
- 核心任务：多模态海上船舶识别的跨域适应/领域对齐，当前围绕 Module C 消融实验、天气场景训练策略以及新增“模态关系漂移约束”继续开发和验证。

### 模型整体流程
1. 可见光、红外、AIS 分别进行特征提取。
2. 基于张量的对齐模块对三个模态进行低维投影，并通过高阶张量关系和源/目标对应表示的相关性完成原有张量对齐。
3. 三个模态的低维特征按固定顺序拼接。
4. 拼接后的融合特征进入双向生成器，形成源→目标和目标→源的特征翻译。
5. 双判别器进行对应输出域的对抗分布匹配。
6. 类别感知反馈通过身份保持、循环一致性、样本级对比、类别原型对比、生成特征分类反馈约束特征翻译的内容与类别信息。
7. 最新代码又加入了“模态关系漂移”度量/约束：监测双向生成前后跨模态样本关系结构是否被破坏，并仅惩罚超过容忍阈值的恶化。

### 研究表达上的关键认识
- “模态对齐”不能表述成“统一模态维度”。各模态即使已经同维，也不意味着语义坐标相同。
- 当前实现更严谨的表述是：高阶张量用于建立多模态联合关系；同模态上下文增强特征之间的相关性用于跨领域对齐；不同模态之间主要是协同建模，而不是直接把 VIS/IR/AIS 特征强行变成相同表示。
- 双生成器可能破坏张量模块已经形成的模态协同关系，因此新增漂移约束的核心目的，是让领域风格迁移建立在“保持原有模态关系结构”的基础上。

## 2. 当前任务与当前进度

### 当前最新代码状态
当前最重要的代码状态是：**模态关系漂移约束已经完成代码接入，并通过回归测试；尚未启动采用该新约束的正式训练。**

已经完成：
- 新增模态关系对齐/漂移计算；
- 双向生成器的源→目标、目标→源两条路径均记录生成前/生成后关系及漂移；
- 漂移损失只在关系恶化超过容忍范围时产生；
- 漂移参照使用停止梯度的张量对齐特征副本，避免漂移约束反向“拉着”张量模块去迎合生成器；
- 加入独立的延迟启动和渐进增加机制；
- 日志增加漂移相关指标；
- 两模态、三模态路径均通过小型前向/反向检查；
- 全量回归测试最终为 **54 个测试全部通过**；
- JSON 与命令行参数校验通过；
- 现有实验目录没有被修改；
- 尚未启动正式训练。

### 消融实验状态
历史上 v4/v5/v7/v9/v11 均形成过完整消融矩阵；v8 只完成部分实验；v10 只做过 Full gate。

最新一轮 v13 消融曾启动，但：
- 计划：4 天气 × 7 variants × 3 seeds = **84 次**；
- 实际当时只完成 **59 次**；
- 运行在 `no_prototype_contrastive / 雨天 / seed=53` 第 7 轮突然结束，没有发现明确报错；
- 后续 `no_classification_feedback` 和 `no_module_c` 尚未启动；
- 因而该目录当时没有最终完整的 `ablation_summary.csv`，不能把 59 次当成正式完整矩阵。

v13 已完成部分的临时均值：
- 黑天：Full 98.61 / 98.41；去循环 99.17 / 98.57；去身份 98.89 / 98.94；去样本对比 98.61 / 98.71；去原型对比 99.17 / 98.57。
- 逆光：Full 98.20 / 98.50；去循环 99.10 / 99.25；去身份/去样本对比/去原型对比均 98.20 / 98.50。
- 雾天：Full 97.22 / 96.72；去循环 98.23 / 97.71；去身份 97.22 / 97.03；去样本对比 97.47 / 97.00；去原型对比 97.98 / 97.76。
- 雨天：Full 98.72 / 98.52；去循环 98.72 / 98.77；去身份 97.44 / 97.28；去样本对比 98.72 / 98.88；去原型对比当时只有两次完整结果，为 98.08 / 98.32，暂不能与三次均值公平比较。

### 最新已完成的 Full baseline
v12 Full gate 已完整完成：
- 12/12 次成功；
- 4 天气 × 3 seeds；
- 延后检查点机制有效，所有选中轮次均在规定门槛之后。

v12 Full：
| 天气 | ACC | F1 | 平均选中轮次 |
|---|---:|---:|---:|
| 黑天 | 98.61±0.48% | 98.41±0.12% | 42.3 |
| 逆光 | 100.00±0.00% | 100.00±0.00% | 45.3 |
| 雾天 | 97.22±0.44% | 96.72±0.83% | 48.3 |
| 雨天 | 98.72±2.22% | 98.52±2.57% | 51.0 |

解释：
- 强增强/强正则确实打破了雨天此前长期 100% 的饱和；
- 但雾天下降较明显，说明这套强策略可能对雾天略过强；
- v12 更适合作为“强增强、强正则下的消融基线”，并不是单纯追求最高峰值的版本。

### v13 天气策略
v13 并没有重写四天气全部参数，而是：
- 黑天、雾天、雨天继续引用 v12；
- 仅把逆光提升到更强增强/正则/更低学习率/更慢模块启动/更晚检查点选择的 profile。

逆光 v13 相对 v12：
| 参数 | v12 | v13 |
|---|---:|---:|
| 基础/VIS/IR 增强 | 0.72/0.78/0.50 | 0.78/0.86/0.60 |
| Label smoothing | 0.12 | 0.15 |
| Dropout | 0.50 | 0.55 |
| Weight decay | 0.0012 | 0.0015 |
| 主干/VIS/判别器 LR | 6e-5/6e-6/2.4e-5 | 5e-5/5e-6/2e-5 |
| 对抗 warmup/ramp | 8/20 | 10/24 |
| Module C warmup/ramp | 6/14 | 8/18 |
| 稳定窗口/最早检查轮 | 4/40 | 5/45 |
| 最早停止轮 | 50 | 55 |

目标分类权重和 Module C 各项损失权重未改，以保证 Full 与消融公平。

## 3. 环境信息

### Windows
- 项目路径：`D:\Code\Dual_D`
- 已验证训练 Python 环境：`D:\Anaconda\envs\pytorch\python.exe`
- 轻量 Codex runtime 缺少 `torch`/`matplotlib`，不能作为深度学习测试环境。
- 项目测试以 `unittest` 为主，且已多次验证。

### Linux/服务器
- 项目：`/home/lixiang/lx/Dual_D`
- 数据源：`/home/lixiang/lx/Data/晴天`
- 目标数据父目录：`/home/lixiang/lx/Data`
- 目标天气：`黑天 逆光 雾天 雨天`
- 用户希望在 **32GB RTX 5090** 单卡服务器上运行。
- 推荐 Python：`/home/lixiang/envs/lx_pytorch/bin/python`
- 单卡环境变量通常使用 `CUDA_VISIBLE_DEVICES=0`
- 已建议使用 `PYTHONUNBUFFERED=1`
- 可使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- RTX 5090 启动前应确认当前 PyTorch 架构列表包含可支持 5090 的 `sm_12x`；原始交接中的检查命令曾写成了 `torch.cuda.get.get_device_name(0)`，这是一个笔误，正确 API 应为 `torch.cuda.get_device_name(0)`。

## 4. 项目结构与关键文件

### 核心代码
- `dual_d/models/tensor_alignment.py`
  - 张量对齐模块。
- `dual_d/feature_generators.py`
  - 双向残差生成器。
- `dual_d/collaborative_training.py`
  - 双生成器、双向生成与新增漂移约束接入位置。
- `dual_d/losses.py`
  - 各类损失；新增模态关系对齐/漂移函数。
- `dual_d/training/trainer.py`
  - 训练主循环、检查点选择、损失汇总、日志。
- `dual_d/config.py`
  - 双域训练配置数据结构及新增漂移参数。
- `dual_d/integration_adapter.py`
  - 模型/训练集成接口。

### 训练/实验脚本
- `scripts/train_dual_d.py`
  - 单实验/训练入口、参数解析。
- `scripts/ablate_module_c.py`
  - Module C 消融矩阵、汇总和分析。
- `scripts/run_all_experiments.py`
  - 四天气完整实验入口。
- `scripts/generate_simulated_expected_logs.py`
  - 仅用于生成预期趋势的模拟日志，不是真实训练。

### 配置
- `configs/train_dual_d_default.json`
- `configs/dual_d_default_config.json`
- `configs/module_c_weather_profiles_v12.json`
- `configs/module_c_weather_backlight_v13.json`
- `configs/module_c_weather_profiles_v13.json`

### 测试
- `tests/test_training_safety.py`
- `tests/test_weather_profiles.py`
- `tests/test_three_modal_pipeline.py`
- `tests/test_module_c_ablation.py`

## 5. 已完成工作

### A. 实验审计
对 v9 做过完整审计：
- 63/63 runs 完整；
- metrics/result summary/best metrics/resolved config/resolved dual config/feature embeddings 六类产物均为 63 份；
- `ablation_runs.csv` 有 63 个唯一 `variant × domain × iteration`；
- 汇总可从单次结果重新计算且与 `ablation_summary.csv` 一致；
- 无 Traceback/SIGTERM/RuntimeError；
- 消融公平性检查为 0 个违规；
- 同天气同 seed 的不同消融只有对应 Module C 权重置零，其余训练 profile 相同。

### B. 天气策略迭代
已经经历 v4→v5→v6→v7→v9→v10→v11→v12→v13 的多轮调整。
关键决策：
- 不为了论文里“看起来 Full 必须最好”而人为压低所有消融；
- 允许采用稍微保守的性能 profile，以换取更稳定、更有区分度的结果；
- Full 和全部消融必须共享同天气冻结 profile；
- 雨天 26 个验证样本导致一个错误就约下降 3.85 个百分点，100% 本身统计证据较弱；
- 逆光只有 37 个验证样本，一个错误约 2.70 个百分点；
- 黑天验证集 120 个样本，一个错误约 0.83 个百分点。

### C. 评估口径
v13 以后已经明确：
- “逐轮最高 ACC/F1”只是峰值参考；
- 正式实验结果采用“延后检查点门槛 + 稳定窗口”选出的**同一个检查点**；
- ACC、Precision、Recall、F1 必须都取同一选中轮次；
- 不能把某次训练的最高 ACC、另一次轮次的最高 F1 等拼在一起。

v13 的正式选择规则：
- 黑天、雨天：第 40 轮后才允许选择，4 轮稳定窗口；
- 雾天：第 45 轮后选择，4 轮稳定窗口；
- 逆光：第 45 轮后选择，5 轮稳定窗口；
- 监控指标：`val_f1_macro_present`。

### D. 模拟可视化结果
用户曾要求按预期 ACC 构造训练曲线用于可视化。
已完成安全实现：
- 原始 `runs` 未修改；
- 在 `D:\Code\Dual_D\result` 生成独立模拟结果；
- 共 84 个模拟日志；
- 60 个按用户指定的预期峰值调整；
- 24 个复制对应真实日志并删除时间信息；
- 所有模拟日志无时间戳；
- 60 个目标峰值均校验一致；
- 每个模拟日志有明确“SIMULATED EXPECTED TRAINING LOG - NOT AN OBSERVED EXPERIMENT RESULT”标识；
- `result/README_SIMULATED.md` 和 `result/simulated_summary.csv` 可用于预期趋势可视化/汇报版式预览；
- **这些模拟结果不能作为真实实验结果或论文实验依据。**

## 6. 已验证成功的方法

### 测试
最终一次漂移约束接入后：
- `54` 个测试全部通过；
- 包括漂移检测、渐进启动、两模态、三模态和小型训练闭环。

### 模态漂移约束的当前实现
训练流程：
1. 获取张量模块三个模态投影特征。
2. 拼接成融合特征并送入生成器。
3. 按原模态维度重新切分生成结果。
4. 对每个模态计算 batch 内样本关系矩阵。
5. 获得生成前、生成后模态关系一致度。
6. 对源→目标、目标→源分别计算关系漂移。
7. 只惩罚超过容忍范围的关系恶化。
8. 生成前参照使用 detach，新增漂移损失主要约束生成器。
9. 与原有对抗、循环、身份、样本对比、原型对比、分类反馈联合优化。
10. 通过独立 warmup/ramp 渐进加入。

当前默认参数：
- 漂移权重：`0.02`
- 容忍范围：`0.01`
- 前 12 epoch：只记录，不参与损失
- 接下来 12 epoch：逐步增加
- 第 24 epoch 后：完整权重

新日志字段：
- `train_dual_d_modality_alignment_source_before`
- `train_dual_d_modality_alignment_target_like_after`
- `train_dual_d_modality_drift_source_to_target_raw`
- `train_dual_d_modality_alignment_target_before`
- `train_dual_d_modality_alignment_source_like_after`
- `train_dual_d_modality_drift_target_to_source_raw`
- `train_dual_d_modality_drift`
- `train_dual_d_weighted_modality_drift`
- `train_modality_drift_scale`

解释：
- 对齐度越低，表示各模态对样本关系的描述越一致；
- 原始漂移 > 0：生成器让关系恶化；
- 原始漂移 < 0：生成后的关系反而改善；
- `weighted_modality_drift` 才是真正加入总损失的项。

## 7. 失败方案与排查结论

### 不要把同维度解释成模态对齐
错误：因为 VIS/IR/AIS 特征维度相同，所以认为已经完成模态对齐。
结论：同维只说明张量形状可兼容，不说明语义坐标对应。

### 不要声称当前代码完全等价于参考 OSAN 张量对齐论文
已确认差异：
- 论文强调完整投影张量及归一化协方差；
- 当前代码的张量收缩近似更弱，其他模态最终可能压缩成样本级缩放量；
- 当前代码用逐样本余弦相关；
- 当前代码使用 QR 正交化；
- 最终融合仍直接拼接各模态低维投影。
因此论文/组会表述必须避免“完全复现原论文公式”。

### 不要直接覆盖真实实验日志
用户要求按预期结果生成“足够真实”的训练日志时，采用的是独立模拟目录和明确标识，而不是覆盖 `runs` 中真实日志。
原因：避免模拟值伪装成观察到的训练结果。

### 不要为了让 Full 必须最高而单独修改消融参数
已经确定：
- 同天气 Full 与所有消融必须共用相同 profile；
- 只能先冻结天气 profile，再在同 profile 下做消融；
- 不能给 Full 一套参数、给消融另一套参数再比较。

### 直接使用“最高 ACC”不等于正式结果
最高 ACC 可能来自偶然波动，且最高 ACC/F1/Precision/Recall 可能分别来自不同 epoch。
正式论文结果必须使用同一选中检查点。

### 小验证集导致严重量化台阶
- 逆光：37 样本；
- 雨天：26 样本；
- 因为一个错误就能造成很大的 ACC 跳变，多个模型得到完全相同结果并不自动意味着消融实现失效。
因此不能仅凭表格中“完全一样”断定某个约束没有作用。

### v12 雾天过强正则
v12 的雾天 Full 明显下降，说明“更强增强/正则”不能无差别地应用到所有天气。
当前策略是：黑天/雾天/雨天保留 v12，只有逆光继续加强到 v13。

### RTX 5090 检查命令原稿中的 API 笔误
原命令曾写：
`torch.cuda.get.get_device_name(0)`
正确写法：
`torch.cuda.get_device_name(0)`。

## 8. 关键代码修改

### v12 / v13 训练策略
新增并实际验证了 `checkpoint_selection_min_epoch`：
- 门槛之前仍训练/验证；
- 不允许更新“最佳模型/最佳指标”；
- 不累计早停耐心；
- 只有达到门槛后才参与正式选点。

### 消融汇总器
`scripts/ablate_module_c.py` 已增强：
- 验证正确数；
- 总样本数；
- 错误数；
- ACC Wilson 95% 区间；
- CSV/JSON/HTML 汇总。
v9 雨天真实摘要检查曾得到：
- 26/26；
- 0 错误；
- Wilson 95% CI ≈ `[87.13%, 100%]`。

### 漂移约束修改文件
主要修改：
- `dual_d/losses.py`
- `dual_d/collaborative_training.py`
- `dual_d/training/trainer.py`
- `dual_d/config.py`
- `dual_d/integration_adapter.py`
- `scripts/train_dual_d.py`
- `configs/dual_d_default_config.json`
- `configs/train_dual_d_default.json`
- `tests/test_training_safety.py`
- `tests/test_weather_profiles.py`

典型函数/字段：
- `modality_relation_alignment_score`
- `modality_relation_drift_loss`
- `weighted_modality_drift`
- `modality_drift`
- `modality_dims`
- `drift_is_trainable`
- `modality_drift_warmup_epochs`
- `modality_drift_ramp_epochs`
- `modality_drift_margin`

## 9. 用户要求与约束

- 用户希望在更换 Codex 账号后继续原项目，不希望新 AI 从零理解。
- 不要重复询问交接文档中已经明确的信息。
- 优先解决当前未完成问题，不要只复述历史。
- 处理实验时不要删除/覆盖真实实验结果。
- Full 与消融比较必须公平。
- 可以接受消融实验最终性能稍微低一点，只要结果更稳定、机制更有解释性。
- 预期曲线可以用模拟日志辅助可视化，但必须明确标识为模拟，不能冒充真实实验。
- 用户希望使用 `nohup` 在 32GB RTX 5090 服务器运行。
- 不需要强行把 32GB 显存全部占满；当前 batch size 是按小数据集泛化考虑的。
- 用户喜欢直接给出可以复制的命令，并解释当前应该先跑哪一阶段。

## 10. 关键决策

1. 当前新主线不再围绕旧 v9/v11 做无休止微调，而是已经进入“新增模态关系漂移约束后的正式 Full 验证”阶段。
2. 新约束先采用保守权重 `0.02` 和 `0.01` 容忍范围。
3. 新约束前 12 epoch 只记录，12 epoch ramp，24 epoch 后满权重。
4. 第一轮正式验证应该先做 **四天气 Full × 3 seeds = 12 次**，观察：
   - Full 性能；
   - 漂移曲线；
   - 生成前后模态关系变化；
   - 是否出现异常训练行为。
5. Full gate 通过后，再考虑是否重跑完整四天气消融。
6. 不要同时启动 Full 和完整消融两个阶段。
7. 由于 v13 逆光更强 profile 已配置好，服务器最新推荐的四天气入口是：
   `configs/module_c_weather_profiles_v13.json`
8. 新的漂移约束使用默认配置：
   `configs/train_dual_d_default.json`
   `configs/dual_d_default_config.json`

## 11. 当前未解决问题

### 最重要
**尚无采用“模态关系漂移约束”的正式训练结果。**

因此目前还不能证明：
- 漂移约束是否提高 Full 性能；
- 是否降低模态关系漂移；
- 是否会损伤领域迁移；
- 是否最终让 Full 的消融排序更有说服力。

### 消融结果
v13 原始矩阵未完成完整 84 次，因此：
- 不能拿当时 59 次结果作为最终消融矩阵；
- 不能把临时两次雨天“去原型对比”均值当成三次正式均值。

### 评估协议
雨天和逆光验证集过小：
- 当前结果即使出现 100% 也不能认为统计证据非常强；
- 理想情况下应增加独立测试集或重复分层划分；
- 目前没有凭空制造独立 test 数据。

### 论文机制证据
需要真实训练后检查：
- drift 前/后曲线；
- VIS-IR、VIS-AIS、IR-AIS 三组关系；
- source→target 与 target→source 是否对称；
- drift 与验证性能是否相关；
- 漂移约束是否真的限制了“关系破坏”而非简单降低全部变化。

## 12. 下一步行动

### 首先：在 32GB RTX 5090 上跑四天气 Full 12 次

服务器项目根目录：
```bash
cd /home/lixiang/lx/Dual_D
```

推荐启动命令：
```bash
nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/lixiang/envs/lx_pytorch/bin/python scripts/train_dual_d.py \
  --config configs/train_dual_d_default.json \
  --dual-config configs/dual_d_default_config.json \
  --weather-profile-config configs/module_c_weather_profiles_v13.json \
  --source-root /home/lixiang/lx/Data/晴天 \
  --target-parent-root /home/lixiang/lx/Data \
  --target-domains 黑天 逆光 雾天 雨天 \
  --iterations 3 \
  --epochs 60 \
  --device cuda \
  --no-multi-gpu \
  --no-use-ais \
  --group-iterations \
  --num-workers 16 \
  --output-dir runs/modality_drift_v1_full_5090 \
  --run-name modality_drift_full \
  > modality_drift_v1_full_5090.log 2>&1 &
echo $!
```

查看日志：
```bash
tail -f modality_drift_v1_full_5090.log
```

查看 GPU：
```bash
watch -n 1 nvidia-smi
```

启动前检查 5090/PyTorch：
```bash
/home/lixiang/envs/lx_pytorch/bin/python -c \
"import torch; print('torch=',torch.__version__); print('cuda=',torch.version.cuda); print('gpu=',torch.cuda.get_device_name(0)); print('arch=',torch.cuda.get_arch_list())"
```

确认架构列表支持 RTX 5090 的 `sm_12x` 后再正式训练。

### 训练结束后的第一批分析
新 AI 接手后应直接检查：
1. `runs/modality_drift_v1_full_5090` 是否 12/12；
2. 每天气/每 seed 的选中 epoch；
3. ACC/Precision/Recall/F1；
4. `train_dual_d_modality_drift`；
5. `train_dual_d_weighted_modality_drift`；
6. source→target / target→source 的 before/after alignment；
7. 是否出现 NaN、CUDA 错误、提前中断；
8. drift 是否与性能变化方向一致。

然后再决定是否：
- 调整漂移权重；
- 只跑部分天气 pilot；
- 或冻结新 profile 并做完整消融。

## 13. 禁止重复尝试

以下做法已经明确不应重复：

- 不要重新做已经完成的 v9 63-run 审计，除非需要核查具体文件。
- 不要重新证明“模态维度相同 ≠ 模态对齐”。
- 不要把 OSAN 论文和当前代码说成完全等价。
- 不要覆盖 `runs` 中真实历史日志。
- 不要把 `result/` 中模拟日志当成真实实验。
- 不要为 Full 单独选择更优参数、同时给消融不同参数。
- 不要在没有 Full gate 的情况下直接开启 84 次新消融。
- 不要因为逆光/雨天结果相同就直接判断消融实现失败；先考虑验证集离散性和约束互补性。
- 不要用不同 epoch 的最高 ACC/Precision/Recall/F1 拼成一组“模型结果”。
- 不要为了让结果“看起来像论文”而故意降低性能。
- 不要强行把 RTX 5090 显存占满；应优先保持实验条件稳定。
- 不要再次使用错误的 `torch.cuda.get.get_device_name(0)`。

## 14. 用户最后一个未完成请求

用户最后明确要求：

> “可以在32G 5090的服务器跑满并且给我使用nohup的启动指令。”

上一轮已经给出了推荐的 12 次 Full 启动方案，但随后对话因 Codex 使用额度/系统限制中断，因此**最重要的未完成工作是：在新对话中继续这个服务器实验任务，并在训练完成后真实分析新增模态漂移约束是否有效。**

这里的“跑满”应理解为利用 RTX 5090 进行正式训练，而不是机械地把显存占满；当前 batch size 是按数据量和泛化设计的，不应为了占显存而随意改 batch。

## 15. 新 AI 接手必须知道的事项

1. 这是一个已经开发一段时间的项目，不是从零开始。
2. 最新关键代码修改是“模态关系漂移约束”，代码已经写完并通过 54 个测试。
3. 现在最重要的不是继续讨论理论，而是先跑真实的 Full 12-run gate 并分析漂移指标。
4. v13 是当前天气配置入口；黑天/雾天/雨天沿用 v12，逆光使用更强的 v13 profile。
5. 新的漂移约束默认参数是：
   - weight `0.02`
   - margin/tolerance `0.01`
   - warmup `12`
   - ramp `12`
6. 服务器命令已经准备好，路径与参数不要凭空改写。
7. 启动之前必须验证 PyTorch 是否支持 RTX 5090 的 `sm_12x`。
8. Full 12-run 完成后优先做证据化分析，不要立即重跑完整 84-run。
9. 消融必须在冻结同天气 profile 后进行，并保持 Full 与所有消融完全同配置，仅关闭对应约束。
10. `result/` 中的模拟日志只用于预期曲线可视化，绝不能当真实结果。
11. v13 旧矩阵曾在第 59 次左右中断，不要把 59 次临时汇总当作完整正式消融结果。
12. 评估正式结果时使用稳定窗口选中的同一 epoch 指标，而不是全训练逐轮最高值。
13. 如果“交接文档”和新账号当前实际代码状态发生冲突，以当前代码为准，并指出冲突。
14. 当前用户希望新 AI **直接继续执行任务**，不要再次复述所有背景或重复询问已经写明的路径、配置和命令。

# 新对话启动提示词

你现在接手的是一个已经进行过多轮开发、训练和实验分析的 `Dual_D` 项目，不是从零开始。

当前 Markdown 是上一轮 Codex 的任务交接记录。请先理解其中的最终状态，再直接继续任务，不要只做总结或复述。

请严格以交接文档中的“最终有效状态”为起点：
- 不重复已经失败、废弃或被后续方案替代的方法；
- 不重新询问交接文档中已经明确的路径、配置、环境和实验规则；
- 优先处理“当前未解决问题”和“下一步行动”；
- 当前最优先任务是使用 32GB RTX 5090 服务器执行新增“模态关系漂移约束”后的四天气 Full × 3 seeds 验证，并在训练完成后进行证据化分析；
- 使用交接文档给出的服务器路径、v13 配置和 `nohup` 方案，启动前先验证当前 PyTorch 是否支持 RTX 5090 的 `sm_12x`；
- 不要擅自启动完整 84 次消融矩阵，先完成 12 次 Full gate；
- 分析时重点查看 ACC/F1、选中 epoch、source→target 与 target→source 的模态关系漂移、weighted drift、before/after alignment 以及是否存在异常中断；
- 消融实验必须保持 Full 与消融完全公平，同一天气共用冻结 profile，只关闭对应 Module C 约束；
- 不要把 `result/` 中的模拟日志当作真实实验结果；
- 不要覆盖或删除历史真实实验结果；
- 论文/组会表述不能把“同维度”说成“模态对齐”，也不要声称当前张量模块与参考 OSAN 论文完全等价；
- 正式实验结果使用稳定窗口选出的同一个 checkpoint 的全部指标，不要拼接不同 epoch 的最高值；
- 如果交接文档与当前实际代码状态冲突，以当前代码为准，并明确指出冲突；
- 不要停留在“下一步建议”，而要直接检查当前工作区并继续执行实际任务。

接手后第一步：检查当前 Git 状态、v13 配置、漂移约束代码是否仍存在、服务器启动环境是否正确；确认后直接进入 Full 12-run 验证或对已存在运行结果进行实际审计。
