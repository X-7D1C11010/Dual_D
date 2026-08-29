# Dual_D项目实验分析任务交接文档

## 1. 项目基本信息

-   项目名称：Dual_D
-   项目路径：`D:\Code\Dual_D`
-   研究/实验方向：
    -   多模态天气域适应与目标识别；
    -   使用可见光、红外以及项目整体设计中的 AIS 特征；
    -   使用 TAL 进行多模态特征对齐；
    -   使用双判别器和双向特征生成进行跨天气域迁移；
    -   研究 Module C 中语义保持及类别约束对模型性能的作用。
-   当前负责的主要任务：
    -   分析最新训练日志和运行产物；
    -   检查训练是否完整结束；
    -   分析日志末尾报错；
    -   汇总 Full 和 Module C 消融实验的平均
        ACC、Precision、Recall、F1；
    -   检查天气专用参数是否真正生效；
    -   为黑天、逆光、雨天分别建立独立参数配置；
    -   优化三种天气的欠拟合问题；
    -   设计能够说明 Module C 各约束作用的特征证据可视化；
    -   冻结并复用已经完成且结果较好的雾天 v5 实验。

------------------------------------------------------------------------

## 2. 当前实验背景

当前正在进行 Dual_D 模型的 Module C 消融实验。

Module C 最终定义为以下五类约束：

1.  `Cycle`：双向循环一致性约束；
2.  `Identity`：身份特征保持约束；
3.  `Paired Contrastive`：类别平衡的跨域配对/多正样本监督对比约束；
4.  `Prototype Contrastive`：基于 EMA 类别原型的对比约束；
5.  `Classification Feedback`：生成特征分类反馈约束。

当前消融实验的目标是验证：

-   完整 Module C 是否能够提高跨天气域分类性能；
-   每个约束是否确实改善了它对应的特征机制；
-   去掉某个约束后，ACC、Precision、Recall、F1 是否下降；
-   Module C
    的优势是否能够分别在黑天、逆光、雾天、雨天上体现，而不只是四种天气的总体平均结果更高；
-   约束对特征空间的影响是否与最终分类性能一致。

双判别器的 adversarial 项不属于 Module C 的五项约束，因此不作为 Module C
的单项消融对象。

`no_module_c` 当前实际表示：

    Cycle、Identity、Paired Contrastive、
    Prototype Contrastive、Classification Feedback
    五项约束权重全部设置为 0

但双判别器及对抗域迁移核心仍然保留。因此，严格来说，这个实验应描述为：

    All Module-C constraints removed

而不是删除整个 Dual_D 域迁移模块。

当前消融采用"将对应损失权重设置为
0"的方式，而不是物理删除生成器或网络结构。这种方式适合研究单项损失贡献，因为：

-   Full 与消融模型的主体结构保持一致；
-   参数量和前向结构基本不变；
-   可以把差异主要归因于对应损失是否参与优化；
-   同一天气下 Full 和所有消融必须使用相同训练参数、相同迭代数和对应
    seed。

需要注意：权重设置为 0
后，部分中间特征可能仍会进行前向计算，但该项不会对总损失和梯度优化产生贡献。

------------------------------------------------------------------------

## 3. 当前重点实验信息

### 3.1 实验名称

Dual_D Module C leave-one-constraint-out ablation experiment。

### 3.2 消融版本

当前要求只运行以下 7 个版本：

    full
    no_cycle
    no_identity
    no_paired_contrastive
    no_prototype_contrastive
    no_classification_feedback
    no_module_c

不需要运行其它额外版本，也不需要把 Full 单独重复训练一次。

### 3.3 使用的数据集

本地数据集：

    D:\Code\JMDA-Net\Data

用户说明该数据集与服务器数据集完全一致。

源域为晴天，目标天气域包括：

    黑天
    逆光
    雾天
    雨天

服务器曾使用的路径形式为：

    /home/lixiang/lx/Data/晴天
    /home/lixiang/lx/Data/黑天
    /home/lixiang/lx/Data/逆光
    /home/lixiang/lx/Data/雾天
    /home/lixiang/lx/Data/雨天

当前消融启动脚本显式使用过：

    --no-use-ais

因此最新 Module C 消融实验主要是 VIS/IR 路径，不应在分析中误称为使用了
AIS 三模态输入。项目整体仍保留 AIS 支持。

### 3.4 实验训练目标

每个实验：

-   训练 60 轮；
-   每个天气独立重复 3 次；
-   相同变体、相同天气的 3 次迭代存放在同一个目录；
-   使用
    `metrics_iter01.csv`、`metrics_iter02.csv`、`metrics_iter03.csv`
    区分迭代；
-   报告 ACC、macro Precision、macro Recall、macro F1；
-   对外汇总时只展示三次平均值，不逐次展示；
-   使用英文绘图文字；
-   不强制指定字体。

用户提出过的性能目标为：

    黑天 ACC >= 97.6%
    逆光 ACC >= 97.3%
    雾天 ACC >= 95.4%
    雨天 ACC >= 96.2%

雨天验证集非常小，已知验证样本数约为 26，一个样本约对应
`3.85% ACC`。用户允许三次 Full 中出现类似：

    96.15%
    96.15%
    100%

但不希望三次都因为单轮峰值选择而得到 100%。

逆光验证集约为 37 个样本，一个样本约对应 `2.70% ACC`，因此 99% 与 100%
之间可能只是一个样本的差异。

### 3.5 v5 实验

v5 实验目录：

    D:\Code\Dual_D\runs\module_c_ablation_60_v5\
        module_c_20260827_121947

整体日志：

    module_c_ablation_60_v5.log

v5 共完成：

    7 个版本 × 4 种天气 × 3 次迭代 = 84 次训练

对话中已经确认：

-   84 次运行完整；
-   没有 OOM；
-   没有未完成的训练运行；
-   v5 汇总按照监控指标选定的最佳轮次统计。

v5 Full 平均结果：

  天气ACCPrecisionRecallF1                                 
  -------------------------- --------- --------- --------- ---------
  黑天                       95.56%    95.54%    97.07%    96.00%
  逆光                       99.10%    99.26%    99.54%    99.34%
  雾天                       98.74%    99.03%    98.17%    98.52%
  雨天                       100.00%   100.00%   100.00%   100.00%

v5 各消融的 ACC/F1 平均值：

  ------------------------------------------------------------------------
  版本黑天 ACC/F1逆光                                          
  ACC/F1雾天 ACC/F1雨天 ACC/F1                                 
  ---------------------------- --------- ----------- --------- -----------
  full                         95.56 /   99.10 /     98.74 /   100.00 /
                               96.00     99.34       98.52     100.00

  no_cycle                     97.22 /   99.10 /     97.98 /   98.72 /
                               96.31     99.34       98.10     98.88

  no_identity                  96.39 /   99.10 /     97.98 /   100.00 /
                               96.10     99.34       97.85     100.00

  no_paired_contrastive        96.67 /   100.00 /    97.98 /   98.72 /
                               96.47     100.00      98.04     98.88

  no_prototype_contrastive     96.67 /   99.10 /     97.73 /   100.00 /
                               96.27     99.34       97.77     100.00

  no_classification_feedback   94.72 /   99.10 /     96.21 /   98.72 /
                               93.71     99.34       96.40     98.88

  no_module_c                  93.61 /   98.20 /     97.47 /   100.00 /
                               94.06     98.81       96.90     100.00
  ------------------------------------------------------------------------

v5 的主要结论：

-   雾天 Full 高于所有单项消融和 `no_module_c`，结果较理想；
-   黑天 Full 低于多个单项消融，但高于 `no_classification_feedback` 和
    `no_module_c`；
-   黑天分类反馈有效，Full 相比去掉分类反馈的 F1 高约 2.29 个百分点；
-   黑天不是简单欠拟合：v5 最佳轮训练 ACC 已接近或达到
    100%，更像约束冲突、泛化波动和 seed 敏感；
-   逆光结果已接近验证集分辨率上限，多项实验相同并不能说明所有约束无效；
-   雨天三次 Full 的单轮最佳结果都是 100%，但达到峰值后验证 F1
    曾下降，存在单轮峰值选择偏差。

### 3.6 雾天冻结结果

雾天 v5 已经冻结，不计划在下一轮重新训练。

归档目录：

    D:\Code\Dual_D\fog_v5_reference_20260827

归档中包含：

-   7 个雾天实验目录；
-   每个版本 3 次迭代，共 21 次运行；
-   训练日志；
-   metrics；
-   result summary；
-   resolved config；
-   数据审计；
-   类别分布；
-   特征快照；
-   雾天特征证据图；
-   雾天汇总 CSV。

冻结参数文件：

    configs/module_c_fog_v5_frozen.json
    configs/dual_d_fog_v5_frozen.json

归档内还包含：

    ablation_summary_fog.csv
    ablation_runs_fog.csv
    constraint_feature_evidence_fog.csv
    README.md

### 3.7 v6 实验

最新实验目录：

    D:\Code\Dual_D\runs\module_c_ablation_60_v6\
        module_c_20260828_191630

最新整体日志：

    D:\Code\Dual_D\module_c_ablation_60_v6.log

v6 设计目标为：

    7 个版本 × 黑天、逆光、雨天 × 3 次 = 63 次新训练

随后在分析阶段合并冻结雾天的 21 次结果：

    63 次新训练 + 21 次冻结雾天 = 84 行最终报告

但是当前对话中尚未成功读取 v6 的实际目录和日志，因此以下内容仍未确认：

-   63 次新训练是否全部完成；
-   日志最后的具体报错文本；
-   报错发生在训练阶段还是最终汇总/可视化阶段；
-   最终 `ablation_runs.csv` 是否生成了 84 行；
-   三种天气实际生效的配置是否不同；
-   v6 的平均 ACC、Precision、Recall、F1；
-   用户判断的"黑天、逆光、雨天均欠拟合"是否与训练曲线、更新步数和验证曲线一致。

------------------------------------------------------------------------

## 4. 当前遇到的问题

### 4.1 v6 日志末尾报错

用户明确表示：

    module_c_ablation_60_v6.log 最后部分有报错

但当前对话中没有读取到具体 traceback，因此不能编造报错内容或原因。

需要新的 Codex 实际检查：

    D:\Code\Dual_D\module_c_ablation_60_v6.log

重点区分：

1.  某次训练本身失败；
2.  63 次训练已经完成，但最终分析失败；
3.  冻结雾天 reference 合并失败；
4.  t-SNE/scikit-learn 等可视化依赖导致分析失败；
5.  实验清单完整性检查失败；
6.  主目录与 reference 目录出现重复天气/变体/迭代键；
7.  特征文件或 result summary 缺失；
8.  配置参数解析失败。

### 4.2 天气专用配置争议

用户认为 v6 没有按天气使用不同参数，而是使用了共同参数，这是错误的。

之前代码设计中使用过一个公共文件：

    configs/module_c_weather_profiles.json

该文件内部原本计划包含黑天、逆光、雾天、雨天四个独立
profile，而不是所有天气共享同一套数值。

但是不能只根据配置文件内容判断是否生效。必须检查 v6
每个天气运行目录中的：

    resolved_config_iter01.json
    resolved_dual_config_iter01.json

需要对比至少以下字段：

    batch_size
    num_workers
    min_steps_per_epoch
    lr_main
    lr_visual
    lr_discriminator
    weight_decay
    augmentation_strength
    label_smoothing
    classifier_dropout
    target_classification_weight
    module_c_warmup_epochs
    module_c_ramp_epochs
    adversarial_warmup_epochs
    adversarial_ramp_epochs
    monitor_stability_window
    early_stopping_patience
    early_stopping_min_epochs
    dual_loss_weights

如果三个天气的 resolved config 完全相同，说明 profile 没有正确应用。

如果 resolved config 不同，只是来源于同一个 JSON
文件，则技术上属于不同天气参数，但用户仍希望改成三个明确的独立配置文件，避免歧义和覆盖。

建议后续创建：

    configs/module_c_weather_night.json
    configs/module_c_weather_backlight.json
    configs/module_c_weather_rain.json

雾天继续使用：

    configs/module_c_fog_v5_frozen.json

### 4.3 三种天气被用户判断为欠拟合

用户最新判断：

    黑天、逆光、雨天三种天气在 v6 中都是欠拟合

这一判断还需要结合实际曲线验证，重点检查：

-   训练 ACC 是否仍在持续上升；
-   训练 ACC 是否明显低于 100%；
-   验证 ACC/F1 是否与训练指标同步缓慢上升；
-   最佳轮是否出现在训练末段；
-   early stopping 是否过早；
-   学习率是否过低；
-   Module C warmup 是否过长；
-   每轮优化 step 数是否过少；
-   正则化是否过强。

一个重要风险是：

    batch_size = 92

虽然提高了显存利用率，但会显著减少每轮 optimizer step 数。如果每轮只有
2--4 步，则 60 轮可能只有约 120--240 次参数更新，容易出现"大 batch
但总更新不足"的欠拟合。

因此后续优化不能只提高学习率，还应优先检查并按天气调整：

    min_steps_per_epoch

必要时还应调整：

    dropout
    label smoothing
    augmentation strength
    weight decay
    Module C warmup/ramp
    early stopping min epochs

### 4.4 早期 dual_g 和 dual_d 为 0

之前日志曾出现：

    dual_g 0.0000
    dual_d 0.0000

发生在最初若干 epoch。

已经分析的原因是：

-   对抗损失有 warmup；
-   Module C 也有 warmup/ramp；
-   在 scale 为 0 的早期 epoch，相关加权损失和判别器更新可能为 0；
-   这不一定是异常。

后续日志已增加类似：

    adv/moduleC
    disc_steps

用于区分"正常 warmup 为 0"和"损失没有接入"。

### 4.5 Full 不一定高于每个单项消融

已确定：

-   去掉某个损失后结果偶尔提高，不一定说明代码错误；
-   多目标损失可能存在梯度竞争；
-   小验证集和 seed 波动会使单项消融偶尔超过 Full；
-   Full 的合理性不能只通过某一次 ACC 证明；
-   应同时看三次平均值、标准差、高维机制指标和分类指标。

不允许为了让 Full 人为胜出而给 Full 与同一天气的消融使用不同训练参数。

正确原则是：

    同一天气：Full 和所有消融使用同一套训练超参数；
    不同天气：允许使用不同超参数。

### 4.6 特征可视化问题

现有 `constraint_feature_evidence`
图确实读取了特征快照，并在原始高维特征空间计算：

-   Cycle cosine error；
-   Identity cosine error；
-   Correct-vs-nearest-wrong prototype margin；
-   Correct-class prototype cosine distance；
-   Generated-feature correct-class logit margin；
-   Generated-feature correct-class confidence。

这些属于"特征机制指标可视化"，不是二维样本分布图。

已有原始空间图比单纯 t-SNE 更适合证明具体约束，因为：

-   Cycle 的作用应看重构误差；
-   Identity 的作用应看身份保持误差；
-   对比约束的作用应看类别 margin 和原型距离；
-   分类反馈应看正确类别 logit margin 和置信度；
-   这些指标与各约束的数学目标一一对应。

### 4.7 t-SNE 的临时修改和待清理状态

在前一次被中断的工作中，曾临时向代码加入配对 t-SNE 可视化：

-   使用 Full 与对应单项消融的相同样本；
-   在同一个联合 t-SNE 空间中投影；
-   为雾天生成了 5 张 t-SNE 图；
-   小范围测试通过；
-   尚未完成全部测试和最终清理。

随后用户明确表示：

    如果 t-SNE 不能很好表现各约束性能，
    就不针对 Module C 使用 t-SNE 可视化。

应采用的结论是：

-   t-SNE 可以直观展示局部聚类；
-   但不能稳定、严格地证明单个 Cycle、Identity 等约束的贡献；
-   t-SNE 会扭曲全局距离；
-   结果受 perplexity、随机初始化和样本数影响；
-   不同 seed 的二维坐标不能直接比较；
-   因此不应把 t-SNE 作为 Module C 各约束的主要证据。

新的 Codex 进入工作区后需要检查当前未提交改动，确认是否仍包含：

    --tsne-feature-view
    _plot_constraint_feature_tsne
    _fit_tsne
    _load_tsne

以及目录：

    fog_v5_reference_20260827\
        constraint_feature_evidence\tsne

根据用户最后决定，应删除或禁用 Module C 的 t-SNE
路径，保留原始高维机制图和类别原型图。

### 4.8 PCA 状态

PCA 不是 Dual_D 模型、训练损失或推理的一部分。

代码中曾保留可选的 post-hoc PCA 二维绘图，但一键实验命令使用：

    --no-pca-feature-view

因此：

-   PCA 不影响训练；
-   PCA 不影响模型性能；
-   PCA 不影响原始空间机制指标；
-   v5 雾天归档中没有生成 `feature_diagnostics` PCA 散点目录。

------------------------------------------------------------------------

## 5. 当前涉及的重要文件

  ----------------------------------------------------------------------------------------------------------------------------------------
  文件/目录作用                                                                    
  -------------------------------------------------------------------------------- -------------------------------------------------------
  `D:\Code\Dual_D\Dual_D_Codex_session_migration_20260820.md`                      2026-08-20 旧 Codex 会话迁移摘要

  `D:\Code\Dual_D\Dual_D_code_modification_conversation_summary_20260820.md`       更早阶段的代码修改和对话摘要，若仍存在应读取

  `D:\Code\Dual_D\configs\train_dual_d_default.json`                               Standalone 训练默认参数

  `D:\Code\Dual_D\configs\dual_d_default_config.json`                              Dual_D 结构及默认损失权重

  `D:\Code\Dual_D\configs\module_c_weather_profiles.json`                          当前公共天气 profile
                                                                                   文件，内部原计划包含不同天气配置；需检查 v6
                                                                                   是否真正应用

  `D:\Code\Dual_D\configs\module_c_fog_v5_frozen.json`                             冻结的雾天 v5 训练参数

  `D:\Code\Dual_D\configs\dual_d_fog_v5_frozen.json`                               冻结的雾天 Dual_D 结构和损失参数

  `D:\Code\Dual_D\scripts\train_dual_d.py`                                         单次及多天气训练入口；负责应用天气 profile

  `D:\Code\Dual_D\scripts\ablate_module_c.py`                                      生成消融配置、运行 7 个版本、分析结果和绘图

  `D:\Code\Dual_D\scripts\run_all_experiments.py`                                  一条命令启动 Full 和全部 Module C 消融

  `D:\Code\Dual_D\dual_d\training\trainer.py`                                      核心训练循环、warmup、稳定
                                                                                   checkpoint、指标和特征快照保存

  `D:\Code\Dual_D\tests\test_module_c_ablation.py`                                 消融矩阵、汇总、特征图、reference 合并等测试

  `D:\Code\Dual_D\tests\test_training_safety.py`                                   warmup、梯度、消融权重保护、稳定窗口等测试

  `D:\Code\Dual_D\tests\test_weather_profiles.py`                                  天气 profile 验证和应用测试

  `D:\Code\Dual_D\tests\test_three_modal_pipeline.py`                              多模态训练和 grouped iterations 烟雾测试

  `D:\Code\Dual_D\runs\module_c_ablation_60_v5\module_c_20260827_121947`           已完整结束的 v5 四天气消融实验

  `D:\Code\Dual_D\module_c_ablation_60_v5.log`                                     v5 整体训练日志

  `D:\Code\Dual_D\fog_v5_reference_20260827`                                       冻结雾天 21 次运行、参数、特征和汇总

  `D:\Code\Dual_D\fog_v5_reference_20260827\ablation_summary_fog.csv`              雾天 7 个版本的三次平均结果

  `D:\Code\Dual_D\fog_v5_reference_20260827\ablation_runs_fog.csv`                 雾天 21 次单次运行结果

  `D:\Code\Dual_D\fog_v5_reference_20260827\constraint_feature_evidence_fog.csv`   雾天各约束的机制增益和 F1 增益

  `D:\Code\Dual_D\fog_v5_reference_20260827\constraint_feature_evidence`           雾天五类原始空间约束证据图

  `D:\Code\Dual_D\fog_v5_reference_20260827\constraint_feature_evidence\tsne`      临时生成的雾天 t-SNE 图；用户已倾向不用于 Module
                                                                                   C，应检查并清理

  `D:\Code\Dual_D\runs\module_c_ablation_60_v6\module_c_20260828_191630`           最新 v6 黑天、逆光、雨天训练信息

  `D:\Code\Dual_D\module_c_ablation_60_v6.log`                                     最新 v6 整体训练日志，尾部有尚未分析的报错

  `metrics_iter01.csv` 等                                                          每次迭代逐 epoch
                                                                                   训练、验证、损失、梯度、学习率和显存数据

  `result_summary_iter01.json` 等                                                  每次迭代最终训练摘要

  `best_metrics_iter01.json` 等                                                    稳定 checkpoint 选择轮的详细指标

  `resolved_config_iter01.json` 等                                                 每次训练实际生效的完整训练参数，是判断天气 profile
                                                                                   是否生效的关键

  `resolved_dual_config_iter01.json` 等                                            实际生效的 Dual_D 损失权重，检查消融权重是否保持为 0

  `feature_embeddings_iter01.npz` 等                                               原始、生成、重构、identity、logits 和标签等特征快照

  `experiment_manifest.json`                                                       声明变体、天气、迭代数和预期运行数

  `ablation_runs.csv`                                                              所有单次运行汇总；v6 加冻结雾天后预期 84 行

  `ablation_summary.csv`                                                           按变体和天气聚合的三次平均结果

  `constraint_feature_evidence.csv`                                                Full 与单项消融的配对机制证据

  `checkpoints`                                                                    消融启动脚本曾使用
                                                                                   `--no-save-checkpoints`，所以最新消融目录可能没有模型
                                                                                   checkpoint；不要默认其一定存在
  ----------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Codex下一步需要执行的任务

进入工作区后需要：

1.  阅读本交接文档，以及：

        Dual_D_Codex_session_migration_20260820.md

    不要重新实现已经存在的 Module C 消融框架。

2.  运行只读检查：

        git status --short
        git diff --check

    重点确认前一次中断后，t-SNE 代码和生成图片是否仍留在工作区。

3.  读取最新 v6 日志：

        D:\Code\Dual_D\module_c_ablation_60_v6.log

    查看最后至少 200 行，保存完整 traceback，并确定报错阶段。

4.  检查最新 v6 实验目录：

        D:\Code\Dual_D\runs\module_c_ablation_60_v6\
            module_c_20260828_191630

5.  读取：

        experiment_manifest.json

    确认预期运行数、天气、版本和迭代数。

6.  分别统计以下文件数量：

        metrics_iter*.csv
        result_summary_iter*.json
        best_metrics_iter*.json
        resolved_config_iter*.json
        resolved_dual_config_iter*.json
        feature_embeddings_iter*.npz

    判断 63 次新训练是否全部结束。

7.  判断日志末尾报错属于：

    -   训练失败；
    -   reference 雾天合并失败；
    -   结果汇总失败；
    -   特征可视化失败；
    -   t-SNE 或 scikit-learn 依赖失败；
    -   manifest 完整性检查失败；
    -   重复运行键错误；
    -   缺失文件错误。

8.  如果训练已经完成，直接从 metrics/result
    文件重新汇总，不因绘图失败而重新训练。

9.  按"实验版本 × 天气"汇总三次平均：

        ACC
        Macro Precision
        Macro Recall
        Macro F1

    只展示平均值，不展示三次单独结果。

10. 检查最终结果是否正确合并：
    `63 次 v6 新训练     + 21 次冻结雾天     = 84 次运行`

11. 检查黑天、逆光、雨天各自的：
    `resolved_config_iter01.json     resolved_dual_config_iter01.json`
    用表格逐字段比较实际参数。

12. 判断天气 profile 是否真正生效：

    -   如果三个天气实际参数相同，修复 profile 应用逻辑；
    -   如果实际参数不同但来源于同一文件，仍按用户要求拆分为三个独立文件。

13. 创建并使用：
    `configs/module_c_weather_night.json     configs/module_c_weather_backlight.json     configs/module_c_weather_rain.json`
    雾天继续使用冻结文件，不重新训练。

14. 根据每个天气的训练曲线分别判断欠拟合原因，不能直接共用一套参数。

15. 优先检查大 batch 导致的更新次数不足。针对每个天气分别调整：
    `min_steps_per_epoch     lr_main     lr_visual     lr_discriminator     dropout     label smoothing     augmentation     weight decay     Module C warmup/ramp     early stopping     monitor stability window`

16. 保证公平性：
    `同一天气的 Full 与所有消融使用完全相同训练参数；     不同天气允许使用不同训练参数。`

17. 检查消融配置中的 0 权重不会被天气 profile 重新启用。

18. 根据用户最后决定，不将 t-SNE 作为 Module C
    各约束的可视化方案。清理或默认禁用临时 t-SNE 代码和图片。

19. 保留以下更有解释力的可视化：

    -   Cycle：原始特征与循环重构特征的 cosine error；
    -   Identity：原始特征与 identity 输出的 cosine error；
    -   Paired Contrastive：正确类别与最近错误类别原型 margin；
    -   Prototype Contrastive：正确类别原型距离和 prototype margin；
    -   Classification Feedback：正确类别 logit margin 和置信度；
    -   类别原型相似度矩阵；
    -   Full 与对应单项消融的机制增益和 F1 增益散点图。

20. 修改后运行完整测试：

            D:\Anaconda\envs\pytorch\python.exe -m unittest discover -s tests -v
            ```
        并运行：

        git diff --check
        ```

------------------------------------------------------------------------

## 7. 当前分析思路和注意事项

### 已确定的信息

-   Module C 已经实现，不应从头重写；
-   最终消融矩阵是 7 个版本；
-   双判别器 adversarial 项不属于 Module C 五项约束；
-   同一天气的 Full 和消融必须使用同一训练配置；
-   不同天气必须允许独立参数；
-   雾天 v5 已经冻结，不应重新训练；
-   v5 共 84 次训练且完整结束；
-   v6 预期只训练黑天、逆光、雨天的 63 次，再合并雾天 21 次；
-   v6 的日志尾部存在报错，但具体内容尚未读取；
-   v6 是否训练完整尚未确认；
-   v6 三种天气平均结果尚未汇总；
-   用户判断 v6 的黑天、逆光、雨天均欠拟合；
-   需要用 resolved config 判断天气专用参数是否真正生效；
-   不能仅根据启动命令或配置文件名称判断实际参数；
-   PCA 不属于模型，也不影响训练；
-   Module C 不再使用 t-SNE 作为主要特征可视化方案。

### 不应该重复分析或重新实现的问题

-   不要重新设计 Module C 的五项约束；
-   不要重新增加 adversarial 单项消融；
-   不要重复运行雾天；
-   不要把 Full 再单独训练一次；
-   不要把每次迭代存放到独立目录；
-   不要只记录 ACC；
-   不要把每次迭代逐项展示给用户；
-   不要通过给 Full 单独设置更优参数来制造消融优势；
-   不要把 t-SNE 聚类图当作各约束有效性的主要证据；
-   不要根据旧 v5 结果推测 v6 结果；
-   不要在没有读取日志的情况下猜测 v6 报错内容。

### 后续重点分析方向

1.  v6 是否完成 63 次新训练；
2.  报错是否只发生在最终分析或绘图；
3.  三种天气是否真的使用了相同有效参数；
4.  每种天气实际 optimizer step 总数；
5.  大 batch 是否导致总更新次数不足；
6.  最佳轮是否位于最后几轮；
7.  train/val 指标是否共同偏低；
8.  Module C warmup 是否启用过晚；
9.  dropout、label smoothing、augmentation 是否同时过强；
10. 稳定 checkpoint 窗口是否在欠拟合情况下选中了过早模型；
11. Full 与消融的高维机制指标是否和分类结果一致；
12. 最终汇总是否正确包含 ACC、Precision、Recall、F1；
13. 冻结雾天是否只读合并，没有被重新训练或覆盖。

### 上一轮曾设计但需要重新验证的参数起点

之前的公共天气 profile 曾计划使用以下不同参数。这些只是 v6
前的设计起点，不代表 v6 实际生效，也不是下一轮最终配置。

黑天曾计划：

    batch_size = 92
    num_workers = 16
    min_steps_per_epoch = 4
    lr_main = 7e-5
    lr_visual = 7e-6
    lr_discriminator = 3.5e-5
    augmentation_strength = 0.65
    label_smoothing = 0.12
    classifier_dropout = 0.48
    target_classification_weight = 0.65
    module_c_warmup/ramp = 5/10
    early_stopping_min_epochs = 48

逆光曾计划：

    batch_size = 92
    num_workers = 16
    min_steps_per_epoch = 2
    lr_main = 6e-5
    lr_visual = 6e-6
    lr_discriminator = 3e-5
    augmentation_strength = 0.70
    label_smoothing = 0.14
    classifier_dropout = 0.50
    module_c_warmup/ramp = 8/12
    monitor_stability_window = 3

雨天曾计划：

    batch_size = 92
    num_workers = 16
    min_steps_per_epoch = 2
    lr_main = 9e-5
    lr_visual = 9e-6
    lr_discriminator = 3.5e-5
    augmentation_strength = 0.65
    label_smoothing = 0.12
    classifier_dropout = 0.50
    module_c_warmup/ramp = 6/12
    monitor_stability_window = 5

v5 在服务器 `batch=92/workers=16` 时曾记录约：

    21.78 GiB allocated
    28.19 GiB reserved

因此该设置已经接近 32GB 显卡的合理安全上限。继续扩大 batch
不一定加快固定更新数训练，反而可能进一步减少每轮更新步数。

------------------------------------------------------------------------

## 8. 新对话启动提示词

你现在接管 `D:\Code\Dual_D` 中的 Dual_D 项目。

请先阅读：

    D:\Code\Dual_D\docs\Dual_D_Codex_task_context.md
    D:\Code\Dual_D\Dual_D_Codex_session_migration_20260820.md

不要从头重新实现已经完成的 Module C 消融框架。

当前最重要的任务是分析最新 v6 实验：

    D:\Code\Dual_D\runs\module_c_ablation_60_v6\
        module_c_20260828_191630

    D:\Code\Dual_D\module_c_ablation_60_v6.log

请按以下顺序处理：

1.  读取日志最后至少 200 行，给出完整报错位置和原因；
2.  检查 `experiment_manifest.json`；
3.  统计
    `metrics_iter*.csv`、`result_summary_iter*.json`、`best_metrics_iter*.json`、`resolved_config_iter*.json`、`resolved_dual_config_iter*.json`
    和 `feature_embeddings_iter*.npz` 的数量；
4.  判断 63 次黑天、逆光、雨天新训练是否全部完成；
5.  判断报错发生在训练阶段，还是训练完成后的雾天 reference
    合并、汇总或可视化阶段；
6.  如果训练已经完成，不要重新训练，直接重新汇总；
7.  按实验版本和天气给出三次迭代的平均 ACC、Macro Precision、Macro
    Recall、Macro F1，不展示单次迭代；
8.  检查黑天、逆光、雨天各自的 `resolved_config_iter01.json` 和
    `resolved_dual_config_iter01.json`，确认三种天气实际使用的参数是否不同；
9.  如果天气 profile 未生效，修复应用逻辑；
10. 无论公共 profile 是否生效，都按用户要求建立三个明确独立的配置文件：

```{=html}
<!-- -->
```
    configs/module_c_weather_night.json
    configs/module_c_weather_backlight.json
    configs/module_c_weather_rain.json

11. 根据每种天气的训练曲线分别分析欠拟合原因，重点检查
    `min_steps_per_epoch`、总 optimizer step、学习率、正则化和 Module C
    warmup；
12. 同一天气的 Full 和所有消融必须使用完全相同的超参数，不允许单独优化
    Full；
13. 雾天使用 `fog_v5_reference_20260827` 中的冻结结果，不重新训练；
14. 检查工作区是否残留前一轮临时加入的 t-SNE 代码和
    `constraint_feature_evidence\tsne` 图片。用户最后决定不针对 Module C
    使用 t-SNE，因此应清理或默认禁用；
15. 保留原始高维空间的约束机制图、类别原型图以及 ACC/Precision/Recall/F1
    汇总；
16. 修改后运行完整单元测试和 `git diff --check`。

分析和修改必须基于实际日志及运行文件，不能根据 v5 或旧对话推测 v6
的结果。
