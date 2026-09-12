# Satellite-Dual_D / So2Sat-LCZ42 代码续接交接文档

> 用途：将本文件直接交给 Codex，作为后续代码修改、数据适配与实验实现的上下文。
>
> 当前目标不是重新设计 Dual_D 的核心方法，而是**将原“海上跨天气多模态领域适应”背景迁移到“卫星遥感时空变化下的 Optical-SAR 多模态跨域图像分类”背景**，并优先接入 **So2Sat LCZ42 v4** 数据集。
>
> 本文档按“已经确认的事实 / 已经确定的设计 / 尚未完成的代码工作”整理。

---

# 1. 当前任务一句话定义

将现有 **Dual_D** 框架从海上 VIS/IR 跨天气目标分类，迁移到：

> **面向时空变化导致领域偏移的 Optical-SAR 多模态卫星遥感跨域图像级分类任务。**

第一阶段数据集采用 **So2Sat LCZ42 v4**：

- Sentinel-1 SAR
- Sentinel-2 Optical
- 17 类 LCZ 场景分类标签
- Source / Target 具有相同类别集合
- 每个 domain 内 SAR 与 Optical 是同一个地理 patch 的多模态对应数据
- Source 与 Target 之间**不需要逐样本一一对应**

当前实验主轴首先验证 **Spatial Domain Shift（空间域偏移）**。

---

# 2. 原始 Dual_D 项目背景与当前方法状态

## 2.1 原任务背景

原始任务为海上目标跨域识别：

- Source：晴天
- Target：黑天 / 逆光 / 雾天 / 雨天
- 多模态：VIS / IR / 可选 AIS
- 任务：图像级分类，不是 YOLO 式目标检测

研究背景核心不是“天气”本身，而是：

> **时空环境变化引起领域分布变化。**

海上场景中最显著的表现是天气变化。

迁移到卫星后，保留这一背景逻辑：

> **空间区域变化、观测时间变化、季节变化、地表状态变化等时空因素引起领域偏移。**

So2Sat 第一阶段主要验证空间变化；后续可再用 SEN12MS 等数据集验证 temporal / spatio-temporal variation。

---

## 2.2 Dual_D 当前核心方法

当前项目不能被简化成普通“双分支分类器”。

核心方法链路为：

1. 多模态特征提取
2. 基于张量的多模态关系 / 对齐模块
3. 多模态特征融合
4. 双向领域特征翻译
5. 双判别器领域匹配
6. Module C 类别感知反馈
7. Modality Relation Drift 约束

当前代码已经经历过多轮海上数据实验与消融，后续迁移卫星背景时，**不能随意替换核心方法**。

---

# 3. 必须保留的 Dual_D 核心模块

## 3.1 Tensor-based Alignment / Multimodal Relation Modeling

这是当前方法的重要核心。

卫星版本仍然必须采用 Dual_D 中已有的：

> **基于张量的多模态关系建模 / 对齐模块**

而不是采用 So2Sat 官方的 decision-level fusion。

卫星版本应把：

- SAR encoder feature
- Optical encoder feature

送入原有 Tensor Alignment / Tensor Relation 模块。

不要改成：

- 直接拼接后分类
- softmax 平均
- attention fusion 替代原 Tensor
- 其他新的多模态融合网络

除非后续明确作为 ablation。

---

## 3.2 Bidirectional Domain Translation

原 Dual_D 中继续保留：

- Source → Target-like Generator
- Target → Source-like Generator
- Source-like discriminator
- Target-like discriminator

卫星背景中的解释：

> 学习不同空间区域 / 时空条件之间的跨域特征转换关系。

Source / Target 不是同一实例，不要求一一对应。

---

## 3.3 Module C

当前 Module C 包含的类别感知约束仍保留：

- Cycle
- Identity
- Paired Contrastive
- Prototype Contrastive
- Classification Feedback

后期版本中 Prototype 已采用 **EMA 类别原型** 的思路。

注意：

- 当前 So2Sat 17 类在 Source / Target 都完整存在
- 因此 closed-set category-aware 训练是合理的
- 若训练使用 Target 标签，可保留 target-side category-aware losses
- 若后续做 UDA，则需要针对 target label 依赖单独改造，不是第一阶段优先事项

---

## 3.4 Modality Relation Drift

这是当前 Dual_D 最新增加的方法。

核心思想：

> Tensor 模块先建立多模态关系，后续领域生成器在进行 Source↔Target 特征转换时，不能严重破坏前面建立的多模态结构。

卫星背景中的解释更加自然：

- Optical 描述光谱 / 纹理 / 地表结构信息
- SAR 描述散射 / 极化 / 几何结构信息
- Tensor 模块建立 Optical-SAR 协同关系
- Generator 进行跨域转换
- Relation Drift 约束迁移过程中不要破坏 Optical-SAR 关系

当前 relation drift 的历史默认配置（来自原 Dual_D 最新阶段）：

- loss weight: 0.02
- tolerance / margin: 0.01
- warmup: 12 epochs
- ramp: 12 epochs
- epoch 24 后完整权重

生成前关系 reference 使用 detach / stop-gradient：

> drift loss 应主要约束 Generator，而不是反向推动 Tensor 模块改变 reference 以“配合” Generator。

迁移到 So2Sat 后第一阶段先保持原逻辑，不主动重写数学实现。

---

# 4. 新任务类型：图像级分类

用户已经明确：

> 继续做 classification，但指的是**图像级 / 场景级分类**，不是 YOLO 式目标检测中的 classification。

每个样本对应一个卫星 patch：

```text
SAR patch + Optical patch → 一个 LCZ 类别
```

例如：

- Compact high-rise
- Open low-rise
- Heavy industry
- Dense trees
- Water
- etc.

不要把任务改成：

- object detection
- semantic segmentation
- bounding box classification

---

# 5. 新背景定义

建议正式表述为：

> 面向时空变化的 Optical-SAR 多模态卫星遥感跨域图像分类。

更加完整的背景逻辑：

```text
Spatio-temporal variation
        ↓
Domain distribution shift
        ↓
Optical + SAR multimodal observation
        ↓
Multimodal relation modeling
        ↓
Bidirectional domain adaptation
        ↓
Cross-domain scene classification
```

So2Sat LCZ42 第一阶段主要验证：

> **Spatial Variation / Cross-region Domain Shift**

后续如果加入 SEN12MS，可验证：

> Temporal / Seasonal / Spatio-temporal Domain Shift

因此当前不要声称 So2Sat 单独验证了 temporal shift。

---

# 6. 多模态配对与 Source/Target 对应关系

这是已经明确确认的设计原则。

## 6.1 Domain 内多模态要对应

理想并且当前 So2Sat 已满足：

```text
SAR_source_i ↔ Optical_source_i
SAR_target_j ↔ Optical_target_j
```

即同一地理 patch 的不同成像模态。

这样 Tensor Relation 和 Relation Drift 才具有清晰的多模态物理含义。

---

## 6.2 Source 与 Target 不要求实例对应

不要求：

```text
Source sample i ↔ Target sample i
```

是同一个地理位置、同一地物实例。

只需要：

- Source 与 Target 属于相同类别空间
- Domain 条件不同
- 可以进行分布级 / 类别级跨域学习

---

## 6.3 类别集合必须完全一致

已经明确要求：

\[
Y_S = Y_T
\]

当前任务采用 closed-set cross-domain classification。

不能出现：

- Source 有某个类别而 Target 没有
- Target 有某类别而 Source 没有

So2Sat 当前官方 split 已经实际验证满足完整 17 类 closed-set。

---

# 7. Target 标签使用策略

当前决定：

> Target 标签“可以使用，也可以不使用”，但数据本身必须有真实 Target 标签。

第一阶段建议优先兼容现有 Dual_D，先做：

## 7.1 Supervised Domain Adaptation

训练时使用：

```text
Source: x_s + y_s
Target-adapt: x_t + y_t
```

这样可以最大程度保留现有：

- target classification feedback
- category-aware sampler
- prototype
- paired contrastive

后续再扩展：

## 7.2 Unsupervised Domain Adaptation

训练时：

```text
Source: x_s + y_s
Target-adapt: x_t
```

Target label 仅用于 evaluation。

第一阶段**不要同时大改 UDA 逻辑**，避免一次引入太多变量。

---

# 8. 已选数据集：So2Sat LCZ42 v4

数据已下载到 Linux 服务器：

```text
/home/lixiang/lx/Data/So2Sat-LCZ42/v4
```

解压后目录约：

```text
56G
```

主要文件：

```text
training.h5
training_geo.h5

validation.h5
validation_geo.h5

testing.h5
testing_geo.h5
```

---

# 9. 已验证的 HDF5 数据结构

本地实际读取：

```text
training.h5

keys:
['label', 'sen1', 'sen2']
```

shape：

```text
label (352366, 17) float64
sen1  (352366, 32, 32, 8) float64
sen2  (352366, 32, 32, 10) float64
```

`training_geo.h5`：

```text
keys:
['city', 'epsg', 'tfw']

city (352366,) |S12
epsg (352366, 1) int32
tfw  (352366, 6) float64
```

说明同一个 index `i` 可解释为：

```text
sen1[i]  → Sentinel-1 SAR
sen2[i]  → Sentinel-2 Optical
label[i] → 17 类 one-hot label
city[i]  → 所属城市
```

即：

```text
SAR_i ↔ Optical_i ↔ label_i ↔ city_i
```

是一一对应的。

---

# 10. 官方 split 与新的 Source / Target 协议

## 10.1 Source Domain

采用：

```text
training.h5
```

已验证：

```text
352,366 samples
42 cities
17 classes
```

42 个城市包括：

```text
amsterdam
beijing
berlin
bogota
buenosaires
cairo
capetown
caracas
changsha
chicago
cologne
dhaka
dongying
hongkong
islamabad
istanbul
kyoto
lima
lisbon
london
losangeles
madrid
melbourne
milan
nanjing
newyork
orangitown
paris
philadelphia
qingdao
quezon
riodejaneiro
rome
salvador
saopaulo
shanghai
shenzhen
tokyo
vancouver
washingtondc
wuhan
zurich
```

注意：部分城市在 training 中样本非常少，但 Source 总体完整覆盖 17 类。

---

## 10.2 Target Adaptation Domain

采用：

```text
validation.h5
```

已验证：

```text
24,119 samples
10 cities
17 classes
```

10 个城市：

```text
guangzhou
jakarta
moscow
mumbai
munich
nairobi
sanfrancisco
santiago
sydney
tehran
```

---

## 10.3 Target Test Domain

采用：

```text
testing.h5
```

已验证：

```text
24,188 samples
10 cities
17 classes
```

城市与 validation 相同：

```text
guangzhou
jakarta
moscow
mumbai
munich
nairobi
sanfrancisco
santiago
sydney
tehran
```

官方 Cultural-10 设计中：

- validation：这些目标城市的一部分空间区域
- testing：同一批目标城市的另一部分空间区域

因此推荐定义：

```text
Source:
42 seen cities

Target-adapt:
10 target cities, one spatial partition

Target-test:
same 10 target cities, independent spatial partition
```

这属于：

> Cross-region Domain Adaptation

不是严格意义的 unseen-city domain generalization。

---

# 11. 17 类 closed-set 已经本地验证

## 11.1 Training

```text
Total: 352366

 0 Compact high-rise          5068
 1 Compact mid-rise         24431
 2 Compact low-rise         31693
 3 Open high-rise            8651
 4 Open mid-rise            16493
 5 Open low-rise            35290
 6 Lightweight low-rise      3269
 7 Large low-rise           39326
 8 Sparsely built           13584
 9 Heavy industry           11954
10 Dense trees              42902
11 Scattered trees           9514
12 Bush/scrub                9165
13 Low plants               41377
14 Bare rock/paved           2392
15 Bare soil/sand            7898
16 Water                    49359
```

All 17 classes are present.

---

## 11.2 Validation

```text
Total: 24119

 0 Compact high-rise          256
 1 Compact mid-rise          1254
 2 Compact low-rise          2353
 3 Open high-rise             849
 4 Open mid-rise              757
 5 Open low-rise             1906
 6 Lightweight low-rise       474
 7 Large low-rise            3395
 8 Sparsely built            1914
 9 Heavy industry             860
10 Dense trees               2287
11 Scattered trees            382
12 Bush/scrub                1202
13 Low plants                2747
14 Bare rock/paved            202
15 Bare soil/sand             672
16 Water                     2609
```

All 17 classes are present.

---

## 11.3 Testing

```text
Total: 24188

 0 Compact high-rise          266
 1 Compact mid-rise          1262
 2 Compact low-rise          2465
 3 Open high-rise             857
 4 Open mid-rise              763
 5 Open low-rise             1936
 6 Lightweight low-rise       503
 7 Large low-rise            3496
 8 Sparsely built            2013
 9 Heavy industry             908
10 Dense trees               2368
11 Scattered trees            433
12 Bush/scrub                1166
13 Low plants                2435
14 Bare rock/paved            205
15 Bare soil/sand             572
16 Water                     2540
```

All 17 classes are present.

因此可以直接保留：

\[
Y_S = Y_{T-adapt} = Y_{T-test} = 17
\]

第一阶段不需要删除类别。

---

# 12. 类别不平衡

So2Sat 是明显 long-tail。

Training：

```text
Water: 49359
Bare rock/paved: 2392
```

约相差 20.6 倍。

Validation：

```text
Large low-rise: 3395
Bare rock/paved: 202
```

约相差 16.8 倍。

建议：

> 保留全部数据，不主动砍大类；训练 sampler 应考虑 class-balanced / category-aware sampling。

这对现有 Dual_D 的：

- Prototype
- Contrastive
- Category-aware feedback

尤其重要。

不要简单使用完全随机 sampler 后让大类长期主导 batch。

---

# 13. Sentinel-1 8 通道定义

So2Sat LCZ42 HDF5 中的 S1 不是普通 8 个频谱波段，而是 SAR 派生特征：

```text
ch0: VH complex real
ch1: VH complex imag
ch2: VV complex real
ch3: VV complex imag
ch4: Lee-filtered VH intensity
ch5: Lee-filtered VV intensity
ch6: covariance real
ch7: covariance imag
```

输入 shape：

```text
32 × 32 × 8
```

第一阶段正式决定：

> **全部保留 8 通道。**

不要只取 VV/VH，不要随意删成 7 通道。

---

# 14. Sentinel-2 10 通道定义

S2 为：

```text
ch0  B2   Blue
ch1  B3   Green
ch2  B4   Red
ch3  B5   Vegetation Red Edge
ch4  B6   Vegetation Red Edge
ch5  B7   Vegetation Red Edge
ch6  B8   NIR
ch7  B8A  Narrow NIR
ch8  B11  SWIR
ch9  B12  SWIR
```

输入 shape：

```text
32 × 32 × 10
```

第一阶段正式决定：

> **全部保留 10 通道。**

不要压成 RGB。

---

# 15. S2 数值尺度

官方 So2Sat S2 已经除以 10000，HDF5 中是 reflectance scale。

本地抽样验证均值约：

```text
0.10 ~ 0.20
```

因此：

> **绝对不要再次 s2 /= 10000**

否则会重复缩放。

---

# 16. S1 不要统一转 dB

S1 中包含：

- real
- imag
- covariance real
- covariance imag

这些通道可能为负。

因此不能对全部 8 通道直接做：

```text
10 * log10(x)
```

第一阶段保持原值，采用 per-channel Z-score。

---

# 17. 已计算并验证的 Source-training normalization statistics

从完整 `training.h5` 352,366 个 Source 样本逐像素计算。

结果与 TorchGeo So2Sat benchmark 统计值逐项一致。

## 17.1 S1 mean

```python
S1_MEAN = [
    -3.591224256609e-05,
    -7.658561276843e-06,
     5.937385747597e-05,
     2.516623153712e-05,
     4.420110659759e-02,
     2.576102708500e-01,
     7.556743372573e-04,
     1.350346683002e-03,
]
```

## 17.2 S1 std

```python
S1_STD = [
    0.175552011374,
    0.175564632750,
    0.459987934178,
    0.455988755730,
    2.855990921313,
    8.324800606440,
    2.449875738256,
    1.464735298451,
]
```

## 17.3 S2 mean

```python
S2_MEAN = [
    0.123756961177,
    0.109277463637,
    0.101085520327,
    0.114239861611,
    0.159265669202,
    0.181472360088,
    0.174574031229,
    0.195016073496,
    0.154284688721,
    0.109050506996,
]
```

## 17.4 S2 std

```python
S2_STD = [
    0.039587959859,
    0.047778262752,
    0.066366167064,
    0.063588749125,
    0.077443871480,
    0.091016350859,
    0.092184665624,
    0.101645812339,
    0.099917730435,
    0.087806325091,
]
```

---

# 18. Normalization 固定方案

第一阶段固定：

\[
x'_c = (x_c - \mu_c) / (\sigma_c + \epsilon)
\]

其中：

- 每个 S1 通道单独 mean/std
- 每个 S2 通道单独 mean/std
- 统计量只来自 `training.h5` Source domain
- validation / testing 统一使用 Source training mean/std

禁止：

- 分别对 Source / Target 计算不同 mean/std
- 18 通道共用一个 global mean/std
- S2 再次除 10000
- S1 全通道统一 dB

---

# 19. S1 极端离群值

本地前 2000 patch 统计发现 S1 有明显极端值，例如：

```text
ch04:
p99 ≈ 0.317
max ≈ 857

ch05:
p99 ≈ 2.217
max ≈ 7012

ch06:
p1 ≈ -0.164
min ≈ -2167

ch07:
p99 ≈ 0.117
max ≈ 1128
```

说明 Z-score 后个别像素可能非常大。

当前决定：

> **第一版不主动 clip。**

原因：

- mean/std 与 benchmark 完全一致
- 需要先建立标准 baseline
- 不希望把 clipping 变成隐藏变量

但是代码建议保留可配置项，例如：

```text
sar_clip_after_normalize = None
```

后续若出现：

- NaN
- gradient spike
- loss explosion
- tensor relation 极端异常

再测试：

```text
clip [-5, 5]
clip [-10, 10]
```

作为独立 robustness ablation。

---

# 20. So2Sat 官方网络调研结论

So2Sat 官方团队提供了分类 Demo / 全球 LCZ 制图网络。

官方处理思路：

```text
S1 → independent ResNet-like classifier → 17-class probability
S2 → independent ResNet-like classifier → 17-class probability
                         ↓
                  decision fusion
```

融合是 prediction / probability 层的后期融合。

---

# 21. 官方 fusion 不用于我们的主模型

Satellite-Dual_D **不采用官方 decision fusion**。

官方 fusion 仅可用于：

- baseline
- architecture reference
- comparison

我们的主模型必须：

```text
SAR encoder
       \
        → Tensor-based Alignment / Relation Modeling
       /
Optical encoder
```

再进入 Dual_D 后续。

---

# 22. 官方 S1 “7 通道模型”不能直接套当前 HDF5

官方全球制图 Demo 中出现过：

```text
IN32-32-7
```

但这不是从 LCZ42 HDF5 8 通道里简单删除一个通道。

全球制图 SAR preprocessing 采用另一组 7 个 analysis-ready 特征，例如：

- unfiltered VH intensity
- unfiltered VV intensity
- unfiltered coherence
- Lee-filtered VH intensity
- Lee-filtered VV intensity
- Lee-filtered coherence
- phase difference

而当前 LCZ42 HDF5 是：

- complex real/imag
- intensity
- covariance real/imag

因此：

> **禁止为了加载官方 7 通道 S1 权重而把当前 8 通道随意删成 7 通道。**

---

# 23. Encoder 设计当前决策

## 23.1 两个独立 encoder

采用：

```text
SAR Encoder:
input_channels = 8

Optical Encoder:
input_channels = 10
```

架构可以相同，但参数独立：

\[
\theta_{SAR} \neq \theta_{Optical}
\]

不要共享权重。

原因：

- SAR / Optical 成像机理不同
- 输入通道不同
- 数值与物理含义不同

---

## 23.2 Source / Target 共用同一模态 encoder

同一 SAR encoder 同时处理：

```text
SAR_source
SAR_target
```

同一 Optical encoder 同时处理：

```text
Optical_source
Optical_target
```

即：

\[
E_{SAR}(x_s^{SAR}), E_{SAR}(x_t^{SAR})
\]

\[
E_{OPT}(x_s^{OPT}), E_{OPT}(x_t^{OPT})
\]

Domain 区别不通过不同 encoder 表示。

---

## 23.3 推荐 ResNet20-style

So2Sat 官方在 32×32 patch 上使用较浅的 ResNet-style 网络。

建议第一版采用：

> ResNet20-style / small-residual encoder

而不是直接使用 ImageNet ResNet50。

理由：

- 输入仅 32×32
- 标准 ResNet50 早期大 stride / pooling 容易丢空间细节
- So2Sat 官方实践本身使用轻量 residual network

---

## 23.4 推荐输出 256-D feature

推荐：

```text
SAR:
8×32×32
→ ResNet20-style
→ feature map
→ GAP
→ 256-D

Optical:
10×32×32
→ ResNet20-style
→ feature map
→ GAP
→ 256-D
```

然后：

```text
256-D SAR feature
256-D Optical feature
        ↓
原 Dual_D Tensor module
```

第一阶段不要在 encoder 末端接官方 17-class softmax 再融合。

---

# 24. Satellite-Dual_D 当前推荐结构

```text
                         SOURCE DOMAIN
────────────────────────────────────────────────

       SAR_s                         Optical_s
    8×32×32                         10×32×32
       │                                │
       ↓                                ↓
 SAR ResNet20                     Optical ResNet20
       │                                │
       ↓                                ↓
     256-D                            256-D
       │                                │
       └──────────────┬─────────────────┘
                      ↓
             Tensor-based Alignment
                      ↓
           Multimodal Source Feature
                      │
                      ↓
             Bidirectional Dual_D
              S→T / T→S Generators
              Dual Discriminators
                      ↑
                      │
           Multimodal Target Feature
                      ↑
             Tensor-based Alignment
                      ↑
       ┌──────────────┴─────────────────┐
       │                                │
     256-D                            256-D
       ↑                                ↑
 SAR ResNet20                     Optical ResNet20
       ↑                                ↑
      SAR_t                          Optical_t
    8×32×32                         10×32×32

────────────────────────────────────────────────
                         TARGET DOMAIN
```

后续继续保留：

```text
Cycle
Identity
Paired Contrastive
Prototype Contrastive
Classification Feedback
Modality Relation Drift
```

---

# 25. Tensor 模块的位置

Tensor Alignment 应位于：

```text
modality-specific encoder
        ↓
semantic feature
        ↓
Tensor Relation / Alignment
```

不建议直接对原始：

```text
8×32×32 SAR
10×32×32 Optical
```

做 Tensor 运算。

原因：

- 原始 SAR 与 Optical 数值量纲和物理意义差异非常大
- Tensor 应建模高层语义特征关系，而不是直接把 complex SAR 数值和 reflectance 做关系计算

---

# 26. 数据读取原则

不要把 HDF5 转成几十万个 PNG。

应保留：

```text
HDF5 + index-based Dataset
```

推荐 Dataset 每次通过 index 读取：

```text
sen1[index]
sen2[index]
label[index]
city[index]
```

然后：

1. `float64 → float32`
2. per-channel Z-score
3. NHWC → NCHW
4. label one-hot → class id (`argmax`)
5. 返回 SAR / Optical / label / metadata

---

# 27. 建议的 metadata / split abstraction

建议构建统一 metadata，例如：

```text
index, split, city, class_id
0, source, london, 5
1, source, beijing, 3
...
0, target_adapt, guangzhou, 1
...
0, target_test, mumbai, 8
```

不一定必须生成实体 CSV，但代码层需要有等价 abstraction。

建议内部统一 domain role：

```python
source
target_adapt
target_test
```

而不是依赖：

```python
training
validation
testing
```

这样可以让训练逻辑与数据集原始命名解耦。

---

# 28. 训练采样建议

因为类别不平衡明显，建议不要简单 random shuffle 后直接训练。

优先考虑：

- category-balanced sampler
- per-class quota
- balanced batch
- 或沿用 Dual_D 现有 category-aware sampler

需要同时满足：

- Source batch 类别可用于 Prototype / Contrastive
- Target batch 若 supervised，可匹配共同类别
- 不要求 Source sample 与 Target sample 是同一实例

对于 paired contrastive：

> 配对仍然是“同类别跨域多正样本 / 类别级关系”，不是物理同位置 Source-Target pairing。

---

# 29. 第一阶段训练模式建议

优先做：

## Supervised Cross-domain Adaptation

```text
Source:
training.h5
SAR + Optical + label

Target-adapt:
validation.h5
SAR + Optical + label

Final evaluation:
testing.h5
SAR + Optical + label
```

这样最小化对原 Dual_D loss pipeline 的修改。

第一阶段跑通以后再增加：

```text
--target-label-mode hidden
```

之类的 UDA 开关。

---

# 30. 评估要求

沿用 Dual_D 后期实验经验：

不要仅报告训练全过程的：

```text
max val ACC
```

应继续使用稳定 checkpoint 选择逻辑，例如：

- minimum checkpoint epoch
- stability window
- val_f1_macro_present
- ACC / Precision / Recall / F1 从同一个 epoch 读取

So2Sat Target test 样本量远大于原海上小验证集，因此指标会更加稳定。

建议最终至少报告：

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- per-class F1
- confusion matrix

若继续 relation drift，还建议记录：

- S→T drift
- T→S drift
- weighted drift
- before/after relation difference

---

# 31. 当前不应做的事情

Codex 修改时请避免：

1. 不要删除 / 替换原 Tensor Alignment 核心方法
2. 不要改成 So2Sat 官方 decision fusion
3. 不要把 S1 8 通道压成官方 Demo 的 7 通道
4. 不要把 S2 压成 RGB
5. 不要再次对 S2 `/10000`
6. 不要对全部 S1 通道统一 log/dB
7. 不要给 Source / Target 分别算 normalization
8. 不要把任务改成 object detection / segmentation
9. 不要要求 Source 与 Target 同 index 一一对应
10. 不要一次同时重写 Tensor、Generator、Module C、Drift 全部逻辑
11. 不要先改成 UDA 再调通数据
12. 不要把 HDF5 全量一次性 preload 到 RAM
13. 不要把 56G HDF5 转成大量 PNG 作为主数据格式

---

# 32. Codex 下一步应做什么：推荐实现顺序

下面按优先级执行。

---

## Phase 1：只做数据层，不碰核心模型

实现一个 So2Sat dataset loader：

建议文件名，例如：

```text
data/so2sat_lcz42.py
```

功能：

- 支持 `training.h5 / validation.h5 / testing.h5`
- 支持对应 `*_geo.h5`
- 返回：
  - `sar`: FloatTensor `[8, 32, 32]`
  - `optical`: FloatTensor `[10, 32, 32]`
  - `label`: LongTensor scalar
  - `city`
  - `index`
  - `domain_role`
- 使用固定 Source mean/std
- HDF5 lazy open
- multiprocessing DataLoader 安全
- 不 preload 全数据

需要特别处理 h5py + multi-worker：

> 不要在 Dataset `__init__` 中长期持有跨进程共享 HDF5 handle。建议 lazy-open / per-worker handle。

---

## Phase 2：建立 So2Sat 配置

建议新增配置：

```text
configs/so2sat_lcz42.yaml
```

至少包含：

```yaml
dataset:
  root: /home/lixiang/lx/Data/So2Sat-LCZ42/v4

  source:
    data: training.h5
    geo: training_geo.h5

  target_adapt:
    data: validation.h5
    geo: validation_geo.h5

  target_test:
    data: testing.h5
    geo: testing_geo.h5

  num_classes: 17

  s1_channels: 8
  s2_channels: 10

normalization:
  use_source_stats: true
  sar_clip_after_normalize: null
```

并写入上述完整 mean/std。

---

## Phase 3：实现 / 适配双 encoder

新增：

```text
SARResNet20Encoder(in_channels=8, out_dim=256)
OpticalResNet20Encoder(in_channels=10, out_dim=256)
```

建议架构相同，但参数不共享。

先写单元测试：

```text
input SAR: [B, 8, 32, 32]
output: [B, 256]

input Optical: [B, 10, 32, 32]
output: [B, 256]
```

---

## Phase 4：对接原 Tensor Alignment

把原：

```text
VIS feature
IR feature
```

替换 / abstract 成：

```text
modality_1 = SAR feature
modality_2 = Optical feature
```

如果原代码写死名称 `vis` / `ir`，建议做最小泛化：

```text
vis → modality_a / optical
ir  → modality_b / sar
```

但第一阶段以少改代码、保证行为一致为原则。

最重要：

> 不改 Tensor 模块内部数学公式，只适配输入维度。

如果 Tensor 模块原来接受投影维度 `d`，检查 256-D encoder output 是否先通过原 projection layer。

---

## Phase 5：只跑前向 smoke test

先不要训练。

验证：

```text
Source batch
Target batch
↓
SAR/Optical encoders
↓
Tensor alignment
↓
Multimodal features
↓
Generator
↓
Discriminator
↓
Classifier
↓
Relation Drift
```

所有 shape 正确。

需要输出一份完整 shape trace。

---

## Phase 6：跑单 batch backward

验证：

- classification loss
- adversarial loss
- Cycle
- Identity
- Contrastive
- Prototype
- Drift

均可 backward。

确认：

- 无 NaN
- 无 Inf
- gradient 非空
- detach reference 行为正确
- Tensor encoder / Generator / classifier 梯度流符合设计

---

## Phase 7：小规模 overfit test

先从 HDF5 抽一个很小子集，例如：

```text
每类 20~50 样本
```

或总计数百 / 数千样本。

验证：

- 分类 loss 能下降
- accuracy 能明显上升
- 模型可过拟合小数据
- 多模态输入 pipeline 正确

---

## Phase 8：小规模 Source→Target adaptation test

不要直接跑 352k full training。

先固定：

```text
Source 每类若干样本
Target-adapt 每类若干样本
```

跑 2~5 epochs。

记录：

- cls source
- cls target
- adversarial
- tensor loss
- cycle
- prototype
- drift
- total loss

观察是否数值稳定。

---

## Phase 9：Full So2Sat supervised adaptation

第一版 Full：

```text
Source = training.h5
Target-adapt = validation.h5
Test = testing.h5
17 classes
8ch S1
10ch S2
Tensor Alignment ON
Relation Drift ON
```

建议先 1 seed smoke，再 3 seeds。

---

# 33. 第一版建议 baseline

为了后续论证方法，不要只跑 Full。

至少应保留以下 baseline：

```text
B0: S1-only classifier
B1: S2-only classifier
B2: S1 + S2 simple concat
B3: S1 + S2 decision/probability fusion
B4: Tensor fusion without domain translation
B5: Dual_D without relation drift
B6: Full Satellite-Dual_D
```

如果计算量允许，再做：

```text
B7: remove Cycle
B8: remove Prototype
B9: remove Tensor Alignment
B10: remove Relation Drift
```

---

# 34. 后续通道消融建议

第一版主模型固定：

```text
S1 all 8
S2 all 10
```

之后可做：

```text
S1 intensity-only: ch4 + ch5
S1 all-8

S2 RGB-only: B4+B3+B2
S2 all-10

S1-only
S2-only
S1 all-8 + S2 all-10
```

这些作为输入信息消融，不要在主实现阶段提前裁剪通道。

---

# 35. 关于原海上 v12/v13 weather profiles

原项目后期海上实验曾采用：

- v12 作为黑天 / 雾天 / 雨天基础 profile
- v13 对逆光单独加大正则 / 延迟模块启动
- stability checkpoint selection

这些天气 profile **不应该直接复制到 So2Sat**。

Satellite 版本需要重新建立：

```text
configs/so2sat_*.yaml
```

原因：

- 数据规模不同
- batch 数量不同
- 输入维度不同
- encoder 不同
- 类别数变成 17
- target test 数量更大
- domain shift 机制不同

但以下训练经验可保留：

- warmup
- module ramp
- delayed adversarial start
- delayed relation drift start
- stability-based checkpoint selection
- multi-seed evaluation

---

# 36. 代码迁移的原则

这次迁移应尽量满足：

> **换背景、换数据、换前端编码器；不换核心方法。**

最小修改路径：

```text
VIS / IR input
      ↓
替换成
SAR / Optical input

旧 feature extractor
      ↓
替换成
8ch / 10ch small ResNet encoders

Tensor Alignment
      ↓
保持原实现

Dual-Domain Translation
      ↓
保持原实现

Module C
      ↓
保持原实现，必要时适配 num_classes=17

Relation Drift
      ↓
保持原实现，适配 2 modalities / feature dim
```

---

# 37. 推荐 Codex 在开始修改前先做代码审计

在真正写代码之前，Codex 应先定位：

1. 当前 VIS encoder 定义位置
2. 当前 IR encoder 定义位置
3. encoder 输出维度
4. Tensor Alignment 输入接口
5. Tensor projection dims
6. 当前 data loader / sampler
7. label representation
8. num_classes 写死的位置
9. Module C 对 class count 的依赖
10. prototype buffer shape
11. relation drift 如何切分 modality feature
12. generator 输入 feature dim
13. classifier 输入 feature dim
14. checkpoint / config 系统
15. train / val / test split 逻辑

输出一个“需要修改文件清单”，再开始 patch。

---

# 38. 强烈建议新增的测试

至少新增：

```text
test_so2sat_dataset_shapes
test_so2sat_label_range
test_so2sat_normalization
test_so2sat_source_target_class_set
test_sar_encoder_shape
test_optical_encoder_shape
test_tensor_with_so2sat_features
test_full_forward_so2sat
test_full_backward_so2sat
test_relation_drift_so2sat
test_h5_dataloader_multiworker
```

---

# 39. 推荐的数据检查脚本

建议保留独立 audit 脚本，例如：

```text
scripts/audit_so2sat.py
```

输出：

- split sample count
- city count
- class count
- class histogram
- NaN / Inf
- channel min/max/mean/std
- Source/Target class-set equality
- random sample shape

这样后续换服务器 / 数据版本时可以快速验证。

---

# 40. 当前最终技术决策汇总

| 项目 | 当前决定 |
|---|---|
| 任务 | 图像级 / 场景级 classification |
| 数据集 | So2Sat LCZ42 v4 |
| Source | training.h5 |
| Target-adapt | validation.h5 |
| Target-test | testing.h5 |
| Source cities | 42 |
| Target cities | 10 |
| 类别 | 17，全部保留 |
| 类别集合 | Source = Target |
| SAR | Sentinel-1，8 通道 |
| Optical | Sentinel-2，10 通道 |
| 模态内对应 | 同 patch，对应 |
| Source-Target 实例对应 | 不要求 |
| normalization | Source per-channel Z-score |
| S2 /10000 | 不做，已经处理 |
| S1 dB | 不统一转换 |
| SAR clipping | v1 不做 |
| Encoder | 两个独立 ResNet20-style |
| Encoder output | 推荐 256-D |
| 多模态融合 | **原 Dual_D Tensor-based Alignment** |
| 官方 decision fusion | 不用于主模型 |
| Domain translation | 保留双向生成器 |
| Discriminator | 保留双判别器 |
| Module C | 保留 |
| Relation Drift | 保留 |
| 第一阶段 DA | supervised adaptation |
| 后续扩展 | UDA / temporal datasets |

---

# 41. Codex 下一步最推荐的第一条指令

建议把下面这段直接作为 Codex 的第一个工作指令：

> 请先审计当前 Dual_D 代码仓库，不要立即大范围修改。定位并列出：
> 1. VIS/IR 数据加载位置；
> 2. VIS/IR encoder；
> 3. Tensor Alignment / Tensor Relation 模块及输入维度；
> 4. Generator / Discriminator 输入维度；
> 5. Module C 各 loss 与类别数的依赖；
> 6. Prototype buffer；
> 7. Modality Relation Drift 对模态切分和 feature dim 的假设；
> 8. train/val/test pipeline；
> 9. config 系统。
>
> 然后给出一份“最小侵入式 So2Sat 迁移修改计划”，目标为：
> - 输入改为 S1 `[B,8,32,32]` 与 S2 `[B,10,32,32]`
> - 使用两个独立 ResNet20-style encoder 输出 256-D
> - 保持原 Tensor-based Alignment 数学实现不变
> - 保持 Dual_D 双向领域翻译 / Module C / Relation Drift 主逻辑不变
> - Source=training.h5，Target-adapt=validation.h5，Target-test=testing.h5
> - num_classes=17
> - 使用本交接文档中的 Source mean/std
>
> 在开始 patch 前先报告需要修改的文件清单、每个文件的修改原因以及可能影响现有海上代码的兼容性风险。

---

# 42. 当前阶段完成度

已经完成：

- [x] 明确新卫星研究背景
- [x] 明确任务继续采用图像级 classification
- [x] 明确 Optical + SAR 双模态
- [x] 明确 domain 仍由时空变化引起
- [x] 明确 Source / Target 类别必须一致
- [x] 明确 Source / Target 不要求实例对应
- [x] 选择 So2Sat LCZ42 v4
- [x] Linux 下载并解压数据
- [x] 验证 HDF5 数据结构
- [x] 验证 geo / city 信息
- [x] 验证 42 Source cities
- [x] 验证 10 Target cities
- [x] 验证 train/validation/testing 均完整 17 类
- [x] 验证 S1 8 通道 / S2 10 通道
- [x] 计算完整 Source mean/std
- [x] 验证统计值与 benchmark 一致
- [x] 确定 normalization 协议
- [x] 确定保留完整 8+10 通道
- [x] 调研 So2Sat 官方网络
- [x] 确认官方 7ch S1 与当前 8ch HDF5 表示不同
- [x] 确定官方 decision fusion 不作为主方法
- [x] 确定继续使用 Dual_D Tensor-based Alignment
- [x] 确定推荐两个独立 ResNet20-style encoder

尚未完成：

- [ ] 审计 Dual_D 当前源码
- [ ] 实现 So2Sat Dataset
- [ ] 实现配置文件
- [ ] 实现 / 适配双 encoder
- [ ] 对接 Tensor Alignment
- [ ] 对接 17 类 classifier / prototype
- [ ] 验证 relation drift feature split
- [ ] 完成 full forward test
- [ ] 完成 backward test
- [ ] 完成小规模 overfit
- [ ] 完成小规模 adaptation
- [ ] 完成 full So2Sat supervised domain adaptation
- [ ] 完成 baseline / ablation
- [ ] 后续 UDA
- [ ] 后续 temporal / spatio-temporal dataset

---

# 43. 最重要的总原则

Codex 后续工作请始终遵循：

> **Satellite-Dual_D 的目标不是重新发明一个 So2Sat 分类网络，而是把 So2Sat 的 SAR/Optical 数据可靠地接入现有 Dual_D，并保留 Dual_D 的“Tensor-based multimodal alignment + bidirectional domain translation + category-aware feedback + relation drift”方法主线。**

第一阶段所有代码决策都应优先服务于：

1. 数据正确
2. shape 正确
3. 数值稳定
4. 原核心方法不被破坏
5. 与原海上代码尽量兼容
6. 能做清晰的 baseline / ablation
7. 后续可扩展到 UDA 与 temporal dataset

