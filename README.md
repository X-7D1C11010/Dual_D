# Dual_D 双判别器扩展包

该目录是针对 `D:\Code\JMDA-Net` 当前多模态域适应方案的独立扩展实现。
所有脚本均为新增文件，不修改原项目中的任何脚本。

## 设计目标

- 在 TAL 张量对齐后的融合特征空间中引入双判别器。
- 使用主判别器约束“目标域 -> 源域稳定特征”的映射。
- 使用辅助判别器约束“源域 -> 目标域天气特征”的映射。
- 通过双向生成、闭环一致性、身份保持、特征级对比损失和分类反馈保持语义稳定。
- 通过配置文件和适配器接口与原有训练流程兼容。

## 推荐接入位置

在原方案完成以下步骤之后接入：

```text
Visual/IR extractor -> TensorBasedAlignmentStable -> feat_src / feat_tgt
```

即在 `feat_src = concat(p_s_vis, p_s_ir)` 和
`feat_tgt = concat(p_t_vis, p_t_ir)` 之后，将二者传入
`dual_d.integration_adapter.DualDTrainingAdapter`。

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
  scripts\
    example_integration_usage.py
```

## 最小使用示例

```python
from dual_d.config import DualDConfig
from dual_d.integration_adapter import DualDTrainingAdapter

config = DualDConfig(feature_dim=256)
adapter = DualDTrainingAdapter(config).to(device)

outputs = adapter.forward_features(feat_src, feat_tgt, labels=s_label)
d_loss, d_logs = adapter.compute_discriminator_loss(outputs)
g_loss, g_logs = adapter.compute_generator_loss(
    outputs=outputs,
    classifier=classifier,
    criterion_cls=criterion_cls,
    source_labels=s_label,
    target_labels=t_label,
)
```

详见 `docs/integration_notes.md` 和 `scripts/example_integration_usage.py`。
