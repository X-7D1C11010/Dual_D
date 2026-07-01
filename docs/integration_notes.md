# Dual_D 与 JMDA-Net 的理论接入说明

本文档说明如何在不修改原始脚本的前提下，将 `D:\Code\Dual_D` 中的新模块接入
`D:\Code\JMDA-Net` 当前训练结构。

## 1. 接入点

Dual_D 的输入不是原始图像，而是原方案中 TAL 张量对齐后的融合特征：

```python
feat_src = torch.cat([p_s_vis, p_s_ir], dim=1)
feat_tgt = torch.cat([p_t_vis, p_t_ir], dim=1)
```

如果当前设置为 `PROJ_DIM = 128`，则 `feat_src` 与 `feat_tgt` 的维度均为 256，
对应默认配置中的：

```json
{
  "feature_dim": 256
}
```

## 2. 模块角色

Dual_D 将论文 TACL 中的双向图像映射改写为特征级双向映射：

| 模块 | 论文语义 | 当前特征语义 |
| --- | --- | --- |
| `target_to_source` | `G_F: X -> Y` | 目标天气特征 -> 源域稳定特征 |
| `source_to_target` | `G_B: Y -> X` | 源域稳定特征 -> 目标天气特征 |
| `PrimarySourceDiscriminator` | `D_F` | 判别真实源域特征与生成源域特征 |
| `AuxiliaryTargetDiscriminator` | `D_B` | 判别真实目标域特征与生成目标域特征 |

## 3. 训练循环建议

在新的训练脚本中，可以沿用原始训练脚本的前半段：

```python
f_s_vis, f_s_ir = net_vis(s_vis), net_ir(s_ir)
f_t_vis, f_t_ir = net_vis(t_vis), net_ir(t_ir)
(p_s_vis, p_s_ir), (p_t_vis, p_t_ir), loss_tal = tal_module(
    [f_s_vis, f_s_ir],
    [f_t_vis, f_t_ir],
)
feat_src = torch.cat([p_s_vis, p_s_ir], dim=1)
feat_tgt = torch.cat([p_t_vis, p_t_ir], dim=1)
```

然后新增：

```python
dual_outputs = dual_adapter.forward_features(feat_src, feat_tgt, labels=s_label)
```

判别器更新：

```python
optimizer_dual_d.zero_grad()
loss_dual_d, d_logs = dual_adapter.compute_discriminator_loss(dual_outputs)
loss_dual_d.backward()
optimizer_dual_d.step()
```

生成器/主干更新：

```python
loss_dual_g, g_logs = dual_adapter.compute_generator_loss(
    outputs=dual_outputs,
    labels=s_label,
    classifier=classifier,
    criterion_cls=criterion_cls,
    source_labels=s_label,
    target_labels=t_label,
)

loss_total = loss_cls_total + 0.3 * loss_tal + loss_dual_g
```

## 4. 与原始单判别器的关系

建议先保留原有三分类域判别器作为 baseline，对 Dual_D 做独立消融。稳定后再考虑
以下两种策略：

- 并联：保留原三分类判别器，同时增加 Dual_D 损失。
- 替代：使用 Dual_D 的方向化对抗约束替代原三分类判别器。

从风险控制角度，建议先采用并联策略，并将 Dual_D 的对抗权重设为较小值。

## 5. 推理建议

目标域验证或测试时，可以使用：

```python
enhanced_feat_tgt = dual_adapter.inference_features(feat_tgt, mode="source_like")
logits = classifier(enhanced_feat_tgt)
```

若担心源域化过强，可使用残差融合：

```python
enhanced_feat_tgt = dual_adapter.inference_features(feat_tgt, mode="residual")
```

