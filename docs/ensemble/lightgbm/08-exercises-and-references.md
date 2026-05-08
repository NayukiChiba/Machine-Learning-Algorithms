---
title: LightGBM — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 LightGBM 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 LightGBM 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 Leaf-wise 生长、GOSS、EFB、直方图、LightGBM vs GBDT 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 LightGBM 行为变化——建立参数-效果的直觉 |
| 参考文献 | 入口 | 提供 LightGBM 原始论文、官方文档和扩展阅读 |

## 1. 自检问题

1. LightGBM 的 Leaf-wise 生长策略与 GBDT 的 Level-wise 生长策略有何本质区别？Leaf-wise 在同等叶子数下为何损失下降更快？

2. 直方图算法如何将连续特征离散化？离散化带来的速度收益和可能的精度损失分别是什么？

3. GOSS（Gradient-based One-Side Sampling）的采样策略与简单的随机子采样（`subsample`）有何本质不同？为什么保留大梯度样本比均匀随机采样更高效？

4. 为什么 LightGBM 使用 `num_leaves` 而非 `max_depth` 来控制树复杂度？在 `max_depth=-1` 的情况下，`num_leaves=31` 的树可能有多深？

5. EFB（Exclusive Feature Bundling）在什么场景下最有效？在当前稠密 20 维数据上，EFB 的收益如何？

6. LightGBM 与 sklearn GBDT 在训练速度、参数体系、默认配置上的核心差异有哪些？

7. 当前 LightGBM 流水线中 `n_jobs=-1` 的并行发生在哪个层面？为什么 Boosting 算法仍能利用多核并行？

## 2. 动手练习

### 练习 1：改变叶子数 `num_leaves`

将 `num_leaves` 分别设为 `7`、`15`、`31`、`63`、`127`，观察特征重要性和混淆矩阵的变化。

```python
model = train_model(X_train_s, y_train, num_leaves=15)
```

回答：`num_leaves` 增大后，模型复杂度如何变化？`num_leaves=127` 在 800 个训练样本上是否会过拟合？

### 练习 2：改变学习率 `learning_rate`

将 `learning_rate` 分别设为 `0.01`、`0.02`、`0.05`、`0.1`、`0.2`，同时保持 `n_estimators=300`，观察混淆矩阵。

```python
model = train_model(X_train_s, y_train, learning_rate=0.01)
```

回答：`learning_rate=0.01` 且 `n_estimators=300` 时，模型是否欠拟合？你需要增加多少棵树来匹配较小学习率？

### 练习 3：改变列采样 `colsample_bytree`

将 `colsample_bytree` 分别设为 `0.3`、`0.5`、`0.7`、`0.9`、`1.0`，观察特征重要性的变化。

```python
model = train_model(X_train_s, y_train, colsample_bytree=0.3)
```

回答：`colsample_bytree` 减小后，特征重要性排序是否发生明显变化？这种变化对理解"哪些特征重要"有什么影响？

### 练习 4：对比 GBDT 训练速度

使用相同数据，分别训练 sklearn GBDT 和 LightGBM，对比训练耗时。

```python
from model_training.ensemble.gbdt import train_model as train_gbdt
from model_training.ensemble.lightgbm import train_model as train_lgbm

# 使用相同数据（需要调整维度匹配）
model_gbdt = train_gbdt(X_train_s, y_train, n_estimators=200, learning_rate=0.1, max_depth=3)
model_lgbm = train_lgbm(X_train_s, y_train, n_estimators=300, learning_rate=0.05, num_leaves=31)
```

回答：在 1000 样本 × 20 特征的数据上，LightGBM 比 GBDT 快多少？随着数据规模增大，这个差距如何变化？

### 练习 5：改变数据规模观察速度优势

修改 `data_generation/ensemble.py` 中的 `n_samples` 参数（分别设为 `500`、`1000`、`2000`、`5000`），重新运行 LightGBM 和 GBDT 流水线。

```python
# 在 data_generation/ensemble.py 的 __init__ 中
class EnsembleData:
    n_samples: int = 5000  # 试试 500, 1000, 2000, 5000
```

回答：数据规模从 500 增加到 5000 时，LightGBM 相对于 GBDT 的速度倍数如何变化？这验证了直方图算法的什么性质？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017. | LightGBM 原始论文——GOSS、EFB 和 Leaf-wise 生长的完整推导和实验验证 |
| 2 | LightGBM 官方文档 — [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) | 全部参数、加速技巧和调参指南的权威参考 |
| 3 | scikit-learn 兼容接口 — [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) | scikit-learn 兼容接口的 API 参考——与 sklearn 无缝集成 |
| 4 | Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189-1232. | GBDT 的理论基础——LightGBM 在此数学框架上进行工程优化 |

## 常见坑

1. 把 `num_leaves=31` 和 `max_depth=-1` 当成"不受控的完全生长"——`num_leaves` 是实际复杂度控制参数，`max_depth=-1` 只是不另设上限。
2. 在新环境中忘记 `lightgbm` 是可选依赖——导入前需 `try/except`，在未安装环境运行会抛出 `ImportError`。
3. 把 LightGBM 的 `n_jobs=-1` 与 Bagging 的并行等同——LightGBM 的并行在直方图构建和特征扫描层面，而非基学习器级。
4. 在极小数据（<100 样本）上使用 LightGBM——直方图离散化损失可能超过精度收益。

## 小结

- 7 个自检问题覆盖 LightGBM 的核心创新：Leaf-wise vs Level-wise、直方图算法、GOSS vs 随机采样、`num_leaves` vs `max_depth`、EFB、与 sklearn GBDT 对比、并行机制。
- 5 个动手练习从不同角度探索 LightGBM 的行为——改变叶子数、学习率、列采样、对比 GBDT 速度、改变数据规模。
- 4 篇参考文献从原始论文（Ke et al. 2017）→ 官方文档 → API 参考 → GBDT 理论基础构成完整的阅读路线。
