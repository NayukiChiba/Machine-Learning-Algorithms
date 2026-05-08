---
title: XGBoost — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 XGBoost 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 XGBoost 的行为——注意回归任务的特点。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对二阶泰勒展开、显式正则化、加权分位数、XGBoost vs GBDT/LightGBM 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 XGBoost 行为变化——建立回归参数-效果的直觉 |
| 参考文献 | 入口 | 提供 XGBoost 原始论文、官方文档和扩展阅读 |

## 1. 自检问题

1. XGBoost 的二阶泰勒展开引入了 Hessian $h_i$——在 MSE 回归任务中 $h_i$ 的值是什么？这对 XGBoost 的二阶优势有何影响？

2. XGBoost 的正则化目标函数包含三个正则化项：$\gamma T$ + $\frac{1}{2}\lambda\|\mathbf{w}\|^2$ + $\alpha\|\mathbf{w}\|_1$。这三个项分别作用于什么层级？在过拟合时应该如何调整？

3. 叶子权重的闭式解 $w_j^* = -\frac{G_j}{H_j + \lambda}$ 中，$\lambda$ 的作用是什么？为什么 GBDT（sklearn）没有类似的闭式解？

4. 分裂增益公式 $\text{Gain} = \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma$ 中，$\gamma$ 和 $\lambda$ 分担了什么不同的角色？

5. XGBoost 的加权分位数草图与 LightGBM 的直方图分桶有何不同？在 MSE 回归下（$h_i=1$），两者是否等价？

6. `min_child_weight` 在回归任务中（$h_i=1$）等价于什么？在分类任务中（$h_i = p_i(1-p_i)$），同一个 `min_child_weight=1` 的含义有何不同？

7. XGBoost 与其他三个集成模型（Bagging/GBDT/LightGBM）在任务类型、数据规模、评估体系上的核心差异有哪些？

## 2. 动手练习

### 练习 1：调整正则化系数 `reg_lambda`

将 `reg_lambda` 分别设为 `0.0`、`0.1`、`1.0`（默认）、`10.0`、`100.0`，观察残差图和特征重要性的变化。

```python
model = train_model(X_train, y_train, reg_lambda=0.0)
```

回答：`reg_lambda=0.0`（关闭 L2）时，残差的分布是否变大？`reg_lambda=100.0` 是否导致欠拟合（残差系统性增宽）？

### 练习 2：调整分裂门槛 `gamma`

将 `gamma` 分别设为 `0.0`、`0.1`、`1.0`、`5.0`，观察树结构的变化。

```python
model = train_model(X_train, y_train, gamma=1.0)
```

回答：`gamma` 增大后，树的数量是否减少？`gamma=5.0` 时是否出现明显的欠拟合？

### 练习 3：改变树深度 `max_depth`

将 `max_depth` 分别设为 `2`、`4`、`6`、`10`、`15`，观察残差图变化。

```python
model = train_model(X_train, y_train, max_depth=2)
```

回答：`max_depth=2` 的残差图是否出现系统性偏差？`max_depth=15` 是否在真实数据上过拟合？

### 练习 4：对比 XGBoost 与 GBDT 在回归任务上的表现

使用相同的加州房价数据，分别训练 XGBoost 和 sklearn `GradientBoostingRegressor`，对比残差。

```python
from sklearn.ensemble import GradientBoostingRegressor

model_gbdt = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
)
model_gbdt.fit(X_train, y_train)
y_pred_gbdt = model_gbdt.predict(X_test)
```

回答：XGBoost 的预测残差是否比 GBDT 更小？正则化（`reg_lambda=1.0`）是否带来了泛化提升？

### 练习 5：改变采样比例

将 `subsample` 和 `colsample_bytree` 分别设为更低的值（如 `0.5`），观察训练耗时和残差的变化。

```python
model = train_model(X_train, y_train, subsample=0.5, colsample_bytree=0.5)
```

回答：采样比例降至 0.5 后，训练是否明显加速？残差是否显著增大？这种采样比例在什么场景下可能有用？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016. | XGBoost 原始论文——二阶泰勒展开、正则化目标函数和系统设计的完整推导 |
| 2 | XGBoost 官方文档 — [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) | 全部参数的官方说明和调参指南 |
| 3 | scikit-learn 兼容接口 — [XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor) | XGBoost 回归模型的 scikit-learn API 参考 |
| 4 | Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. | GBDT 的理论基础——XGBoost 在此基础上引入二阶展开和显式正则化 |

## 常见坑

1. 在 MSE 回归中期待二阶展开带来巨大提升——MSE 的 Hessian 是常数 1，二阶优势主要体现于分类（$h_i = p_i(1-p_i)$ 非均匀）。
2. 忽略 `reg_lambda=1.0` 的默认值——与 GBDT/LightGBM 不同，XGBoost 的 L2 默认开启，调参时应优先调整它。
3. 混淆 `min_child_weight` 与 `min_samples_leaf`——两者仅在 Hessian 恒为 1（MSE 回归）时等价。
4. 在 `train_test_split` 中误传入 `stratify=y`——回归无类别可分层。

## 小结

- 7 个自检问题覆盖 XGBoost 的核心创新：二阶泰勒展开、三重正则化、闭式解、加权分位数、`min_child_weight` 含义、与其他集成模型对比。
- 5 个动手练习从不同角度探索 XGBoost 的行为——调整 L2 正则化、gamma 门槛、树深度、对比 GBDT 回归、改变采样比例。
- 4 篇参考文献从原始论文（Chen & Guestrin 2016）→ 官方文档 → API 参考 → GBDT 理论基础构成完整的阅读路线。
