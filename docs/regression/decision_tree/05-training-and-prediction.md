---
title: 决策树回归 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解决策树回归流水线的完整执行顺序——从数据加载到四类可视化输出。
2. 理解决策树的训练过程——递归寻找最优 $(j, s)$ 分裂直到满足停止条件。
3. 理解 `predict` 的预测方式——输入沿树从根走到叶，输出该叶子的局部常数。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainDecisionTreeRegressionModel(...)` | 函数 | 构建并训练决策树回归模型——递归二分特征空间 |
| `model.fit(X_train, y_train)` | 方法 | CART 算法——每次遍历所有特征的候选分裂点，选平方误差最小的 $(j, s)$ |
| `model.predict(X_test)` | 方法 | 对新样本沿树从根走到叶——返回叶子节点训练集目标均值 |
| `plot_residuals(...)` | 函数 | 绘制预测-真实散点图 + 残差分布图 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证 R² 随样本量变化的曲线 |
| `plot_tree_structure(...)` | 函数 | 绘制决策树的可视化结构图 |

## 1. 完整流水线流程

### 流程概述

```
fetch_california_housing(as_frame=True)
    │
    ├─ ① X = data.drop(columns=["price"]), y = data["price"]
    ├─ ② X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
    ├─ ③ model = trainDecisionTreeRegressionModel(X_train, y_train)
    ├─ ④ y_pred = model.predict(X_test)
    ├─ ⑤ plot_residuals(y_test, y_pred)
    ├─ ⑥ plot_feature_importance(model, feature_names)
    ├─ ⑦ plot_learning_curve(DecisionTreeRegressor(...), X_train, y_train, scoring="r2")
    └─ ⑧ plot_tree_structure(model, feature_names)
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 加载数据 | `fetch_california_housing` | — | `DataFrame`，`(20640, 9)` | 真实加州房价数据 |
| 特征标签拆分 | `drop` + 列选择 | `DataFrame` | `X` `(20640, 8)`、`y` `(20640,)` | 标签列 `price` |
| 数据切分 | `train_test_split` | `X`、`y` | `X_train`、`X_test`、`y_train`、`y_test` | `test_size=0.2`，无标准化 |
| 训练 | `trainDecisionTreeRegressionModel` | `X_train`、`y_train` | `DecisionTreeRegressor` | CART 递归分裂 |
| 预测 | `model.predict` | `X_test` | `y_pred` `(4128,)` | 叶子局部常数 |
| 残差图 | `plot_residuals` | `y_test`、`y_pred` | PNG 图像 | 误差分布诊断 |
| 特征重要性 | `plot_feature_importance` | `model`、`feature_names` | PNG 图像 | 特征贡献排名 |
| 学习曲线 | `plot_learning_curve` | 新 `DecisionTreeRegressor`、`X_train`、`y_train` | PNG 图像 | 样本量-得分趋势 |
| 树结构图 | `plot_tree_structure` | `model`、`feature_names` | PNG 图像 | 树的分裂结构可视化 |

### 理解重点

- 当前流水线**无标准化步骤**——决策树基于特征阈值的相对排序分裂，特征尺度不影响分裂选择。
- 学习曲线传入的是**新的** `DecisionTreeRegressor(...)` 实例，而非已训练的 `model`——因为学习曲线内部需要在不同训练子集上重新拟合。
- 树结构图是本流水线特有的可视化——大多数回归模型（线性回归、SVR）没有结构图。

## 2. 训练细节：CART 递归分裂

### 算法流程

```
从根节点开始（包含全部训练样本）
    ↓
对当前节点：
    ① 检查停止条件（深度 ≥ max_depth？样本数 < min_samples_split？）
       是 → 创建叶子节点，预测值 = 区域内样本 y 的均值
       否 → 继续
    ② 对每个特征 j：
        排序所有样本的 x_j 取值
        遍历候选分割点 s
        计算分裂后的平方误差降低：Δ = MSE(parent) - (n₁/N)·MSE(child₁) - (n₂/N)·MSE(child₂)
    ③ 选 Δ 最大的 (j*, s*)
    ④ 按 (j*, s*) 将样本分为左子节点和右子节点
    ⑤ 对左右子节点递归执行 ①-④
    ↓
达到停止条件 → 树生长完成
```

### 理解重点

- CART（Classification and Regression Tree）是二叉分裂——每个节点恰好产生两个子节点，不会同时分裂出多路。
- 平方误差降低 $\Delta$ 等价于"分裂后方差减少量"——CART 贪婪地选择每步方差降低最大的分裂。
- 停止条件是**预剪枝**（pre-pruning）——在生长过程中提前阻止，而非长成后再剪（post-pruning）。

## 3. 预测细节：从根走到叶

对测试样本 $\mathbf{x}$：

```
从根节点开始
    ↓
while 当前节点不是叶子:
    if xⱼ ≤ s:  去左子节点
    else:        去右子节点
    ↓
到达叶子 → 返回该叶子的预测值（训练时该区域样本 y 的均值）
```

### 理解重点

- 预测只需沿树走一条路径——复杂度为 $O(\text{depth})$，极快。
- 同一叶子内的所有测试样本得到完全相同的预测值——这就是"分段常数"在预测端的体现。
- 树模型天然支持缺失值处理（寻找替代分裂），但当前 California Housing 数据无缺失值，此能力未展示。

## 4. 与线性回归训练流程的对比

| 步骤 | 线性回归 | 决策树回归 |
|---|---|---|
| 数据 | 手动合成 $(200, 3)$ | **真实数据 $(20640, 8)$** |
| 标准化 | 有（`StandardScaler`） | **无** |
| 训练算法 | 闭式解（正规方程）或梯度下降 | **CART 贪心递归分裂** |
| 训练复杂度 | $O(d^3 + Nd^2)$（闭式解） | **$O(d \cdot N \log N)$** |
| 模型结构 | 系数向量 $\boldsymbol{\beta}$ | **二叉树——节点 + 阈值 + 叶子值** |
| 预测 | $\hat{y} = \mathbf{x}^T \boldsymbol{\beta}$ | **沿树走到叶子 → 返回叶子均值** |
| 预测复杂度 | $O(d)$ | **$O(\text{depth})$** |
| 评估可视化 | 残差图 + 学习曲线 | **残差图 + 特征重要性 + 学习曲线 + 树结构图** |

## 常见坑

1. 在决策树流水线中引入标准化——树模型不需要，且标准化不会改变树结构（只会改变阈值的数值表示）。
2. 忘记额外保存 `feature_names`——`feature_importances_` 只返回数值数组，没有特征名。
3. 把已训练的 `model` 直接传给学习曲线——学习曲线需要在不同子集上重新训练，必须传未训练的模型实例。

## 小结

- 决策树回归流水线为 8 步：加载 → 拆分 → 切分 → 训练 → 预测 → 残差图 → 特征重要性 → 学习曲线 → 树结构图——无标准化。
- `fit()` 的核心流程：检查停止条件 → 遍历特征和阈值 → 选平方误差降低最大的 $(j, s)$ → 递归分裂左右子节点 → 直到触达约束。
- `predict()` 极为高效——沿树走一条路径（$O(\text{depth})$），到达叶子后输出该区域的训练均值。
