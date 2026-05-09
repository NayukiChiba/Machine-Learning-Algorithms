---
title: 决策树回归 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `trainDecisionTreeRegressionModel(...)` 如何构建并训练 `DecisionTreeRegressor`。
2. 理解三个复杂度超参数（`max_depth`、`min_samples_split`、`min_samples_leaf`）的默认值与作用。
3. 看清训练完成后最重要的模型属性——`feature_importances_`、`get_depth()`、`get_n_leaves()`。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainDecisionTreeRegressionModel(...)` | 函数 | 构建并训练一个 `sklearn.tree.DecisionTreeRegressor` 模型 |
| `DecisionTreeRegressor(...)` | 类 | scikit-learn 提供的决策树回归器——CART 算法 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上递归生长决策树——每次选最优 $(j, s)$ 分裂 |
| `model.get_depth()` | 方法 | 返回当前树的最大深度——复杂度诊断 |
| `model.get_n_leaves()` | 方法 | 返回当前树的叶子节点总数——分段数量诊断 |
| `model.feature_importances_` | 属性 | 各特征在分裂中对误差降低的累积贡献 |

## 1. `trainDecisionTreeRegressionModel(...)` 的函数签名

### 参数速览

适用函数：`trainDecisionTreeRegressionModel(XTrain, yTrain, randomState=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `XTrain` | `ndarray`，形状 `(16512, 8)` | 训练特征矩阵——California Housing 的 8 个特征 | `X_train` |
| `yTrain` | `ndarray`，形状 `(16512,)` | 训练目标值——房价中位数 | `y_train` |
| `randomState` | `int` | 随机种子。`42`——保证分裂过程中的随机性可复现 | `42` |
| 返回值 | `DecisionTreeRegressor` | 已完成 `fit()` 的回归树模型 | — |

### 示例代码

```python
from src.mlAlgorithms.training.regression.regressionModels import (
    trainDecisionTreeRegressionModel,
)

model = trainDecisionTreeRegressionModel(X_train, y_train)
```

### 理解重点

- 当前训练入口返回的是**单个**决策树模型——不是集成（如随机森林、GBDT），更强调单棵树的结构可解释性。
- 函数内部不切分数据——训练/测试切分在流水线层完成，训练层只负责接收训练数据并拟合。
- `randomState=42` 保证分裂过程中的随机性可复现——虽然 CART 的 $(j, s)$ 搜索是确定性的，但某些内部实现细节涉及随机性。

## 2. `DecisionTreeRegressor` 的构造器参数

### 参数速览

适用 API：`DecisionTreeRegressor(max_depth=6, min_samples_split=6, min_samples_leaf=3, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `max_depth` | `int` | 树的最大深度。`6`——限制从根到叶子的最长路径为 6 层 | `3`、`6`、`10`、`None` |
| `min_samples_split` | `int` | 节点继续分裂所需的最小样本数。`6`——节点样本数 < 6 时停止分裂 | `2`、`6`、`20` |
| `min_samples_leaf` | `int` | 叶子节点最少样本数。`3`——分裂后任一子节点样本数 < 3 则拒绝该分裂 | `1`、`3`、`10` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| `criterion` | `str` | 分裂准则。默认 `"squared_error"`——当前源码未显式写出，使用默认值 | `"squared_error"`、`"friedman_mse"`、`"absolute_error"` |

### 示例代码

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=6,
    min_samples_leaf=3,
    random_state=42,
)
model.fit(X_train, y_train)
```

### 理解重点

- 当前源码没有显式设置 `criterion`——使用 scikit-learn 默认的 `"squared_error"`（平方误差）。这是回归树最经典的分裂准则。
- `max_depth=6` 是中等深度——在 8 维特征空间中最深可产生 $2^6 = 64$ 个叶子（实际通常远少于此，因为 `min_samples_split` 和 `min_samples_leaf` 会提前停止）。
- 与分类树的关键区别：回归树没有 `class_weight`、`criterion="gini"` 等分类特有参数。

## 3. 三个复杂度超参数的联合作用

### 理解重点

- `max_depth` 是最直观的上限约束——无论样本多充足，根到叶子的路径不超过 6 层。
- `min_samples_split` 是"是否继续切"的门槛——节点样本数太小说明已经切得够细，继续切意义不大。
- `min_samples_leaf` 是"切了之后叶子是否可靠"的门槛——如果切完后某边只剩 1-2 个样本，这个分裂的统计意义存疑。
- 三者联合作用：即使 `max_depth` 还有余量，如果样本数不满足 `min_samples_split` 或 `min_samples_leaf`，树也会提前停止。

## 4. 训练完成后的关键属性

### 参数速览

| 属性/方法 | 类型 | 含义 | 示例取值 |
|---|---|---|---|
| `get_depth()` | `int` | 实际树深度——≤ `max_depth` | `6` |
| `get_n_leaves()` | `int` | 叶子节点总数——即分段预测函数的常数段数 | `20`~`50` |
| `feature_importances_` | `ndarray`，形状 `(8,)` | 各特征在分裂中对平方误差降低的累积贡献，和为 1 | `[0.45, 0.05, 0.03, ...]` |
| `tree_` | 内部对象 | Cython 实现的树结构——含节点信息、阈值、左右子节点索引等 | — |

### 示例代码

```python
print(f"树深度: {model.get_depth()}")
print(f"叶子节点数: {model.get_n_leaves()}")
print(f"特征重要性: {model.feature_importances_}")
```

### 理解重点

- `get_depth()` 和 `get_n_leaves()` 是训练后最先关注的结构性指标——深度表示模型复杂度，叶子数表示分段数。
- `feature_importances_` 反映特征在分裂中的相对贡献——值越大，该特征在树中被用于分裂的次数越多、每次分裂的误差降低越大。
- 与线性回归的 `coef_` 有本质区别：`feature_importances_` 衡量的是"分裂贡献"而非"线性效应大小和方向"。

## 5. 决策树回归 vs 线性回归 vs SVR 模型参数对比

| 参数/属性 | 线性回归 | SVR | 决策树回归 |
|---|---|---|---|
| 核心超参数 | 无（仅 `fit_intercept`） | `C`、`epsilon`、`kernel`、`gamma` | **`max_depth`、`min_samples_split`、`min_samples_leaf`** |
| 训练输入 | `fit(X, y)` | `fit(X, y)` | `fit(X, y)` |
| 模型属性 | `coef_`、`intercept_` | `support_vectors_`、`dual_coef_` | **`feature_importances_`、`get_depth()`、`get_n_leaves()`** |
| 预测输出 | 连续值 | 连续值 | 连续值（分段常数） |
| 标准化 | 有 | 有 | **无** |
| 依赖 | sklearn 内置 | sklearn 内置 | sklearn 内置 |

## 常见坑

1. 只看 `max_depth` 不看 `min_samples_split` 和 `min_samples_leaf`——实际树结构是由三者联合决定的。
2. 把 `feature_importances_` 解读为"特征对目标的正负影响"——重要性只衡量分裂贡献，不表示影响方向。
3. 在极深树（`max_depth=None`）上期待稳定的特征重要性——树过深时重要性可能分散到多个相关特征。

## 小结

- `trainDecisionTreeRegressionModel(...)` 是本仓库决策树回归的核心训练入口——对 `DecisionTreeRegressor` 的薄封装，传递三个复杂度超参数。
- `DecisionTreeRegressor(max_depth=6, min_samples_split=6, min_samples_leaf=3)` 是当前默认配置——中等保守，在 California Housing 上平衡了灵活性与稳定性。
- 训练完成后的核心属性：`get_depth()`（复杂度）、`get_n_leaves()`（分段数）、`feature_importances_`（特征贡献）——三者构成回归树的结构诊断三件套。
