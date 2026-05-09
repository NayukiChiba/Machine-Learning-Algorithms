---
title: 线性回归 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `trainLinearRegressionModel(...)` 如何构建并训练 `LinearRegression`——本仓库最简训练函数。
2. 理解 `coef_` 和 `intercept_` 的含义及其与真实生成公式的对照关系。
3. 看清 `feature_names` 的处理逻辑——如何让训练日志中的系数与中文列名一一对应。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainLinearRegressionModel(...)` | 函数 | 构建并训练一个 `sklearn.linear_model.LinearRegression` 模型——最简薄封装 |
| `LinearRegression()` | 类 | scikit-learn 提供的普通最小二乘线性回归器——无超参数 |
| `model.fit(X_train, y_train)` | 方法 | 基于 SVD 求解 OLS——返回 `coef_` 和 `intercept_` |
| `model.coef_` | 属性 | 各特征对应的线性系数 $\mathbf{w}$——形状 `(3,)` |
| `model.intercept_` | 属性 | 截距 $b$——标量 |

## 1. `trainLinearRegressionModel(...)` 的函数签名

### 参数速览

适用函数：`trainLinearRegressionModel(XTrain, yTrain)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `XTrain` | `ndarray`，形状 `(160, 3)` | 训练特征矩阵——面积、房间数、房龄 | `X_train` |
| `yTrain` | `ndarray`，形状 `(160,)` | 训练目标值——房价 | `y_train` |
| 返回值 | `LinearRegression` | 已完成 `fit()` 的模型对象——含 `coef_` 和 `intercept_` | — |

### 示例代码

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
# model.coef_      ≈ [2.0, 10.0, -3.0]
# model.intercept_ ≈ 50.0
```

### 理解重点

- 这是本仓库**最简训练函数**——没有超参数、没有装饰器、没有耗时统计，仅 3 行代码。
- 与决策树回归的 `trainDecisionTreeRegressionModel` 形成鲜明对比——后者有 3 个复杂度超参数。
- `LinearRegression()` 的无参设计是因为 OLS 无需调参——最优解由数据通过 SVD 唯一确定。

## 2. `LinearRegression()` 的构造器参数

### 参数速览

适用 API：`LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fit_intercept` | `bool` | 是否拟合截距。默认 `True`——当前源码未显式写出 | `True` |
| `copy_X` | `bool` | 是否复制输入数据。默认 `True` | `True` |
| `n_jobs` | `int` 或 `None` | 并行计算线程数。默认 `None`——单线程 | `None`、`-1` |

### 理解重点

- 当前源码使用全部默认参数——`LinearRegression()` 无参构造是 scikit-learn 中最简洁的模型之一。
- `fit_intercept=True` 意味着模型会学习截距 $b$——不需要手动在数据中加一列 1。
- 没有 `random_state` 参数——因为 OLS 的解是确定性的（给定相同数据，结果永远相同），不存在随机性。

## 3. 训练完成后的关键属性

### 参数速览

| 属性 | 类型 | 数学含义 | 示例取值 |
|---|---|---|---|
| `coef_` | `ndarray`，形状 `(3,)` | 系数向量 $\mathbf{w} = [w_1, w_2, w_3]$ | `[2.03, 9.87, -2.94]`（接近 $[2, 10, -3]$） |
| `intercept_` | `float` | 截距 $b$ | `51.23`（接近 $50$） |
| `rank_` | `int` | 设计矩阵 $\mathbf{X}$ 的秩 | `3`（= 特征数，满秩） |
| `singular_` | `ndarray` | $\mathbf{X}$ 的奇异值 | 内部使用——通常不需要关注 |

### 示例代码

```python
print(f"截距(intercept): {model.intercept_:.2f}")
print("斜率(coefficients):")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name}: {coef:.2f}")
```

### 输出

```text
截距(intercept): 51.23
斜率(coefficients):
  面积: 2.03
  房间数: 9.87
  房龄: -2.94
```

### 理解重点

- `coef_` 的值应与真实系数 `[2, 10, -3]` 接近——正负方向应完全一致，数值因噪声而有小幅偏差。
- `intercept_` 应接近 `50`——偏差同样来自噪声和有限样本。
- 三个系数的正负号正确（面积+、房间数+、房龄-）比数值精确更重要——方向正确说明模型学到了真实的数据模式。

## 4. `feature_names` 的处理

`trainLinearRegressionModel` 在打印系数日志时，需要特征名来提升可读性。当前源码在流水线层处理特征名：

### 示例代码

```python
feature_names = list(X.columns)  # ["面积", "房间数", "房龄"]
# 训练后将系数与特征名一一对应打印
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name}: {coef:.2f}")
```

### 理解重点

- 特征名处理在流水线层面而非训练函数内部——训练函数只关心数值矩阵，不关心列名。
- 中文列名（`面积`、`房间数`、`房龄`）在日志中直接显示——比英文列名更具可读性。
- `feature_names` 是贯穿流水线的重要中间变量——在训练日志和后续可视化的标题中都会用到。

## 5. 线性回归 vs 决策树回归 模型参数对比

| 参数/属性 | 线性回归 | 决策树回归 |
|---|---|---|
| 构造器参数 | `fit_intercept`（1 个可选） | **`max_depth`、`min_samples_split`、`min_samples_leaf`、`random_state`（4 个）** |
| 训练方式 | SVD 闭式解——确定性 | **CART 贪心递归——含随机性** |
| 核心属性 | `coef_`、`intercept_` | **`feature_importances_`、`get_depth()`、`get_n_leaves()`** |
| 属性数量 | 4（含 `rank_`、`singular_`） | **多个——`tree_` 含完整节点结构** |
| 超参数调优 | 不需要（无超参数） | **需要——深度和叶子约束直接影响泛化** |
| 训练耗时 | 极短（$O(d^3 + Nd^2)$，$d=3$ 时毫秒级） | 短（$O(d \cdot N \log N)$） |
| 预测输出 | 连续值（线性函数） | 连续值（分段常数） |

## 常见坑

1. 期待 `LinearRegression()` 有丰富的超参数——它是 scikit-learn 中最简模型之一，仅 `fit_intercept` 一个实质参数。
2. 把 `coef_` 的返回值顺序搞错——`coef_[0]` 对应 `X` 的第一列，需与 `feature_names` 对齐。
3. 忽略 `feature_names` 的作用——数组输入时日志只会显示 `Feature_0, Feature_1, ...`，丧失可读性。

## 小结

- `trainLinearRegressionModel(...)` 是本仓库最简训练函数——仅 3 行，对 `LinearRegression()` 做最薄的调用封装。
- `LinearRegression()` 使用 SVD 求解 OLS——无超参数、无随机性、确定性输出——`coef_` 和 `intercept_` 是唯一的训练结果。
- 与决策树回归的模型构建形成清晰对比：线性回归追求"极简 + 可解释"，决策树回归追求"灵活 + 需约束"。
