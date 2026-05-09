---
title: 正则化回归 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 理解 `trainRegularizationModels(...)` 如何一次性构建并训练三个正则化模型。
2. 理解 Lasso、Ridge、ElasticNet 的超参数及其默认值的选取理由。
3. 理解 `coef_`、`intercept_` 和近零系数计数在模型构建层的角色。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainRegularizationModels(...)` | 函数 | 构建并训练 Lasso、Ridge、ElasticNet——返回 `dict[str, 模型]` |
| `Lasso(alpha=0.15)` | 类 | L1 正则化线性模型——坐标下降求解 |
| `Ridge(alpha=2.0)` | 类 | L2 正则化线性模型——闭式解 |
| `ElasticNet(alpha=0.2, l1_ratio=0.5)` | 类 | L1+L2 混合正则化——坐标下降求解 |
| `model.coef_` | 属性 | 21 维系数向量——正则化的核心输出 |
| `np.sum(np.abs(coef) < 1e-3)` | 派生量 | 近零系数计数——量化稀疏化程度 |

## 1. `trainRegularizationModels(...)` 的函数签名

### 参数速览

适用函数：`trainRegularizationModels(XTrain, yTrain, randomState=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `XTrain` | `ndarray`，形状 `(353, 21)` | 标准化后的训练特征矩阵 | `X_train_s` |
| `yTrain` | `ndarray`，形状 `(353,)` | 训练目标值 | `y_train` |
| `randomState` | `int` | 随机种子——保证坐标下降可复现 | `42` |
| 返回值 | `dict[str, 模型]` | `{"lasso": Lasso, "ridge": Ridge, "elasticnet": ElasticNet}` | — |

### 示例代码

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

def trainRegularizationModels(XTrain, yTrain, randomState: int = 42):
    models = {
        "lasso": Lasso(alpha=0.15, max_iter=10000, random_state=randomState),
        "ridge": Ridge(alpha=2.0, random_state=randomState),
        "elasticnet": ElasticNet(
            alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=randomState
        ),
    }
    for model in models.values():
        model.fit(XTrain, yTrain)
    return models
```

### 理解重点

- 这是本仓库**唯一返回多个模型的训练函数**——其他回归训练函数（线性回归、决策树、SVR）都只返回单个模型。
- 三个模型共享同一份标准化后的训练数据——确保对比的公平性。
- 函数签名比 `trainLinearRegressionModel` 多了 `randomState` 参数——Lasso 和 ElasticNet 的坐标下降涉及随机性。

## 2. Lasso 的构造器参数

### 参数速览

适用 API：`Lasso(alpha=0.15, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | L1 正则化强度——越大清零越激进 | `0.15` |
| `max_iter` | `int` | 坐标下降最大迭代次数——防止未收敛 | `10000` |
| `random_state` | `int` | 坐标下降的随机种子——保证可复现 | `42` |
| `fit_intercept` | `bool` | 是否拟合截距——默认 `True` | `True`（默认） |
| `coef_` | `ndarray`，形状 `(21,)` | 训练后的系数向量——部分可能精确为零 | — |

### 理解重点

- `alpha=0.15` 是经过调试的取值——在 diabetes 数据上既能展示稀疏化，又不至于把所有系数清零。
- `max_iter=10000` 远大于默认值（1000）——确保在 21 维特征 + 坐标下降下充分收敛。
- Lasso 没有 `l1_ratio` 参数——它是纯 L1 惩罚，与 ElasticNet 不同。

## 3. Ridge 的构造器参数

### 参数速览

适用 API：`Ridge(alpha=2.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | L2 正则化强度——越大收缩越狠 | `2.0` |
| `random_state` | `int` | 随机种子——Ridge 使用闭式解，此参数仅影响特定求解器 | `42` |
| `fit_intercept` | `bool` | 是否拟合截距——默认 `True` | `True`（默认） |
| `solver` | `str` | 求解器——默认 `'auto'`，根据数据自动选择 | `'auto'`（默认） |

### 理解重点

- `alpha=2.0` 明显大于 Lasso 的 `0.15`——Ridge 使用平方惩罚，需要更大的 α 才能产生等量的收缩效果。
- Ridge 有闭式解 $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$——不需要 `max_iter`。
- Ridge 的 `random_state` 仅在 `solver='sag'` 或 `'saga'` 时生效——当前使用默认 `'auto'`，通常选择闭式求解器。

## 4. ElasticNet 的构造器参数

### 参数速览

适用 API：`ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | 总正则化强度——越大惩罚越重 | `0.2` |
| `l1_ratio` | `float` | L1 占比——0 = 纯 Ridge，1 = 纯 Lasso | `0.5` |
| `max_iter` | `int` | 坐标下降最大迭代次数 | `10000` |
| `random_state` | `int` | 坐标下降的随机种子 | `42` |
| `fit_intercept` | `bool` | 是否拟合截距——默认 `True` | `True`（默认） |

### 理解重点

- `l1_ratio=0.5` 使 ElasticNet 处于 L1 和 L2 的正中间——不是极端偏向任一侧。
- `alpha=0.2` 介于 Lasso（0.15）和 Ridge（2.0）之间——总强度适中。
- 坐标下降同时处理 L1 和 L2 分量——`max_iter=10000` 与 Lasso 一致，确保充分收敛。
- ElasticNet 的关键优势：在 `bmi`/`bmi_corr` 这种共线特征对上，比纯 Lasso 更稳定（L2 分量分摊权重），比纯 Ridge 更稀疏（L1 分量清零噪声）。

## 5. 训练后的关键属性

### 参数速览

| 属性 | 类型 | Lasso | Ridge | ElasticNet |
|---|---|---|---|---|
| `coef_` | `ndarray(21,)` | 部分系数精确为 0 | 所有系数非零但收缩 | 介于两者之间 |
| `intercept_` | `float` | 截距项 | 截距项 | 截距项 |
| `alpha` | `float` | `0.15` | `2.0` | `0.2` |
| `l1_ratio` | `float` | —（无此属性） | —（无此属性） | `0.5` |
| `n_iter_` | `int` | 实际迭代次数 | —（闭式解） | 实际迭代次数 |

### 理解重点

- `coef_` 是正则化回归最重要的输出——不仅是预测参数，更是特征选择的直接证据。
- Lasso 的 `coef_` 中精确为零的位置对应被淘汰的特征——这是 L1 正则化的核心价值。
- ElasticNet 的 `coef_` 中零的数量取决于 `l1_ratio`——越接近 1 越像 Lasso，越接近 0 越像 Ridge。
- `intercept_` 不受正则化惩罚——只有系数向量被惩罚，截距项始终自由。

## 6. 正则化回归 vs 线性回归 vs 决策树回归 模型构建对比

| 模型维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 模型类 | `LinearRegression` | `DecisionTreeRegressor` | **`Lasso` / `Ridge` / `ElasticNet`** |
| 训练函数 | `trainLinearRegressionModel` | `trainDecisionTreeRegressionModel` | **`trainRegularizationModels`** |
| 返回值 | 单个模型 | 单个模型 | **`dict`——三个模型** |
| 超参数数 | 0 | 3 | **Lasso: 1, Ridge: 1, EN: 2** |
| `random_state` | 不需要 | 需要 | **需要（Lasso/EN）** |
| 核心属性 | `coef_`, `intercept_` | `feature_importances_`, `tree_` | **`coef_`, `intercept_` + 近零计数** |
| 训练方式 | SVD 闭式解 | CART 贪心递归 | **坐标下降（Lasso/EN）/ 闭式解（Ridge）** |
| 是否需要标准化 | 否 | 否 | **是——强制要求** |

## 常见坑

1. 误以为 `trainRegularizationModels` 只训练一个模型——它返回 `dict`，是三个模型的容器。
2. 将三个模型的 `alpha` 值直接比较大小——Lasso 的 0.15 和 Ridge 的 2.0 产生不同的惩罚效果（L1 vs L2），不能直接比较数值。
3. 期待 Ridge 也有 `max_iter` 参数——Ridge 使用闭式解，不需要迭代。
4. 忘记 ElasticNet 有 `l1_ratio` 而 Lasso 没有——Lasso 是纯 L1，不需要混合比例。

## 小结

- `trainRegularizationModels(...)` 是本仓库唯一返回多模型的训练函数——一次性构建 Lasso、Ridge、ElasticNet 三个模型。
- 三个模型的超参数各有侧重：Lasso 强调稀疏化（α=0.15），Ridge 强调收缩（α=2.0），ElasticNet 折中（α=0.2, l1_ratio=0.5）。
- `coef_` 不仅是预测参数，更是正则化回归的"成绩单"——观测系数结构比观测预测分数更重要。
- 标准化是模型构建的隐含前提——`trainRegularizationModels` 的输入必须是标准化后的数据。
