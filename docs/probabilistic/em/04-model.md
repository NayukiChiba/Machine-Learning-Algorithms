---
title: EM 与 GMM — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `GaussianMixture`——无监督，不需要 `y_train` 参数。
2. 理解 `GaussianMixture` 的核心构造器参数（`n_components`、`covariance_type`、`max_iter`）及其概率含义。
3. 看清训练完成后最重要的模型属性——`weights_`（混合权重）、`means_`（分量均值）、`covariances_`（分量协方差）、`lower_bound_`（对数似然下界）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.mixture.GaussianMixture` 模型——EM 算法 |
| `GaussianMixture(...)` | 类 | scikit-learn 提供的 GMM 实现——通过 EM 迭代估计分量参数 |
| `model.fit(X)` | 方法 | EM 迭代训练——最多 200 次 E-M 循环，收敛则提前停止 |
| `model.weights_` | 属性 | 混合权重 $\pi_k$——3 个分量的先验概率 |
| `model.means_` | 属性 | 分量均值 $\boldsymbol{\mu}_k$——3 个椭圆中心的坐标 |
| `model.covariances_` | 属性 | 分量协方差 $\boldsymbol{\Sigma}_k$——3 个椭圆的形状和方向 |
| `model.lower_bound_` | 属性 | 对数似然下界——训练收敛的诊断指标 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, n_components=3, covariance_type="full", max_iter=200, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(500, 2)` | 标准化后的特征矩阵——注意无 `y_train` 参数，EM 是无监督学习 | `X_scaled` |
| `n_components` | `int` | 高斯分量数。`3`——与真实分量数一致，是已知先验知识 | `2`、`3`、`5` |
| `covariance_type` | `str` | 协方差类型。`"full"`——每个分量有独立的完全协方差矩阵 | `"full"`、`"tied"`、`"diag"`、`"spherical"` |
| `max_iter` | `int` | EM 最大迭代次数。`200`——足够 2 维 3 分量数据收敛 | `100`、`200`、`500` |
| `random_state` | `int` | 随机种子，保证初始化可复现。`42` | `42` |
| 返回值 | `GaussianMixture` | 已完成 `fit()` 的模型对象——含 `weights_`、`means_`、`covariances_` | — |

### 示例代码

```python
from model_training.probabilistic.em import train_model

model = train_model(X_scaled)
```

### 理解重点

- `train_model(...)` 是**无监督训练**——没有 `y_train` 参数。这是 EM 与集成分类模型（Bagging/GBDT）的最根本差异。
- `n_components=3` 需要作为先验知识给定——与 KMeans 的 `n_clusters=3` 相同。在实际应用中，$K$ 需要通过 BIC 或交叉验证选择。
- `covariance_type="full"` 是 GMM 最灵活的配置——允许 3 个分量各自由学习形状。

## 2. `GaussianMixture` 构造器参数

### 参数速览

适用 API：`GaussianMixture(n_components=3, covariance_type="full", max_iter=200, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 高斯分量数。默认 `3`——EM 算法的 $K$ 值，需预先指定 | `2`、`3`、`5` |
| `covariance_type` | `str` | 协方差约束类型。`"full"`——每个分量有独立的完全协方差矩阵 | `"full"`、`"tied"`、`"diag"`、`"spherical"` |
| `tol` | `float` | 收敛阈值。默认 `1e-3`——对数似然变化小于此值则停止 | `1e-3`、`1e-4` |
| `max_iter` | `int` | EM 最大迭代次数。`200`——安全上限，2 维 3 分量数据通常在 100 次内收敛 | `100`、`200`、`500` |
| `n_init` | `int` | 随机初始化的次数。默认 `1`——只用一次 k-means 初始化 | `1`、`5`、`10` |
| `init_params` | `str` | 初始化方法。默认 `"kmeans"`——先用 KMeans 聚类作为初始参数 | `"kmeans"`、`"random"` |
| `reg_covar` | `float` | 协方差对角线的非负正则化。默认 `1e-6`——防止协方差奇异 | `1e-6`、`1e-4` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| `verbose` | `int` | 日志级别。默认 `0` | `0`、`1`、`2` |

### 示例代码

```python
from sklearn.mixture import GaussianMixture

model = GaussianMixture(
    n_components=3,
    covariance_type="full",
    max_iter=200,
    random_state=42,
)
model.fit(X_scaled)
```

### 理解重点

- `covariance_type` 是 GMM 最关键的选择——`full` 参数最多但最灵活；`spherical` 参数最少但退化为 KMeans-like。
- `init_params="kmeans"`（默认）使用 KMeans 聚类结果初始化 GMM 均值和协方差——这提供了一个"合理的起点"。
- `reg_covar=1e-6` 是一个数值稳定技巧——在协方差对角线上加一个小值，防止矩阵奇异。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `weights_` | `ndarray`，形状 `(3,)` | 混合权重 $\pi_k$ | 3 个分量的先验概率——和为 1，反映各分量的体量 |
| `means_` | `ndarray`，形状 `(3, 2)` | 分量均值 $\boldsymbol{\mu}_k$ | 3 个椭圆中心在 2 维空间中的坐标 |
| `covariances_` | `ndarray`，形状 `(3, 2, 2)` | 分量协方差 $\boldsymbol{\Sigma}_k$ | 3 个 2×2 协方差矩阵——描述各椭圆的形状和方向 |
| `lower_bound_` | `float` | 对数似然下界 $\log p(\mathbf{X} \mid \Theta)$ | 收敛时当前参数下的数据对数似然 |
| `converged_` | `bool` | EM 是否收敛 | `True` 表示在 `max_iter` 内达到容差收敛 |
| `n_iter_` | `int` | 实际 EM 迭代次数 | 可能小于 `max_iter`（提前收敛） |

### 示例代码

```python
print(f"n_components: {n_components}")
print(f"covariance_type: {covariance_type}")
print(f"log-likelihood: {model.lower_bound_:.4f}")
print(f"混合权重: {model.weights_}")
print(f"分量均值:\n{model.means_}")
print(f"是否收敛: {model.converged_}")
```

### 理解重点

- `weights_`、`means_`、`covariances_` 是 GMM 的"三件套"——完全描述了 $K$ 个高斯分量的混合模型。
- `lower_bound_` 是 EM 训练的诊断指标——值越大（越接近 0），模型对数据的拟合越好。
- `covariances_` 的 `(3, 2, 2)` 形状——3 个分量，每个有一个 $2 \times 2$ 协方差矩阵。

## 4. `predict()` 与 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 2)` | `ndarray`，形状 `(n,)`，取值 $\{0, 1, 2\}$ | 硬预测——取 $\gamma_{ik}$ 最大的分量索引 |
| `predict_proba(X)` | `array_like`，形状 `(n, 2)` | `ndarray`，形状 `(n, 3)` | 软预测——每个样本对 3 个分量的归属概率，行和为 1.0 |

### 理解重点

- `predict()` 是对软赋值做硬截断——等价于 $\arg\max_k \gamma_{ik}$。与 KMeans 的 `predict()` 输出格式一致。
- `predict_proba()` 是 GMM 独有的输出——直接返回 E 步计算的责任矩阵 $\gamma_{ik}$，提供了归属不确定性信息。
- 与集成分类的 `predict_proba` 不同——这里的概率是"属于哪个高斯分量"，不是"属于哪个类别"。

## 常见坑

1. 在 `n_components` 未知的情况下盲目设定——GMM 需要预先知道分量数，可借助 BIC/AIC 选择。
2. 混淆 `predict()` 和 `predict_proba()` 的输出——前者是硬标签（与 KMeans 相同），后者是软归属（GMM 独有）。
3. 在 `covariance_type="spherical"` 下期待椭圆簇——此时 GMM 退化为概率版 KMeans。
4. 忽略 `lower_bound_` 的符号——它是对数似然，始终为负数（密度小于 1），越接近 0 拟合越好。

## 小结

- `train_model(...)` 是本仓库 EM 的核心训练入口——是对 `sklearn.mixture.GaussianMixture` 的薄封装，无监督（无 `y_train`）。
- `GaussianMixture` 的核心参数是 `n_components`（分量数）、`covariance_type`（协方差约束）、`max_iter`（迭代上限）——三者共同决定模型的灵活性和收敛行为。
- 训练完成后的核心属性：`weights_` / `means_` / `covariances_`（三件套描述 GMM）和 `lower_bound_`（收敛诊断）——构成了完整的概率模型描述。
