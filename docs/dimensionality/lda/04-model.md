---
title: LDA 线性判别分析 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LinearDiscriminantAnalysis`。
2. 理解 `LinearDiscriminantAnalysis` 的核心构造器参数（`n_components`、`solver`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`explained_variance_ratio_`（判别方向贡献）、`scalings_`（判别向量）、`means_`（类均值）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 模型，打印解释方差比日志 |
| `LinearDiscriminantAnalysis(...)` | 类 | scikit-learn 提供的线性判别分析器——求解广义特征值问题以找到判别方向 |
| `model.fit(X_train, y_train)` | 方法 | 学习判别方向——有监督，标签用于计算类均值、类内散度和类间散度 |
| `model.explained_variance_ratio_` | 属性 | 各判别方向的特征值占比——反映每个方向的相对判别能力 |
| `model.scalings_` | 属性 | 判别向量矩阵——将原始特征空间映射到判别子空间的线性变换 |
| `model.transform(X)` | 方法 | 将数据投影到判别子空间——生成降维后的坐标 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_components=2, solver='svd')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 $(178, 13)$ | 标准化后的特征矩阵，传入 `LinearDiscriminantAnalysis.fit()` | `X_scaled` |
| `y_train` | `array_like`，形状 $(178,)$ | 类别标签 $\{0, 1, 2\}$——LDA 训练必需的监督信息，用于定义类间/类内散度 | `y` |
| `n_components` | `int` | 保留的判别方向数。当前设为 `2`——$K=3$ 类数据的理论上限 | `1`、`2` |
| `solver` | `str` | 求解器。`'svd'`（默认）通过 SVD 求解，无需显式计算 $\mathbf{S}_W^{-1}$；`'eigen'` 直接特征分解；`'lsqr'` 最小二乘 | `'svd'`、`'eigen'`、`'lsqr'` |
| 返回值 | `LinearDiscriminantAnalysis` | 已完成 `fit()` 的模型对象，含 `explained_variance_ratio_`、`scalings_`、`means_` 等属性 | — |

### 示例代码

```python
from model_training.dimensionality.lda import train_model

model = train_model(X_scaled, y, n_components=2)
```

### 理解重点

- 和 PCA 分册不同，`train_model(...)` **必须有 `y_train` 参数**——LDA 是有监督降维，标签用于定义类内散度 $\mathbf{S}_W$ 和类间散度 $\mathbf{S}_B$。
- `n_components=2` 不是随意选的默认值——Wine 数据 $K=3$ 类，理论上限恰好就是 $K-1=2$。
- `train_model(...)` 是对 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 的薄封装——算法本体是 scikit-learn 基于 SVD 的高效实现。

## 2. `LinearDiscriminantAnalysis` 构造器参数

### 参数速览

适用 API：`LinearDiscriminantAnalysis(n_components=2, solver='svd')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 保留的判别方向数。必须 $\leq K-1$。当前设为 `2` | `1`、`2` |
| `solver` | `str` | 求解器选择。`'svd'`（默认）通过 SVD 求解，数值稳定性好，支持 `explained_variance_ratio_`；`'eigen'` 直接特征分解；`'lsqr'` 最小二乘，不支持 `explained_variance_ratio_` | `'svd'`、`'eigen'`、`'lsqr'` |
| `priors` | `array_like` 或 `None` | 类先验概率。`None`（默认）使用各类样本频率 $N_k/N$ | `None`、`[0.3, 0.3, 0.4]` |
| `shrinkage` | `float` 或 `str` 或 `None` | 收缩参数——仅在 `solver='lsqr'` 或 `'eigen'` 时可用，用于正则化 $\mathbf{S}_W$ 估计 | `None`、`'auto'`、`0.5` |
| `tol` | `float` | 特征值筛选的数值容忍度——仅 `solver='eigen'` 时使用。默认 `1e-4` | `1e-3`、`1e-4` |
| `covariance_estimator` | `CovarianceEstimator` 或 `None` | 协方差估计器——scikit-learn 1.2+ 新增参数 | `None` |

### 示例代码

```python
model = LinearDiscriminantAnalysis(
    n_components=2,
    solver="svd",
)
model.fit(X_train, y_train)
```

### 理解重点

- LDA 的核心参数是 `n_components`——它直接决定了降维后的维数，但受 $K-1$ 上限约束。
- `solver='svd'` 是 scikit-learn 的默认黄金标准——数值稳定、不经由显式矩阵求逆、且支持 `explained_variance_ratio_`。
- LDA 的 `fit()` 是解析求解（广义特征值分解）——与 KMeans 的迭代优化和 DBSCAN 的密度扩展在计算特征上截然不同。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `explained_variance_ratio_` | `ndarray`，形状 `(n_components,)` | $\lambda_j / \sum_i \lambda_i$ | 各判别方向的特征值占比——反映每个方向对类别分离的相对贡献。仅 `solver='svd'` 和 `'eigen'` 时可用 |
| `scalings_` | `ndarray`，形状 `(n_features, n_components)` | 判别向量 $\mathbf{w}_1, \dots, \mathbf{w}_q$ | 将 13 维特征映射到 2 维判别子空间的线性变换矩阵 |
| `means_` | `ndarray`，形状 `(n_classes, n_features)` | $\boldsymbol{\mu}_k$ | 各类在原始特征空间中的均值向量——当前为 $3 \times 13$ 矩阵 |
| `priors_` | `ndarray`，形状 `(n_classes,)` | $\pi_k = N_k/N$ | 各类的先验概率 |
| `classes_` | `ndarray`，形状 `(n_classes,)` | 类别标签 | 训练数据中出现的类别标签——当前为 `[0, 1, 2]` |
| `xbar_` | `ndarray`，形状 `(n_features,)` | $\boldsymbol{\mu}$ | 全局均值向量——$\mathbf{S}_B$ 计算的基准 |

### 示例代码

```python
print(f"n_components: {n_components}")
if hasattr(model, "explained_variance_ratio_"):
    print(f"解释方差比: {model.explained_variance_ratio_.round(4)}")
    print(f"累计解释方差: {model.explained_variance_ratio_.sum():.4f}")
```

### 理解重点

- `explained_variance_ratio_` 是理解判别方向相对重要性的关键——第一个方向通常捕获绝大部分类别分离能力。
- `scalings_` 是 LDA 区别于 PCA 的标志性属性——PCA 有 `components_`（主成分方向），LDA 有 `scalings_`（判别方向）。名称不同，数学含义也不同。
- `explained_variance_ratio_` 的条件可用性（`hasattr` 检查）反映了不同求解器的工程差异——`svd` 和 `eigen` 支持，`lsqr` 不支持。

## 4. `transform()` ：从模型训练到降维输出的桥梁

### 参数速览

适用方法：`model.transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n, 13)` | 经过同一 `scaler` 标准化后的特征矩阵 | `X_scaled` |
| 返回值 | `ndarray`，形状 `(n, 2)` | 投影到判别子空间后的坐标——$\mathbf{X} \cdot \text{scalings\_}$ | `X_transformed` |

### 示例代码

```python
X_transformed = model.transform(X_scaled)
```

### 理解重点

- `fit()` 学习判别方向（`scalings_`），`transform()` 执行投影——两者分离的设计使模型可以对新数据重复投影。
- 流水线中 `X_transformed` 是 `plot_dimensionality(...)` 的直接输入——它是训练和可视化的桥梁。
- 与 PCA 的 `transform()` 语法相同、语义不同——PCA 投影到方差最大方向，LDA 投影到类别最可分方向。

## 5. 训练阶段的工程封装

除了 `LinearDiscriminantAnalysis(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察判别方向学习耗时——通常极快（毫秒级） |
| `n_components` 日志 | 确认当前判别方向数 |
| `explained_variance_ratio_` 日志 | 若求解器支持，打印各方向解释占比和累计值 |
| `timer(...)` 上下文 | 单独测量 `fit()` 阶段的耗时 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过条件判断保护性地输出解释比例。
- `explained_variance_ratio_` 的条件打印（`hasattr` 检查）是 LDA 特有的工程边界处理——不同求解器的属性可用性不同。
- LDA 不打印簇数量（不是聚类）、不打印准确率（当前目标是降维而非分类）——与监督分类和聚类分册的输出各有侧重。

## 常见坑

1. 误以为 `train_model(...)` 不需要传 `y_train`——LDA 是有监督算法，标签是训练必需的输入。
2. 忽略 `n_components` 受 $K-1$ 约束——对 3 类数据传 `n_components=3` 会报错。
3. 把 `explained_variance_ratio_` 当成所有求解器都支持的属性——`lsqr` 求解器没有此属性。
4. 把 `scalings_` 当成 PCA 的 `components_`——两者优化目标不同，方向含义不同。

## 小结

- `train_model(...)` 是本仓库 LDA 的核心训练入口，是对 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 的薄封装。
- `LinearDiscriminantAnalysis` 的核心参数是 `n_components`（判别方向数，受 $K-1$ 约束）和 `solver`（求解路径，决定属性可用性）。
- 训练完成后的核心属性：`explained_variance_ratio_`（方向贡献）、`scalings_`（判别向量）、`means_`（类均值）——三者分别回答了"哪个方向更重要""怎么投影""各类在哪"。
- LDA 有 `scalings_`、有 `explained_variance_ratio_`、有 `transform()` 方法、标签必需参与 `fit()`——这四点构成了它与 PCA 最核心的工程差异。
