---
title: LDA 线性判别分析 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 LDA 的优化目标——最大化类间散度与类内散度之比（Fisher 判别准则）。
2. 理解类内散度矩阵 $\mathbf{S}_W$ 和类间散度矩阵 $\mathbf{S}_B$ 的构造与含义。
3. 理解广义特征值问题 $\mathbf{S}_B\mathbf{w} = \lambda \mathbf{S}_W\mathbf{w}$ 如何给出判别方向，以及 $K-1$ 维上限的秩论证。
4. 理解 `solver='svd'` 的求解路径，以及与 PCA 在优化目标上的根本区别。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 类内散度矩阵 $\mathbf{S}_W$ | 数学对象 | 衡量同一类别内部样本围绕类均值的分散程度——LDA 希望将其最小化 |
| 类间散度矩阵 $\mathbf{S}_B$ | 数学对象 | 衡量各类别均值围绕全局均值的分散程度——LDA 希望将其最大化 |
| Fisher 判别准则 $J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$ | 优化目标 | 广义瑞利商——同时最大化类间分离度与最小化类内散布 |
| 广义特征值问题 $\mathbf{S}_B\mathbf{w} = \lambda \mathbf{S}_W\mathbf{w}$ | 求解形式 | 判别方向是 $\mathbf{S}_W^{-1}\mathbf{S}_B$ 的特征向量，对应最大的 $q$ 个特征值 |
| $K-1$ 维上限 | 理论约束 | $\text{rank}(\mathbf{S}_B) \leq K-1$，因此 $K$ 类 LDA 最多提取 $K-1$ 个判别方向 |
| `solver='svd'` | 工程求解 | 通过 SVD 分解直接求解，无需显式计算 $\mathbf{S}_W^{-1}$，数值稳定性更好 |

## 1. LDA 的优化目标

给定 $N$ 个样本 $\mathbf{x}_i \in \mathbb{R}^d$ 及其类别标签 $y_i \in \{1, \dots, K\}$，LDA 寻找投影方向 $\mathbf{w}$，使得投影后类间散度最大、类内散度最小。

### 二分类 Fisher 准则

将数据投影到方向 $\mathbf{w}$ 后，第 $k$ 类的投影均值和投影散度为：

$$
\tilde{\mu}_k = \mathbf{w}^T \boldsymbol{\mu}_k, \quad \tilde{s}_k^2 = \sum_{\mathbf{x}_i \in C_k} (\mathbf{w}^T \mathbf{x}_i - \tilde{\mu}_k)^2 = \mathbf{w}^T \mathbf{S}_k \mathbf{w}
$$

Fisher 准则定义为类间距离与类内散度之比：

$$
J(\mathbf{w}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2} = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}
$$

### 理解重点

- 分子是类间距离的平方——不同类别的投影中心应尽可能远离。
- 分母是各类内部的投影散度之和——同类别样本投影后应尽可能聚集。
- 这个比值越大，方向 $\mathbf{w}$ 的判别能力越强。当前源码中 `LinearDiscriminantAnalysis` 寻找的就是最大化此比值的 $\mathbf{w}$。

## 2. 散度矩阵

### 类内散度矩阵（Within-Class Scatter Matrix）

$$
\mathbf{S}_W = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T
$$

其中 $\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$ 为第 $k$ 类的均值向量，$C_k$ 为第 $k$ 类的样本集合。

### 类间散度矩阵（Between-Class Scatter Matrix）

$$
\mathbf{S}_B = \sum_{k=1}^{K} N_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T
$$

其中 $\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$ 为全局均值向量，$N_k = |C_k|$ 为第 $k$ 类样本数。

### 理解重点

- $\mathbf{S}_W$ 汇总了各类内部的协方差结构——它是 $K$ 个类内协方差矩阵的加权和（在等协方差假设下，各类协方差相同）。
- $\mathbf{S}_B$ 汇总了类中心之间的方差——它的秩不超过 $K-1$，因为 $K$ 个类中心满足一个线性关系（均值的加权平均等于全局均值）。
- 当前 Wine 数据集（$d=13, K=3$）下，$\mathbf{S}_W$ 是 $13 \times 13$ 矩阵，$\mathbf{S}_B$ 的秩为 2。

## 3. 广义瑞利商与广义特征值问题

### 从优化到特征值问题

最大化 $J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$ 等价于求解广义特征值问题：

$$
\boxed{\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}}
$$

若 $\mathbf{S}_W$ 可逆，即化为标准特征值问题：

$$
\mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}
$$

判别方向取 $\mathbf{S}_W^{-1}\mathbf{S}_B$ 最大的 $q$ 个特征值对应的特征向量，其中 $q \leq K-1$。

### 拉格朗日推导

以 $\mathbf{w}^T \mathbf{S}_W \mathbf{w} = 1$ 为约束，最大化 $\mathbf{w}^T \mathbf{S}_B \mathbf{w}$：

$$
\mathcal{L} = \mathbf{w}^T \mathbf{S}_B \mathbf{w} - \lambda(\mathbf{w}^T \mathbf{S}_W \mathbf{w} - 1)
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{S}_B \mathbf{w} - 2\lambda \mathbf{S}_W \mathbf{w} = 0 \quad\Rightarrow\quad \mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}
$$

### 理解重点

- 这不是经验规则——判别方向是严格优化问题的解析结果。
- 特征值 $\lambda$ 就是该判别方向对应的 Fisher 准则值 $J(\mathbf{w})$——特征值越大，该方向的判别能力越强。
- `explained_variance_ratio_` 就是各特征值占总特征值之和的比例。

## 4. 二分类闭式解

二分类时 $\text{rank}(\mathbf{S}_B) = 1$，存在闭式解：

$$
\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
$$

### 理解重点

- 二分类 LDA 的判别方向非常直观：就是"类中心差异"经"类内协方差结构"修正后的方向。
- $\mathbf{S}_W^{-1}$ 的作用是白化——消除特征间的相关性，使几何距离在各方向上等价。
- 这是理解多分类 LDA 的最佳起点：多分类只是将"一对多"的概念推广到多个判别方向。

## 5. 多分类推广与 $K-1$ 维上限

### 秩论证

$\mathbf{S}_B$ 是 $K$ 个秩-1 外积 $\{(\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T\}_{k=1}^{K}$ 的加权和，且 $\sum_k N_k (\boldsymbol{\mu}_k - \boldsymbol{\mu}) = \mathbf{0}$，因此：

$$
\text{rank}(\mathbf{S}_B) \leq K - 1
$$

进而 $\text{rank}(\mathbf{S}_W^{-1}\mathbf{S}_B) \leq K-1$，最多只有 $K-1$ 个非零特征值。

### 对当前源码的影响

| 类别数 $K$ | 最大判别方向数 $K-1$ | 当前 `n_components` |
|---|---|---|
| 2 | 1 | — |
| 3（Wine） | 2 | `2`（达理论上限） |
| 4 | 3 | — |
| 10 | 9 | — |

### 理解重点

- 当前 Wine 数据（$K=3$）下 `n_components=2` 不是随意选择——它恰好达到理论上限。
- 这是区分 LDA 与 PCA 的核心数学特征之一：PCA 维数无类别限制，LDA 受 $K-1$ 约束。
- 当前流水线只输出 2D 图（不输出 3D 图）的数学根源即在于此。

## 6. SVD 求解器

当前源码使用 `solver='svd'`，其求解路径为：

1. 对类内散度 $\mathbf{S}_W$ 做 Cholesky 分解或直接取逆的平方根
2. 将广义特征值问题转化为普通特征值问题
3. 通过 SVD 求解，避免显式计算 $\mathbf{S}_W^{-1}\mathbf{S}_B$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `solver` | `str` | 求解器选择。`'svd'`（默认）通过 SVD 求解，无需显式计算散度矩阵逆；`'eigen'` 直接做特征分解；`'lsqr'` 使用最小二乘 | `'svd'`、`'eigen'`、`'lsqr'` |

### 理解重点

- `solver='svd'` 是 scikit-learn 的默认选择——数值稳定性好，且不要求 $\mathbf{S}_W$ 满秩。
- 不同求解器最显著的工程差异是 `explained_variance_ratio_` 是否可用——`'svd'` 支持此属性，`'lsqr'` 不支持。
- 当前源码用 `hasattr(model, "explained_variance_ratio_")` 做保护式输出，正因求解器差异。

## 7. LDA 与 PCA 的数学对比

| 维度 | PCA | LDA |
|---|---|---|
| 监督方式 | 无监督 | 有监督（需要 $y$） |
| 优化目标 | $\max \mathbf{w}^T \mathbf{S}_T \mathbf{w}$（最大化投影方差） | $\max \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$（最大化类间/类内比） |
| 核心矩阵 | 总散度矩阵 $\mathbf{S}_T = \sum_i (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T$ | $\mathbf{S}_W$ 和 $\mathbf{S}_B$，满足 $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B$ |
| 降维上限 | 最多 $\min(d, N)$ 维 | 最多 $K-1$ 维 |
| 标签参与 | 否 | 是（定义类结构） |
| 适用场景 | 数据压缩、无监督可视化、去噪 | 分类预处理、判别式降维、特征提取 |

### 理解重点

- $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B$ 表明总方差可分解为"类内方差 + 类间方差"——PCA 最大化总方差，LDA 最大化类间方差占类内方差的比例。
- 这是两者数学本质差异的集中体现：PCA 不关心类别，LDA 以类别为核心。

## 8. 标准化对 LDA 的数学必要性

LDA 的核心操作涉及散度矩阵的计算：

$$
\mathbf{S}_W = \sum_k \sum_{\mathbf{x} \in C_k} (\mathbf{x} - \boldsymbol{\mu}_k)(\mathbf{x} - \boldsymbol{\mu}_k)^T
$$

若特征 $x_1$ 的量纲是 $x_2$ 的 100 倍，则 $x_1$ 方向的方差将主导散度矩阵——判别方向被尺度最大的特征绑架。

### 理解重点

- 标准化后每个特征对散度矩阵的贡献均等——判别方向反映真实的类别可分性结构。
- Wine 数据集中 `alcohol`（~13）和 `proline`（~746）的数值范围差异巨大，不标准化将导致 `proline` 主导全部判别方向。
- 这与此前所有基于距离/散度的算法（KMeans、DBSCAN、SVC）的逻辑完全一致——标准化是几何意义的前置条件。

## 9. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 类内散度矩阵 | $\mathbf{S}_W$ | `LinearDiscriminantAnalysis` 内部计算 |
| 类间散度矩阵 | $\mathbf{S}_B$ | `LinearDiscriminantAnalysis` 内部计算 |
| Fisher 判别准则 | $J(\mathbf{w}) = \mathbf{w}^T\mathbf{S}_B\mathbf{w} / \mathbf{w}^T\mathbf{S}_W\mathbf{w}$ | LDA 优化核心 |
| 广义特征值问题 | $\mathbf{S}_B\mathbf{w} = \lambda \mathbf{S}_W\mathbf{w}$ | `solver` 内部求解 |
| 判别方向数 | $q \leq K-1$ | `n_components=2` |
| 求解器 | — | `solver='svd'` |
| 判别方向 | $\mathbf{w}_1, \dots, \mathbf{w}_q$ | `model.scalings_` |
| 解释方差比 | $\lambda_j / \sum_i \lambda_i$ | `model.explained_variance_ratio_`（若 solver 支持） |
| 类均值 | $\boldsymbol{\mu}_k$ | `model.means_` |
| 先验概率 | $\pi_k = N_k / N$ | `model.priors_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |
| 投影 | $\mathbf{X}\mathbf{W}$ | `model.transform(X)` |

## 常见坑

1. 混淆 PCA 与 LDA 的优化目标——PCA 最大化投影方差（无监督），LDA 最大化类间/类内散度比（有监督）。
2. 忽略 $K-1$ 维上限，误以为 LDA 可以像 PCA 一样自由增加输出维度。
3. 把 `explained_variance_ratio_` 当成所有求解器都支持的属性——`lsqr` 求解器不提供此属性。
4. 在不标准化的数据上运行——不同量纲的特征绑架散度矩阵计算。

## 小结

- LDA 的数学核心链：类内/类间散度矩阵 $\mathbf{S}_W, \mathbf{S}_B$ → Fisher 准则 $\max \mathbf{w}^T\mathbf{S}_B\mathbf{w} / \mathbf{w}^T\mathbf{S}_W\mathbf{w}$ → 广义特征值问题 $\mathbf{S}_B\mathbf{w} = \lambda \mathbf{S}_W\mathbf{w}$ → $\mathbf{S}_W^{-1}\mathbf{S}_B$ 特征分解 → 取最大 $q \leq K-1$ 个特征向量作为判别方向。
- $K-1$ 维上限来自 $\text{rank}(\mathbf{S}_B) \leq K-1$ 的秩论证——这是 LDA 区别于 PCA 最核心的数学约束。
- 当前源码 `LinearDiscriminantAnalysis(n_components=2, solver='svd')` 针对 Wine 数据（$K=3$）是最经典的监督降维配置。
