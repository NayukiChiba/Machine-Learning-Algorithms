---
title: PCA 主成分分析 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 PCA 的优化目标——最大化投影方差（寻找数据变化最大的方向）。
2. 理解协方差矩阵特征值分解与 SVD 之间的等价关系。
3. 理解 `explained_variance_ratio_` 的数学含义及其随 `n_components` 变化的单调性。
4. 理解标准化对 PCA 的数学必要性——协方差矩阵的几何意义依赖特征量纲一致。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 投影方差 $\text{Var}(\mathbf{X}\mathbf{w})$ | 优化目标 | PCA 寻找最大化投影方差的方向——方差越大，"信息量"越大 |
| 协方差矩阵 $\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ | 核心矩阵 | 其特征向量即主成分方向，特征值即该方向的方差 |
| 特征值分解 $\mathbf{S}\mathbf{u}_k = \lambda_k \mathbf{u}_k$ | 求解形式 | 第 $k$ 个主成分是第 $k$ 大特征值对应的特征向量 |
| SVD $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | 数值求解 | 右奇异向量 $\mathbf{V}$ 的列即主成分方向，$\sigma_k^2/N = \lambda_k$ |
| `explained_variance_ratio_` | 评估指标 | 第 $k$ 个主成分的解释方差占比 $\lambda_k / \sum_j \lambda_j$ |
| `svd_solver='auto'` | 工程选择 | scikit-learn 自适应选择最优 SVD 求解器 |

## 1. PCA 的优化目标

给定 $N$ 个样本 $\mathbf{x}_i \in \mathbb{R}^d$（已去均值），PCA 寻找投影方向 $\mathbf{w}$（$\|\mathbf{w}\|=1$），使得投影后数据的方差最大：

$$
\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X}\mathbf{w}) = \max_{\|\mathbf{w}\|=1} \frac{1}{N} \sum_{i=1}^{N} (\mathbf{w}^T \mathbf{x}_i)^2 = \max_{\|\mathbf{w}\|=1} \mathbf{w}^T \mathbf{S} \mathbf{w}
$$

其中 $\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ 为协方差矩阵（数据已中心化）。

### 拉格朗日推导

约束 $\mathbf{w}^T\mathbf{w} = 1$ 下最大化 $\mathbf{w}^T\mathbf{S}\mathbf{w}$：

$$
\mathcal{L} = \mathbf{w}^T\mathbf{S}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{S}\mathbf{w} - 2\lambda\mathbf{w} = 0 \quad\Rightarrow\quad \boxed{\mathbf{S}\mathbf{w} = \lambda \mathbf{w}}
$$

### 理解重点

- 主成分方向是协方差矩阵的特征向量——这不是经验规则，而是拉格朗日乘子法的严格推导结果。
- 特征值 $\lambda_k$ 恰好等于该方向的投影方差——$\text{Var}(\mathbf{X}\mathbf{u}_k) = \lambda_k$。
- 因此取最大的 $q$ 个特征值对应的特征向量，即得到保留方差最多的 $q$ 个投影方向。

## 2. 多个主成分

第一个主成分 $\mathbf{u}_1$（最大特征值 $\lambda_1$）捕获最多的方差。第二个主成分 $\mathbf{u}_2$ 在与 $\mathbf{u}_1$ 正交的约束下最大化方差（对应于第二大特征值 $\lambda_2$），以此类推。

$$
\text{Var}(\mathbf{X}\mathbf{u}_1) = \lambda_1 \geq \text{Var}(\mathbf{X}\mathbf{u}_2) = \lambda_2 \geq \dots \geq \text{Var}(\mathbf{X}\mathbf{u}_d) = \lambda_d
$$

### 解释方差比

第 $k$ 个主成分的解释方差比定义为：

$$
\text{explained\_variance\_ratio\_}[k] = \frac{\lambda_k}{\sum_{j=1}^{d} \lambda_j}
$$

### 理解重点

- 每个主成分的解释方差比反映了它在总方差中的"份额"——值越大，该方向承载的信息越多。
- 累计解释方差比 $\sum_{k=1}^{q} \lambda_k / \sum_j \lambda_j$ 反映了前 $q$ 个主成分总共保留了多少信息。
- 当前源码打印这两项——它们是 PCA 训练完成后最重要的定量输出。

## 3. SVD 与 PCA 的等价性

在实际计算中，通常不显式构造协方差矩阵 $\mathbf{S}$ 再做特征分解，而是通过 SVD 直接求解：

$$
\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T
$$

其中 $\mathbf{U}$（$N \times N$）和 $\mathbf{V}$（$d \times d$）为正交矩阵，$\boldsymbol{\Sigma}$ 为对角奇异值矩阵。代入协方差矩阵：

$$
\mathbf{S} = \frac{1}{N} \mathbf{X}^T \mathbf{X} = \frac{1}{N} \mathbf{V} \boldsymbol{\Sigma}^T \mathbf{U}^T \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T = \mathbf{V} \frac{\boldsymbol{\Sigma}^2}{N} \mathbf{V}^T
$$

因此：
- $\mathbf{V}$ 的列即主成分方向（特征向量）
- $\sigma_k^2 / N = \lambda_k$（奇异值的平方除以 $N$ 等于特征值）

### 理解重点

- SVD 路径不需要显式计算 $\mathbf{X}^T\mathbf{X}$——这在 $d$ 很大时（如 $d = 10^5$）极大降低了计算量和数值误差。
- `svd_solver='auto'` 是 scikit-learn 的默认选择——它会根据 $N$ 和 $d$ 的大小自动选择 full / randomized / arpack 中最合适的 SVD 实现。
- 这个等价性是 PCA 实现层最重要的数学性质——它把"特征分解协方差矩阵"转化为"SVD 分解数据矩阵"。

## 4. PCA 与 LDA 的数学对比

| 维度 | PCA | LDA |
|---|---|---|
| 监督方式 | 无监督 | 有监督（需要 $y$） |
| 优化目标 | $\max \mathbf{w}^T\mathbf{S}_T\mathbf{w}$（最大化投影方差） | $\max \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$（最大化类间/类内比） |
| 核心矩阵 | 总协方差矩阵 $\mathbf{S}_T$ | 类内散度 $\mathbf{S}_W$ + 类间散度 $\mathbf{S}_B$，满足 $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B$ |
| 降维上限 | $\min(d, N)$ | $K-1$ |
| 求解方式 | 特征分解或 SVD | 广义特征值问题 |
| 特征值含义 | 投影方差 | Fisher 准则值 $J(\mathbf{w})$ |

### 理解重点

- PCA 的特征值 $\lambda_k$ 是"该方向保留了多少方差"——这在物理意义上非常直观。
- LDA 的特征值是"该方向的类间/类内散度比"——同样 $[0.6, 0.4]$ 的两个数字，在 PCA 中表示方差占比，在 LDA 中表示判别能力占比。
- 两者同名属性 `explained_variance_ratio_` 在语义上是不同的——这是对比阅读时最容易混淆的地方。

## 5. 标准化对 PCA 的数学必要性

PCA 的核心操作是计算协方差矩阵：

$$
\mathbf{S} = \frac{1}{N} \mathbf{X}^T\mathbf{X}, \quad S_{ij} = \frac{1}{N}\sum_{n=1}^{N} x_{ni} x_{nj}
$$

若特征 $x_1$ 的取值量纲是 $x_2$ 的 100 倍，则 $S_{11}$ 将是 $S_{22}$ 的约 $100^2 = 10000$ 倍——协方差矩阵被尺度最大的特征完全主导。

### 理解重点

- 标准化后每个特征均值为 0、方差为 1——协方差矩阵退化为相关系数矩阵，各特征平等参与。
- 当前数据是合成数据——各特征量纲本身相近（均为高斯噪声的线性组合），但标准化仍然是最佳实践。
- 这与 LDA、KMeans、DBSCAN、SVC 的逻辑完全一致——任何基于距离或协方差的算法都需要标准化。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 中心化数据 | $\mathbf{X} - \bar{\mathbf{x}}$ | `StandardScaler` 处理后均值为 0 |
| 协方差矩阵 | $\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ | PCA 内部计算（经由 SVD 间接得到） |
| 主成分方向 | $\mathbf{u}_1, \dots, \mathbf{u}_q$ | `model.components_`，形状 `(q, d)` |
| 特征值（方差） | $\lambda_k$ | `model.explained_variance_` |
| 解释方差比 | $\lambda_k / \sum_j \lambda_j$ | `model.explained_variance_ratio_` |
| 累计解释方差 | $\sum_{k=1}^{q} \lambda_k / \sum_j \lambda_j$ | `model.explained_variance_ratio_.sum()` |
| 保留主成分数 | $q$ | `n_components=2` 或 `3` |
| SVD 求解器 | — | `svd_solver='auto'` |
| 投影 | $\mathbf{X}\mathbf{U}_q$ | `model.transform(X)` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |
| 随机种子 | — | `random_state=42` |

## 常见坑

1. 混淆 PCA 与 LDA 的优化目标——PCA 最大化投影方差（无监督），LDA 最大化类间/类内散度比（有监督）。
2. 忽略标准化——协方差矩阵对特征量纲高度敏感，不标准化的 PCA 本质上是"尺度最大的特征主导的 PCA"。
3. 把 PCA 的 `explained_variance_ratio_` 与 LDA 的同名属性当成同一种含义——前者是方差占比，后者是判别能力占比。
4. 误认为 `n_components` 加 1 必然带来显著信息增益——信息增益的边际递减率取决于数据的固有秩结构。

## 小结

- PCA 的数学核心链：中心化数据 → 协方差矩阵 $\mathbf{S}$ → 特征值问题 $\mathbf{S}\mathbf{u} = \lambda\mathbf{u}$ → 等价于 SVD $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ → 取最大 $q$ 个特征向量作为主成分方向。
- 特征值 $\lambda_k$ 就是该方向的投影方差，`explained_variance_ratio_` 就是各方向方差占总方差的比例。
- 当前源码 `PCA(n_components=2, svd_solver='auto', random_state=42)` 针对低秩合成数据（3 个真实方向 + 10 维表面特征）是展示方差压缩最经典的教学配置。
