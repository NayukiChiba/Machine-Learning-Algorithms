# LDA 线性判别分析 (Linear Discriminant Analysis)

## 核心思想

LDA 是一种**有监督降维**方法。它寻找一个投影方向，使得**类间散度最大、类内散度最小**，从而在降维的同时最大化类别可分性。

## Fisher 判别准则

### 二分类情形

将数据投影到方向 $\mathbf{w}$ 上后，第 $k$ 类的投影均值和投影方差为：

$$
\tilde{\mu}_k = \mathbf{w}^T \boldsymbol{\mu}_k, \quad \tilde{\sigma}_k^2 = \mathbf{w}^T \mathbf{S}_k \mathbf{w}
$$

Fisher 准则最大化**类间距离与类内方差之比**：

$$
J(\mathbf{w}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{\sigma}_1^2 + \tilde{\sigma}_2^2} = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}
$$

### 散度矩阵

**类内散度矩阵** (Within-Class Scatter)：

$$
\mathbf{S}_W = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T
$$

**类间散度矩阵** (Between-Class Scatter)：

$$
\mathbf{S}_B = \sum_{k=1}^{K} N_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T
$$

其中 $\boldsymbol{\mu}$ 为全局均值，$N_k$ 为第 $k$ 类样本数。

## 瑞利商最优化

$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$ 是**广义瑞利商** (Generalized Rayleigh Quotient)。

### 推导

令 $\mathbf{w}^T \mathbf{S}_W \mathbf{w} = 1$（归一化约束），用拉格朗日乘子法：

$$
\mathcal{L} = \mathbf{w}^T \mathbf{S}_B \mathbf{w} - \lambda(\mathbf{w}^T \mathbf{S}_W \mathbf{w} - 1)
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{S}_B \mathbf{w} - 2\lambda \mathbf{S}_W \mathbf{w} = 0
$$

$$
\boxed{\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}}
$$

即广义特征值问题。若 $\mathbf{S}_W$ 可逆：

$$
\mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}
$$

### 二分类闭式解

$\text{rank}(\mathbf{S}_B) = 1$，$\mathbf{S}_B \mathbf{w} \propto (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$，因此：

$$
\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
$$

### 多分类推广

$K$ 个类最多可降至 $K-1$ 维（因为 $\text{rank}(\mathbf{S}_B) \leq K-1$）。选取 $\mathbf{S}_W^{-1}\mathbf{S}_B$ 最大的 $q$ 个特征值对应的特征向量。

## PCA vs LDA

| 特性 | PCA | LDA |
|------|-----|-----|
| 监督方式 | 无监督 | 有监督 |
| 优化目标 | 最大投影方差 | 最大类间/类内散度比 |
| 降维上限 | $\min(N, d)$ | $K - 1$ |
| 适用场景 | 数据压缩、可视化 | 分类预处理 |

## 代码对应

```bash
python -m pipelines.dimensionality.lda
```
