# PCA 主成分分析 (Principal Component Analysis)

## 核心思想

PCA 是一种**线性降维**方法。它寻找数据方差最大的方向（主成分），将高维数据投影到低维子空间中，在最大化保留信息的前提下压缩维度。

## 最大投影方差推导

### 中心化

设 $\mathbf{X} \in \mathbb{R}^{N \times d}$ 已中心化（即 $\sum_i \mathbf{x}_i = \mathbf{0}$）。

### 投影到单位向量

将数据投影到方向 $\mathbf{u}$（$\|\mathbf{u}\| = 1$），投影后方差为：

$$
\text{Var}(\mathbf{X}\mathbf{u}) = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{x}_i^T \mathbf{u})^2 = \mathbf{u}^T \left(\frac{1}{N}\mathbf{X}^T\mathbf{X}\right) \mathbf{u} = \mathbf{u}^T \mathbf{S} \mathbf{u}
$$

其中 $\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ 是**协方差矩阵**。

### 约束优化

最大化投影方差：

$$
\max_{\mathbf{u}} \mathbf{u}^T \mathbf{S} \mathbf{u} \quad \text{s.t.} \quad \mathbf{u}^T \mathbf{u} = 1
$$

使用拉格朗日乘子法：

$$
\mathcal{L}(\mathbf{u}, \lambda) = \mathbf{u}^T \mathbf{S} \mathbf{u} - \lambda(\mathbf{u}^T \mathbf{u} - 1)
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}} = 2\mathbf{S}\mathbf{u} - 2\lambda\mathbf{u} = 0
$$

$$
\boxed{\mathbf{S}\mathbf{u} = \lambda\mathbf{u}}
$$

这正是**特征值问题**！$\mathbf{u}$ 是 $\mathbf{S}$ 的特征向量，$\lambda$ 是对应特征值。

最大方差 = 最大特征值对应的特征向量方向。

### 多个主成分

第 $k$ 个主成分为第 $k$ 大特征值对应的特征向量。前 $q$ 个主成分张成的子空间可解释的方差比例：

$$
\text{解释方差比} = \frac{\sum_{k=1}^{q} \lambda_k}{\sum_{k=1}^{d} \lambda_k}
$$

## SVD 与 PCA 的关系

对中心化数据 $\mathbf{X}$ 做奇异值分解 $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，则：

$$
\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X} = \frac{1}{N}\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^T
$$

$\mathbf{V}$ 的列即为主成分方向，$\frac{\sigma_k^2}{N}$ 为对应特征值。SVD 在数值上比直接计算 $\mathbf{X}^T\mathbf{X}$ 更稳定。

## 代码对应

```bash
python -m pipelines.dimensionality.pca
```
