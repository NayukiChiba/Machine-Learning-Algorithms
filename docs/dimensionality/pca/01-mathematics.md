---
title: PCA — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/dimensionality/pca.py`
>  
> 相关对象：`PCA`、`train_model(...)`

## 本章目标

1. 理解 PCA 为什么可以形式化为“最大化投影方差”的优化问题。
2. 理解协方差矩阵特征值问题和 SVD 与 PCA 之间的关系。
3. 把这些数学表达和当前源码中的 `n_components`、`explained_variance_ratio_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 协方差矩阵 | 数学对象 | 描述特征之间的方差与相关结构 |
| 主成分方向 | 优化结果 | 使投影方差最大的单位方向 |
| 特征值问题 | 数学形式 | 给出主成分与解释方差的解析结构 |
| SVD | 数值分解工具 | 更稳定地求解 PCA |
| `explained_variance_ratio_` | 源码属性 | 各主成分解释方差占比的直接输出 |

## 1. 核心思想

PCA 是一种线性降维方法。它寻找数据方差最大的方向（主成分），将高维数据投影到低维子空间中，在最大化保留信息的前提下压缩维度。

### 理解重点

- 当前源码中的 `PCA(...)`，本质上就是在求这些主成分方向。\n+- 文档里的“解释方差比”与“累计解释方差”，在数学上都来自主成分对应的特征值大小。\n+- 这也是为什么 PCA 分册的数学核心不是损失函数，而是方差与投影结构。

## 2. 最大投影方差推导

### 中心化

设 $\mathbf{X} \in \mathbb{R}^{N \times d}$ 已中心化（即 $\sum_i \mathbf{x}_i = \mathbf{0}$）。

### 投影到单位向量

将数据投影到方向 $\mathbf{u}$（$\|\mathbf{u}\| = 1$），投影后方差为：

$$
\text{Var}(\mathbf{X}\mathbf{u}) = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{x}_i^T \mathbf{u})^2 = \mathbf{u}^T \left(\frac{1}{N}\mathbf{X}^T\mathbf{X}\right) \mathbf{u} = \mathbf{u}^T \mathbf{S} \mathbf{u}
$$

其中 $\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ 是协方差矩阵。

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

这正是特征值问题。$\mathbf{u}$ 是 $\mathbf{S}$ 的特征向量，$\lambda$ 是对应特征值。

最大方差对应最大特征值的特征向量方向。

### 理解重点

- PCA 的关键结果不是拍脑袋找到的，而是一个严格的约束优化问题。\n+- 当前训练模型学习到的主成分方向，本质上就是协方差矩阵的主特征向量。\n+- 这也解释了为什么主成分有明确的排序：特征值越大，说明该方向解释的方差越多。

## 3. 多个主成分

第 $k$ 个主成分为第 $k$ 大特征值对应的特征向量。前 $q$ 个主成分张成的子空间可解释的方差比例为：

$$
\text{解释方差比} = \frac{\sum_{k=1}^{q} \lambda_k}{\sum_{k=1}^{d} \lambda_k}
$$

### 理解重点

- 这条公式直接对应当前源码里的 `explained_variance_ratio_` 和累计解释方差。\n+- 当你保留的主成分数量越多，累计解释方差通常越高。\n+- 当前流水线同时训练 2D 和 3D PCA，本质上就是在比较不同 $q$ 下的信息保留量。

## 4. SVD 与 PCA 的关系

对中心化数据 $\mathbf{X}$ 做奇异值分解：

$$
\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T
$$

则：

$$
\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X} = \frac{1}{N}\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^T
$$

$\mathbf{V}$ 的列即为主成分方向，$\frac{\sigma_k^2}{N}$ 为对应特征值。SVD 在数值上比直接计算 $\mathbf{X}^T\mathbf{X}$ 更稳定。

### 理解重点

- 数学上 PCA 可以通过协方差矩阵特征分解理解。\n+- 工程上则常常更适合通过 SVD 来稳定求解。\n+- 当前源码里的 `svd_solver='auto'`，正对应了这类数值求解层面的自动选择。

## 5. 数学原理如何映射到当前源码

### 理解重点

- 主成分数量 $q$ 在工程里对应 `n_components`。\n+- 每个主成分对应的解释方差比例，在工程里对应 `model.explained_variance_ratio_`。\n+- 前几个主成分的累计解释方差，在工程里对应 `model.explained_variance_ratio_.sum()`。\n+- 当前训练代码没有手写特征值分解或 SVD，而是由 `PCA.fit(...)` 内部完成。

## 常见坑

1. 把 PCA 误写成“直接学习低维坐标”，却忽略它先学习的是主成分方向。\n+2. 只记住“最大方差”口号，却不理解它背后是一个特征值问题。\n+3. 把解释方差比误当成监督学习里的精度指标。

## 小结

- PCA 的数学核心，是在单位约束下寻找投影方差最大的方向。\n+- 这会自然导向协方差矩阵的特征值问题，而 SVD 则提供了更稳定的数值求解路径。\n+- 当前源码中的 `n_components`、`explained_variance_ratio_` 和累计解释方差，正是这些数学思想在工程层面的直接映射。
