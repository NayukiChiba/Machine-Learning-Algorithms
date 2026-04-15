---
title: LDA — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/dimensionality/lda.py`
>  
> 相关对象：`LinearDiscriminantAnalysis`、`train_model(...)`

## 本章目标

1. 理解 LDA 为什么可以形式化为“最大类间散度、最小类内散度”的优化问题。
2. 理解 Fisher 判别准则、广义特征值问题和 `K-1` 维上限之间的关系。
3. 把这些数学表达和当前源码中的 `n_components`、`solver`、`explained_variance_ratio_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 类内散度矩阵 | 数学对象 | 描述同类样本内部的分散程度 |
| 类间散度矩阵 | 数学对象 | 描述不同类别中心之间的分离程度 |
| Fisher 判别准则 | 优化目标 | 最大化类间 / 类内散度比 |
| 广义特征值问题 | 数学形式 | 给出判别方向的求解路径 |
| `n_components` | 源码参数 | 当前保留多少个判别方向 |

## 1. 核心思想

LDA 是一种有监督降维方法。它寻找一个投影方向，使得类间散度最大、类内散度最小，从而在降维的同时最大化类别可分性。

### 理解重点

- 当前源码中的 `LinearDiscriminantAnalysis(...)`，本质上就是在寻找这样的判别方向。
- 这和 PCA 的“最大方差”目标不同，LDA 关心的是类别能否被更好地区分。
- 当前分册的所有工程输出，都应围绕“可分性增强”来理解。

## 2. Fisher 判别准则

### 二分类情形

将数据投影到方向 $\mathbf{w}$ 上后，第 $k$ 类的投影均值和投影方差为：

$$
\tilde{\mu}_k = \mathbf{w}^T \boldsymbol{\mu}_k, \quad \tilde{\sigma}_k^2 = \mathbf{w}^T \mathbf{S}_k \mathbf{w}
$$

Fisher 准则最大化类间距离与类内方差之比：

$$
J(\mathbf{w}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{\sigma}_1^2 + \tilde{\sigma}_2^2} = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}
$$

### 理解重点

- 这个目标函数直观上很合理：希望不同类别中心尽量远，同类内部尽量紧。
- 当前 LDA 分册所有“判别方向”的直觉，都是从这个比例目标出发的。
- 这也是为什么标签在 LDA 中是训练必需信息。

## 3. 散度矩阵

类内散度矩阵（Within-Class Scatter）：

$$
\mathbf{S}_W = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T
$$

类间散度矩阵（Between-Class Scatter）：

$$
\mathbf{S}_B = \sum_{k=1}^{K} N_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T
$$

其中 $\boldsymbol{\mu}$ 为全局均值，$N_k$ 为第 $k$ 类样本数。

### 理解重点

- `S_W` 想表达的是“同一类内部有多散”。
- `S_B` 想表达的是“不同类别中心有多远”。
- 当前 Wine 数据集类别差异较明显，因此这个目标在可视化上通常更容易看到效果。

## 4. 瑞利商最优化

$$
J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}
$$

这是广义瑞利商（Generalized Rayleigh Quotient）。

### 推导

令 $\mathbf{w}^T \mathbf{S}_W \mathbf{w} = 1$，用拉格朗日乘子法：

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

### 理解重点

- 这说明 LDA 的判别方向不是经验规则，而是一个严格优化问题的解析结果。
- 工程上并不会手写这些矩阵求解，而是由 `LinearDiscriminantAnalysis.fit(...)` 内部完成。
- 但数学上理解这一步，能帮助你真正看懂“判别方向”从哪里来。

## 5. 二分类闭式解

由于 $\text{rank}(\mathbf{S}_B) = 1$，$\mathbf{S}_B \mathbf{w} \propto (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$，因此：

$$
\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
$$

### 理解重点

- 这个闭式解把二分类 LDA 的判别方向写得非常直接：就是“类中心差异”乘上“类内结构修正”。
- 它帮助你直观理解为什么 LDA 既看中心距离，也看类内散布。
- 当前分册虽然是 3 分类，但这个二分类情形仍然是理解多分类推广的最好起点。

## 6. 多分类推广与 `K - 1` 上限

$K$ 个类最多可降至 $K - 1$ 维，因为：

$$
\text{rank}(\mathbf{S}_B) \leq K - 1
$$

因此只需选取 $\mathbf{S}_W^{-1}\mathbf{S}_B$ 最大的 $q$ 个特征值对应的特征向量。

### 理解重点

- 这条性质直接解释了为什么当前 Wine 数据 3 类时，最多只能降到 2 维。
- 当前源码中的 `n_components=2`，并不是随便选的默认值，而是正好达到这个理论上限。
- 这也是当前流水线只输出 2D 图而没有 3D 图的根本原因。

## 7. PCA vs LDA

| 特性 | PCA | LDA |
|------|-----|-----|
| 监督方式 | 无监督 | 有监督 |
| 优化目标 | 最大投影方差 | 最大类间 / 类内散度比 |
| 降维上限 | 可由数据维度决定 | $K - 1$ |
| 适用场景 | 数据压缩、可视化 | 分类预处理、判别式降维 |

### 理解重点

- 这张表是当前 LDA 分册最容易和 PCA 分册混淆时的核心澄清。
- 当前实现之所以需要 `y`，而 PCA 不需要，根本原因就在这里。
- 因此文档里所有“label 的作用”都要围绕这个监督差异来写。

## 8. 数学原理如何映射到当前源码

### 理解重点

- 判别方向数量在工程里对应 `n_components`。
- 求解路径在工程里对应 `solver`。
- 若当前求解器支持，判别方向解释比例会在工程里对应 `model.explained_variance_ratio_`。
- 当前训练代码没有手写散度矩阵求解，而是由 `LinearDiscriminantAnalysis.fit(...)` 内部完成。

## 常见坑

1. 把 LDA 误写成“带标签版 PCA”，却不解释类内 / 类间散度结构。
2. 忽略 `K - 1` 上限，误以为 LDA 可以像 PCA 一样自由增加维度。
3. 把 `explained_variance_ratio_` 当成一定存在的属性，而不考虑求解器差异。

## 小结

- LDA 的数学核心，是通过最大化类间散度与类内散度之比来寻找判别方向。
- 这一优化最终导向广义特征值问题，并自然给出 `K - 1` 的降维上限。
- 当前源码中的 `n_components`、`solver` 和可选 `explained_variance_ratio_`，正是这些数学思想在工程层面的直接映射。
