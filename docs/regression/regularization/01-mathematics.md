---
title: 正则化回归 — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/regression/regularization.py`
>  
> 相关对象：`Lasso`、`Ridge`、`ElasticNet`

## 本章目标

1. 理解正则化为什么要在平方误差之外再加入参数惩罚项。
2. 理解 Ridge、Lasso、ElasticNet 三种目标函数的数学差异。
3. 把这些公式和当前源码中的 `alpha`、`l1_ratio` 参数对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| Ridge | 模型 | 在平方误差外加入 L2 惩罚 |
| Lasso | 模型 | 在平方误差外加入 L1 惩罚 |
| ElasticNet | 模型 | 同时加入 L1 与 L2 惩罚 |
| `alpha` | 超参数 | 控制正则化强度 |
| `l1_ratio` | 超参数 | 控制 ElasticNet 中 L1 所占比例 |

## 1. 核心思想

当特征之间存在多重共线性，或特征数接近甚至超过样本数时，最小二乘解往往不稳定且容易过拟合。正则化的做法是在损失函数中加入惩罚项，主动限制参数大小，从而提高模型的泛化能力。

### 理解重点

- 普通最小二乘只关心训练误差是否足够小。
- 正则化回归则额外关心“系数是不是过大、过于依赖某些特征”。
- 当前源码中的 `alpha` 就是把这种惩罚强度显式参数化后的结果。

## 2. Ridge 回归（L2 正则化）

### 参数速览（本节）

适用目标函数：Ridge

| 参数名 | 数学含义 | 在源码中的对应 |
|---|---|---|
| $\mathbf{w}$ | 模型系数向量 | `model.coef_` |
| $\lambda$ | L2 正则化强度 | `alpha` |
| $\mathbf{X}, \mathbf{y}$ | 训练数据与标签 | `X_train`、`y_train` |

### 目标函数

$$
\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^{N}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_2^2
= (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w}) + \lambda \mathbf{w}^T\mathbf{w}
$$

### 闭式解推导

$$
\frac{\partial \mathcal{L}_{\text{Ridge}}}{\partial \mathbf{w}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w} + 2\lambda\mathbf{w} = 0
$$

$$
\boxed{\mathbf{w}^*_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}}
$$

$\lambda\mathbf{I}$ 使矩阵更稳定、通常也更容易求逆，因此 Ridge 对多重共线性尤其有效。

### 贝叶斯解释

Ridge 等价于对 $\mathbf{w}$ 施加高斯先验 $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \frac{\sigma^2}{\lambda}\mathbf{I})$ 后的 MAP 估计。

### 理解重点

- L2 惩罚会让系数整体变小，但通常不会把它们直接压成 0。
- 这也是为什么 Ridge 更擅长“稳定系数”，而不是“筛掉特征”。
- 在当前源码里，`Ridge(alpha=2.0)` 对应的就是这一类目标函数。

## 3. Lasso 回归（L1 正则化）

### 参数速览（本节）

适用目标函数：Lasso

| 参数名 | 数学含义 | 在源码中的对应 |
|---|---|---|
| $\mathbf{w}$ | 模型系数向量 | `model.coef_` |
| $\lambda$ | L1 正则化强度 | `alpha` |
| $\mathbf{X}, \mathbf{y}$ | 训练数据与标签 | `X_train`、`y_train` |

### 目标函数

$$
\mathcal{L}_{\text{Lasso}} = \sum_{i=1}^{N}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_1
= (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w}) + \lambda \sum_{j=1}^d |w_j|
$$

### 稀疏性

L1 罚项在原点不可微，产生的约束区域更容易在坐标轴方向产生尖点。当损失等高线与这些尖点相切时，部分 $w_j$ 会被直接压到 0，从而自动完成特征选择。

### 次梯度

L1 正则项的次梯度为：

$$
\partial |w_j| =
\begin{cases}
\{-1\} & w_j < 0 \\
[-1, 1] & w_j = 0 \\
\{+1\} & w_j > 0
\end{cases}
$$

Lasso 没有像 Ridge 那样简单的闭式解，通常使用坐标下降等迭代方法求解。

### 贝叶斯解释

Lasso 等价于对 $\mathbf{w}$ 施加拉普拉斯先验 $P(w_j) = \frac{\lambda}{2\sigma^2}\exp(-\frac{\lambda}{\sigma^2}|w_j|)$ 后的 MAP 估计。

### 理解重点

- L1 惩罚最显著的效果，就是让部分系数精确为 0 或非常接近 0。
- 这也是当前源码里要统计“接近 0 的系数数量”的原因。
- 在当前分册中，Lasso 的数学特性会直接映射到 `noise_*` 特征是否被压缩掉。

## 4. ElasticNet（弹性网）

### 参数速览（本节）

适用目标函数：ElasticNet

| 参数名 | 数学含义 | 在源码中的对应 |
|---|---|---|
| $\lambda$ | 总体正则化强度 | `alpha` |
| $\rho$ | L1 所占比例 | `l1_ratio` |
| $\mathbf{w}$ | 模型系数向量 | `model.coef_` |

### 目标函数

组合 L1 和 L2 罚项：

$$
\mathcal{L}_{\text{EN}} = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2
$$

等价形式（使用混合比 $\rho \in [0,1]$）：

$$
\mathcal{L}_{\text{EN}} = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda \big[\rho \|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2 \big]
$$

- $\rho = 1$：退化为 Lasso
- $\rho = 0$：退化为 Ridge

ElasticNet 在相关特征成组出现时通常比纯 Lasso 更稳定，同时又保留了一部分稀疏化能力。

### 理解重点

- ElasticNet 的核心不是“介于两者之间”这么简单，而是试图同时利用 L1 和 L2 的优点。
- 当前源码中的 `l1_ratio=0.5`，就是让它处在一个折中状态。
- 这也解释了为什么当前实现要同时构造相关特征和噪声特征，方便观察 ElasticNet 的双重作用。

## 5. L1 与 L2 的行为对比

| 特性 | Ridge (L2) | Lasso (L1) |
|------|-----------|-----------|
| 惩罚项 | $\sum w_j^2$ | $\sum |w_j|$ |
| 参数趋势 | 收缩但通常不为零 | 可精确为零 |
| 特征选择 | 不擅长 | 擅长 |
| 闭式解 | 有 | 无 |
| 贝叶斯先验 | 高斯分布 | 拉普拉斯分布 |
| 多重共线性 | 处理稳定 | 可能偏向只选部分特征 |

### 理解重点

- Ridge 更适合“保留全部信息，但限制幅度”。
- Lasso 更适合“主动筛掉一部分特征”。
- ElasticNet 则是在这两种倾向之间建立平衡，这与当前源码中的三模型对比完全一致。

## 常见坑

1. 把 `alpha` 理解成“训练轮数”或“学习率”，它在这里表示正则化强度。
2. 误以为正则化一定让预测更好，实际上它是在偏差与方差之间做权衡。
3. 只记住公式形式，不把它们与 `coef_`、`near_zero`、`l1_ratio` 这些源码对象对应起来。

## 小结

- 正则化回归的本质，是在拟合误差之外增加对参数复杂度的约束。
- Ridge、Lasso、ElasticNet 的差异，最终都会体现为系数形态和泛化行为的差异。
- 当前源码中的 `alpha`、`l1_ratio`、系数打印和近零计数，正是这些数学思想在工程层面的直接映射。
