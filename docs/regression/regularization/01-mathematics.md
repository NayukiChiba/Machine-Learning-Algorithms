---
title: 正则化回归 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解正则化为什么在 OLS 损失函数中加入参数惩罚项。
2. 理解 Ridge（L2）、Lasso（L1）、ElasticNet（L1+L2）三种目标函数的数学差异。
3. 将数学公式中的 $\lambda$、$\rho$ 与源码中的 `alpha`、`l1_ratio` 精确对应。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| L2 正则化（Ridge） | 惩罚项 | $\lambda \|\mathbf{w}\|_2^2$——平方惩罚，系数整体收缩 |
| L1 正则化（Lasso） | 惩罚项 | $\lambda \|\mathbf{w}\|_1$——绝对值惩罚，系数可精确归零 |
| ElasticNet | 惩罚项 | $\lambda[\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2]$——L1+L2 混合 |
| `alpha` | 超参数 | 对应数学中的 $\lambda$——控制正则化总强度 |
| `l1_ratio` | 超参数 | 对应数学中的 $\rho$——控制 L1 在 ElasticNet 中的占比 |

## 1. 从 OLS 到正则化：为什么需要惩罚项

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| OLS 损失 | 数学表达式 | $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$——仅最小化残差平方和 | — |
| 正则化损失 | 数学表达式 | OLS 损失 $+ \lambda \cdot$ 惩罚项——在拟合与复杂度之间权衡 | — |
| $\lambda$ | 标量 | 正则化强度——$\lambda=0$ 退化为 OLS，$\lambda \to \infty$ 系数全部归零 | `alpha=0.15`（Lasso） |

### 理解重点

- OLS 的唯一目标是让训练误差最小——在高维或共线场景下，系数可能剧烈波动。
- 正则化在 OLS 损失上叠加系数惩罚——"你可以拟合数据，但要为系数过大付出代价"。
- 当前源码中 Lasso/Ridge/ElasticNet 的 `alpha` 值不同（0.15 / 2.0 / 0.2），说明不同惩罚类型需要不同的强度来展示典型行为。

## 2. Ridge 回归（L2 正则化）

### 参数速览

适用模型：`Ridge(alpha=2.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 目标函数 | 数学表达式 | $\mathcal{L}_{\text{Ridge}} = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$ | — |
| $\lambda$ | 标量 | L2 正则化强度 | `alpha=2.0` |
| $\|\mathbf{w}\|_2^2$ | 标量 | $\sum_{j=1}^d w_j^2$——所有系数的平方和 | — |
| 闭式解 | 数学表达式 | $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ | — |

### 理解重点

- L2 惩罚对每个系数施加平方代价——大系数付出更大的代价，因此所有系数被**均匀收缩**。
- 闭式解中 $\lambda\mathbf{I}$ 使 $\mathbf{X}^T\mathbf{X}$ 始终可逆——这是 Ridge 处理共线性的数学基础。
- 平方惩罚在零点可微——系数可以趋于零但**不会精确为零**。
- 贝叶斯视角：Ridge 等价于对 $\mathbf{w}$ 施加高斯先验 $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \frac{\sigma^2}{\lambda}\mathbf{I})$ 后的 MAP 估计。

## 3. Lasso 回归（L1 正则化）

### 参数速览

适用模型：`Lasso(alpha=0.15, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 目标函数 | 数学表达式 | $\mathcal{L}_{\text{Lasso}} = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_1$ | — |
| $\lambda$ | 标量 | L1 正则化强度 | `alpha=0.15` |
| $\|\mathbf{w}\|_1$ | 标量 | $\sum_{j=1}^d \|w_j\|$——所有系数的绝对值之和 | — |
| 次梯度 | 数学表达式 | $\partial \|w_j\| = \text{sign}(w_j)$ 当 $w_j \neq 0$；$\partial \|w_j\| \in [-1, 1]$ 当 $w_j = 0$ | — |

### 理解重点

- L1 惩罚在原点**不可微**——正是这个不可微性使得系数可以被精确驱动到零。
- 次梯度条件：当某个 $w_j$ 对降低残差的贡献不足以抵消 $\lambda$ 时，$w_j$ 被置为零——自动特征选择。
- Lasso 没有闭式解——scikit-learn 使用坐标下降法迭代求解。
- 贝叶斯视角：Lasso 等价于对 $\mathbf{w}$ 施加拉普拉斯先验 $P(w_j) \propto \exp(-\lambda\|w_j\|)$ 后的 MAP 估计。
- 拉普拉斯分布在零点有尖峰——比高斯分布更倾向于让参数恰好为零。

## 4. ElasticNet（弹性网）

### 参数速览

适用模型：`ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 目标函数 | 数学表达式 | $\mathcal{L}_{\text{EN}} = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda[\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2]$ | — |
| $\lambda$ | 标量 | 总正则化强度 | `alpha=0.2` |
| $\rho$ | 标量 | L1 占比——$\rho=1$ 退化为 Lasso，$\rho=0$ 退化为 Ridge | `l1_ratio=0.5` |

### 理解重点

- ElasticNet 通过 $\rho$ 在 L1 和 L2 之间插值——兼具稀疏性和共线性稳定性。
- 纯 Lasso 在相关特征组中可能随机只选一个——ElasticNet 的 L2 分量鼓励同组特征共享权重。
- $\rho=0.5$ 表示当前实现在稀疏性和稳定性之间取折中——不极端偏向任一侧。
- 也是坐标下降求解——`max_iter=10000` 确保在 $\rho$ 非极端值时充分收敛。

## 5. L1 vs L2 行为对比

| 特性 | L2（Ridge） | L1（Lasso） |
|---|---|---|
| 惩罚项 | $\sum w_j^2$ | $\sum \|w_j\|$ |
| 原点可微性 | 可微 | **不可微** |
| 系数归零 | 否——仅收缩 | **是——可精确归零** |
| 闭式解 | 有 | 无（需迭代） |
| 共线性处理 | 系数分摊到同组特征 | 可能只选部分特征 |
| 贝叶斯先验 | 高斯分布 | 拉普拉斯分布 |
| 特征选择 | 不擅长 | **擅长** |

### 理解重点

- L1 产生稀疏解的根本原因是原点不可微——不是"惩罚得更狠"，而是惩罚的**形状**不同。
- L2 的圆形等高线容易与损失等高线在任意点相切——系数非零但较小。
- L1 的菱形等高线容易在坐标轴尖点处与损失等高线相切——系数精确为零。

## 6. 数学概念与代码实现的映射

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 特征矩阵 | $\mathbf{X} \in \mathbb{R}^{N \times d}$ | `X_train_s`——标准化后形状 `(353, 21)` |
| 目标向量 | $\mathbf{y} \in \mathbb{R}^N$ | `y_train`——形状 `(353,)` |
| 系数向量 | $\mathbf{w} \in \mathbb{R}^d$ | `model.coef_`——形状 `(21,)` |
| 截距 | $b$ | `model.intercept_`——标量 |
| L2 惩罚强度 | $\lambda$（Ridge） | `Ridge(alpha=2.0)` |
| L1 惩罚强度 | $\lambda$（Lasso） | `Lasso(alpha=0.15)` |
| 总惩罚强度 | $\lambda$（ElasticNet） | `ElasticNet(alpha=0.2)` |
| L1 混合比例 | $\rho$ | `ElasticNet(l1_ratio=0.5)` |
| 系数绝对值 < $10^{-3}$ | $\{j : \|w_j\| < 10^{-3}\}$ | `np.sum(np.abs(coef) < 1e-3)` |
| 标准化 | $z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$ | `StandardScaler().fit_transform(X_train)` |

## 7. 正则化回归 vs 线性回归 数学对比

| 数学维度 | 线性回归 | 正则化回归 |
|---|---|---|
| 损失函数 | $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$ | OLS 损失 $+ \lambda \cdot R(\mathbf{w})$ |
| 优化问题 | 无约束最小化 | **带惩罚的约束最小化** |
| 求解方法 | SVD 闭式解（一步到位） | **坐标下降 / 闭式解（Ridge）+ 迭代（Lasso/EN）** |
| 唯一解 | 是（当 $\mathbf{X}$ 满秩） | **是——惩罚项使问题强凸（Ridge/EN）或凸（Lasso）** |
| 对共线性的数值稳定性 | 低——$\mathbf{X}^T\mathbf{X}$ 可能接近奇异 | **高——$\lambda\mathbf{I}$ 改善条件数（Ridge/EN）** |
| 特征选择 | 无——所有系数非零 | **有——L1 罚项可将系数精确归零** |
| 超参数数量 | 0 | **1~2（α + 可选的 l1_ratio）** |
| 尺度敏感性 | 不敏感——闭式解是尺度等变的 | **敏感——惩罚项对系数量级敏感，必须标准化** |

## 常见坑

1. 把 `alpha` 理解成学习率或迭代轮数——它在这里是正则化强度 $\lambda$，越大惩罚越重。
2. 认为"正则化一定让预测更准"——正则化在偏差和方差之间权衡，过强的正则化会导致欠拟合。
3. 只记公式不记代码映射——`alpha` ↔ $\lambda$、`l1_ratio` ↔ $\rho$、`np.sum(np.abs(coef) < 1e-3)` ↔ 稀疏性。

## 小结

- 正则化回归的数学本质是在 OLS 损失上叠加系数惩罚——从无约束优化变为带惩罚优化。
- L2 惩罚产生收缩（Ridge），L1 惩罚产生稀疏（Lasso），L1+L2 混合产生折中（ElasticNet）。
- L1 产生稀疏解的根本原因是绝对值函数在原点不可微——而非"惩罚更重"。
- 数学公式中的 $\lambda$、$\rho$ 直接映射到源码中的 `alpha`、`l1_ratio`——理解这个映射是读懂后续章节的前提。
