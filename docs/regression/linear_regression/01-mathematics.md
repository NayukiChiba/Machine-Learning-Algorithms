---
title: 线性回归 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解线性回归的模型形式——目标值是特征的线性组合加上截距和噪声。
2. 理解最小二乘法（OLS）的目标函数、正规方程闭式解及其成立条件。
3. 理解极大似然估计视角——高斯噪声假设下 OLS 等价于 MLE。
4. 把这些数学表达和当前源码中的 `coef_`、`intercept_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 线性模型 | 模型形式 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$——用线性函数拟合连续值目标 |
| OLS | 优化目标 | $\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$——最小化残差平方和 |
| 正规方程 | 闭式解 | $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$——OLS 的解析解 |
| MLE | 概率视角 | 高斯噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 下最大化似然等价于最小化 RSS |
| `coef_` / `intercept_` | 源码属性 | 训练后 $\mathbf{w}$ 和 $b$ 在工程中的直接映射 |

## 1. 模型定义

线性回归假设目标变量 $y$ 与特征 $\mathbf{x} = (x_1, \dots, x_d)^T$ 之间存在线性关系：

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b
$$

引入扩展向量 $\tilde{\mathbf{x}} = (1, x_1, \dots, x_d)^T$，$\tilde{\mathbf{w}} = (b, w_1, \dots, w_d)^T$，可统一写为：

$$
\hat{y} = \tilde{\mathbf{w}}^T \tilde{\mathbf{x}}
$$

其中 $\mathbf{w} \in \mathbb{R}^d$ 是系数向量，$b \in \mathbb{R}$ 是截距。

### 理解重点

- 线性回归预测的本质是"特征乘系数，再加截距"——每个特征独立贡献，最终求和。
- 当前数据中的 `面积`、`房间数`、`房龄` 分别乘上自己的系数再相加，加上截距得到预测房价。
- 训练完成后，`model.coef_` 和 `model.intercept_` 就是学到的 $\mathbf{w}$ 和 $b$。

## 2. 最小二乘法（OLS）

对 $N$ 个训练样本，定义残差平方和（RSS）为损失函数：

$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2 = \|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|^2
$$

其中 $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ 是扩展设计矩阵（第一列全为 1），$\mathbf{y} \in \mathbb{R}^N$ 是目标向量。

OLS 的目标是找到使 $\mathcal{L}$ 最小的 $\tilde{\mathbf{w}}$。

### 理解重点

- OLS 的目标非常直接——让所有样本的预测误差平方和尽可能小。
- 平方惩罚意味着大误差会受到不成比例的重罚——一个偏离 10 的样本比十个偏离 1 的样本对损失函数的贡献更大。
- 当前代码没有手写这个损失函数——`LinearRegression()` 内部求解的正是这个问题。

## 3. 正规方程：闭式解

展开损失函数并求导：

$$
\mathcal{L} = \mathbf{y}^T\mathbf{y} - 2\tilde{\mathbf{w}}^T\mathbf{X}^T\mathbf{y} + \tilde{\mathbf{w}}^T\mathbf{X}^T\mathbf{X}\tilde{\mathbf{w}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{w}}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\tilde{\mathbf{w}} = 0
$$

得到**正规方程**（Normal Equation）：

$$
\boxed{\tilde{\mathbf{w}}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}
$$

### 理解重点

- 正规方程给出了 OLS 的**闭式解**——不需要迭代，一次矩阵运算即可得到最优参数。
- 对于 $d=3$、$N=160$（训练集大小），$\mathbf{X}^T\mathbf{X}$ 是 $4 \times 4$ 矩阵——求逆计算几乎瞬间完成。
- 当前仓库没有手写矩阵求逆——scikit-learn 使用 `scipy.linalg.lstsq`（基于 SVD）求解，数值更稳定。

## 4. 正规方程的成立条件

$\mathbf{X}^T\mathbf{X}$ 必须可逆。当：

- 特征数 $d$ 大于样本数 $N$
- 特征间存在精确线性关系（多重共线性）

时，$\mathbf{X}^T\mathbf{X}$ 奇异或近似奇异，正规方程数值不稳定。

### 理解重点

- 当前数据只有 3 个特征、160 个训练样本——$N \gg d$，$\mathbf{X}^T\mathbf{X}$ 通常满秩。
- 特征之间相对独立（面积、房间数、房龄来自独立均匀采样）——无严重共线性。
- 当特征高度相关或 $d > N$ 时，需要正则化（Ridge/Lasso）——这是后续正则化分册要解决的问题。

## 5. 极大似然估计视角

从概率视角看，假设目标值由线性函数加高斯噪声生成：

$$
y_i = \mathbf{w}^T \mathbf{x}_i + b + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

单个样本的似然：

$$
P(y_i \mid \mathbf{x}_i, \mathbf{w}, b) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i - b)^2}{2\sigma^2}\right)
$$

对数似然：

$$
\ln L = -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \mathbf{w}^T\mathbf{x}_i - b)^2
$$

最大化 $\ln L$ 等价于最小化 $\sum (y_i - \hat{y}_i)^2$——即 OLS。

### 理解重点

- 当前数据生成函数**确实含有高斯噪声** `rng.normal(0, 10, size=n_samples)`——MLE 的高斯假设与数据生成过程完全一致。
- OLS 不只是一个代数技巧——它是高斯噪声假设下最自然的参数估计方法。
- 理解这层关系，有助于后续过渡到正则化（从 MLE 到 MAP）和贝叶斯线性回归。

## 6. 常见评估指标（理论层）

| 指标 | 公式 | 含义 |
|---|---|---|
| MSE | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ | 均方误差——预测误差平方的平均 |
| RMSE | $\sqrt{\text{MSE}}$ | 均方根误差——与目标同量纲 |
| MAE | $\frac{1}{N}\sum|y_i - \hat{y}_i|$ | 平均绝对误差——对异常值不敏感 |
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 决定系数——模型解释的目标方差比例，$\le 1$，越接近 1 越好 |

### 理解重点

- 这些是线性回归理论上最常用的评估指标——但**当前流水线未显式打印**任何数值指标。
- 当前评估侧重图形化诊断（残差图 + 学习曲线）——学习曲线内部使用 $R^2$ 作为评分。
- 区分"理论上常见"与"当前实现真实输出"——不可将公式表当成已实现的功能。

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 线性模型 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$ | `LinearRegression()` |
| 系数向量 | $\mathbf{w} = (w_1, w_2, w_3)^T$ | `model.coef_`——长度为 3 的数组 |
| 截距 | $b$ | `model.intercept_`——标量 |
| 设计矩阵 | $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ | `X_train`（扩展列由 scikit-learn 内部处理） |
| 目标向量 | $\mathbf{y} \in \mathbb{R}^N$ | `y_train` |
| OLS 求解 | $\tilde{\mathbf{w}}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ | `model.fit(X_train, y_train)`——基于 SVD 的数值求解 |
| 预测 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$ | `model.predict(X_test)` |
| 残差 | $e_i = y_i - \hat{y}_i$ | `y_test - y_pred`——残差图的数据源 |
| 训练样本数 | $N$ | `n_samples=200`——`test_size=0.2` 后训练集约 160 |
| 特征维度 | $d$ | `3`——面积、房间数、房龄 |

## 8. 线性回归 vs 决策树回归 数学对比

| 维度 | 线性回归 | 决策树回归 |
|---|---|---|
| 模型形式 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$——全局线性 | **$\hat{y} = \sum_m \hat{c}_m \mathbb{1}(\mathbf{x} \in R_m)$——分段常数** |
| 目标函数 | $\min \|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|^2$——凸优化 | **$\min_{j,s} \sum_{R_1,R_2} (y_i - \hat{c})^2$——贪心搜索** |
| 求解方式 | 闭式解（正规方程或 SVD） | **贪心递归——无闭式解** |
| 参数数 | $d + 1$（系数 + 截距）——固定 | **叶子节点数——随数据增长** |
| 非线性处理 | 需基函数展开或特征工程 | **天然支持——递归分裂即是非线性** |
| 特征交互 | 需手工构造交互项 | **自然通过条件分支捕获** |
| 可解释性 | 极强——一个系数一个影响方向 | 中等——重要性无方向 |
| 外推能力 | 有——可线性外推 | **无——叶子边界外预测为常数** |

## 常见坑

1. 把正规方程当成当前源码显式写出的训练逻辑——实际上仓库调用的是 scikit-learn 基于 SVD 的实现。
2. 忽略高斯噪声假设与 OLS 的关系——只把最小二乘当成纯公式，错过了概率建模的统一视角。
3. 把理论上的 MSE/MAE/RMSE/$R^2$ 指标表误读成当前流水线已打印输出——实际只用了图形化评估。
4. 看到当前实现没有标准化就认为"线性回归永远不需要标准化"——当使用梯度下降求解或正则化时标准化是必需的。

## 小结

- 线性回归的数学核心链：线性模型 $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ → OLS $\min \|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|^2$ → 正规方程闭式解 → MLE 概率解释 → `coef_`/`intercept_` 工程映射。
- 与决策树回归的根本区别：全局线性函数 vs 分段常数，闭式解 vs 贪心搜索，系数可解释 vs 重要性无方向。
- 当前源码 `LinearRegression()` 是 OLS 的最简教学实现——无超参数、无标准化、关系透明，是回归学习的逻辑起点。
