---
title: 线性回归 — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/regression/linear_regression.py`
>  
> 相关对象：`LinearRegression`、`train_model(...)`

## 本章目标

1. 理解线性回归为什么可以写成“系数向量和特征向量的线性组合”。
2. 理解最小二乘法、正规方程和极大似然估计之间的关系。
3. 把这些数学表达和当前源码中的 `coef_`、`intercept_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 线性模型 | 数学形式 | 用线性函数拟合连续值目标 |
| OLS | 优化目标 | 最小化残差平方和 |
| 正规方程 | 闭式解 | 给出最小二乘的解析解 |
| MLE | 概率视角 | 解释 OLS 与高斯噪声假设的关系 |
| `coef_` / `intercept_` | 源码属性 | 数学参数在工程实现中的直接映射 |

## 1. 核心思想

线性回归假设目标变量 $y$ 与特征 $\mathbf{x}$ 之间存在线性关系，通过最小化预测误差的平方和来拟合参数。

### 理解重点

- 当前源码中的 `LinearRegression()`，本质上就是在求这组线性参数。
- 文档里的“系数”和“截距”，在数学上对应模型参数，在工程里对应 `coef_` 和 `intercept_`。
- 线性回归之所以常被当作起点，就是因为这种参数形式简单、清晰、可解释。

## 2. 模型定义

### 参数速览（本节）

适用模型：线性回归

| 参数名 | 数学含义 | 在源码中的对应 |
|---|---|---|
| $\mathbf{w}$ | 系数向量 | `model.coef_` |
| $b$ | 截距 | `model.intercept_` |
| $\mathbf{x}$ | 输入特征向量 | 单个样本特征 |
| $\hat{y}$ | 预测值 | `model.predict(...)` 输出 |

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b
$$

引入扩展向量 $\tilde{\mathbf{x}} = (1, x_1, \dots, x_d)^T$，$\tilde{\mathbf{w}} = (b, w_1, \dots, w_d)^T$：

$$
\hat{y} = \tilde{\mathbf{w}}^T \tilde{\mathbf{x}}
$$

### 理解重点

- 这说明线性回归预测的本质，就是“特征乘系数，再加上截距”。
- 当前数据里的 `面积`、`房间数`、`房龄`，都会分别乘上自己的系数再相加。
- 这也是为什么训练完成后，只看 `coef_` 和 `intercept_` 就能直接解释模型行为。

## 3. 最小二乘法（OLS）

### 参数速览（本节）

适用目标函数：OLS

| 参数名 | 数学含义 | 在源码中的对应 |
|---|---|---|
| $\mathbf{X}$ | 设计矩阵 | `X_train` |
| $\mathbf{y}$ | 目标向量 | `y_train` |
| $\mathbf{w}$ | 参数向量 | `coef_` 与截距合并后的参数 |

### 损失函数

对 $N$ 个样本，残差平方和（RSS）为：

$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 = (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})
$$

其中 $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ 是设计矩阵，$\mathbf{y} \in \mathbb{R}^{N}$ 是目标向量。

### 理解重点

- OLS 的目标很直接：让预测值和真实值之间的平方误差尽可能小。
- 当前代码没有手写这个损失函数，但 `LinearRegression()` 内部求解的正是这个问题。
- 这也解释了为什么线性回归天然关注“残差”这一概念。

## 4. 矩阵求导与正规方程

### 示例推导

展开损失函数：

$$
\mathcal{L} = \mathbf{y}^T\mathbf{y} - 2\mathbf{w}^T\mathbf{X}^T\mathbf{y} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w}
$$

对 $\mathbf{w}$ 求导：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w}
$$

令其为零，得到正规方程（Normal Equation）：

$$
\mathbf{X}^T\mathbf{X}\mathbf{w}^* = \mathbf{X}^T\mathbf{y}
$$

$$
\boxed{\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}
$$

### 理解重点

- 正规方程给出了最小二乘解的闭式表达。
- 当前仓库没有自己写矩阵求逆代码，而是交给 scikit-learn 内部处理。
- 但理解这个推导，能帮助你把“训练得到系数”这件事和数学闭式解联系起来。

## 5. 正规方程成立的条件与局限

### 理解重点

$\mathbf{X}^T\mathbf{X}$ 必须可逆。当特征间存在多重共线性时，$\mathbf{X}^T\mathbf{X}$ 可能近似奇异，正规方程就会变得数值不稳定。这正是后续正则化回归分册要解决的问题。

- 当前线性回归数据只有 3 个特征，关系也较清晰，因此很适合先讲 OLS 本身。
- 如果把这套方法直接搬到高度相关特征场景，就可能出现系数不稳定问题。
- 这也是为什么“先学线性回归，再学正则化”是很自然的阅读顺序。

## 6. 极大似然估计视角

假设 $y_i = \mathbf{w}^T\mathbf{x}_i + \epsilon_i$，$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$，则：

$$
P(y_i \mid \mathbf{x}_i, \mathbf{w}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)
$$

对数似然：

$$
\ln L = -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \mathbf{w}^T\mathbf{x}_i)^2
$$

最大化对数似然等价于最小化 RSS，因此 OLS 等价于高斯噪声假设下的 MLE。

### 理解重点

- 当前数据生成函数确实显式加入了高斯噪声 `rng.normal(...)`，所以这个概率视角与源码背景是对得上的。
- 这说明 OLS 不只是一个代数技巧，也能从概率建模角度得到解释。
- 文档里强调这一点，是为了把“合成噪声数据”与“最小二乘求解”连起来看。

## 7. 常见评估指标（理论层）

下表是线性回归理论上常见的评估指标：

| 指标 | 公式 |
|------|------|
| MSE | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ |
| RMSE | $\sqrt{\text{MSE}}$ |
| MAE | $\frac{1}{N}\sum\lvert y_i - \hat{y}_i\rvert$ |
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ |

### 理解重点

- 这些指标是线性回归里常见的理论工具。
- 但当前仓库的 linear regression 流水线并没有显式打印这些数值，而是使用残差图和学习曲线做主要诊断。
- 因此要区分“理论上常见”与“当前实现真实输出”这两个层次。

## 常见坑

1. 把正规方程看成当前源码里显式写出的训练逻辑，实际上仓库只是调用了 scikit-learn 实现。
2. 忽略高斯噪声假设与 OLS 的关系，只把最小二乘当成纯公式记忆。
3. 把理论上的常见指标误读成当前流水线已经全部打印输出。

## 小结

- 线性回归的数学核心，是用最小二乘法求解一组线性参数。
- 正规方程给出了解析形式，MLE 则给出了概率解释。
- 当前源码中的 `coef_`、`intercept_`、残差图和学习曲线，正是这些数学思想在工程层面的直接映射。
