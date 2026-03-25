# 线性回归 (Linear Regression)

## 核心思想

线性回归假设目标变量 $y$ 与特征 $\mathbf{x}$ 之间存在线性关系，通过最小化预测误差的平方和来拟合参数。

## 模型定义

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b
$$

引入扩展向量 $\tilde{\mathbf{x}} = (1, x_1, \dots, x_d)^T$，$\tilde{\mathbf{w}} = (b, w_1, \dots, w_d)^T$：

$$
\hat{y} = \tilde{\mathbf{w}}^T \tilde{\mathbf{x}}
$$

## 最小二乘法 (OLS)

### 损失函数

对 $N$ 个样本，残差平方和 (RSS) 为：

$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 = (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})
$$

其中 $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ 是设计矩阵，$\mathbf{y} \in \mathbb{R}^{N}$ 是目标向量。

### 矩阵求导推导

展开损失函数：

$$
\mathcal{L} = \mathbf{y}^T\mathbf{y} - 2\mathbf{w}^T\mathbf{X}^T\mathbf{y} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w}
$$

对 $\mathbf{w}$ 求导（利用矩阵微分恒等式 $\frac{\partial \mathbf{a}^T\mathbf{w}}{\partial \mathbf{w}} = \mathbf{a}$，$\frac{\partial \mathbf{w}^T\mathbf{A}\mathbf{w}}{\partial \mathbf{w}} = 2\mathbf{A}\mathbf{w}$）：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w}
$$

令其为零，得到**正规方程** (Normal Equation)：

$$
\mathbf{X}^T\mathbf{X}\mathbf{w}^* = \mathbf{X}^T\mathbf{y}
$$

$$
\boxed{\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}
$$

### 条件

$\mathbf{X}^T\mathbf{X}$ 必须可逆。当特征间存在多重共线性（$\mathbf{X}^T\mathbf{X}$ 近似奇异）时，正规方程数值不稳定 — 这正是正则化的动机。

## 极大似然估计视角

假设 $y_i = \mathbf{w}^T\mathbf{x}_i + \epsilon_i$，$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$，则：

$$
P(y_i \mid \mathbf{x}_i, \mathbf{w}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)
$$

对数似然：

$$
\ln L = -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \mathbf{w}^T\mathbf{x}_i)^2
$$

最大化对数似然等价于最小化 RSS，因此 **OLS 等价于高斯噪声下的 MLE**。

## 评估指标

| 指标 | 公式 |
|------|------|
| MSE | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ |
| RMSE | $\sqrt{\text{MSE}}$ |
| MAE | $\frac{1}{N}\sum\lvert y_i - \hat{y}_i\rvert$ |
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ |

## 代码对应

```bash
python -m pipelines.regression.linear_regression
```
