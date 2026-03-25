# 正则化回归 (Regularization)

## 核心思想

当特征之间存在多重共线性或特征数量接近/超过样本数时，最小二乘解不稳定且易过拟合。**正则化**通过在损失函数中添加惩罚项来约束参数大小，从而提高模型的泛化能力。

## Ridge 回归 (L2 正则化)

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

$\lambda\mathbf{I}$ 使得矩阵永远可逆，解决了共线性问题。

### 贝叶斯解释

Ridge 等价于对 $\mathbf{w}$ 施加高斯先验 $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \frac{\sigma^2}{\lambda}\mathbf{I})$ 后的 MAP 估计。

## Lasso 回归 (L1 正则化)

### 目标函数

$$
\mathcal{L}_{\text{Lasso}} = \sum_{i=1}^{N}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_1
= (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w}) + \lambda \sum_{j=1}^d |w_j|
$$

### 稀疏性

L1 罚项在原点不可微，产生的约束区域为"菱形"。当等高线（椭圆）与菱形的**顶点**相切时，部分 $w_j$ 恰好为零 — 自动实现**特征选择**。

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

Lasso 没有闭式解，通常使用**坐标下降法**求解。

### 贝叶斯解释

Lasso 等价于对 $\mathbf{w}$ 施加拉普拉斯先验 $P(w_j) = \frac{\lambda}{2\sigma^2}\exp(-\frac{\lambda}{\sigma^2}|w_j|)$ 后的 MAP 估计。

## ElasticNet (弹性网)

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

ElasticNet 在相关特征分组时优于 Lasso（Lasso 只倾向于选出同组中的一个）。

## L1 vs L2 对比

| 特性 | Ridge (L2) | Lasso (L1) |
|------|-----------|-----------|
| 惩罚项 | $\sum w_j^2$ | $\sum \lvert w_j\rvert$ |
| 参数趋势 | 收缩但不为零 | 可精确为零 |
| 特征选择 | ❌ 不具备 | ✅ 自动选择 |
| 闭式解 | ✅ 有 | ❌ 无 |
| 贝叶斯先验 | 高斯分布 | 拉普拉斯分布 |
| 多重共线性 | 效果好 | 不稳定 |

## 代码对应

```bash
python -m pipelines.regression.regularization
```
