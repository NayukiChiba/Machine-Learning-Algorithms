# SVR 支持向量回归 (Support Vector Regression)

## 核心思想

SVR 将 SVM 的最大间隔思想推广到回归：构建一个以 $\epsilon$ 为半宽的"管道"，允许管道内的预测误差为零，仅惩罚落在管道外的样本。

## $\epsilon$-不敏感损失函数

$$
L_\epsilon(y, f(\mathbf{x})) = \max(0, |y - f(\mathbf{x})| - \epsilon)
$$

- 预测值与真实值之差在 $[-\epsilon, +\epsilon]$ 内时，损失为 0
- 超出管道的部分按线性计损

## 优化问题

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N} (\xi_i + \xi_i^*)
$$

$$
\text{s.t.} \quad
\begin{cases}
y_i - \mathbf{w}^T\mathbf{x}_i - b \leq \epsilon + \xi_i \\
\mathbf{w}^T\mathbf{x}_i + b - y_i \leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* \geq 0
\end{cases}
$$

### 对偶问题

引入拉格朗日乘子 $\alpha_i, \alpha_i^*$：

$$
\max_{\boldsymbol{\alpha}, \boldsymbol{\alpha}^*} -\frac{1}{2}\sum_{i,j}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)\mathbf{x}_i^T\mathbf{x}_j
- \epsilon\sum_i(\alpha_i + \alpha_i^*) + \sum_i y_i(\alpha_i - \alpha_i^*)
$$

$$
\text{s.t.} \quad \sum_i(\alpha_i - \alpha_i^*) = 0, \quad 0 \leq \alpha_i, \alpha_i^* \leq C
$$

### 预测函数

$$
f(\mathbf{x}) = \sum_{i=1}^{N}(\alpha_i - \alpha_i^*)\mathbf{x}_i^T\mathbf{x} + b
$$

同样可以使用核函数 $K(\mathbf{x}_i, \mathbf{x})$ 替换内积以处理非线性回归。

## 代码对应

```bash
python -m pipelines.regression.svr
```
