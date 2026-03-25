# 逻辑回归 (Logistic Regression)

## 核心思想

逻辑回归是一种**广义线性模型**，通过 Sigmoid 函数将线性组合映射到概率空间 $(0, 1)$，用于二分类或多分类任务。虽名为"回归"，实则是一个分类器。

## 模型定义

### 线性部分

给定输入 $\mathbf{x} \in \mathbb{R}^d$ ，线性得分为：

$$
z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b
$$

### Sigmoid 函数

Sigmoid（逻辑函数）将任意实数映射到 $(0, 1)$：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

其导数具有优雅的自引用形式：

$$
\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))
$$

### 概率输出

正类后验概率：

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

负类后验概率：

$$
P(y=0 \mid \mathbf{x}) = 1 - \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

## 对数几率 (Log-Odds)

逻辑回归对**对数几率**（logit）建模为线性函数：

$$
\ln \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = \mathbf{w}^T \mathbf{x} + b
$$

这意味着**决策边界** $\mathbf{w}^T \mathbf{x} + b = 0$ 是一个超平面。

## 极大似然估计 (MLE)

### 似然函数

对于训练集 $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$，假设样本独立同分布，似然函数为：

$$
L(\mathbf{w}, b) = \prod_{i=1}^{N} P(y_i \mid \mathbf{x}_i) = \prod_{i=1}^{N} \hat{p}_i^{\,y_i} (1 - \hat{p}_i)^{1 - y_i}
$$

其中 $\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$。

### 对数似然 → 交叉熵损失

取对数并加负号，得到要**最小化**的交叉熵损失：

$$
\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \Big]
$$

## 梯度推导

交叉熵对参数 $w_j$ 的偏导数：

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i) \, x_{ij}
$$

向量形式：

$$
\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{p}} - \mathbf{y})
$$

**推导过程**：

$$
\frac{\partial \mathcal{L}}{\partial w_j}
= -\frac{1}{N} \sum_{i=1}^N \left[ y_i \frac{\sigma'(z_i)}{\sigma(z_i)} x_{ij} + (1-y_i) \frac{-\sigma'(z_i)}{1-\sigma(z_i)} x_{ij} \right]
$$

利用 $\sigma'(z) = \sigma(z)(1-\sigma(z))$：

$$
= -\frac{1}{N} \sum_{i=1}^N \left[ y_i (1-\hat{p}_i) - (1-y_i) \hat{p}_i \right] x_{ij}
= \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i) x_{ij}
$$

## 梯度下降更新

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla_{\mathbf{w}} \mathcal{L}
$$

其中 $\eta$ 为学习率。

## 多分类扩展：Softmax 回归

对于 $K$ 个类别，使用 Softmax 函数：

$$
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}
$$

损失函数变为多类交叉熵：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \ln P(y_i = k \mid \mathbf{x}_i)
$$

## 代码对应

```bash
python -m pipelines.classification.logistic_regression
```
