# 朴素贝叶斯 (Naive Bayes)

## 核心思想

朴素贝叶斯基于**贝叶斯定理**与**特征条件独立假设**，是一种高效的生成式分类器。虽然"朴素"假设在实践中几乎不成立，但朴素贝叶斯在文本分类等场景通常表现优异。

## 贝叶斯定理

$$
P(Y = c_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid Y = c_k) \, P(Y = c_k)}{P(\mathbf{x})}
$$

- **先验概率**：$P(Y = c_k)$
- **似然**：$P(\mathbf{x} \mid Y = c_k)$
- **后验概率**：$P(Y = c_k \mid \mathbf{x})$
- **证据**：$P(\mathbf{x})$（对所有类别相同，可忽略）

## 条件独立假设

假设给定类别 $c_k$ 后，所有特征相互独立：

$$
P(\mathbf{x} \mid Y = c_k) = P(x_1, x_2, \dots, x_d \mid Y = c_k) = \prod_{j=1}^{d} P(x_j \mid Y = c_k)
$$

这极大地降低了需要估计的参数数量：从 $O(K \cdot |\mathcal{X}|^d)$ 降为 $O(K \cdot d)$。

## 分类决策

$$
\hat{y} = \arg\max_{c_k} P(Y = c_k) \prod_{j=1}^{d} P(x_j \mid Y = c_k)
$$

取对数避免下溢：

$$
\hat{y} = \arg\max_{c_k} \left[ \ln P(Y = c_k) + \sum_{j=1}^{d} \ln P(x_j \mid Y = c_k) \right]
$$

## 不同分布假设下的似然模型

### 高斯朴素贝叶斯

假设每个特征服从正态分布：

$$
P(x_j \mid Y = c_k) = \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)
$$

参数估计：

$$
\mu_{kj} = \frac{1}{|D_k|} \sum_{i: y_i = c_k} x_{ij}, \quad \sigma_{kj}^2 = \frac{1}{|D_k|} \sum_{i: y_i = c_k} (x_{ij} - \mu_{kj})^2
$$

### 多项式朴素贝叶斯

适用于离散计数特征（如词频）：

$$
P(x_j \mid Y = c_k) = \frac{N_{kj} + \alpha}{N_k + \alpha \cdot d}
$$

其中 $N_{kj}$ 为类别 $c_k$ 中特征 $j$ 出现的总次数，$\alpha$ 为平滑参数。

### 伯努利朴素贝叶斯

适用于二值特征（出现/不出现）：

$$
P(x_j \mid Y = c_k) = p_{kj}^{x_j} (1 - p_{kj})^{1 - x_j}
$$

## 拉普拉斯平滑

当某个特征值在某类别中从未出现时，$P(x_j \mid Y=c_k) = 0$，导致整个后验概率为零。

**拉普拉斯平滑**为每个计数加 $\alpha$（通常 $\alpha = 1$）：

$$
\hat{P}(x_j = a_{jl} \mid Y = c_k) = \frac{\sum_{i: y_i = c_k} \mathbb{1}(x_{ij} = a_{jl}) + \alpha}{|D_k| + \alpha \cdot S_j}
$$

其中 $S_j$ 是特征 $j$ 的可能取值数。

## 代码对应

```bash
python -m pipelines.classification.naive_bayes
```
