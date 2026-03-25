# KNN (K-Nearest Neighbors) K 近邻分类

## 核心思想

KNN 是一种**基于实例的懒惰学习**算法：它不构建显式模型，而是在预测时直接从训练集中找到与待分类样本"最近"的 $k$ 个邻居，以多数投票决定分类。

## 距离度量

### 闵可夫斯基距离

给定 $n$ 维空间中两点 $\mathbf{x} = (x_1, x_2, \dots, x_n)$ 与 $\mathbf{y} = (y_1, y_2, \dots, y_n)$，**闵可夫斯基距离**（Minkowski Distance）定义为：

$$
d_p(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}, \quad p \geq 1
$$

特殊情况：

| $p$ 值 | 名称 | 公式 |
|--------|------|------|
| $p=1$ | 曼哈顿距离 | $d_1 = \sum_{i=1}^{n} \lvert x_i - y_i \rvert$ |
| $p=2$ | 欧几里得距离 | $d_2 = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$ |
| $p \to \infty$ | 切比雪夫距离 | $d_\infty = \max_i \lvert x_i - y_i \rvert$ |

### 为什么需要标准化？

当不同特征的量纲差异悬殊时（例如"年收入"以万为单位、"年龄"以十为单位），大值特征将完全主导距离计算。因此必须对所有特征执行 **Z-score 标准化**：

$$
x_i' = \frac{x_i - \mu_i}{\sigma_i}
$$

使得每个特征均值为 0、标准差为 1，确保距离度量对所有特征公平。

## 分类决策规则

### 多数投票法

对于待预测点 $\mathbf{x}$，定义其 $k$ 近邻集合为 $\mathcal{N}_k(\mathbf{x})$，则预测类别为：

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}(y_i = c)
$$

其中 $\mathbb{1}(\cdot)$ 是指示函数，$\mathcal{C}$ 为类别集合。

### 加权投票法

可以考虑距离越近权重越大的加权方案：

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \frac{\mathbb{1}(y_i = c)}{d(\mathbf{x}, \mathbf{x}_i)^2}
$$

## $k$ 值选择的偏差-方差权衡

| $k$ 值 | 偏差 | 方差 | 表现 |
|--------|------|------|------|
| 小 $k$ | 低偏差 | 高方差 | 对噪声敏感，容易过拟合 |
| 大 $k$ | 高偏差 | 低方差 | 决策边界过于平滑，欠拟合 |

最佳 $k$ 值通常通过**交叉验证**确定。

## KD-Tree 加速搜索

暴力搜索的时间复杂度为 $O(n \cdot d)$（$n$ 为样本数，$d$ 为维度）。**KD-Tree** 是一种二叉空间划分树：

1. 选择方差最大的维度作为当前划分维度
2. 找到该维度的中位数，将数据一分为二
3. 递归构建子树

查询时复杂度平均为 $O(\log n)$，但维度 $d$ 较高时退化为接近线性。

## 代码对应

```bash
python -m pipelines.classification.knn
```
