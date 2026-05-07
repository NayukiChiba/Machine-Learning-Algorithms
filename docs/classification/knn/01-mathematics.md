---
title: KNN K 近邻分类 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 KNN 为什么不通过显式参数优化边界，而是通过邻域关系完成分类。
2. 理解闵可夫斯基距离、多数投票、加权投票和 $k$ 值在当前实现中的数学角色。
3. 理解为什么标准化会直接影响 KNN 的预测结果——距离型模型对特征尺度敏感。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 闵可夫斯基距离 $d_p(\mathbf{x}, \mathbf{y})$ | 距离度量 | 当前实现 `metric='minkowski'` 的底层框架，$p=2$ 时退化为欧几里得距离 |
| 多数投票 | 决策规则 | 当前默认 `weights='uniform'` 对应的预测方式，$k$ 个邻居等权投票 |
| 加权投票 | 决策规则 | `weights='distance'` 对应的加权方式，邻居越近权重越大 |
| $k$ 值 | 超参数 | 决定投票邻域范围，直接控制偏差-方差权衡 |
| 标准化 $x_i' = (x_i - \mu_i)/\sigma_i$ | 预处理 | KNN 的距离计算依赖特征尺度，不标准化会导致量纲大的特征主导近邻判断 |
| KD-Tree | 加速结构 | 通过空间划分在低维场景中加速近邻查询，$O(\log n)$ 平均复杂度 |

## 1. KNN 的核心思想

KNN（K-Nearest Neighbors）是一种基于实例的懒惰学习算法。它不通过最小化损失函数来学习一组显式参数，而是在预测时直接在训练集中寻找距离最近的 $k$ 个样本，由它们的类别分布决定输出。

### 理解重点

- KNN 的核心不是先学一条全局边界，而是"看待预测点周围有哪些样本"。
- 这使 KNN 对局部结构非常敏感，天然能适应非线性边界。
- 同时，这也意味着它对距离定义和数据尺度特别敏感——距离变了，近邻关系就变了。

## 2. 闵可夫斯基距离：定义"近"的数学框架

闵可夫斯基距离是当前源码 `metric='minkowski'` 对应的底层框架，通过参数 $p$ 控制距离类型。

### 参数速览

适用参数：`metric`、`p`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `metric` | `str` | 距离度量方式。默认为 `"minkowski"`，是一条通用距离框架：$d_p(\mathbf{x}, \mathbf{y}) = (\sum_{i=1}^{d} \vert x_i - y_i\vert^p)^{1/p}$。$p$ 通过独立参数 `p` 控制 | `"minkowski"`、`"euclidean"`、`"manhattan"` |
| `p` | `int` | 闵可夫斯基距离的幂参数。$p=1$ 为曼哈顿距离，$p=2$ 为欧几里得距离。默认为 `2` | `1`、`2` |

闵可夫斯基距离的一般形式：

$$
d_p(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{d} |x_i - y_i|^p \right)^{1/p}, \quad p \geq 1
$$

三种常见特例：

| $p$ 值 | 名称 | 公式 | 几何直觉 |
|---|---|---|---|
| $p=1$ | 曼哈顿距离 | $d_1 = \sum_{i=1}^{d} \vert x_i - y_i \vert$ | 只能沿坐标轴移动 |
| $p=2$ | 欧几里得距离 | $d_2 = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}$ | 直线距离（默认） |
| $p \to \infty$ | 切比雪夫距离 | $d_\infty = \max_i \vert x_i - y_i \vert$ | 只考虑最大分量差 |

### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier

# 默认使用闵可夫斯基距离，p=2（等同欧几里得）
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# 显式使用曼哈顿距离
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)

# 等价写法
model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
```

### 理解重点

- 当前源码没有显式设置 `p`，因此使用默认值 $p=2$（欧几里得距离）。
- 一旦距离定义改变，近邻集合和最终分类结果也会随之变化。
- `metric='minkowski'` 是 sklearn 的默认值，它不独立指定距离类型，而是和 `p` 参数配合使用。

## 3. 为什么必须标准化

当不同特征的量纲差异悬殊时，大值特征会主导距离计算（在闵可夫斯基公式中，量纲大的分量对和的贡献更大），因此标准化对 KNN 是必需的：

$$
x_i' = \frac{x_i - \mu_i}{\sigma_i}
$$

其中 $\mu_i$ 和 $\sigma_i$ 是特征 $i$ 在训练集上的均值和标准差。标准化后每个特征均值为 0、标准差为 1，所有特征在距离计算中得到平等对待。

### 参数速览

适用类：`sklearn.preprocessing.StandardScaler`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `with_mean` | `bool` | 是否中心化（减去均值）。默认为 `True` | `True` |
| `with_std` | `bool` | 是否缩放（除以标准差）。默认为 `True` | `True` |
| `copy` | `bool` | 是否复制输入数据。默认为 `True` | `True` |

### 示例代码

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # 训练集上拟合统计量并变换
X_test_s = scaler.transform(X_test)        # 测试集只变换，用训练集的统计量
```

### 理解重点

- 对 KNN 来说，标准化不是锦上添花，而是距离型模型几乎必备的预处理。
- 如果不标准化，量纲大的特征会主导 $\vert x_i - y_i \vert^p$ 的计算，使得远近关系完全失真。
- 这也是 KNN 与决策树在当前仓库中最关键的工程差异之一——决策树基于阈值切分，不依赖距离尺度。

## 4. 分类决策规则

### 多数投票法（当前默认）

对待预测点 $\mathbf{x}$，定义其 $k$ 个最近邻集合为 $\mathcal{N}_k(\mathbf{x})$，则预测类别为：

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}(y_i = c)
$$

$k$ 个邻居每人一票，得票最多的类别胜出。平票时按 sklearn 内部规则处理（默认选择类别标签最小的那个）。

### 加权投票法

考虑距离越近权重越大的方案（`weights='distance'`）：

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \frac{\mathbb{1}(y_i = c)}{d(\mathbf{x}, \mathbf{x}_i)}
$$

权重与距离成反比——邻居越近，投票权重越大。

### 参数速览

适用参数：`weights`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `weights` | `str` | 投票权重方式。`"uniform"` 为等权投票 $\hat{y} = \arg\max_c \sum \mathbb{1}(y_i=c)$；`"distance"` 为距离倒数加权 $\hat{y} = \arg\max_c \sum \frac{\mathbb{1}(y_i=c)}{d(\mathbf{x}, \mathbf{x}_i)}$。默认为 `"uniform"` | `"uniform"`、`"distance"` |

### 示例代码

```python
# 多数投票（当前默认）
model = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# 距离加权投票
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

### 理解重点

- 当前源码默认 `weights='uniform'`，对应多数投票直觉。
- 如果改成 `'distance'`，邻居越近投票影响越大，边界通常更精细，但对噪声也更敏感。
- 这是 KNN 分册里应该重点解释的两个投票策略之一。

## 5. $k$ 值的偏差-方差权衡

$k$ 是 KNN 最核心的超参数，它直接控制决策的局部性程度。

| $k$ 值 | 偏差 | 方差 | 决策行为 |
|---|---|---|---|
| 小 $k$（如 $k=1$） | 低偏差 | 高方差 | 边界紧密贴合训练样本，对噪声异常敏感，容易过拟合 |
| 大 $k$（如 $k=50$） | 高偏差 | 低方差 | 边界过度平滑，丢失局部结构信息，容易欠拟合 |

### 参数速览

适用参数：`n_neighbors`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_neighbors` | `int` | 近邻数量 $k$。$k$ 越小 → 偏差低、方差高、边界精细、易过拟合；$k$ 越大 → 偏差高、方差低、边界平滑、易欠拟合。默认 `5` | `1`、`5`、`15`、`50` |

### 示例代码

```python
model = KNeighborsClassifier(n_neighbors=5)
```

### 理解重点

- $k=1$ 时每个训练样本自身就是一个 Voronoi 区域中心，训练误差为 0 但泛化差。
- $k=5$ 是 sklearn 默认值，也是教学上最常见的起点，兼顾了局部性和稳定性。
- $k$ 通常设为奇数以避免二分类平票，但多分类场景中平票仍可能发生。

## 6. 概率估计

KNN 的概率输出基于邻域内各类别占比：

$$
P(\hat{y} = c \mid \mathbf{x}) = \frac{1}{k} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}(y_i = c)
$$

对于 `weights='distance'` 的情况，概率为加权占比：

$$
P(\hat{y} = c \mid \mathbf{x}) = \frac{\sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} w_i \cdot \mathbb{1}(y_i = c)}{\sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}
$$

### 理解重点

- KNN 的概率输出本质上是邻域内的类别频率，这不同于逻辑回归通过 sigmoid 映射得分到概率。
- 由于 $k$ 较小，概率值只取离散值（如 $k=5$ 时概率只能是 $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$），看起来不如其他模型的概率"平滑"。
- 这些概率是 ROC 曲线的直接输入——需要连续变化的阈值才能画出 TPR/FPR 轨迹。

## 7. 数学原理如何映射到当前源码

以下表格将本章涉及的数学概念与当前仓库的代码实现一一对应：

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 闵可夫斯基距离 | $d_p(\mathbf{x}, \mathbf{y}) = (\sum \vert x_i - y_i \vert^p)^{1/p}$ | `metric='minkowski'`，$p=2$（默认） |
| 欧几里得距离 | $d_2 = \sqrt{\sum (x_i - y_i)^2}$ | `p=2`（默认，未显式写出） |
| 多数投票 | $\hat{y} = \arg\max_c \sum \mathbb{1}(y_i = c)$ | `weights='uniform'`（默认） |
| 加权投票 | $\hat{y} = \arg\max_c \sum \frac{\mathbb{1}(y_i=c)}{d(\mathbf{x}, \mathbf{x}_i)}$ | `weights='distance'` |
| 邻域大小 | $k$ | `n_neighbors=5` |
| 概率估计 | $P(c\vert\mathbf{x}) = \frac{1}{k} \sum \mathbb{1}(y_i=c)$ | `model.predict_proba(X)` |
| 标准化 | $x_i' = (x_i - \mu_i) / \sigma_i$ | `StandardScaler().fit_transform(X_train)` |
| KD-Tree 查询 | — | `algorithm='auto'`（默认，自动选择） |

## 常见坑

1. 把 KNN 当成"会自动学出参数边界"的模型——它是懒惰学习，`fit()` 只存储数据，不做优化。
2. 忽略标准化，让距离关系完全失真——量纲大的特征会主导 $\vert x_i - y_i \vert^p$。
3. 只会机械调 `k`，却不理解它对应偏差-方差权衡——小 $k$ 低偏差高方差，大 $k$ 高偏差低方差。
4. 把加权投票写成当前默认行为——源码默认 `weights='uniform'`，加权投票需要显式设置。
5. 混淆概率估计的来源——KNN 概率来自邻域频率，不是连续函数映射，取值是离散的（分母为 $k$）。

## 小结

- KNN 的核心数学：用闵可夫斯基距离 $d_p(\mathbf{x}, \mathbf{y})$ 定义近邻，再用投票规则 $\arg\max_c \sum \mathbb{1}(y_i=c)$ 完成分类。
- $k$（`n_neighbors`）、`weights`、`metric`/`p` 和标准化共同决定模型行为——哪一个变了，近邻关系和分类结果都会变。
- KNN 不通过最小化损失函数学习参数，`fit()` 只是存储训练数据，所有计算发生在 `predict()` 阶段。
