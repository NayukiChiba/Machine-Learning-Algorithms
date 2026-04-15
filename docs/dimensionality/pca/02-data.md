---
title: PCA — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/dimensionality.py`、`data_generation/__init__.py`、`pipelines/dimensionality/pca.py`
>  
> 相关对象：`DimensionalityData.pca()`、`pca_data`

## 本章目标

1. 明确本仓库 PCA 数据来自 `DimensionalityData.pca()` 的低秩高维构造逻辑。
2. 明确特征列、可视化标签列以及它们在流水线中的边界。
3. 明确当前流程的标准化顺序，以及当前实现没有 train/test split 这一事实。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `DimensionalityData.pca()` | 方法 | 生成 PCA 使用的高维合成数据 |
| `pca_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `x1` ~ `x10` | 列名 | 当前流水线中的原始高维特征 |
| `label` | 列名 | 仅用于可视化着色的伪标签 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `pca_data`
- 生成来源：`data_generation/dimensionality.py` 中的 `DimensionalityData.pca()`
- 流水线使用：`pipelines/dimensionality/pca.py` 中的 `data = pca_data.copy()`

### 理解重点

- `pca_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 使用 `.copy()` 的目的，是避免后续标准化、投影或调试过程意外修改原始数据对象。

## 2. 数据生成函数 `DimensionalityData.pca()`

### 参数速览（本节）

适用参数（本节）：

1. `n_samples`
2. `pca_n_features`
3. `pca_n_informative`
4. `pca_noise_std`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_samples` | `400` | 样本总数 |
| `pca_n_features` | `10` | 原始特征维度 |
| `pca_n_informative` | `3` | 真正有信息的独立方向数 |
| `pca_noise_std` | `0.5` | 各向同性高斯噪声标准差 |
| `random_state` | `42` | 随机种子，保证可复现 |
| 返回值 | `DataFrame` | 含 `x1` ~ `x10` 与 `label` 的数据表 |

### 示例代码

```python
base = rng.randn(self.n_samples, self.pca_n_informative)
projection = rng.randn(self.pca_n_informative, self.pca_n_features)
X = base @ projection
X += rng.randn(self.n_samples, self.pca_n_features) * self.pca_noise_std
```

### 理解重点

- 当前数据不是普通随机高维点，而是“低秩结构先生成，再映射到高维空间”的合成数据。
- 这样做的目的，是让数据虽然表面上有 10 维，但真正有信息的方向只有 3 个左右。
- 这正好非常适合演示 PCA 如何通过主成分提取主要方差结构。

## 3. `label` 的语义与边界

当前数据表结构如下：

- 特征列：`x1` ~ `x10`
- 可视化标签列：`label`

### 参数速览（本节）

适用列组（本节）：

1. 原始特征列
2. 伪标签列

| 列名 | 当前作用 |
|---|---|
| `x1` ~ `x10` | 真正参与 PCA 训练的高维特征 |
| `label` | 仅用于降维图着色，帮助观察结构 |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"].values
```

### 理解重点

- `label` 不是监督学习里的训练标签，它只用于降维图着色和结构观察。
- 当前训练函数 `train_model(...)` 只接收标准化后的特征矩阵，不会使用 `label`。
- 这一点必须反复区分清楚，否则很容易把当前 PCA 流程误读成监督学习。

## 4. 为什么当前数据特别适合讲解释方差比

### 参数速览（本节）

适用结构（本节）：

1. 低秩主方向
2. 噪声维度

| 结构 | 当前含义 |
|---|---|
| 主方向 | 3 个真正有信息的独立方向 |
| 其余维度 | 被投影扩展出来并掺杂噪声的高维表现 |

### 理解重点

- 如果数据每个维度都同样重要，PCA 的主成分解释就不会特别集中。
- 当前数据故意让信息主要集中在少数方向上，因此解释方差比会更有教学意义。
- 这也是为什么当前分册会把 `explained_variance_ratio_` 和累计解释方差作为重要输出。

## 5. 标准化与当前流程边界

### 参数速览（本节）

适用 API（分项）：

1. `StandardScaler().fit_transform(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | `x1` ~ `x10` 组成的特征矩阵 | 参与标准化的原始特征 |
| 返回值 | `X_scaled` | 标准化后的高维特征 |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- 当前 PCA 流水线会先对全量特征做标准化，再训练 PCA。
- 由于这里没有 train/test split，因此不存在“只在训练集拟合标准化器”这一监督学习式边界。
- 对 PCA 来说，标准化尤其重要，否则大尺度特征会直接主导主成分方向。

## 常见坑

1. 把 `label` 当成训练标签，误以为当前 PCA 是监督学习流程。
2. 忽略当前数据是“低秩结构 + 噪声”的教学构造，误把它当成普通随机高维数据。
3. 把监督学习里的 train/test split 惯性写进当前 PCA 流程，和源码不一致。

## 小结

- 当前 PCA 数据来自 `DimensionalityData.pca()`，底层是手工构造的低秩高维数据。
- 数据表结构很清晰：`x1` ~ `x10` 是训练输入，`label` 只用于可视化着色。
- 读懂数据来源、低秩结构设计和标准化顺序，是理解后续 PCA 训练与降维图像的前提。
