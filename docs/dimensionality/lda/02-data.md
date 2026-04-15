---
title: LDA — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/dimensionality.py`、`data_generation/__init__.py`、`pipelines/dimensionality/lda.py`
>  
> 相关对象：`DimensionalityData.lda()`、`lda_data`

## 本章目标

1. 明确本仓库 LDA 数据来自 `DimensionalityData.lda()` 的真实数据加载逻辑。
2. 明确特征列、标签列以及标签在当前 LDA 流程中的真实作用。
3. 明确当前流程的标准化顺序，以及当前实现没有 train/test split 这一事实。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `DimensionalityData.lda()` | 方法 | 加载 LDA 使用的 Wine 真实数据集 |
| `load_wine(as_frame=True)` | 函数 | scikit-learn 提供的红酒数据集加载器 |
| `lda_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `label` | 列名 | 当前 LDA 训练与可视化都要使用的类别标签 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `lda_data`
- 生成来源：`data_generation/dimensionality.py` 中的 `DimensionalityData.lda()`
- 流水线使用：`pipelines/dimensionality/lda.py` 中的 `data = lda_data.copy()`

### 理解重点

- `lda_data` 在导入时就已经加载完成，因此流水线里直接 `.copy()` 使用即可。
- 使用 `.copy()` 的目的，是避免后续标准化、投影或调试过程意外修改原始数据对象。

## 2. 数据生成函数 `DimensionalityData.lda()`

### 参数速览（本节）

适用 API（分项）：

1. `DimensionalityData.lda()`
2. `load_wine(as_frame=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `as_frame` | `True` | 直接返回带列名的 `DataFrame` |
| 返回值 | `DataFrame` | 含 13 个特征和 `label` 的数据表 |
| `n_samples` | 无效 | 当前方法使用真实数据集，不受该属性控制 |

### 示例代码

```python
data = load_wine(as_frame=True)
df = data.frame.copy().rename(columns={"target": "label"})
return df
```

### 理解重点

- 当前数据不是手工合成结构，而是 Wine 真实数据集。
- 标签列在源码里被统一重命名为 `label`，这让降维分册中的数据接口更一致。
- 这份数据本身类别差异比较明显，非常适合展示 LDA 判别方向。

## 3. 特征列与标签列

当前数据表来自 Wine 数据集，包含 13 个特征和 1 个标签列。

### 参数速览（本节）

适用列组（本节）：

1. 原始特征列
2. 标签列

| 列组 | 当前内容 | 作用 |
|---|---|---|
| 特征列 | `alcohol`、`malic_acid`、`ash`、`alcalinity_of_ash`、... | 提供化学成分信息 |
| 标签列 | `label` | 3 分类目标，也是 LDA 训练所需监督信息 |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"].values
```

### 理解重点

- 和 PCA 不同，当前 LDA 分册里的 `label` 不是只用于着色，而是会真正参与模型训练。
- 这也是 LDA 被称为“有监督降维”的根本原因。
- 当前 Wine 数据集恰好有 3 个类别，因此后续最多只能降到 `K-1 = 2` 维。

## 4. 标准化与当前流程边界

### 参数速览（本节）

适用 API（分项）：

1. `StandardScaler().fit_transform(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | 13 个原始特征组成的矩阵 | 参与标准化的原始特征 |
| 返回值 | `X_scaled` | 标准化后的输入特征 |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- 当前 LDA 流水线会先对全量特征做标准化，再训练 LDA。
- 由于这里没有 train/test split，因此不存在“只在训练集拟合标准化器”这一监督学习式边界。
- 标准化在这里也很重要，否则不同量纲会影响类内/类间散度结构。

## 5. 为什么当前只训练 2D LDA

### 参数速览（本节）

适用事实（本节）：

1. 当前类别数 `K = 3`
2. 降维上限 `K - 1 = 2`

| 项目 | 当前含义 |
|---|---|
| 类别数 | 3 |
| 理论最大判别维度 | 2 |

### 理解重点

- 当前实现只训练 `n_components=2`，并不是随意选择，而是因为 3 类 LDA 的理论上限本来就是 2 维。
- 这和 PCA 可以继续训练 3D 模型的逻辑不同。
- 文档必须把这个数学和工程上的一致性明确写出来。

## 常见坑

1. 把 `label` 当成像 PCA 那样只用于着色的辅助列，忽略它其实参与训练。
2. 忽略当前数据是 3 类真实数据集，误写成任意维度都可以继续降。
3. 把监督学习里的 train/test split 惯性写进当前 LDA 流程，和源码不一致。

## 小结

- 当前 LDA 数据来自 `DimensionalityData.lda()`，底层使用的是 `load_wine(as_frame=True)`。
- 数据表结构很清晰：13 个特征是训练输入，`label` 既参与训练，也用于可视化着色。
- 读懂数据来源、标签边界和标准化顺序，是理解后续 LDA 训练与判别投影图的前提。
