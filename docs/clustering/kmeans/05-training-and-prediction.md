---
title: KMeans K 均值聚类 — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/clustering/kmeans.py`、`model_training/clustering/kmeans.py`
>
> 运行方式：`python -m pipelines.clustering.kmeans`

## 本章目标

1. 按源码顺序看清当前 KMeans 流水线到底执行了哪些步骤。
2. 理解聚类场景下“训练结果输出”和监督学习里的“预测”有何不同。
3. 明确 `labels_`、`cluster_centers_` 与可视化之间的连接关系。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `kmeans_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `data.drop(columns=["true_label"])` | 操作 | 去掉参考标签列，保留训练特征 |
| `StandardScaler().fit_transform(X)` | 方法 | 标准化特征 |
| `train_model(X_scaled)` | 函数 | 训练 KMeans 模型 |
| `model.labels_` | 属性 | 返回训练样本的簇分配结果 |
| `model.cluster_centers_` | 属性 | 返回最终簇中心 |
| `plot_clusters(...)` | 函数 | 绘制聚类结果与真实标签对照图 |

## 1. 流水线从复制数据开始

### 示例代码

```python
data = kmeans_data.copy()
```

### 理解重点

- 当前流水线先复制 `kmeans_data`，这样后续即使做列拆分或其他处理，也不会影响原始数据对象。
- 这种写法和回归分册保持一致，体现了“原始数据只读、流程内部再处理”的习惯。

## 2. 拆出 `y_true`，但不把它送进模型

### 示例代码

```python
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `y_true` 在当前仓库里只用于训练后的结果对照，不参与 `fit(...)`。
- `X` 才是送进 KMeans 的真正输入特征。
- 这是聚类文档里最需要反复强调的一点：当前任务是无监督训练，不是分类。

## 3. 训练前先做标准化

### 参数速览（本节）

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 当前对象 | 说明 |
|---|---|---|
| `X` | `x1`、`x2` 组成的特征表 | KMeans 训练输入 |
| 返回值 | `X_scaled` | 标准化后的特征矩阵 |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- KMeans 通过距离做簇分配，因此特征必须处于可比较的尺度。
- 当前流水线没有拆训练集/测试集，所以直接对全量 `X` 做 `fit_transform(...)`。
- 这里和监督学习分册不同，不涉及“只在训练集上拟合标准化统计量”的问题。

## 4. 训练步骤实际只有一次 `train_model(...)`

### 示例代码

```python
model = train_model(X_scaled)
```

### 理解重点

- 这里没有 `y_train`，也没有单独验证集。
- `train_model(...)` 内部会创建 `KMeans(...)` 并执行 `model.fit(X_train)`。
- 训练结束后，模型对象内部已经包含当前样本的聚类结果和簇中心。

## 5. 当前实现里的“预测”如何体现

在监督学习里，“预测”通常指对未见样本调用 `predict(...)`。当前 KMeans 流水线没有单独写这一步，而是直接使用训练完成后的模型属性：

### 示例代码

```python
labels_pred = model.labels_
centers = model.cluster_centers_
```

### 理解重点

- `model.labels_` 是对训练样本的簇分配结果，可以理解为“当前数据的聚类输出”。
- `model.cluster_centers_` 是聚类中心坐标，用来说明每个簇最终落在什么位置。
- 当前文档要如实说明：本流水线没有单独展示 `model.predict(new_X)`，但这并不影响读者理解 KMeans 的核心流程。

## 6. 训练结果如何进入可视化

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    centers=model.cluster_centers_,
    title="KMeans 聚类分布",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- `labels_pred` 决定左图的预测簇着色。
- `labels_true` 决定右图的真实标签着色，用于对照观察。
- `centers` 会以红色 `X` 标记显示在预测图中，帮助读者直观看到每个簇中心的位置。

## 7. 如果要对新样本分簇，应如何理解

当前源码没有单独演示这一点，但从 `KMeans` 的常规用法看，流程通常是：

1. 使用训练阶段相同的 `scaler` 变换新样本。
2. 调用 `model.predict(new_X_scaled)` 获取簇编号。

### 理解重点

- 这属于 KMeans 的通用能力，但不是当前仓库这条教学流水线的重点。
- 文档可以简要提及，但不能写成“当前实现的主流程”。

## 常见坑

1. 把 `labels_` 误认为真实类别标签。
2. 误以为聚类场景必须像监督学习一样区分 `fit` 和 `predict` 两个阶段。
3. 忘记训练前先删除 `true_label`。
4. 把新样本预测流程写成当前流水线里已经存在的实现。

## 小结

- 当前 KMeans 流水线的训练过程非常直接：复制数据、拆出 `y_true`、标准化、训练模型、绘图展示。
- 对本仓库而言，`model.labels_` 就是训练样本的聚类输出，`model.cluster_centers_` 是最核心的学习结果。
- 把这条链路看清楚后，再读评估与工程实现章节会更容易建立全局理解。
