---
title: DBSCAN 密度聚类 — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/clustering/dbscan.py`、`model_training/clustering/dbscan.py`
>
> 运行方式：`python -m pipelines.clustering.dbscan`

## 本章目标

1. 按源码顺序看清当前 DBSCAN 流水线到底执行了哪些步骤。
2. 理解聚类场景下“训练结果输出”和监督学习里的“预测”有何不同。
3. 明确 `labels_` 与可视化之间的连接关系，以及为什么当前实现不强调新样本预测。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `dbscan_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `data.drop(columns=["true_label"])` | 操作 | 去掉参考标签列，保留训练特征 |
| `StandardScaler().fit_transform(X)` | 方法 | 标准化特征 |
| `train_model(X_scaled)` | 函数 | 训练 DBSCAN 模型 |
| `model.labels_` | 属性 | 返回训练样本的簇分配结果 |
| `plot_clusters(...)` | 函数 | 绘制聚类结果与真实标签对照图 |

## 1. 流水线从复制数据开始

### 示例代码

```python
data = dbscan_data.copy()
```

### 理解重点

- 当前流水线先复制 `dbscan_data`，这样后续即使做列拆分或其他处理，也不会影响原始数据对象。
- 这种写法和回归分册保持一致，体现了“原始数据只读、流程内部再处理”的习惯。

## 2. 拆出 `y_true`，但不把它送进模型

### 示例代码

```python
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `y_true` 在当前仓库里只用于训练后的结果对照，不参与 `fit(...)`。
- `X` 才是送进 DBSCAN 的真正输入特征。
- 这是聚类文档里最需要反复强调的一点：当前任务是无监督训练，不是分类。

## 3. 训练前先做标准化

### 参数速览（本节）

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 当前对象 | 说明 |
|---|---|---|
| `X` | `x1`、`x2` 组成的特征表 | DBSCAN 训练输入 |
| 返回值 | `X_scaled` | 标准化后的特征矩阵 |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- DBSCAN 用 `eps` 来定义邻域半径，因此特征必须处于可比较的尺度。
- 当前流水线没有拆训练集/测试集，所以直接对全量 `X` 做 `fit_transform(...)`。
- 这里和监督学习分册不同，不涉及“只在训练集上拟合标准化统计量”的问题。

## 4. 训练步骤实际只有一次 `train_model(...)`

### 示例代码

```python
model = train_model(X_scaled)
```

### 理解重点

- 这里没有 `y_train`，也没有单独验证集。
- `train_model(...)` 内部会创建 `DBSCAN(...)` 并执行 `model.fit(X_train)`。
- 训练结束后，模型对象内部已经包含当前样本的聚类标签结果。

## 5. 当前实现里的“预测”如何体现

在监督学习里，“预测”通常指对未见样本调用 `predict(...)`。当前 DBSCAN 流水线没有单独写这一步，而是直接使用训练完成后的模型属性：

### 示例代码

```python
labels_pred = model.labels_
```

### 理解重点

- `model.labels_` 是对训练样本的簇分配结果，可以理解为“当前数据的聚类输出”。
- 与 KMeans 不同，标准 `sklearn.cluster.DBSCAN` 并不提供同风格的 `predict(new_X)` 接口。
- 因此本分册更适合把重点放在“训练后标签输出”而不是“新样本预测”。

## 6. 训练结果如何进入可视化

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    title="DBSCAN 聚类分布",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- `labels_pred` 决定左图的预测簇着色，其中 `-1` 对应噪声点。
- `labels_true` 决定右图的真实标签着色，用于对照观察。
- 当前 DBSCAN 流水线没有传入 `centers`，因为 DBSCAN 本身没有簇中心这一结果对象。

## 7. 为什么这里不强调新样本预测

### 理解重点

- 对当前实现来说，最重要的是理解 DBSCAN 如何给现有样本分簇并识别噪声。
- 新样本如何增量归类，不是当前源码展示的重点，也不是标准 `DBSCAN` 在 scikit-learn 中最直接提供的接口能力。
- 因此“训练与预测”这一章在 DBSCAN 分册里，应被理解为“训练与标签输出”。

## 常见坑

1. 把 `labels_` 误认为真实类别标签。
2. 误以为 DBSCAN 必须像监督学习一样区分 `fit` 和 `predict` 两个阶段。
3. 忘记训练前先删除 `true_label`。
4. 把新样本预测流程写成当前流水线里已经存在的实现。

## 小结

- 当前 DBSCAN 流水线的训练过程非常直接：复制数据、拆出 `y_true`、标准化、训练模型、绘图展示。
- 对本仓库而言，`model.labels_` 就是训练样本的聚类输出，而噪声点会通过 `-1` 显式体现。
- 把这条链路看清楚后，再读评估与工程实现章节会更容易建立全局理解。
