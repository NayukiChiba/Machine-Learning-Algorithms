---
title: KMeans K 均值聚类 — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 KMeans 流水线从数据复制到聚类输出的完整步骤。
2. 理解 KMeans 无训练/测试切分、但拥有 `predict()` 方法的工程特征——与 DBSCAN 有本质差异。
3. 理解 `fit()` 即训练、`labels_` + `cluster_centers_` + `inertia_` 三者共同构成聚类输出的流程特征。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `kmeans_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `StandardScaler` | 类 | 对全量特征做一致性标准化——欧氏距离计算的前置条件 |
| `train_model(...)` | 函数 | 调用 `KMeans.fit()` 执行分配-更新交替迭代，返回模型对象 |
| `model.fit(X_scaled)` | 方法 | 在标准化特征上执行 KMeans 聚类——迭代优化质心位置以最小化簇内平方和 |
| `model.labels_` | 属性 | 每个样本的簇分配标签——KMeans 强制每个点必属一簇，无噪声标记 |
| `model.cluster_centers_` | 属性 | $K$ 个簇的质心坐标——KMeans 区别于 DBSCAN 的标志性输出 |
| `model.inertia_` | 属性 | 收敛时的簇内平方和——衡量聚类紧密度，用于肘部法则选 $K$ |

## 1. 流水线起点：复制数据并拆出特征与对照标签

### 示例代码

```python
data = kmeans_data.copy()
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `.copy()` 确保后续处理不修改全局 `kmeans_data`。
- `true_label` 被单独保存为 `y_true`——它**不参与**后续的 `fit()`，只在最后的 `plot_clusters(...)` 中作为对照显示。
- 这是聚类与分类最根本的工程差异——没有"标签=训练目标"的概念。

## 2. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(400, 2)$ | 去掉 `true_label` 后的全量特征矩阵 | `X` |
| 输出 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- KMeans 流水线**没有**训练/测试切分——无监督聚类不需要验证集。
- `fit_transform` 直接在全量数据上计算统计量并变换——不存在测试集数据泄露的风险。
- 标准化是必须的——欧氏距离 $\|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ 的几何意义依赖于各维度尺度一致。

## 3. 模型训练：`fit()` 执行分配-更新交替迭代

### 参数速览

适用 API：`train_model(X_scaled)` → `model.fit(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 2)$ | 标准化后的全量特征矩阵——KMeans 的唯一输入 | `X_scaled` |
| 无 `y` 参数 | — | KMeans 是无监督算法——`fit(X)` 不接受标签 | — |

### 示例代码

```python
model = train_model(X_scaled)
```

### 理解重点

- `KMeans.fit(X_scaled)` 内部流程：`k-means++` 初始化 $K$ 个质心 → 分配每个点到最近质心 → 更新质心为簇内均值 → 重复直到收敛或达到 `max_iter`。
- 与 DBSCAN 不同——KMeans 的 `fit()` 是迭代优化过程（有 `n_iter_` 属性记录迭代次数），而非一次性密度扩展。
- 与分类模型比较：分类流程是 `fit(X, y)` → `predict(X_test)`，KMeans 流程是 `fit(X)` → `labels_` + `cluster_centers_` + `inertia_`。

## 4. 获取聚类结果

### 参数速览

| 属性名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `labels_` | `ndarray`，形状 $(400,)$ | 每个样本的簇分配标签 $\{0, 1, 2, 3\}$——KMeans 强制分配，无噪声标记 | `model.labels_` |
| `cluster_centers_` | `ndarray`，形状 $(4, 2)$ | $K$ 个簇的质心坐标——KMeans 区别于 DBSCAN 的标志性属性 | `model.cluster_centers_` |
| `inertia_` | `float` | 收敛时的簇内平方和 $\sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | 当前打印到 4 位小数 |
| `n_iter_` | `int` | 收敛所用迭代次数——通常远小于 `max_iter=300` | `model.n_iter_` |

### 示例代码

```python
labels_pred = model.labels_
centers = model.cluster_centers_
```

### 理解重点

- `model.labels_` 对训练样本的簇分配——每个点必属一簇，没有噪声标记。这是 KMeans 与 DBSCAN（有 $-1$ 噪声标签）的关键差异。
- `model.cluster_centers_` 是 KMeans 最有教学意义的输出——它把"中心式聚类"这一直觉直接映射为可观察的坐标。
- `model.inertia_` 在训练日志中打印——它是 KMeans 独有的定量输出，DBSCAN 没有对应物。

## 5. KMeans 的 `predict()` 方法

KMeans 与 DBSCAN 的一个重要工程差异：KMeans **支持**对新样本的簇归属预测。

### 参数速览

适用 API：`model.predict(X_new)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_new` | `array_like`，形状 `(n, 2)` | 经过**同一 `scaler` 标准化后**的新样本特征 | `X_new_scaled` |
| 返回值 | `ndarray`，形状 `(n,)` | 每个新样本的簇编号 $\{0, 1, 2, 3\}$——归属最近质心 | — |

### 示例代码

```python
# 新样本必须先经过训练阶段相同的 scaler 变换
X_new_scaled = scaler.transform(X_new)
new_labels = model.predict(X_new_scaled)
```

### 理解重点

- `predict()` 的原理很简单——计算新样本到 $K$ 个质心的距离，返回最近质心的编号。不需要重新迭代。
- 当前流水线没有演示这一步——教学重点在聚类本身，而非对新样本的预测。
- 这与 DBSCAN 形成鲜明对比——sklearn 的 DBSCAN 根本没有 `predict()` 方法。

## 6. 聚类结果可视化

### 参数速览

适用函数：`plot_clusters(X_scaled, labels_pred=model.labels_, labels_true=y_true, centers=model.cluster_centers_, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 2)$ | 标准化后的全量特征，用于散点图的坐标 | `X_scaled` |
| `labels_pred` | `ndarray`，形状 $(400,)$ | KMeans 的预测簇标签——来自 `model.labels_` | `model.labels_` |
| `labels_true` | `ndarray`，形状 $(400,)$ | 真实簇标签（仅用于视觉对照） | `y_true` |
| `centers` | `ndarray`，形状 $(4, 2)$ | 质心坐标——以红色 `X` 标记显示在预测图中 | `model.cluster_centers_` |

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

- `plot_clusters(...)` 是当前 KMeans 分册唯一的可视化函数——与分类分册的四类评估（混淆矩阵+ROC+决策边界+学习曲线）完全不同。
- `centers` 参数是 KMeans 独有的——DBSCAN 调用 `plot_clusters` 时不传此参数。红色 `X` 标记直观展示每个簇的中心位置。
- 双侧对照布局：左侧显示 `labels_pred`（KMeans 聚类结果 + 质心标记），右侧显示 `labels_true`（真实簇标签）——帮助读者判断算法是否恢复了真实的 4 簇结构。

## 常见坑

1. 期望当前流水线有训练/测试切分——无监督聚类不划分训练集和验证集。
2. 把 `true_label` 当成 `fit()` 的输入——它仅用于最终的可视化对照。
3. 忘记 `predict()` 前需要对新样本做**同一 `scaler` 的 `transform`**——新样本必须经过与训练数据相同的标准化。
4. 把 KMeans 的 `predict()` 与分类模型的 `predict()` 混为一谈——KMeans 只是返回最近质心的编号，没有概率输出。

## 小结

- 当前 KMeans 流水线极为简洁：复制数据 → 剥离 `true_label` → 全量标准化 → `fit(X)` 交替迭代 → `labels_` + `cluster_centers_` + `inertia_` 三输出 → 可视化对照（含质心标记）。
- 与 DBSCAN 的核心差异：KMeans 有 `cluster_centers_`（DBSCAN 没有）、有 `inertia_`（DBSCAN 没有）、有 `predict()`（DBSCAN 没有）、强制分配无噪声点（DBSCAN 有 $-1$ 噪声）、$K$ 必须预设（DBSCAN 由密度自动决定）。
- 与分类分册的核心差异：无切分（无 `train_test_split`）、无监督标签、无 `predict_proba`、无混淆矩阵/ROC/学习曲线。
