---
title: KMeans K 均值聚类 — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/clustering.py`、`data_generation/__init__.py`、`pipelines/clustering/kmeans.py`
>
> 相关对象：`ClusteringData.kmeans()`、`kmeans_data`

## 本章目标

1. 明确本仓库 KMeans 数据来自 `ClusteringData.kmeans()` 的 blob 生成逻辑。
2. 明确特征列与 `true_label` 在当前流水线中的角色差异。
3. 明确标准化发生在什么位置，以及它为什么对 KMeans 很重要。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClusteringData.kmeans()` | 方法 | 生成 KMeans 使用的二维聚类数据 |
| `make_blobs(...)` | 函数 | scikit-learn 提供的多簇高斯 blob 数据生成器 |
| `kmeans_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `true_label` | 列名 | 真实簇标签，仅用于结果对照，不参与训练 |
| `StandardScaler` | 类 | 对特征做标准化，避免距离被量纲主导 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `kmeans_data`
- 生成来源：`data_generation/clustering.py` 中的 `ClusteringData.kmeans()`
- 流水线使用：`pipelines/clustering/kmeans.py` 中的 `data = kmeans_data.copy()`

### 理解重点

- `kmeans_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 用 `.copy()` 的目的，是避免后续处理意外修改原始数据对象。
- 当前数据是为 KMeans 教学场景专门构造的，因此和算法假设高度匹配。

## 2. 数据生成函数 `ClusteringData.kmeans()`

### 参数速览（本节）

适用 API（分项）：

1. `ClusteringData.kmeans()`
2. `make_blobs(n_samples=self.n_samples, centers=self.kmeans_centers, cluster_std=self.kmeans_cluster_std, random_state=self.random_state)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_samples` | `400` | 样本数，来自 `ClusteringData` 默认属性 |
| `centers` | `4` | 生成 4 个簇中心 |
| `cluster_std` | `0.8` | 控制簇内离散程度 |
| `random_state` | `42` | 随机种子，保证数据可复现 |
| 返回值 | `DataFrame` | 含 `x1`、`x2` 与 `true_label` 的数据表 |

### 示例代码

```python
X, y = make_blobs(
    n_samples=self.n_samples,
    centers=self.kmeans_centers,
    cluster_std=self.kmeans_cluster_std,
    random_state=self.random_state,
)
return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})
```

### 理解重点

- `make_blobs(...)` 生成的是多个近似球形的高斯簇，这正是 KMeans 最擅长处理的数据形态。
- 当前数据只有两个特征维度，因此聚类结果可以直接画成二维散点图。
- `true_label` 是数据生成器自带的参考答案，方便后续和预测簇结果做视觉对照。

## 3. 特征列与 `true_label` 的角色

当前数据表结构如下：

- 特征列：`x1`、`x2`
- 参考标签列：`true_label`

### 示例代码

```python
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `true_label` 不是监督学习里的训练标签，不能直接送进 KMeans。
- 当前流水线先把 `true_label` 单独保存到 `y_true`，只用于绘图对照。
- 如果把 `true_label` 误当成普通特征一起输入模型，相当于把答案信息泄露给聚类过程，会破坏演示意义。

## 4. 标准化为什么放在训练前

### 参数速览（本节）

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | 去掉 `true_label` 后的二维特征 | KMeans 的实际输入 |
| 返回值 | `X_scaled` | 标准化后的特征矩阵 |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- KMeans 的簇分配完全依赖距离，因此特征量纲会直接影响结果。
- 当前 `x1`、`x2` 都来自同一个数据生成器，尺度问题不一定严重，但流水线仍然保留标准化步骤，体现规范做法。
- 在更一般的业务数据里，如果不先标准化，数值范围大的特征会主导聚类中心的位置。

## 常见坑

1. 把 `true_label` 当成普通特征一起送进模型。
2. 误以为 `true_label` 是当前 KMeans 的训练目标。
3. 忽略标准化，让距离计算被量纲主导。
4. 看到二维散点图效果很好，就误以为 KMeans 在任意形状数据上都同样有效。

## 小结

- 当前 KMeans 数据来自 `ClusteringData.kmeans()`，底层使用的是 `make_blobs(...)`。
- 数据表结构清晰：`x1`、`x2` 是特征，`true_label` 是仅用于对照的真实簇标签。
- 读懂数据来源、标签角色和标准化位置，是理解后续模型构建与训练流程的前提。
