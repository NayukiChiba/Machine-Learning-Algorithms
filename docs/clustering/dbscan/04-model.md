---
title: DBSCAN 密度聚类 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `DBSCAN`。
2. 理解 `DBSCAN` 的核心构造器参数（`eps`、`min_samples`、`metric`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`labels_`、`core_sample_indices_`、以及衍生的簇数量和噪声点数量。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.cluster.DBSCAN` 模型，打印聚类统计日志 |
| `DBSCAN(...)` | 类 | scikit-learn 提供的密度聚类器——基于 $\epsilon$ 邻域和密度连通关系 |
| `model.fit(X_train)` | 方法 | 在训练数据上执行密度聚类——注意无监督：只传特征不传标签 |
| `model.labels_` | 属性 | 每个训练样本的簇分配结果，噪声点标记为 $-1$ |
| `model.core_sample_indices_` | 属性 | 核心点在训练数组中的索引位置 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, eps=0.3, min_samples=5, metric='euclidean')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的特征矩阵，形状 $(400, 2)$，传入 `DBSCAN.fit()` | `X_scaled` |
| `eps` | `float` | $\epsilon$ 邻域半径。$\epsilon \uparrow$ → 更大邻域、更多核心点、簇数减少。默认 `0.3` | `0.2`、`0.3`、`0.5`、`1.0` |
| `min_samples` | `int` | 核心点判定阈值 $\text{MinPts}$。值越大，成为核心点的门槛越高。默认 `5` | `3`、`5`、`10` |
| `metric` | `str` | 距离度量方式。默认 `'euclidean'`（欧氏距离 $d = \sqrt{\sum (x_j - z_j)^2}$） | `'euclidean'`、`'manhattan'` |
| 返回值 | `DBSCAN` | 已完成 `fit()` 的模型对象，含 `labels_`、`core_sample_indices_` 等属性 | — |

### 示例代码

```python
from model_training.clustering.dbscan import train_model

model = train_model(X_scaled)
```

### 理解重点

- 当前入口只负责构建一个 `DBSCAN` 并 `fit`——没有参数网格搜索或多度量对比。
- 和监督学习分册的 `train_model` 不同，这里**没有 `y_train` 参数**——DBSCAN 是无监督算法。
- `train_model(...)` 是对 `sklearn.cluster.DBSCAN` 的薄封装——算法本体是 scikit-learn 的 C++ 实现。

## 2. `DBSCAN` 构造器参数

### 参数速览

适用 API：`DBSCAN(eps=0.3, min_samples=5, metric='euclidean')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `eps` | `float` | $\epsilon$ 邻域半径。决定了"多近算邻居"。默认 `0.5`，当前源码显式设为 `0.3` | `0.2`、`0.3`、`0.5`、`1.0` |
| `min_samples` | `int` | 核心点判定阈值 $\text{MinPts}$。决定了"多密算高密度区域"。$d$ 维数据建议 $\geq d+1$，默认 `5` | `3`、`5`、`10`、`20` |
| `metric` | `str` 或 `callable` | 距离度量。`'euclidean'`（欧氏距离）、`'manhattan'`（曼哈顿距离）、`'cosine'`（余弦距离）等。默认 `'euclidean'` | `'euclidean'`、`'manhattan'`、`'cosine'` |
| `algorithm` | `str` | 最近邻搜索算法。`'auto'` 自动选择最优；`'ball_tree'` 球树；`'kd_tree'` KD 树；`'brute'` 暴力搜索。默认 `'auto'` | `'auto'`、`'ball_tree'`、`'kd_tree'`、`'brute'` |
| `leaf_size` | `int` | BallTree 或 KDTree 的叶子节点大小。对构建索引速度和查询速度有影响，不影响聚类结果。默认 `30` | `20`、`30`、`50` |
| `p` | `float` | Minkowski 距离的指数参数。`p=2` 等价于欧氏距离，`p=1` 等价于曼哈顿距离。仅当 `metric='minkowski'` 时生效。默认 `2` | `1`、`2` |
| `n_jobs` | `int` 或 `None` | 并行计算的作业数。`None` 表示 1，`-1` 表示使用所有 CPU。默认 `None` | `None`、`-1`、`4` |

### 示例代码

```python
model = DBSCAN(
    eps=0.3,
    min_samples=5,
    metric="euclidean",
)
model.fit(X_train)
```

### 理解重点

- DBSCAN 的核心参数是 `eps` 和 `min_samples`——两者联合决定了点类型的划分（核心/边界/噪声）和最终的簇结构。
- `eps` 的默认值是 `0.5`，但当前源码显式设为 `0.3`——这是针对标准化后双月牙数据的定制选择（月牙间距约 0.5，`0.3 < 0.5` 避免跨月牙连接）。
- `min_samples=5` 对二维数据是一个合理起点——经验规则 $2d$ 到 $2d+1$，即二维下 4~5。
- `algorithm='auto'`（默认）会根据数据量和特征维度自动选择最近邻搜索方式——对当前 400 样本 2 维数据，通常选用 KD 树。
- 与分类模型（逻辑回归、SVC）的关键差异：DBSCAN 的 `fit()` 不接受标签参数 `y`——`fit(X)` 而非 `fit(X, y)`。

## 3. 训练完成后的关键属性与统计量

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `labels_` | `ndarray`，形状 `(n_samples,)` | 簇分配标签 | 每个训练样本所属簇的编号 $\{0, 1, \dots, k-1\}$，噪声点为 $-1$ |
| `core_sample_indices_` | `ndarray`，形状 `(n_core_samples,)` | 核心点索引集 | 所有核心点在原始训练数组中的下标位置 |
| `components_` | `ndarray`，形状 `(n_core_samples, n_features)` | 核心点特征值 | 所有核心点的特征向量——仅在内存高效模式下可用 |
| `n_features_in_` | `int` | 特征维度 $d$ | 训练时输入的特征维数，当前为 `2` |

### 衍生统计量

| 统计量 | 计算方式 | 说明 |
|---|---|---|
| `n_clusters` | `len(set(labels_)) - (1 if -1 in labels_ else 0)` | 排除噪声后的簇数量——当前期望为 2（两个月牙） |
| `n_noise` | `(labels_ == -1).sum()` | 被标记为噪声的样本数量——反映数据中密度不足的点规模 |

### 示例代码

```python
labels = model.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"簇数量: {n_clusters}")
print(f"噪声点数量: {n_noise}")
```

### 理解重点

- `labels_` 是 DBSCAN 最重要的输出——它不是单个预测值，而是对所有 400 个训练样本的簇分配。
- `-1` 是 DBSCAN 的独特标签——与其他分类器的 `classes_` 不同，它专门标记不满足密度连通条件的噪声点。
- DBSCAN **没有** `cluster_centers_`（KMeans 有）——因为密度聚类不依赖簇中心。
- DBSCAN **没有** `predict()` 方法——sklearn 的 DBSCAN 只能对训练数据本身做聚类，不能预测新样本。这是一个常见的工程限制。

## 4. 训练阶段的工程封装

除了 `DBSCAN(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察密度聚类执行时间 |
| `eps` / `min_samples` 日志 | 确认当前参数配置 |
| `簇数量` 日志 | 快速查看算法发现的簇数——与真实类别数对比 |
| `噪声点数量` 日志 | 观察被识别为离群点的样本规模 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出聚类统计量。
- `簇数量` 和 `噪声点数量` 是 DBSCAN 独有的日志输出——它们直接反映算法的聚类行为。
- 这一层封装把"构建模型""训练模型""打印统计"收在一个函数里，方便文档和流水线复用。

## 常见坑

1. 误以为 `train_model(...)` 需要传入 `y_train`——DBSCAN 是无监督算法，不接受标签。
2. 误以为 DBSCAN 训练完成后能得到簇中心——它没有 `cluster_centers_` 属性。
3. 期望能用 `model.predict(X_new)` 预测新样本——sklearn 的 DBSCAN 不支持，需结合 `NearestNeighbors` 等后处理。
4. 只看 `labels_`，却忽略 `-1`（噪声）、`n_clusters` 和 `n_noise` 才是理解聚类行为的关键统计量。
5. 忘记当前 `X_train` 应该是标准化后的特征——`eps` 是绝对数值，未经标准化的数据会让邻域判定失真。

## 小结

- `train_model(...)` 是本仓库 DBSCAN 的核心训练入口，是对 `sklearn.cluster.DBSCAN` 的薄封装。
- `DBSCAN` 的核心参数是 `eps`（$\epsilon$ 邻域半径）和 `min_samples`（核心点阈值）——两者联合决定簇的形态和噪声规模。
- 训练完成后的关键属性：`labels_`（含噪声标签 $-1$）、`core_sample_indices_`（核心点索引）——通过它们推导 `n_clusters` 和 `n_noise`。
- DBSCAN 没有簇中心、没有 `predict()`、不接收 `y` 标签——这三个"没有"是它与分类模型和 KMeans 最核心的工程差异。
