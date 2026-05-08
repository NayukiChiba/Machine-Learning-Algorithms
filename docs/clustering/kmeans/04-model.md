---
title: KMeans K 均值聚类 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `KMeans`。
2. 理解 `KMeans` 的核心构造器参数（`n_clusters`、`init`、`n_init`、`max_iter`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`cluster_centers_`（质心）、`labels_`（簇分配）、`inertia_`（簇内平方和）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.cluster.KMeans` 模型，打印 `inertia_` 日志 |
| `KMeans(...)` | 类 | scikit-learn 提供的 K 均值聚类器——基于交替最小化簇内平方和 |
| `model.fit(X_train)` | 方法 | 执行分配-更新交替迭代直至收敛——无监督，不传入标签 |
| `model.cluster_centers_` | 属性 | $K$ 个簇的质心坐标——KMeans 区别于 DBSCAN 的标志性属性 |
| `model.inertia_` | 属性 | 收敛时的簇内平方和——衡量簇紧密度，用于肘部法则选 $K$ |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的特征矩阵，形状 $(400, 2)$，传入 `KMeans.fit()` | `X_scaled` |
| `n_clusters` | `int` | 预设簇数 $K$。当前设为 `4`，与 `make_blobs(centers=4)` 一致 | `3`、`4`、`5` |
| `init` | `str` | 质心初始化策略。`'k-means++'`（默认）使用加权随机采样使初始质心分散 | `'k-means++'`、`'random'` |
| `n_init` | `int` | 不同初始质心下独立运行 KMeans 的次数，返回 `inertia_` 最小的结果。默认 `10` | `1`、`10`、`20` |
| `max_iter` | `int` | 单次运行的最大迭代次数。默认 `300`——对 400 样本通常远不需要这么多 | `100`、`300` |
| `random_state` | `int` | 随机种子，保证质心初始化和结果可复现。默认 `42` | `42` |
| 返回值 | `KMeans` | 已完成 `fit()` 的模型对象，含 `cluster_centers_`、`labels_`、`inertia_` | — |

### 示例代码

```python
from model_training.clustering.kmeans import train_model

model = train_model(X_scaled)
```

### 理解重点

- 和监督学习分册不同，`train_model(...)` **没有 `y_train` 参数**——KMeans 是无监督算法。
- `n_clusters=4` 是必须预设的前置条件——KMeans 不会自己决定簇数，需要用户根据领域知识或肘部法则选定。
- `train_model(...)` 是对 `sklearn.cluster.KMeans` 的薄封装——算法本体是 scikit-learn 基于 Lloyd 算法的高效实现。

## 2. `KMeans` 构造器参数

### 参数速览

适用 API：`KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_clusters` | `int` | 预设簇数 $K$。当前设为 `4`。这是 KMeans 最重要的参数——选错直接导致错误合并或分裂 | `3`、`4`、`5`、`8` |
| `init` | `str` | 质心初始化方法。`'k-means++'`（默认）加权随机采样；`'random'` 纯随机选 $K$ 个点。scikit-learn 1.2+ 还支持传入 `ndarray` 自定义初始质心 | `'k-means++'`、`'random'` |
| `n_init` | `int` 或 `'auto'` | 不同初始质心下的独立运行次数，返回 `inertia_` 最小的结果。`'auto'`（scikit-learn 1.4+ 默认）在 `init='k-means++'` 时自动设为 `1`。当前显式设为 `10` | `1`、`10`、`'auto'` |
| `max_iter` | `int` | 单次运行的最大迭代次数。默认 `300`——大多数数据远在此之前就收敛了 | `100`、`300`、`500` |
| `tol` | `float` | 收敛容忍度——当质心位移的 Frobenius 范数相对变化小于此值时停止迭代。默认 `1e-4` | `1e-3`、`1e-4` |
| `algorithm` | `str` | 计算后端。`'lloyd'`（经典 EM 风格）、`'elkan'`（三角不等式加速，适合密集数据）、`'auto'`。默认 `'lloyd'` | `'lloyd'`、`'elkan'` |
| `random_state` | `int` | 随机种子，保证质心初始化可复现。当前设为 `42` | `42` |

### 示例代码

```python
model = KMeans(
    n_clusters=4,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42,
)
model.fit(X_train)
```

### 理解重点

- KMeans 的核心参数是 `n_clusters`——它必须在训练前确定，算法不会自行推断。这是 KMeans 与 DBSCAN 最根本的工程差异。
- `init='k-means++'` 是默认值的黄金标准——在绝大多数情况下比 `'random'` 收敛更快、结果更好。
- `n_init=10`（当前显式设为 10）是 scikit-learn 较旧版本的默认值——用计算量换取更可靠的局部最优。
- KMeans 的 `fit()` 是迭代优化（分配-更新交替），这与 DBSCAN 和 GaussianNB（一步式）在计算特征上截然不同。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `cluster_centers_` | `ndarray`，形状 `(n_clusters, n_features)` | $\boldsymbol{\mu}_k$ | $K$ 个簇的质心坐标——当前为 $4 \times 2$ 矩阵 |
| `labels_` | `ndarray`，形状 `(n_samples,)` | 簇分配标签 | 每个样本所属簇的编号 $\{0, 1, 2, 3\}$ |
| `inertia_` | `float` | $\sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | 收敛时的簇内平方和——值越小簇越紧凑 |
| `n_iter_` | `int` | 收敛所用迭代次数 | 反映收敛速度——通常远小于 `max_iter=300` |
| `n_features_in_` | `int` | 特征维度 $d$ | 训练时输入的特征维数，当前为 `2` |

### 示例代码

```python
print(f"n_clusters: {n_clusters}")
print(f"inertia: {model.inertia_:.4f}")
print(f"质心坐标:\n{model.cluster_centers_}")
```

### 理解重点

- `cluster_centers_` 是 KMeans 最有教学意义的属性——它把"中心式聚类"这一直觉直接映射为可观察的坐标。
- `inertia_` 是 KMeans 的核心输出——它量化了整个聚类的紧密度。当前源码打印到 4 位小数。
- 与 DBSCAN 的关键对比：KMeans 有 `cluster_centers_` 和 `inertia_`，DBSCAN 有 `core_sample_indices_` 和 `-1` 噪声标签——前者反映"中心在哪、多紧"，后者反映"核心在哪、谁被排除"。

## 4. 训练阶段的工程封装

除了 `KMeans(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察迭代优化耗时——通常极快（毫秒级） |
| `n_clusters` 日志 | 确认预设簇数 |
| `inertia_` 日志 | 观察最终簇内平方和——聚类紧密度定量参考 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出关键统计量。
- `inertia_` 是 KMeans 独有的日志输出——它直接反映聚类紧密度，是肘部法则的基础。
- KMeans 不打印簇数量（因为 $K$ 是预设的），不打印噪声点数量（因为每个点都被强制分配）——这两点与 DBSCAN 形成鲜明对比。

## 常见坑

1. 误以为 `train_model(...)` 需要传入 `y_train`——KMeans 是无监督算法，不接受标签。
2. 忽略 `n_clusters` 是必须预设的前置约束——选错 $K$ 意味着预设的分组方式与数据真实结构不匹配。
3. 把不同运行中 `labels_` 编号的变化理解为模型不稳定——编号是任意的，重要的是簇的结构和边界。
4. 忘记 `inertia_` 随 $K$ 单调递减——不能直接用它选择 $K$，需配合肘部法则。

## 小结

- `train_model(...)` 是本仓库 KMeans 的核心训练入口，是对 `sklearn.cluster.KMeans` 的薄封装。
- `KMeans` 的核心参数是 `n_clusters`（预设 $K$）、`init`（初始化策略）、`n_init`（多轮择优次数）——三者共同决定了聚类的质量和稳定性。
- 训练完成后的核心属性：`cluster_centers_`（质心坐标）、`labels_`（簇分配）、`inertia_`（簇内平方和）——三者分别回答了"中心在哪""谁属于谁""有多紧"。
- KMeans 有 `cluster_centers_` 和 `inertia_`、有 `predict()` 方法、但需要预设 $K$——这三点构成了它与 DBSCAN 最核心的工程差异。
