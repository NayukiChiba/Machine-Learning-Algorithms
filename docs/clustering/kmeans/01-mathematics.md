---
title: KMeans K 均值聚类 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 KMeans 的优化目标——最小化簇内平方和（Inertia / WCSS）。
2. 理解分配-更新两步骤的交替迭代机制，以及为什么算法保证收敛（到局部最优）。
3. 理解 `k-means++` 初始化策略的数学动机——如何减少不良局部最优的风险。
4. 理解 `inertia_` 作为损失函数的含义及其随 $k$ 增大单调递减的性质。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 簇内平方和 $\sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | 优化目标 | KMeans 尝试最小化的损失函数——质心代表簇成员越紧密 |
| 分配步骤（E-step） | 迭代步骤 | 固定质心，将每个点分配给最近的质心 |
| 更新步骤（M-step） | 迭代步骤 | 固定分配，将每个质心更新为簇内所有点的均值 |
| `k-means++` | 初始化策略 | 加权随机选择初始质心，使其尽可能分散——显著减少不良局部最优 |
| `n_init` | 鲁棒机制 | 多次运行取最佳结果——以计算量换局部最优质量的提升 |
| `inertia_` | 收敛指标 | 最终簇内平方和——用于评估聚类紧密度和肘部法则选 $k$ |

## 1. KMeans 的优化目标

给定 $N$ 个样本 $\mathbf{x}_i \in \mathbb{R}^d$ 和簇数 $K$，KMeans 将数据划分为 $K$ 个不相交的集合 $C_1, C_2, \dots, C_K$，最小化簇内平方和（Within-Cluster Sum of Squares, WCSS）：

$$
\min_{C_1, \dots, C_K} \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

其中 $\boldsymbol{\mu}_k$ 是第 $k$ 个簇的质心（该簇内所有点的均值）：

$$
\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i
$$

### 理解重点

- 这是一个组合优化问题——同时优化质心位置和分配方案。直接求解是 NP-hard，因此使用交替迭代的启发式方法。
- 目标函数使用平方欧氏距离——这隐含假设簇在各方向上的方差相近（各向同性），因此 KMeans 偏好球形簇。
- `inertia_` 就是优化目标在收敛处的值——它是 KMeans 训练完成后最重要的标量输出。

## 2. 分配-更新交替迭代

KMeans 使用 EM 风格的交替最小化来逼近最优解。

### 分配步骤（Assignment Step）

固定 $K$ 个质心 $\{\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K\}$，将每个样本分配给最近的质心：

$$
C_k = \{\mathbf{x}_i : \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2 \leq \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2, \; \forall j \neq k\}
$$

### 更新步骤（Update Step）

固定簇分配，重新计算每个簇的质心为簇内所有点的均值：

$$
\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i
$$

### 收敛性

每一步都保证不增加目标函数值——分配步取最近距离，更新步的均值为该簇 SSE 的全局最小化点。由于只有有限种分配方式，算法在有限步内收敛到**局部最优**。

### 理解重点

- 每次迭代目标函数单调不增——这是一个收敛保证，但收敛到的是局部最优而非全局最优。
- 最终结果高度依赖初始质心的选择——这正是 `k-means++` 和 `n_init` 存在的理由。
- 当前源码 `max_iter=300` 设置了迭代上限，防止在病态数据上无限循环。

## 3. `k-means++` 初始化

随机选择初始质心容易导致不良局部最优（例如两个质心落在同一簇内）。`k-means++` 通过加权随机采样使初始质心尽可能分散：

1. 从数据中随机选择第一个质心
2. 对每个点 $\mathbf{x}_i$，计算其到已选质心的最小平方距离 $D(\mathbf{x}_i)^2$
3. 以概率 $\frac{D(\mathbf{x}_i)^2}{\sum_j D(\mathbf{x}_j)^2}$ 选择下一个质心——距离已有质心越远的点越可能被选中
4. 重复 2-3 直到选满 $K$ 个质心

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `init` | `str` | 初始化方法。`'k-means++'`（默认）使用加权随机采样策略；`'random'` 从数据中纯随机选择 $K$ 个点 | `'k-means++'`、`'random'` |
| `n_init` | `int` 或 `'auto'` | 使用不同初始质心运行 KMeans 的次数，返回 `inertia_` 最小的结果。默认 `10` | `1`、`10`、`20` |

### 理解重点

- `k-means++` 是 KMeans 从"频繁得到差结果"到"实践中稳定可靠"的关键改进——它将选到不良初始质心的概率降低了多个数量级。
- `n_init=10` 是额外保险——以约 10 倍计算量换取更好的局部最优。对当前 400 样本 2 维 4 簇数据，10 次运行几乎总能找到正确的聚类结构。

## 4. 质心、标签与 `inertia_`

训练完成后，KMeans 生成三项核心输出：

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `cluster_centers_` | `ndarray`，形状 `(n_clusters, n_features)` | $\boldsymbol{\mu}_k$ | 每个簇的质心坐标——当前为 $4 \times 2$ 矩阵 |
| `labels_` | `ndarray`，形状 `(n_samples,)` | 簇分配标签 | 每个样本所属簇的编号 $\{0, 1, 2, 3\}$ |
| `inertia_` | `float` | $\sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | 最终簇内平方和——值越小表示簇越紧凑 |

### 理解重点

- `cluster_centers_` 是 KMeans 区别于 DBSCAN 的标志性属性——KMeans 有显式质心，DBSCAN 没有。
- `inertia_` 随 $K$ 增大**单调递减**——当 $K=N$ 时惯性为 0（每个点自成簇）。因此它不能直接用于选择最优 $K$，需配合肘部法则（Elbow Method）使用。
- 当前源码打印 `inertia_` 到 4 位小数——这是一项聚类紧密度的定量参考。

## 5. 标准化对 KMeans 的数学必要性

KMeans 的核心操作是计算点到质心的欧氏距离：

$$
\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2 = \sum_{j=1}^{d} (x_{ij} - \mu_{kj})^2
$$

若特征 $x_1$ 的取值量纲是 $x_2$ 的 100 倍，则 $x_1$ 的差异平方将主导整个距离计算——聚类结果实际上只由 $x_1$ 决定，$x_2$ 的贡献被淹没。

### 理解重点

- 标准化后每个特征对距离的贡献均等——聚类结果反映所有特征维度的信息。
- 对 KMeans 而言标准化是必须的——它直接依赖于距离度量的几何意义。
- 这与 DBSCAN 和 SVC（RBF 核）的逻辑完全一致——任何基于距离度量的算法都需要标准化。

## 6. 为什么适合 `make_blobs` 数据

`make_blobs` 从各向同性高斯分布 $\mathcal{N}(\mathbf{c}_k, \sigma^2 \mathbf{I})$ 采样生成簇：

- 簇内样本在质心周围球形散布——与 KMeans 的平方欧氏距离假设完美匹配
- 各簇方差统一（`cluster_std=0.8`）——避免了 KMeans 在方差差异大时偏向大方差簇的问题
- 4 个质心分布在二维平面的不同象限——分配步骤容易做出正确判断

### 理解重点

- `make_blobs` 是为 KMeans "量身定制"的数据——它满足了 KMeans 的所有隐假设（球形、等方差）。
- 这种设计在教学上有意为之——先在理想数据上展示算法优势，再通过练习引导理解边界条件。
- 对比 DBSCAN 的 `make_moons`——不同聚类算法需要不同的数据形态来展示各自最强的一面。

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 预设簇数 | $K$ | `n_clusters=4` |
| 优化目标 | $\min \sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | KMeans 算法核心 |
| 分配步骤 | $C_k = \{\mathbf{x}_i : \arg\min_j \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2 = k\}$ | KMeans 内部迭代 |
| 更新步骤 | $\boldsymbol{\mu}_k = \frac{1}{\vert C_k \vert} \sum_{\mathbf{x} \in C_k} \mathbf{x}$ | KMeans 内部迭代 |
| 质心初始化 | `k-means++` 加权采样 | `init='k-means++'` |
| 多轮初始化 | 运行 $n$ 次取惯性最小者 | `n_init=10` |
| 最大迭代次数 | — | `max_iter=300` |
| 质心坐标 | $\boldsymbol{\mu}_k$ | `model.cluster_centers_` |
| 簇分配标签 | $\{0, 1, \dots, K-1\}$ | `model.labels_` |
| 簇内平方和 | $\sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | `model.inertia_` |
| 迭代次数 | — | `model.n_iter_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 常见坑

1. 不理解 KMeans 收敛到局部最优而非全局——`k-means++` 和 `n_init` 旨在缓解而非根除。
2. 用 `inertia_` 直接比较不同 $K$ 的模型——惯性随 $K$ 单调递减，需配合肘部法则或轮廓系数。
3. 在不标准化的数据上运行——距离计算被量纲绑架，聚类结果由尺度最大的特征主导。
4. 混淆 `labels_` 编号与 `true_label` 编号——簇标签是任意的，0 不一定对应真实标签 0。

## 小结

- KMeans 的数学核心链：簇内平方和 $\min\sum\|\mathbf{x}-\boldsymbol{\mu}\|^2$ → 分配-更新交替迭代 → `k-means++` 加权初始化 → `n_init` 多轮择优 → 收敛到局部最优。
- KMeans 有显式质心（`cluster_centers_`）和可量化的损失（`inertia_`）——这是它区别于 DBSCAN 最核心的数学特征。
- 当前源码 `KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300)` 是针对 `make_blobs` 球形高斯簇的最经典配置。
