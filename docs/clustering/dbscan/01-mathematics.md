---
title: DBSCAN 密度聚类 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 DBSCAN 如何用 $\epsilon$ 邻域和密度关系定义簇——而非像 KMeans 那样依赖质心。
2. 理解核心点、边界点、噪声点的数学定义及其与 `eps` 和 `min_samples` 的关系。
3. 理解密度直达、密度可达、密度相连三种关系如何将散点组织成簇。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| $\epsilon$ 邻域 $N_{\epsilon}(\mathbf{x})$ | 基础定义 | 以 $\mathbf{x}$ 为中心、$\epsilon$ 为半径的超球体内所有点 |
| 核心点 | 点类型 | $\vert N_{\epsilon}(\mathbf{x}) \vert \geq \text{MinPts}$——邻域内点数达到阈值，可以向外扩展簇 |
| 边界点 | 点类型 | 自身非核心点，但落在某核心点的 $\epsilon$ 邻域内 |
| 噪声点 | 点类型 | 既非核心点也不属于任何簇——`labels_ == -1` |
| 密度直达 | 关系 | 核心点向其 $\epsilon$ 邻域内任意点的单向一步关系 |
| 密度可达 | 关系 | 通过有限步密度直达串联而成的传递关系 |
| 密度相连 | 关系 | 两点通过同一核心点桥接——这是簇的连通性基础 |

## 1. $\epsilon$ 邻域与点类型

给定数据集 $D = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$、邻域半径 $\epsilon > 0$ 和最小邻域点数 $\text{MinPts}$。

### $\epsilon$ 邻域

点 $\mathbf{x}$ 的 $\epsilon$ 邻域定义为：

$$
N_{\epsilon}(\mathbf{x}) = \{\mathbf{x}' \in D \mid d(\mathbf{x}, \mathbf{x}') \leq \epsilon\}
$$

其中 $d(\cdot, \cdot)$ 是距离度量——当前源码默认使用欧氏距离 $d(\mathbf{x}, \mathbf{x}') = \|\mathbf{x} - \mathbf{x}'\|_2$。

### 核心点（Core Point）

若点 $\mathbf{x}$ 的 $\epsilon$ 邻域内包含至少 $\text{MinPts}$ 个样本（含自身），则 $\mathbf{x}$ 为核心点：

$$
|N_{\epsilon}(\mathbf{x})| \geq \text{MinPts}
$$

### 边界点（Border Point）

点 $\mathbf{x}$ 不是核心点（$|N_{\epsilon}(\mathbf{x})| < \text{MinPts}$），但位于某个核心点的 $\epsilon$ 邻域内。

### 噪声点（Noise Point）

点 $\mathbf{x}$ 既不是核心点，也不属于任何核心点扩展出的簇。

### 理解重点

- `eps` 控制"多近算邻居"——$\epsilon$ 越大越宽松，越小越严格。
- `min_samples` 控制"多密才算核心"——$\text{MinPts}$ 越大，成为核心点的门槛越高。
- 核心点是簇扩展的"种子"——只有核心点能向外扩展，边界点只能被包含，噪声点被排除。

## 2. 三种密度关系

### 密度直达（Directly Density-Reachable）

$\mathbf{x}'$ 从 $\mathbf{x}$ 密度直达，当且仅当：

1. $\mathbf{x}$ 是核心点
2. $\mathbf{x}' \in N_{\epsilon}(\mathbf{x})$

密度直达**不对称**——如果 $\mathbf{x}'$ 是边界点（非核心点），则 $\mathbf{x}$ 不能从 $\mathbf{x}'$ 密度直达。

### 密度可达（Density-Reachable）

$\mathbf{x}_n$ 从 $\mathbf{x}_1$ 密度可达，当存在一条点链 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n$，使得对每个 $i$，$\mathbf{x}_{i+1}$ 从 $\mathbf{x}_i$ 密度直达。

密度可达**不对称**——边界点可以被核心点密度可达，但反过来不成立。

### 密度相连（Density-Connected）

$\mathbf{x}$ 和 $\mathbf{x}'$ 密度相连，当存在点 $\mathbf{o}$，使得 $\mathbf{x}$ 和 $\mathbf{x}'$ 都从 $\mathbf{o}$ 密度可达。

密度相连是**对称的**——这是簇定义的连通性基础。

### 理解重点

- 密度直达是"一步扩展"（微观），密度可达是"沿链扩展"（中观），密度相连是"桥接扩展"（宏观）。
- 只有密度相连关系是对称的——这正是 DBSCAN 能把点归入同一个簇的数学保证。

## 3. 簇的数学定义

基于以上关系，DBSCAN 定义的簇 $C$ 满足两个性质：

1. **最大性（Maximality）**：若 $\mathbf{x} \in C$ 且 $\mathbf{x}'$ 从 $\mathbf{x}$ 密度可达，则 $\mathbf{x}' \in C$。
2. **连通性（Connectivity）**：$C$ 中任意两点都是密度相连的。

不属于任何簇的点被标记为噪声（标签 $-1$）。

### 理解重点

- 最大性保证簇"收齐"所有能连通到的点——不会遗漏。
- 连通性保证簇内部的点在密度上是连通的——不会错误合并。
- 噪声点不是算法失败——它是 DBSCAN 设计的固有输出，对应数据中密度不足以形成簇的离群点。

## 4. 参数 `eps` 与 `min_samples`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `eps` | `float` | $\epsilon$ 邻域半径。$\epsilon \uparrow$ → 更多点被纳入邻域，大簇倾向、噪声点减少；$\epsilon \downarrow$ → 邻居门槛变严，小簇倾向、噪声点增多。默认 `0.3` | `0.2`、`0.3`、`0.5`、`1.0` |
| `min_samples` | `int` | 核心点判定阈值 $\text{MinPts}$。值越大，成为核心点的门槛越高，簇更保守、噪声可能更多。默认 `5` | `3`、`5`、`10`、`20` |

### 理解重点

- `eps` 和 `min_samples` 是联动参数——不能孤立调参。增大 `eps` 同时可能需要增大 `min_samples` 以避免过度合并。
- 对于二维数据，`min_samples` 的经验值通常是 $2 \times d$ 到 $2 \times d + 1$（$d$ 为特征维度）——当前 `min_samples=5` 对二维数据是合理起点。
- 当前 `eps=0.3` 是针对标准化后双月牙数据的选择——两月牙内侧最小距离约 0.5，0.3 的 $\epsilon$ 小于此间距，避免两月牙被错误连成一个簇。

## 5. 距离度量 `metric`

### 参数速览

适用 API：`DBSCAN(metric='euclidean')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `metric` | `str` 或 `callable` | 距离度量方式。`'euclidean'`（默认）使用欧氏距离 $d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum (x_j - z_j)^2}$；`'manhattan'` 使用曼哈顿距离 $d(\mathbf{x}, \mathbf{z}) = \sum \vert x_j - z_j \vert$；`'cosine'` 使用余弦距离 | `'euclidean'`、`'manhattan'`、`'cosine'` |

### 理解重点

- 距离度量的选择直接影响 $\epsilon$ 邻域的形状——欧氏距离产生超球邻域，曼哈顿距离产生超菱面邻域。
- 当前源码使用默认的 `'euclidean'`，与标准化后的二维特征匹配。
- 不同度量下相同的 `eps` 值对应不同的实际邻域范围——切换度量时需重新调整 `eps`。

## 6. 标准化对 DBSCAN 的数学必要性

`eps` 是一个在特征空间中定义邻域半径的绝对数值。如果特征 $x_1$ 取值在 $[-2, 2]$ 而 $x_2$ 取值在 $[-100, 100]$，则：

- $\epsilon = 0.3$ 对 $x_1$ 来说覆盖了其取值范围的 7.5%
- 但同样的 $\epsilon = 0.3$ 对 $x_2$ 来说仅覆盖了其取值范围的 0.15%

这意味着 $\epsilon$ 邻域在不同维度上的实际含义不同——距离计算被量纲绑架。

### 理解重点

- 标准化后每个特征均值为 0、方差为 1，$\epsilon$ 在所有维度上的意义一致。
- 对于 DBSCAN 而言，标准化不是可选的优化手段——它是 `eps` 参数几何意义正确的前提。
- 这与 SVC（RBF 核距离敏感）的逻辑一致——任何基于距离度量的方法都需要标准化。

## 7. 为什么适合双月牙数据

`make_moons` 生成的双月牙数据具有以下数学特征：

- 两个月牙内部的点密度较高且均匀——满足核心点的判定条件
- 两个月牙之间的最小间距（约 0.5 标准化单位）大于 `eps=0.3`——密度扩展不会跨月牙跳跃
- 月牙内部沿弧形方向密度连通——单个月牙内的任意两点可以通过密度可达/密度相连关系归入同一簇

### 理解重点

- DBSCAN 的密度扩展天然适合月牙的弯曲形状——不需要任何全局形状假设。
- KMeans 依赖到中心的欧氏距离划分，会将弯月沿中心连线切分成两个半球形区域——这是算法本质差异。

## 8. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| $\epsilon$ 邻域半径 | $\epsilon$ | `eps=0.3` |
| 最小邻域点数 | $\text{MinPts}$ | `min_samples=5` |
| 距离度量 | $d(\mathbf{x}, \mathbf{x}')$ | `metric='euclidean'` |
| 核心点判定 | $\vert N_{\epsilon}(\mathbf{x}) \vert \geq \text{MinPts}$ | DBSCAN 算法内部逻辑 |
| 密度直达 | $\mathbf{x}' \in N_{\epsilon}(\mathbf{x})$，$\mathbf{x}$ 为核心点 | DBSCAN 算法扩展步骤 |
| 密度可达链 | $\mathbf{x}_1 \to \mathbf{x}_2 \to \dots \to \mathbf{x}_n$ | DBSCAN 的 BFS/DFS 扩展 |
| 簇标签 | $\{0, 1, \dots, k-1\}$ | `model.labels_` |
| 噪声标签 | $-1$ | `labels_ == -1` |
| 簇数量 | $k$ | `n_clusters = len(set(labels_)) - (1 if -1 in labels_ else 0)` |
| 噪声点数量 | — | `n_noise = (labels_ == -1).sum()` |
| 核心点索引 | — | `model.core_sample_indices_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 常见坑

1. 把 DBSCAN 当成中心式聚类——它没有簇中心（无 `cluster_centers_`），簇由密度连通关系定义。
2. 孤立调 `eps` 而忽略 `min_samples`——两者联动，增大一个时通常需调整另一个。
3. 看到噪声点（`labels_ == -1`）就认为模型失败——噪声识别是 DBSCAN 的核心设计，不是 bug。
4. 不标准化数据——`eps` 是绝对数值，在未标准化的数据上其几何意义被量纲扭曲。
5. 期望 DBSCAN 能像 KMeans 一样预测新点的簇归属——sklearn 的 DBSCAN 没有 `predict()` 方法，只能对训练数据做 `fit_predict`。

## 小结

- DBSCAN 的数学核心链：$\epsilon$ 邻域 $\to$ 核心点判定 $\to$ 密度直达/可达/相连 $\to$ 最大性 + 连通性定义簇 $\to$ 噪声点为 $-1$。
- `eps` 和 `min_samples` 联合决定点类型的划分和簇的形态——这是 DBSCAN 仅有的两个核心超参数。
- 当前源码 `DBSCAN(eps=0.3, min_samples=5, metric='euclidean')` 是二维标准化双月牙数据的合理配置——`eps` 小于月牙间距，`min_samples` 匹配二维特征的经验建议。
