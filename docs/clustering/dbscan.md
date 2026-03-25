# DBSCAN (Density-Based Spatial Clustering)

## 核心思想

DBSCAN 是一种**基于密度**的聚类算法。它将"密度足够高"的区域视为簇，能发现任意形状的簇，并自动识别噪声点。无需预先指定簇数 $K$。

## 关键概念

给定参数 $\epsilon$（邻域半径）和 $\text{MinPts}$（最少点数）：

### $\epsilon$-邻域

$$
N_\epsilon(\mathbf{x}) = \{\mathbf{x}' \in D : d(\mathbf{x}, \mathbf{x}') \leq \epsilon\}
$$

### 核心对象 (Core Point)

$$
|N_\epsilon(\mathbf{x})| \geq \text{MinPts}
$$

$\epsilon$-邻域内至少有 $\text{MinPts}$ 个样本（包含自身）的点。

### 密度直达 (Directly Density-Reachable)

若 $\mathbf{x}$ 是核心对象，且 $\mathbf{x}' \in N_\epsilon(\mathbf{x})$，则 $\mathbf{x}'$ 从 $\mathbf{x}$ 密度直达。

::: warning 注意
密度直达**不对称**：$\mathbf{x}'$ 从 $\mathbf{x}$ 密度直达，不意味着 $\mathbf{x}$ 从 $\mathbf{x}'$ 密度直达（因为 $\mathbf{x}'$ 可能不是核心对象）。
:::

### 密度可达 (Density-Reachable)

存在链 $\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n$，使得 $\mathbf{p}_{i+1}$ 从 $\mathbf{p}_i$ 密度直达，则 $\mathbf{p}_n$ 从 $\mathbf{p}_1$ 密度可达。

### 密度相连 (Density-Connected)

若存在 $\mathbf{o}$ 使得 $\mathbf{x}$ 和 $\mathbf{x}'$ 都从 $\mathbf{o}$ 密度可达，则它们密度相连。

## 簇的定义

簇 $C$ 满足两个性质：

1. **最大性**：若 $\mathbf{x} \in C$ 且 $\mathbf{x}'$ 从 $\mathbf{x}$ 密度可达，则 $\mathbf{x}' \in C$
2. **连通性**：$C$ 中任意两点密度相连

不属于任何簇的点被标记为**噪声**。

## 算法流程

1. 标记所有核心对象
2. 从任一未访问的核心对象出发，通过密度可达扩展簇
3. 将不属于任何簇的点标记为噪声（标签 $-1$）

## 参数选取

### $\epsilon$ 的选择

使用 **k-距离图**：对每个点计算到其第 $k$ 近邻的距离（$k = \text{MinPts}$），排序后画图。选择曲线的"拐点"作为 $\epsilon$。

### MinPts 的选择

经验法则：$\text{MinPts} \geq d + 1$（$d$ 为数据维度），通常取 $2d$ 或更大。

## 优缺点

| 优点 | 缺点 |
|------|------|
| 无需指定 $K$ | 对 $\epsilon$ 和 MinPts 敏感 |
| 可发现任意形状簇 | 高维数据效果差 |
| 能识别噪声 | 密度不均匀时困难 |

## 代码对应

```bash
python -m pipelines.clustering.dbscan
```
