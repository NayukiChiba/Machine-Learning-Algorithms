# K-Means 聚类

## 核心思想

K-Means 将 $N$ 个样本划分为 $K$ 个簇，使得每个样本属于最近的簇中心，目标是**最小化簇内平方和** (Within-Cluster Sum of Squares, WCSS)。

## 数学定义

### 目标函数（畸变函数）

$$
J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

其中 $C_k$ 为第 $k$ 个簇，$\boldsymbol{\mu}_k$ 为第 $k$ 个簇中心：

$$
\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i
$$

### NP-Hard 问题

精确最小化 $J$ 是 NP-Hard 的。K-Means 使用**交替优化**（类似 EM 算法的思想）来逼近局部最优。

## 算法流程

1. **初始化**：选择 $K$ 个初始中心 $\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K$
2. **E 步（分配）**：将每个样本分配到最近中心：$c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$
3. **M 步（更新）**：重新计算中心：$\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i: c_i=k} \mathbf{x}_i$
4. 重复 2-3 直到收敛

### 收敛性

每次迭代 $J$ 单调不增（E 步不增、M 步不增），且 $J$ 有下界 0，因此算法必然收敛。但只保证**局部最优**。

## K-Means++ 初始化

为避免差的初始化，K-Means++ 按概率选择初始中心：

1. 随机选择第一个中心 $\boldsymbol{\mu}_1$
2. 对于每个样本 $\mathbf{x}_i$，计算距已选中心的最短距离 $D(\mathbf{x}_i)$
3. 以概率 $\frac{D(\mathbf{x}_i)^2}{\sum_j D(\mathbf{x}_j)^2}$ 选择下一个中心
4. 重复直到选出 $K$ 个中心

K-Means++ 保证期望的初始目标值为 $O(\log K)$ 倍最优值。

## 如何选 $K$

### 肘部法则 (Elbow Method)

绘制 $K$ vs $J(K)$ 曲线，选择"拐点"（肘部）作为最佳 $K$。

### 轮廓系数

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

- $a(i)$：样本 $i$ 到同簇其他点的平均距离
- $b(i)$：样本 $i$ 到最近异簇的平均距离
- $s(i) \in [-1, 1]$，越接近 1 越好

## 代码对应

```bash
python -m pipelines.clustering.kmeans
```
