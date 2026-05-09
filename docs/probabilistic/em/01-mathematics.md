---
title: EM 与 GMM — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解高斯混合模型（GMM）的生成过程——$\pi_k$ 选分量 → $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ 生成样本。
2. 理解 EM 算法的两步迭代——E 步（计算责任）和 M 步（最大化参数）。
3. 理解对数似然的下界保证——EM 保证对数似然单调不减。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 高斯混合模型 | 生成模型 | $p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$——$K$ 个高斯分布的加权和 |
| 隐变量 $z_{ik}$ | 概率框架 | 指示样本 $i$ 是否由分量 $k$ 生成——GMM 的"未观测变量" |
| E 步 | 期望计算 | 计算后验责任 $\gamma(z_{ik})$——"在当前参数下，样本 $i$ 属于分量 $k$ 的概率" |
| M 步 | 参数最大化 | 用责任加权更新 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$、$\pi_k$——最大化完全数据对数似然的期望 |
| 对数似然下界 | 收敛保证 | $\log p(\mathbf{X} \mid \Theta)$ 在每次 EM 迭代中单调不减 |
| 协方差类型 | 模型假设 | `full`（完全协方差）允许椭圆形簇——比 KMeans 的球面假设更灵活 |

## 1. 高斯混合模型的生成过程

GMM 假设数据由 $K=3$ 个高斯分量按以下过程生成：

1. 以概率 $\pi_k$ 选择一个高斯分量：
   $$
   p(z_k = 1) = \pi_k, \quad \sum_{k=1}^{K} \pi_k = 1
   $$
2. 从所选分量的高斯分布中采样：
   $$
   p(\mathbf{x} \mid z_k = 1) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
   $$

边缘分布为：
$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

### 理解重点

- $\pi_k$ 是**混合权重**——$\pi_k \ge 0$，$\sum_k \pi_k = 1$。当前数据 $\pi = [0.5, 0.3, 0.2]$。
- $\boldsymbol{\mu}_k = [\mu_{k1}, \mu_{k2}]^T$ 是第 $k$ 个分量的均值（2 维）。
- $\boldsymbol{\Sigma}_k$ 是 $2 \times 2$ 的协方差矩阵——`covariance_type="full"` 允许每个分量的协方差各不相同。

## 2. 最大似然的挑战

直接最大化对数似然：
$$
\log p(\mathbf{X} \mid \Theta) = \sum_{i=1}^{N} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

困难在于：$\log \sum$ 内部有求和——导数为零的方程没有闭式解，因为隐变量 $z_{ik}$ 未被观测。

### 理解重点

- 如果有标签（知道每个样本属于哪个分量），参数估计简化为加权样本均值和协方差——有闭式解。
- 无标签时，EM 通过**迭代猜测**（E 步）和**用猜测更新参数**（M 步）来绕过这个困难。

## 3. E 步：计算后验责任

给定当前参数 $\Theta^{(t)}$，计算每个样本属于每个分量的后验概率（责任）：

$$
\gamma(z_{ik})^{(t+1)} = \frac{\pi_k^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}
{\sum_{j=1}^{K} \pi_j^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}
$$

- $\gamma(z_{ik}) \in [0, 1]$，且 $\sum_k \gamma(z_{ik}) = 1$——每个样本对各分量的责任和为 1
- 高斯密度：$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$

### 理解重点

- 责任 $\gamma(z_{ik})$ 就是**软赋值**——样本 $i$ 对三个分量各有部分归属。
- 与 KMeans 的硬赋值对比：KMeans 输出 $\gamma_{ik} \in \{0, 1\}$，EM 输出 $\gamma_{ik} \in [0, 1]$。
- 当前 `covariance_type="full"` 使 $\boldsymbol{\Sigma}_k$ 可以是任意正定矩阵——每个分量的高斯密度是倾斜的椭圆形。

## 4. M 步：最大化参数

用 E 步计算的责任 $\gamma_{ik}$ 作为权重，重新估计参数：

**有效样本数**：
$$
N_k = \sum_{i=1}^{N} \gamma(z_{ik})
$$

**均值更新**：
$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma(z_{ik}) \mathbf{x}_i
$$

**协方差更新**（`covariance_type="full"`）：
$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma(z_{ik}) (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^T
$$

**混合权重更新**：
$$
\pi_k^{(t+1)} = \frac{N_k}{N}
$$

### 理解重点

- 每个参数更新都是**责任加权**——$\gamma_{ik}$ 越大的样本对分量 $k$ 的参数更新贡献越大。
- 这相当于"软计数"——不是每个点固定属于一个分量，而是按比例贡献于多个分量。
- `full` 协方差给每个分量最大自由度——可以学习任意方向的椭圆形状。

## 5. 对数似然的单调性

EM 算法保证对数似然在每次迭代中**单调不减**：
$$
\log p(\mathbf{X} \mid \Theta^{(t+1)}) \ge \log p(\mathbf{X} \mid \Theta^{(t)})
$$

这是因为 EM 实际上在最大化对数似然的一个**下界**函数（ELBO）：
$$
\mathcal{L}(\Theta, q) = \sum_i \sum_k \gamma_{ik} \log \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\gamma_{ik}} \le \log p(\mathbf{X} \mid \Theta)
$$

### 理解重点

- 对数似然单调不减是 EM 收敛的保证——但只保证收敛到**局部最大值**，不保证全局最优。
- 当前源码中 `model.lower_bound_` 记录了收敛时的对数似然下界值。
- 在实际中，初始化的均值和协方差可能会使 EM 收敛到不同的局部最优——这类似于 KMeans 的 `n_init`。

## 6. 协方差类型对比

| `covariance_type` | 协方差约束 | 簇形状 | 参数数（$K$ 分量、$d$ 维） |
|---|---|---|---|
| `full` | 无约束 | 任意椭圆 | $K \times \frac{d(d+1)}{2}$ |
| `tied` | 所有分量共享 | 相同椭圆 | $\frac{d(d+1)}{2}$ |
| `diag` | 对角矩阵 | 轴对齐椭圆 | $K \times d$ |
| `spherical` | $\sigma_k^2 \mathbf{I}$ | 球形（同 KMeans） | $K \times 1$ |

当前源码使用 `full`——每个分量有独立的 $2 \times 2$ 协方差矩阵（3 个参数每个）。

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 生成模型 | $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | `GaussianMixture(n_components=3, covariance_type="full")` |
| 隐变量 | $z_{ik}$ | 内部矩阵——E 步计算 |
| 后验责任 | $\gamma(z_{ik})$ | `model.predict_proba(X)` |
| 混合权重 | $\pi_k$ | `model.weights_` |
| 分量均值 | $\boldsymbol{\mu}_k$ | `model.means_` |
| 分量协方差 | $\boldsymbol{\Sigma}_k$ | `model.covariances_` |
| 对数似然下界 | $\log p(\mathbf{X} \mid \Theta)$ | `model.lower_bound_` |
| 最大迭代 | $t_{\max}$ | `max_iter=200` |
| 收敛判断 | $\|\Theta^{(t+1)} - \Theta^{(t)}\| < \epsilon$ | 内部自动判断 |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 8. EM vs KMeans 数学对比

| 维度 | KMeans | EM (GMM) |
|---|---|---|
| 目标函数 | $\min \sum_{i} \sum_{k} r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$ | $\max \sum_i \log \sum_k \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ |
| 赋值 | 硬赋值 $r_{ik} \in \{0, 1\}$ | 软赋值 $\gamma_{ik} \in [0, 1]$ |
| 簇形状 | 球形（等距离衰减各向同性） | 椭圆形（全协方差各向异性） |
| 不确定性 | 无 | 有——$1 - \max_k \gamma_{ik}$ 量化置信度 |
| 参数数 | $K \times d$（均值） | $K \times (1 + d + d(d+1)/2)$（权重 + 均值 + 协方差） |

## 常见坑

1. 混淆 EM 与 KMeans——EM 输出概率归属（软聚类），KMeans 输出确定归属（硬聚类）。
2. 在 `covariance_type="spherical"` 下期待椭圆形簇——球形协方差等价于 KMeans 的簇形状假设。
3. 忽略 EM 收敛到局部最优的风险——不同初始化可能导致不同的聚类结果。
4. 认为 `max_iter=200` 不够——200 次对于 2 维 3 分量数据通常足够收敛。

## 小结

- EM 算法的数学核心链：GMM 生成模型 $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ → 隐变量 → E 步计算责任 $\gamma(z_{ik})$ → M 步责任加权更新参数 → 对数似然单调递增 → 局部收敛。
- 与 KMeans 的根本区别：概率软赋值（$\gamma_{ik}$ 连续）vs 距离硬赋值（$r_{ik}$ 离散）、椭圆协方差 vs 球形距离。
- 当前源码 `GaussianMixture(n_components=3, covariance_type="full", max_iter=200)` 是 GMM 最灵活的教学配置——允许每个分量有独立的全协方差矩阵。
