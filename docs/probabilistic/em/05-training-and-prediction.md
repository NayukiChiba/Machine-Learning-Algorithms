---
title: EM 与 GMM — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解 `pipelines/probabilistic/em.py` 的 `run()` 流水线——无监督聚类下的端到端流程（无训练/测试切分）。
2. 理解 EM 算法的 `fit()` 训练过程——E 步计算责任 + M 步更新参数，对数似然单调不减。
3. 理解 `predict()` 和 `predict_proba()` 的输出差异——硬标签 vs 软归属。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 无监督聚类流水线编排——5 步串联标准化、EM 训练、预测和聚类可视化 |
| `model.fit(X_scaled)` | 方法 | EM 迭代训练——E 步（计算责任）+ M 步（更新参数），直到收敛或达到 `max_iter` |
| `model.predict(X_scaled)` | 方法 | 硬聚类标签——对每个样本取 $\gamma_{ik}$ 最大分量的索引 |
| `model.predict_proba(X_scaled)` | 方法 | 软归属——每个样本对 3 个分量的概率，返回后验责任矩阵 $\gamma_{ik}$ |
| `plot_clusters(X_scaled, labels_pred, labels_true, ...)` | 函数 | 双面板对比——预测标签 vs 真实标签的聚类分布 |

## 1. 完整流水线流程

### 流程概述

```
em_data.copy()
    │
    ├─ ① y_true = data["true_label"].values  # 仅用于评估对比
    ├─ ② X = data.drop(columns=["true_label"])
    ├─ ③ X_scaled = scaler.fit_transform(X)  # 全量标准化
    ├─ ④ model = train_model(X_scaled)        # 无 y_train
    └─ ⑤ labels_pred = model.predict(X_scaled) → plot_clusters
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 复制数据 | `em_data.copy()` | 全局 `DataFrame` | 本地 `DataFrame`，`(500, 3)` | 避免修改全局变量 |
| 提取真实标签 | `data["true_label"].values` | `DataFrame` | `ndarray`，`(500,)` | 仅用于评估对比——EM 训练时不使用 |
| 分离 X | `data.drop(columns=["true_label"])` | `DataFrame` | `DataFrame`，`(500, 2)` | 特征 `x1`、`x2` |
| 标准化 | `scaler.fit_transform(X)` | `DataFrame` | `ndarray`，`(500, 2)` | 全量数据 Z-score——无训练/测试切分 |
| 训练 | `train_model(X_scaled)` | `ndarray` | `GaussianMixture` | EM 迭代——无监督，无 `y_train` |
| 预测 | `model.predict(X_scaled)` | `ndarray` | `ndarray`，`(500,)` | 硬聚类标签 $\{0, 1, 2\}$ |
| 可视化 | `plot_clusters(X_scaled, labels_pred, y_true, ...)` | `(ndarray, ndarray, ndarray)` | PNG 文件 | 双面板聚类分布对比 |

### 理解重点

- 流水线只有 5 步——比分类流水线更简洁（无 `train_test_split`、无 `stratify`）。
- `y_true` 在步骤 1 就提取完毕——全程不参与训练，只在最后传入 `plot_clusters` 做可视化对比。
- 标准化使用 `fit_transform`（全量数据一次性完成）——聚类没有"将训练统计量应用于测试集"的需求。

## 2. 训练细节：`model.fit(X_scaled)`

### EM 迭代流程

```
初始化（KMeans 聚类 → 初始均值和协方差）
    ↓
E 步：对每个样本 i 和分量 k，计算责任 γ(z_ik)
    γ(z_ik) = π_k * N(x_i|μ_k, Σ_k) / Σ_j π_j * N(x_i|μ_j, Σ_j)
    ↓
M 步：用 γ(z_ik) 作为权重，更新参数
    μ_k = Σ_i γ(z_ik) * x_i / N_k
    Σ_k = Σ_i γ(z_ik) * (x_i - μ_k)(x_i - μ_k)^T / N_k
    π_k = N_k / N
    ↓
检查收敛：|log p(X|Θ_new) - log p(X|Θ_old)| < tol ？
    是 → 停止
    否 → 回到 E 步
    ↓
达到 max_iter=200 → 终止
```

### 参数速览

| 参数名 | 当前取值 | 训练中的作用 |
|---|---|---|
| `n_components` | `3` | 高斯分量数——EM 的预设 K，决定了责任矩阵列数 |
| `covariance_type` | `"full"` | 每个分量学习独立的 2×2 协方差矩阵 |
| `max_iter` | `200` | E-M 循环最大次数——通常远小于此即收敛 |
| `tol` | `1e-3`（默认） | 对数似然变化阈值——连续两次小于此值则收敛 |
| `init_params` | `"kmeans"`（默认） | 初始参数来自 KMeans 聚类——提供较好的起点 |
| `reg_covar` | `1e-6`（默认） | 协方差对角线的正则化——防止奇异矩阵 |

### 理解重点

- EM 训练是**最小化对数似然的负数**——每次迭代保证数据似然不降，但只收敛到局部最优。
- `init_params="kmeans"` 意味着初始均值来自 KMeans 聚类——这比随机初始化收敛更快且更稳定。
- 对于 2 维 3 分量数据，EM 通常会在 50-100 次迭代内收敛——远小于 `max_iter=200`。

## 3. 预测细节

### `model.predict(X_scaled)` — 硬聚类

对每个样本 $i$，返回后验概率最大的分量索引：
$$
\hat{y}_i = \arg\max_k \gamma(z_{ik}) = \arg\max_k p(z_{ik} = 1 \mid \mathbf{x}_i, \Theta)
$$

### `model.predict_proba(X_scaled)` — 软聚类

直接返回后验责任矩阵：
$$
[\gamma(z_{ik})]_{N \times K} = \left[\frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}\right]
$$

### 参数速览

| 方法 | 输入形状 | 输出形状 | 输出含义 |
|---|---|---|---|
| `predict(X)` | `(n, 2)` | `(n,)` | 硬聚类标签——$\{0, 1, 2\}$ |
| `predict_proba(X)` | `(n, 2)` | `(n, 3)` | 软归属概率——行和为 1.0，每列对应一个分量 |

### 理解重点

- `predict()` 的输出与 KMeans 的 `predict()` 格式完全一致——都是 $\{0, 1, 2\}$ 的整数标签。
- `predict_proba()` 是 GMM 独有的——KMeans 没有此方法。它提供了每个样本对各分量归属的"确定性"。
- 高不确定性样本：如果某样本的 $\max_k \gamma_{ik} < 0.6$，说明它在两个分量边界处"摇摆不定"。

## 4. 与 KMeans 训练流程的对比

| 步骤 | KMeans | EM (GMM) |
|---|---|---|
| 初始化 | k-means++（质心） | KMeans 聚类（均值和协方差） |
| 赋值 | 硬——最小欧氏距离 | **软——最大后验概率 $\gamma_{ik}$** |
| 更新 | 算术平均（等权更新质心） | **加权平均（$\gamma_{ik}$ 加权更新均值和协方差）** |
| 收敛条件 | 标签不再变化 | **对数似然变化 < tol** |
| 标准化 | fit_transform（全量） | fit_transform（全量）——相同 |
| 训练数据 | X_scaled（无 y） | X_scaled（无 y）——相同 |
| 迭代上限 | `max_iter=300` | `max_iter=200` |

## 常见坑

1. 没有标准化就直接 `fit`——不同特征尺度导致协方差估计偏向尺度大的维度。
2. 在 `n_components` 不匹配真实分量数时硬用——分量数需要作为先验知识或通过 BIC 选择。
3. 忽略 EM 的局部最优风险——不同的 `random_state` 可能给出不同的聚类结果。
4. 混淆 `predict` 和 `predict_proba` 的用途——评估聚类效果用前者，分析归属不确定性用后者。

## 小结

- EM 流水线仅有 5 步——是最简洁的聚类流水线之一：提取标签（仅供评估）→ 分离特征 → 标准化 → 训练 → 预测 → 可视化。
- `fit()` 的核心流程：KMeans 初始化 → E 步（计算责任 $\gamma_{ik}$）→ M 步（责任加权更新 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$、$\pi_k$）→ 检查对数似然收敛 → 循环。
- `predict()` 和 `predict_proba()` 分别提供硬聚类标签和软归属概率——后者是 GMM 区别于 KMeans 的关键输出。
