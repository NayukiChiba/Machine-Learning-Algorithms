---
title: PCA 主成分分析 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 PCA 实现。
2. 给出继续深入阅读主成分分析与相关数据工具的可靠入口。

## 自检题

1. 为什么 PCA 的 `fit()` 不需要 `y` 参数，而 LDA 需要？PCA 的优化目标（最大化投影方差）为什么天然不依赖标签？
2. 当前 PCA 流水线为什么分别训练 2D 和 3D 两个独立模型？2D PCA 的前两个主成分与 3D PCA 的前两个主成分是否相同？为什么？
3. 当前数据是低秩合成结构（3 个真实方向隐藏在 10 维中）——这在 `explained_variance_ratio_` 的输出中如何体现？方差断层应该出现在第几个主成分之后？
4. `svd_solver='auto'` 与 `'full'`、`'randomized'`、`'arpack'` 有什么区别？为什么 `'auto'` 是大多数情况下的最佳选择？
5. PCA 的 `components_` 与 LDA 的 `scalings_` 在数学含义上有何不同？为什么不能用同一套术语描述它们？
6. 为什么标准化对 PCA 是硬性要求？如果去掉标准化，协方差矩阵会怎样被大尺度特征主导？
7. PCA 和 LDA 共用同一个 `plot_dimensionality` 函数——同名的 `explained_variance_ratio_` 属性在两种场景下语义有何不同？

## 练习方向

### 1. 改变 `n_components`

- 对 2D 模型：把 `n_components=2` 改成 `1`、`2`、`3`、`5`
- 观察变化：
  - `explained_variance_ratio_` 各值的变化
  - 累计解释方差从 ~75%（2D）→ ~94%（3D）→ ~98%（5D）
  - 降维图的视觉信息量变化
- 核心理解：`n_components` 增加带来的信息增益是边际递减的——这正是数据低秩特性的体现

### 2. 改变 `pca_n_informative`

- 修改 `data_generation/dimensionality.py` 中 `DimensionalityData` 的 `pca_n_informative`（`2`、`3`、`5`）
- 观察变化：
  - 方差断层从第几个主成分之后开始
  - `pca_n_informative=2` 时，前 2 个主成分累计方差是否更高
  - `pca_n_informative=5` 时，方差下降是否更平缓
- 核心理解：`pca_n_informative` 直接决定数据的固有秩——`explained_variance_ratio_` 的断层位置应与其对应

### 3. 改变 `pca_noise_std`

- 修改 `pca_noise_std`（`0.0`、`0.2`、`0.5`、`1.0`、`2.0`）
- 观察变化：
  - 无噪声（`0.0`）——前 3 个主成分累计方差 = 100%，后续主成分方差 = 0
  - 高噪声（`2.0`）——方差下降非常平缓，难以识别固有秩
  - 降维图上样本的散布程度——高噪声下结构被淹没
- 核心理解：噪声水平决定了 PCA 从数据中能否可靠地识别低秩结构

### 4. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用原始 `X` 训练
- 观察变化：
  - `components_` 方向是否变化——大尺度特征是否主导了第一主成分
  - `explained_variance_ratio_` 分布是否不同
- 体会：标准化确保每个特征在协方差矩阵中权重均等——PCA 结果反映数据的相关结构而非量纲差异

### 5. 用 PCA 降维后接 LDA 做对比

- 先用 PCA 降到 5 维，再在 PCA 投影特征上用 LDA 降到 2 维
- 对比直接在原始 10 维数据上用 LDA 降到 2 维
- 观察变化：
  - 两种路径下的判别投影图有何差异
  - PCA 作为 LDA 的预处理步骤，是否丢失了类别区分信息
- 核心理解：PCA 的方差最大化 + LDA 的判别最大化可以串联——这是特征工程的常见组合

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`PCA` | 完整构造器参数（`n_components`、`svd_solver`、`random_state`、`whiten`、`tol`、`iterated_power`、`n_oversamples`、`power_iteration_normalizer`）、属性（`components_`、`explained_variance_`、`explained_variance_ratio_`、`singular_values_`、`mean_`、`n_features_in_`、`n_samples_`）与方法（`fit`、`transform`、`fit_transform`、`inverse_transform`、`get_covariance`、`get_precision`）说明 |
| 2 | scikit-learn 用户指南：Decomposing signals in components (matrix factorization problems) → PCA | PCA 原理、SVD 求解器选择指南、`n_components` 选择策略、增量 PCA（`IncrementalPCA`）和核 PCA（`KernelPCA`）的适用场景 |
| 3 | Jolliffe, I. T. and Cadima, J. (2016). *Principal component analysis: a review and recent developments*. Philosophical Transactions of the Royal Society A. | PCA 综述——从经典推导到稀疏 PCA、鲁棒 PCA 等现代变体，涵盖选主成分数量的多种准则 |
| 4 | Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 12: Continuous Latent Variables. | PCA 的概率视角——概率 PCA（PPCA）、EM 算法求解、与因子分析的关系，为理解贝叶斯 PCA 提供基础 |

- scikit-learn `PCA`：https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn 用户指南 PCA：https://scikit-learn.org/stable/modules/decomposition.html#pca

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 PCA 分册的核心内容：
  - PCA 是无监督降维——`fit()` 不接收 `y`，优化目标是最大化投影方差
  - 主成分是协方差矩阵的特征向量——可通过 SVD 数值稳定地求解
  - 当前流水线有独特的双模型设计——分别训练 2D 和 3D PCA，对比不同降维程度的效果
  - `explained_variance_ratio_` 反映各主成分的方差占比——方差断层揭示数据的固有秩
  - 标准化对基于协方差矩阵的 PCA 是硬性要求——特征量纲差异会绑架主成分方向
  - PCA 的 `components_`（主成分方向）与 LDA 的 `scalings_`（判别方向）名称不同、数学含义不同
  - PCA 与 LDA 共用可视化模块 `plot_dimensionality`——同函数、同属性名、不同语义
  - `n_components` 在 PCA 中可自由选择（1 到 $\min(d,N)$），在 LDA 中受 $K-1$ 约束
  - 低秩合成数据（`base @ projection + noise`）是展示 PCA 优势的理想教学数据
