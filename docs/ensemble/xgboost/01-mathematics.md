---
title: XGBoost — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 XGBoost 与 GBDT 共享的数学基础——加法模型、梯度提升、收缩步长。
2. 理解 XGBoost 独有的数学创新——二阶泰勒展开（Hessian）、显式正则化目标函数、分位数加权草图。
3. 理解 XGBoost 的回归目标（MSE）与分类目标（交叉熵）在数学形式上的差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 二阶泰勒展开 | 目标函数近似 | 使用 Hessian（二阶导数）比仅用梯度更精确地近似损失变化 |
| 正则化目标函数 | 模型正则化 | $\Omega(f) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2 + \alpha\|\mathbf{w}\|_1$——剪枝 + 权重收缩 |
| 分裂增益公式 | 分裂决策 | 精确计算每次分裂的损失下降——直接最大化增益 |
| 加权分位数草图 | 近似分裂点搜索 | 用二阶梯度加权的分位数确定候选分裂点——比等频分桶更高效 |
| 稀疏感知分裂 | 缺失值处理 | 自动学习缺失值的最优分裂方向 |
| 列块并行 | 计算加速 | 预排序后按列分块——在分裂搜索层面并行 |

## 1. 加法模型与目标函数

### 加法模型

与 GBDT 一致：

$$
\hat{y}_i^{(M)} = \sum_{m=1}^{M} f_m(\mathbf{x}_i), \quad f_m \in \mathcal{F}
$$

其中 $\mathcal{F}$ 是回归树空间，$f_m$ 是第 $m$ 棵树。

### 正则化目标函数（XGBoost 独有）

XGBoost 的核心创新——在损失函数外显式加入正则项：

$$
\text{Obj}(\Theta) = \sum_{i=1}^{N} \ell(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(f_m)
$$

其中单棵树的正则项为：

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \|\mathbf{w}\|^2 + \alpha \|\mathbf{w}\|_1
$$

- $T$：叶子节点数，$\gamma$（`gamma`）控制分裂的"代价"——分裂增益必须超过 $\gamma$ 才执行
- $\mathbf{w}$：叶子权重的向量，$\lambda$（`reg_lambda=1.0`）做 L2 收缩，$\alpha$（`reg_alpha=0.0`）做 L1 稀疏

### 理解重点

- GBDT（sklearn）的"正则化"主要是学习率收缩——XGBoost 在此基础上加入显式的 L1/L2 惩罚项。
- `gamma=0.0` 表示当前源码不要求分裂有最低增益——调大 `gamma` 是防止过拟合的有效手段。
- `reg_lambda=1.0`（L2 默认开启）是 XGBoost 泛化性能好的重要原因——它对叶子权重做持续的收缩约束。

## 2. 二阶泰勒展开（XGBoost 独有）

在第 $m$ 轮，XGBoost 对损失函数做二阶泰勒展开：

$$
\text{Obj}^{(m)} \approx \sum_{i=1}^{N} \left[ \ell(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \Omega(f_m)
$$

其中：

$$
g_i = \frac{\partial \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i} \bigg|_{\hat{y}=\hat{y}^{(m-1)}}, \quad
h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i^2} \bigg|_{\hat{y}=\hat{y}^{(m-1)}}
$$

### 回归（MSE）下的 $g_i$ 和 $h_i$

对于当前回归任务，$\ell = \frac{1}{2}(y_i - \hat{y}_i)^2$：

$$
g_i = \hat{y}_i - y_i, \quad h_i = 1
$$

### 理解重点

- 二阶泰勒展开是 XGBoost 最核心的数学创新——Hessian $h_i$ 提供了损失函数曲率信息，使目标函数近似比 GBDT 的一阶近似更精确。
- 在 MSE 回归下，$h_i = 1$（常数），二阶信息退化——但 XGBoost 的框架对任意可微损失函数都适用。
- 对于分类（对数损失），$h_i = p_i(1-p_i)$——此时二阶信息提供了预测不确定性的加权。

## 3. 叶子权重的闭式解

将目标函数按叶子重组，对第 $j$ 个叶子：

$$
\text{Obj}_{\text{leaf}}^{(m)} = \sum_{i \in I_j} \left( g_i w_j + \frac{1}{2} h_i w_j^2 \right) + \frac{1}{2} \lambda w_j^2 + \alpha |w_j|
$$

L1 正则化 $\alpha=0.0$ 时（当前源码），对 $w_j$ 求导为零得最优权重：

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

代入得最优叶子对应的目标函数值：

$$
\text{Obj}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$

### 理解重点

- 叶子权重的闭式解存在，是因为 XGBoost 的二次近似目标函数——GBDT（sklearn）没有这样的闭式解。
- `reg_lambda=1.0` 在分母中——它抑制大权重，防止单片叶子主导预测。
- 在 MSE 回归中，$w_j^* = -\frac{\sum g_i}{\vert I_j\vert + \lambda}$——即该叶子内残差均值的 L2 压缩版。

## 4. 分裂增益公式

给定一个叶子节点，将其分裂为左右子节点 $L$ 和 $R$，分裂增益为：

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

其中 $G = \sum_{i \in I} g_i$，$H = \sum_{i \in I} h_i$。

当 $\text{Gain} > 0$ 时执行分裂；`gamma` 增大要求更高的最小增益——做预剪枝。

### 理解重点

- 分裂增益公式使 XGBoost 能**精确评估**每次候选分裂的效果——最大化增益等价于最小化目标函数。
- $\gamma=0.0$（当前源码）意味着只要增益为正就分裂——这是最小限制。
- 这个公式也是特征重要性的计算基础——特征在所有分裂中的增益累加。

## 5. 加权分位数草图

XGBoost 寻找候选分裂点时，不用简单的等频分桶（直方图），而是用二阶梯度加权分位数：

按 $h_i$ 加权排序后取分位数——样本的 Hessian 越大，在分位数计算中的权重越大。

### 理解重点

- $h_i$ 反映了样本对损失函数的"重要性"——Hessian 大的样本，损失在该点变化剧烈，分裂点应该更精确地考虑它们。
- 在 MSE 回归中 $h_i=1$，加权分位数退化为等频分位数——此时近似分裂点搜索与直方图分桶等价。
- 在分类场景下，$h_i = p_i(1-p_i)$——接近决策边界（$p \approx 0.5$）的样本有更大权重。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 加法模型 | $\hat{y}_i^{(M)} = \sum_{m=1}^{M} f_m(\mathbf{x}_i)$ | `XGBRegressor(n_estimators=300, learning_rate=0.05)` |
| 正则化目标 | $\text{Obj} = \sum\ell + \sum\Omega(f)$ | `reg_lambda=1.0, reg_alpha=0.0, gamma=0.0` |
| 梯度（MSE） | $g_i = \hat{y}_i - y_i$ | 内部自动计算 |
| Hessian（MSE） | $h_i = 1$ | 内部自动计算 |
| 叶子权重闭式解 | $w_j^* = -\frac{G_j}{H_j + \lambda}$ | 内部自动计算 |
| 分裂增益 | $\text{Gain} = \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma$ | 内部自动计算 |
| 行采样 | 随机子集 | `subsample=0.9` |
| 列采样 | 随机特征子集 | `colsample_bytree=0.9` |
| 最小叶子权重和 | $\sum_{i \in I_j} h_i \ge$ `min_child_weight` | `min_child_weight=1` |
| 学习率收缩 | $\eta \cdot f_m$ | `learning_rate=0.05` |
| 列块并行 | 预排序后按列分块 | `n_jobs=-1` |

## 7. XGBoost vs GBDT vs LightGBM 数学对比

| 维度 | GBDT (sklearn) | LightGBM | XGBoost |
|---|---|---|---|
| 目标函数近似 | 一阶（仅梯度） | 一阶（仅梯度） | **二阶（梯度 + Hessian）** |
| 正则化 | 学习率收缩 | 学习率收缩 | **学习率 + L1 + L2 + gamma 剪枝** |
| 叶子权重 | 逐点线搜索 | 逐点线搜索 | **闭式解**（二次近似） |
| 分裂点搜索 | 预排序 → 逐一计算 | 直方图分桶 | **加权分位数草图** |
| 缺失值 | 不支持 | 不支持 | **稀疏感知——自动学习最优方向** |
| 并行 | 无 | 直方图构建级 | **列块级** |
| 树生长 | Level-wise | Leaf-wise | Level-wise（近似） |

## 常见坑

1. 在 MSE 回归场景下，$h_i=1$ 是常数——XGBoost 的二阶展开近似退化为与牛顿法而非梯度下降对应的形式，理解这一点很重要。
2. 忽略 `reg_lambda=1.0` 的默认值——XGBoost 默认 L2 正则化已开启，与 GBDT/LightGBM 的默认行为不同。
3. 混淆 `min_child_weight=1` 与 `min_samples_leaf`——前者是 Hessian 和的最小值（MSE 下等价于叶子最小样本数），非样本计数。
4. 认为 `gamma` 和 `reg_lambda` 功能重叠——`gamma` 做分裂级剪枝（分裂是否值得），`reg_lambda` 做权重级收缩（叶子值是否过大）。

## 小结

- XGBoost 的数学核心链：加法模型 → 二阶泰勒展开（$g_i + \frac{1}{2}h_i f^2$）→ 正则化目标（$+\gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2 + \alpha\|\mathbf{w}\|_1$）→ 叶子权重闭式解 $w_j^* = -\frac{G_j}{H_j+\lambda}$ → 分裂增益公式 → 精确剪枝。
- 与 GBDT/LightGBM 的最关键区别：二阶展开 + 显式正则化项——前者提供更精确的目标近似，后者提供更强的过拟合控制。
- 当前源码 `XGBRegressor(n_estimators=300, max_depth=6, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0)` 是回归任务的经典配置——L2 默认开启、无 L1 稀疏、无最低分裂增益。
