---
title: LightGBM — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 LightGBM 与 GBDT 共享的数学基础——加法模型、负梯度拟合、多类对数损失。
2. 理解 LightGBM 独有的工程优化：Leaf-wise 生长策略、直方图算法、GOSS 采样、EFB 特征捆绑。
3. 理解 LightGBM 的参数选择如何在数学上影响模型——`num_leaves` vs `max_depth`、`learning_rate` 收缩。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 加法模型 | 数学框架 | $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu h_m(\mathbf{x})$——GBDT 系列共享的建模方式 |
| 负梯度 | 优化理论 | 每棵树拟合 $\tilde{y}_i^{(m)} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}$——函数空间的梯度下降 |
| Leaf-wise 生长 | 树生长策略 | 每次选择损失下降最多的叶子分裂——更快的收敛速度和更深的树 |
| 直方图算法 | 加速技术 | 连续特征离散化为 $k$ 个 bins——分割点搜索从 $O(n\log n)$ 降到 $O(k)$ |
| GOSS | 采样策略 | 保留所有大梯度样本 + 从小梯度样本中随机采样——在信息损失很小的前提下加速训练 |
| EFB | 降维技术 | 将互斥特征捆绑为一个特征——减少直方图构建开销 |

## 1. GBDT 数学基础（与 LightGBM 共享）

LightGBM 在数学框架上与 GBDT 完全一致——都是加法模型 + 负梯度拟合。

### 加法模型

$$
F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu \cdot h_m(\mathbf{x}; \Theta_m)
$$

其中 $h_m$ 是第 $m$ 棵回归树，$\nu$ 是学习率（`learning_rate`），$\Theta_m$ 是树的结构参数。

### 多类对数损失

对于 $K=4$ 类分类问题，使用多类对数损失（交叉熵）：

$$
L(\{y_i\}, \{F(\mathbf{x}_i)\}) = -\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log p_k(\mathbf{x}_i)
$$

其中 $p_k(\mathbf{x}_i) = \frac{\exp(F_k(\mathbf{x}_i))}{\sum_{j=1}^{K} \exp(F_j(\mathbf{x}_i))}$（softmax），$y_{ik}$ 是 one-hot 编码。

### 负梯度（残差近似）

第 $m$ 轮对第 $k$ 类的负梯度：

$$
\tilde{y}_{ik}^{(m)} = -\left[\frac{\partial L}{\partial F_k(\mathbf{x}_i)}\right]_{F=F^{(m-1)}} = y_{ik} - p_k^{(m-1)}(\mathbf{x}_i)
$$

即**真实概率与当前预测概率之差**——新树拟合这个差值。

### 理解重点

- LightGBM 在数学上等价于 GBDT——差异全在工程实现，不在数学框架。
- 负梯度 $\tilde{y}_i$ 在分类场景下恰好是"残差概率"——当前预测的 softmax 概率与真实 one-hot 的偏差。
- 学习率 $\nu=0.05$ 表示每棵树只修正残差概率的 5%——防止单棵树修正过猛。

## 2. Leaf-wise 生长（LightGBM 独有）

### Level-wise（sklearn GBDT）的局限

传统 GBDT 按层生长（Level-wise）：每层所有叶子同时分裂——不分"重要"和"不重要的"叶子。

### Leaf-wise 策略

LightGBM 按叶子生长（Leaf-wise）：在所有叶子中，选择**分裂后损失下降最多**的叶子进行分裂。

数学上：设叶子的分裂增益为 $\Delta L_j$，选择

$$
j^* = \arg\max_j \Delta L_j
$$

重复此过程直到叶子数达到 `num_leaves=31`。

### 参数关系

| Leaf-wise 关键参数 | 数学含义 |
|---|---|
| `num_leaves=31` | 最大叶子数——复杂度上限 |
| `max_depth=-1` | 不限制深度——Leaf-wise 树可能很深但叶子数固定 |

### 理解重点

- Leaf-wise 使 Loss 下降更高效——同等叶子数下，Leaf-wise 树的损失低于 Level-wise 树。
- 代价是可能生成极深的树（深度 $\gg \log_2(\text{num\_leaves})$）——因此需要 `min_child_samples=20` 等正则化手段防止叶子过小。
- 与 Bagging 的完全生长树不同——Leaf-wise 仍受 `num_leaves` 限制，不会无限生长。

## 3. 直方图算法

### 传统方法：预排序

sklearn GBDT 对每个特征的每个分裂点，排序后逐一计算损失——复杂度 $O(n_{\text{unique}})$。

### LightGBM：直方图分桶

将连续特征值离散化为 $k$ 个 bins（直方图桶），只在桶边界搜索分裂点——复杂度 $O(k)$，$k \ll n_{\text{unique}}$。

数学上：

$$
\text{bin}(x_j) = \lfloor k \cdot \frac{x_j - x_{\min}}{x_{\max} - x_{\min}} \rfloor
$$

### 理解重点

- 直方图加速是 LightGBM 快于 sklearn GBDT 3-5 倍的核心原因——在大数据上差距更大。
- 分桶带来轻微的正则化效果——离散化后的分割点更粗糙，有助于防止过拟合。
- 代价是牺牲了极细粒度的分割点——但在实践中，256 个桶通常足够（默认 `max_bin=255`）。

## 4. GOSS（Gradient-based One-Side Sampling）

### 动机

在 Boosting 中，大梯度样本（$|\tilde{y}_i|$ 大）对训练更重要——它们是"还没学好的样本"。

### GOSS 策略

1. 按梯度绝对值 $|\tilde{y}_i|$ 排序所有样本
2. 保留前 $a \times 100\%$ 的大梯度样本（不采样）
3. 从剩余小梯度样本中随机采样 $b \times 100\%$
4. 为小梯度样本乘以权重 $\frac{1-a}{b}$ 以补偿

当前源码 `subsample=0.9`（全局采样）——未显式启用 GOSS（需要分别设置 `top_rate` 和 `other_rate`）。但 `subsample` 机制与 GOSS 的思想一致：利用梯度信息偏向保留重要样本。

### 理解重点

- GOSS 使得 LightGBM 在保持训练精度的前提下，减少了参与分裂计算的样本数。
- 梯度是样本"重要性"的天然代理——大梯度样本是当前模型处理不好的样本。

## 5. EFB（Exclusive Feature Bundling）

### 动机

高维稀疏数据中，许多特征互斥（不会同时为非零值）。EFB 将互斥特征捆绑为一个特征，减少直方图构建开销。

对于当前 20 维稠密数据，EFB 的收益有限——但这是 LightGBM 在处理稀疏高维数据时的关键加速手段。

### 理解重点

- EFB 本质上是一个图着色问题——将互斥特征（冲突少的特征）分到同一组，每组构建一个共享直方图。
- 在当前数据上 `n_features=20`，EFB 的收益不大——但数据维度提升到数千维时，EFB 的降维效果显著。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 加法模型 | $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu h_m(\mathbf{x})$ | `LGBMClassifier(n_estimators=300, learning_rate=0.05)` |
| 多类对数损失 | $L = -\sum_i \sum_k y_{ik} \log p_k(\mathbf{x}_i)$ | `objective='multiclass'`（内部默认） |
| 负梯度 | $\tilde{y}_{ik} = y_{ik} - p_k(\mathbf{x}_i)$ | 内部自动计算 |
| Leaf-wise 生长 | $\arg\max_j \Delta L_j$ | `num_leaves=31, max_depth=-1` |
| 直方图分桶 | $\text{bin}(x) = \lfloor k \cdot (x - x_{\min})/(x_{\max} - x_{\min})\rfloor$ | `max_bin=255`（内部默认） |
| 行采样 | 按梯度采样 | `subsample=0.9` |
| 列采样 | 随机选择特征子集 | `colsample_bytree=0.9` |
| Softmax 概率 | $p_k = \exp(F_k)/\sum_j \exp(F_j)$ | `model.predict_proba(X)` |
| 学习率收缩 | $\nu \cdot h_m$ | `learning_rate=0.05` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 7. LightGBM vs GBDT 数学对比

| 维度 | GBDT (sklearn) | LightGBM |
|---|---|---|
| 加法模型 | $F_M = \sum \nu h_m$ | $F_M = \sum \nu h_m$——相同 |
| 损失函数 | 多类对数损失 | 多类对数损失——相同 |
| 负梯度 | $\tilde{y} = y - p$ | $\tilde{y} = y - p$——相同 |
| 树生长策略 | Level-wise（按层） | Leaf-wise（按叶子）——**不同** |
| 分裂点搜索 | 预排序 → 逐一计算 | 直方图分桶 → 桶边界搜索——**不同** |
| 样本采样 | 随机子采样 | GOSS（梯度加权采样）——**不同** |
| 特征降维 | 无 | EFB（互斥特征捆绑）——**不同** |
| 树复杂度控制 | `max_depth=3` | `num_leaves=31`——**不同** |

### 理解重点

- LightGBM 在数学主链上与 GBDT 完全相同——差异全在算法实现的四个环节：生长策略、分裂点搜索、样本采样、特征处理。
- 这四个差异使得 LightGBM 在训练速度上有数量级优势——但预测精度与调好参的 GBDT 通常相当。

## 常见坑

1. 把 `max_depth=-1` 当成"树可以无限大"——Leaf-wise 生长下，`num_leaves` 才是真正的复杂度上限。
2. 把 GOSS 当成普通的随机子采样——GOSS 保留所有大梯度样本，不是均匀随机采样。
3. 以为 EFB 总是有效——在稠密低维数据上，特征间几乎没有互斥关系，EFB 收益极小。
4. 忽略学习率与树数量的耦合——$\nu$ 和 $M$ 共同决定总修正量 $M \times \nu$。

## 小结

- LightGBM 的数学核心链与 GBDT 完全一致：加法模型 + 负梯度拟合 + 多类对数损失 + softmax 输出。
- LightGBM 的工程优化链：Leaf-wise 生长（损失下降更高效）→ 直方图分桶（分裂搜索加速）→ GOSS（梯度加权采样）→ EFB（互斥特征捆绑）——四项优化在不改变数学框架的前提下大幅提升训练速度。
- 当前源码 `LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9)` 是轻量级高维数据的经典配置。
