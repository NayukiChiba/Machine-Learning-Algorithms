---
title: GBDT 梯度提升树 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 GBDT 的加法模型结构——$M$ 棵树按学习率加权累加，逐步逼近真实函数。
2. 理解梯度提升的核心思想——每棵新树拟合前 $M-1$ 棵树的负梯度（残差方向）。
3. 理解学习率（shrinkage）的数学作用——控制每步更新幅度，防止过拟合。
4. 理解为什么 GBDT 使用浅层决策树（`max_depth=3`）——弱学习器是偏差缩减的前提。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 加法模型 | 模型结构 | $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu \cdot h_m(\mathbf{x})$——$M$ 棵树按学习率加权累加 |
| 梯度提升 | 训练策略 | 第 $m$ 棵树拟合前 $m-1$ 棵树集成在损失函数上的负梯度方向 |
| 学习率收缩 | 正则化 | $\nu \in (0, 1]$ 控制每棵树的贡献——$\nu$ 越小越保守，泛化越好但需要更多树 |
| 对数损失 | 损失函数 | 多分类的交叉熵损失——$L = -\sum_{k=1}^{K} y_k \log p_k(\mathbf{x})$ |
| 随机梯度提升 | 采样策略 | `subsample < 1.0` 时每棵树只使用部分样本——引入随机性增强泛化 |

## 1. 加法模型

GBDT 的核心是一个**加法模型**——$M$ 棵决策树按学习率加权后累加：

$$
F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu \cdot h_m(\mathbf{x}; \Theta_m)
$$

其中：
- $F_M(\mathbf{x})$ 是 $M$ 轮迭代后的集成模型输出
- $h_m(\mathbf{x}; \Theta_m)$ 是第 $m$ 棵决策树（当前为浅层树，`max_depth=3`）
- $\nu$ 是学习率（`learning_rate=0.1`）
- $\Theta_m$ 是第 $m$ 棵树的参数（分裂点、叶节点值等）

### 理解重点

- 加法模型意味着每棵树**直接与前序所有树的输出相加**——不是投票，不是平均，是累加。
- 学习率 $\nu$ 控制每棵树的贡献幅度——$\nu=0.1$ 意味着每棵树只贡献其完整输出的 10%。
- 与 Bagging 的对比：Bagging 是 $f_{\text{bag}} = \frac{1}{n}\sum f_b$（等权平均），GBDT 是 $F_M = \sum \nu h_m$（学习率加权累加）。

## 2. 梯度提升——在函数空间做梯度下降

GBDT 的训练策略可以理解为**在函数空间中执行梯度下降**。

### 前向分步算法

GBDT 以贪心方式逐棵添加树。第 $m$ 步，已知前 $m-1$ 棵树的集成 $F_{m-1}(\mathbf{x})$，寻找最优的新树 $h_m$ 使损失最小：

$$
h_m = \underset{h}{\arg\min} \sum_{i=1}^{N} L\big(y_i, F_{m-1}(\mathbf{x}_i) + \nu \cdot h(\mathbf{x}_i)\big)
$$

### 负梯度——"残差"的方向

直接求解上述优化问题很困难。GBDT 的巧妙之处在于——将损失函数 $L$ 对当前预测值 $F_{m-1}(\mathbf{x}_i)$ 求负梯度，作为新树的拟合目标：

$$
r_{im} = -\left[ \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F = F_{m-1}}
$$

这 $N$ 个负梯度值 $\{(\mathbf{x}_i, r_{im})\}_{i=1}^{N}$ 构成了第 $m$ 棵树的训练目标——树 $h_m$ 的任务是**逼近负梯度方向**。

### 理解重点

- 负梯度指向损失函数下降最快的方向——GBDT 在函数空间中向这个方向迈出步长 $\nu$。
- 对回归任务（平方损失），负梯度恰好等于残差 $y_i - F_{m-1}(\mathbf{x}_i)$——这也是"拟合残差"这一直觉说法的来源。
- 对分类任务（对数损失），负梯度是"伪残差"——不是简单的 $y_i - p$，而是损失对 log-odds 的导数。
- 这就是为什么 GBDT 的核心是**降偏差**——每棵新树专门修正前序集成犯的错误。

## 3. 对数损失（多分类）

当前 GBDT 处理的是 3 分类问题（$K = 3$），使用多分类对数损失（交叉熵）：

$$
L = -\sum_{k=1}^{K} y_k \log p_k(\mathbf{x})
$$

其中 $p_k(\mathbf{x})$ 是模型对类别 $k$ 的预测概率，由 softmax 函数从 $F_M$ 的原始输出转换而来。

### 理解重点

- 多分类 GBDT 内部实际上训练了 $K$ 组树——每组对应一个类别（one-vs-rest 风格，但共享梯度结构）。
- `GradientBoostingClassifier` 使用 `loss='log_loss'`（默认）——即多分类对数损失。
- 负梯度的形式取决于损失函数的选择——对数损失的负梯度是 $y_k - p_k$，即"真实概率 - 预测概率"。

## 4. 学习率收缩（Shrinkage）

学习率 $\nu$（`learning_rate`）是 GBDT 最重要的正则化参数：

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})
$$

### 理解重点

- $\nu$ 越小，每棵树的影响越小——需要更多的树（更大的 $M$）才能达到相同的拟合程度。
- 经验上，小 $\nu$ + 大 $M$ 的组合泛化效果更好——这就是为什么当前源码 `n_estimators=200` 搭配 `learning_rate=0.1`。
- $\nu$ 与 $M$ 存在权衡：$\nu=0.01$ 可能需要 $M=2000$ 棵树，$\nu=1.0$ 可能 $M=50$ 就过拟合。
- 与 Bagging 的对比：Bagging 没有学习率——每棵树等权投票，不需要缩放。

## 5. 随机梯度提升（Stochastic GBDT）

当 `subsample < 1.0` 时，每棵树只在随机抽取的部分训练样本上拟合——这被称为随机梯度提升：

$$
\mathcal{D}_m \subset \mathcal{D}, \quad |\mathcal{D}_m| = \text{subsample} \times N
$$

### 理解重点

- `subsample < 1.0` 同时在两个方面起作用：降低计算量、增加模型多样性（类似 Bagging 的 Bootstrap 思路）。
- 当前源码 `subsample=1.0`（默认值）——不使用随机梯度提升。设置为 `0.8` 可获得额外的方差缩减效果。
- 与 Bagging 的 `max_samples` 对比：Bagging 的每个子集完全独立且并行，GBDT 的子集是串行的——第 $m$ 棵树看到的数据子集不影响第 $m+1$ 棵树所见。

## 6. GBDT 与 Bagging 的数学对比

| 维度 | Bagging | GBDT |
|---|---|---|
| 模型结构 | $f_{\text{bag}} = \frac{1}{n}\sum f_b$（等权平均） | $F_M = \sum_{m=1}^{M} \nu h_m$（学习率加权累加） |
| 训练方式 | 并行——$n$ 棵树独立训练 | 串行——第 $m$ 棵树拟合前 $m-1$ 棵的负梯度 |
| 核心目标 | 降方差——$\text{Var}[f_{\text{bag}}] = \rho\sigma^2 + (1-\rho)\sigma^2/n$ | 降偏差——$F_M$ 逐步逼近 $F^*$ |
| 基学习器 | 强学习器（完全生长树，低偏差高方差） | 弱学习器（浅层树 `max_depth=3`，高偏差低方差） |
| 核心参数 | `n_estimators`、`max_samples` | `n_estimators`、`learning_rate`、`max_depth` |
| 正则化 | 并行平均天然正则化 | 学习率收缩 + 树深度限制 + subsample |
| 过拟合风险 | 低——投票平均天然平滑 | 较高——串行拟合可能过度追逐训练噪声 |
| 并行能力 | 天然可并行——各树独立 | 必须串行——每棵树依赖前序结果 |
| 独有诊断 | OOB 得分 | 特征重要性（`feature_importances_`） |

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 加法模型 | $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu h_m(\mathbf{x})$ | `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)` |
| 基学习器（浅层树） | $h_m$：`max_depth=3` | `max_depth=3`（对比 Bagging 的 `max_depth=None`） |
| 学习率收缩 | $\nu$ | `learning_rate=0.1` |
| 负梯度（伪残差） | $r_{im} = -\partial L / \partial F$ | GBDT 内部自动计算——用户不可见 |
| 对数损失（多分类） | $L = -\sum_k y_k \log p_k$ | `loss='log_loss'`（默认值） |
| 随机梯度提升 | $\mathcal{D}_m \subset \mathcal{D}$ | `subsample=1.0`（当前未启用） |
| 特征重要性 | 基于分裂增益的加权平均 | `model.feature_importances_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler`（训练集拟合/测试集变换） |
| 分层抽样 | 保持类别比例 | `train_test_split(stratify=y)` |

## 常见坑

1. 混淆 GBDT 与 Bagging 的基学习器选择——GBDT 用弱学习器（浅层树）降偏差，Bagging 用强学习器（深层树）降方差。
2. 把 `learning_rate` 设得过大——$\nu=1.0$ 时 GBDT 退化为无收缩的简单加法模型，极易过拟合。
3. 把 `n_estimators` 设得过小——$\nu=0.1$ 时 200 棵树是合理起点，10 棵树远不足以收敛。
4. 忽略学习率与树数量的耦合关系——$\nu$ 越小需要越大的 $M$，两者必须协同调整。

## 小结

- GBDT 的数学核心链：加法模型 $\to$ 函数空间负梯度 $\to$ 每棵树拟合伪残差 $\to$ 学习率收缩控制步长 $\to$ $M$ 轮后得到低偏差集成 $\to$ 特征重要性基于分裂增益。
- GBDT 降偏差而不降方差——因此需要高偏差低方差的基学习器（浅层决策树 `max_depth=3`）。
- 当前源码 `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)` 是针对多类别分类数据最经典的 GBDT 配置。
