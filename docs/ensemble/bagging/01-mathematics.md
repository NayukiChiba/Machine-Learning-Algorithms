---
title: Bagging 集成学习 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 Bootstrap 抽样的概率基础——每个样本被选入训练子集的概率约为 63.2%。
2. 理解 Bagging 为何能降低方差——通过并行训练多个不相关（或弱相关）模型并投票平均。
3. 理解 OOB（Out-of-Bag）误差估计的数学原理——利用未参与训练的样本做无偏估计。
4. 理解为什么 Bagging 选择完全生长的决策树（高方差低偏差）作为基学习器。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| Bootstrap 采样 | 抽样方法 | 从 $N$ 个样本中有放回地抽取 $m$ 个样本——每个子训练集约含 63.2% 的原始样本 |
| 方差缩减 | 核心原理 | 对 $n$ 个方差均为 $\sigma^2$、两两相关系数为 $\rho$ 的模型取平均，集成方差为 $\rho\sigma^2 + (1-\rho)\sigma^2/n$ |
| 投票聚合 | 输出方式 | 分类任务：$n$ 个基学习器投票，多数票决定最终预测 |
| OOB 误差 | 评估指标 | 用未参与训练的约 36.8% 样本评估每个基学习器——等价于交叉验证 |
| `n_estimators` | 源码参数 | 基学习器数量——越多方差越低，但边际收益递减 |
| `max_samples` | 源码参数 | 每个 Bootstrap 子集的样本比例——控制训练子集与原始数据的差异度 |

## 1. Bootstrap 抽样

给定 $N$ 个样本的数据集 $\mathcal{D} = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_N, y_N)\}$，Bootstrap 抽样从中**有放回**地抽取 $m$ 个样本，构成一个训练子集 $\mathcal{D}_b$。

### 单个样本未被抽中的概率

每个样本在一次抽取中被选中的概率为 $1/N$，未被选中的概率为 $1 - 1/N$。$m$ 次独立抽取后：

$$
P(\text{样本 } i \text{ 未被抽中}) = \left(1 - \frac{1}{N}\right)^m
$$

当 $m = N$ 时（即子集大小等于原始数据大小），取极限：

$$
\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.368
$$

### 理解重点

- 每个 Bootstrap 子集约含原始数据中约 **63.2%** 的样本——剩余的约 **36.8%** 就是 OOB 样本。
- 当前源码 `max_samples=0.8` 表示 $m = 0.8N$——子集比原始数据稍小，进一步增加了子集间的差异性。
- Bootstrap 采样的随机性使每个基学习器看到的数据分布略有不同——这是"模型多样性"的来源。

## 2. 方差缩减原理

### 独立模型的方差

设 $n$ 个基学习器 $f_1, \dots, f_n$，每个的预测方差均为 $\sigma^2$，两两之间的相关系数为 $\rho$。Bagging 通过投票（分类）或平均（回归）聚合：

$$
f_{\text{bag}}(\mathbf{x}) = \frac{1}{n} \sum_{b=1}^{n} f_b(\mathbf{x})
$$

集成模型的方差为：

$$
\text{Var}[f_{\text{bag}}] = \frac{1}{n^2} \left( \sum_{b=1}^{n} \text{Var}[f_b] + \sum_{b \neq c} \text{Cov}[f_b, f_c] \right) = \rho \sigma^2 + \frac{1 - \rho}{n} \sigma^2
$$

### 两种极端情况

- $\rho = 1$（完全相关——所有基学习器完全相同）：$\text{Var}[f_{\text{bag}}] = \sigma^2$——Bagging 无帮助
- $\rho = 0$（完全不相关——基学习器完全独立）：$\text{Var}[f_{\text{bag}}] = \sigma^2 / n$——方差随 $n$ 线性下降

实际情况下 $0 < \rho < 1$，Bagging 在方差缩减和基学习器多样性之间取得平衡。

### 理解重点

- Bagging **降低方差，不降低偏差**——集成模型的偏差约等于单个基学习器的偏差。
- 这就是为什么 Bagging 选择**完全生长的决策树**（`max_depth=None`）——它们偏差极低但方差极高，正是 Bagging 最受益的对象。
- `n_estimators=80` 意味着理论上方差约缩减为 $\rho\sigma^2 + (1-\rho)\sigma^2/80$——当 $\rho$ 较小时，方差大幅下降。
- 若基学习器本身偏差就很高（如浅层决策树），Bagging 无法纠正——低偏差是基学习器的必要前提。

## 3. OOB（Out-of-Bag）误差估计

对每个样本 $(\mathbf{x}_i, y_i)$，找到所有未使用该样本训练的基学习器 $\{b : (\mathbf{x}_i, y_i) \notin \mathcal{D}_b\}$，仅用这些基学习器预测 $\hat{y}_i^{\text{OOB}}$，计算：

$$
\text{OOB Error} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}[y_i \neq \hat{y}_i^{\text{OOB}}]
$$

等价地，OOB 得分：

$$
\text{OOB Score} = 1 - \text{OOB Error}
$$

### 理解重点

- OOB 误差等价于**对每个样本做一次留出验证**——无需额外划分验证集。
- 与交叉验证不同，OOB 误差在训练过程中"免费"获得——不需要额外训练。
- 当前源码 `oob_score=True` 启用此功能——`model.oob_score_` 打印到 4 位小数。
- OOB 得分可以直接作为模型泛化能力的参考——当它与测试集准确率接近时，说明模型泛化良好。

## 4. 为什么选择完全生长的决策树

Bagging 的方差缩减依赖于基学习器满足两个条件：

1. **低偏差**——基学习器必须能拟合训练数据（偏差小）
2. **高方差**——不同的训练子集应导致明显不同的模型（方差大）

完全生长的决策树（`max_depth=None`）完美满足这两个条件：
- 能完美拟合训练数据（偏差 $\approx 0$）
- 对训练数据的微小变化极其敏感（方差极大）

### 理解重点

- 如果使用浅层决策树或线性模型（低方差），Bagging 的方差缩减效果非常有限——因为没有方差可缩减。
- 当前源码中的基学习器参数 `max_depth=None, min_samples_split=2, min_samples_leaf=1`——刻意让每棵树完全生长，最大化方差。
- 这与 Boosting 形成对比——Boosting 通常使用浅层决策树（弱学习器），因为它的目标是**降低偏差**而非方差。

## 5. Bagging 与 Boosting 的数学对比

| 维度 | Bagging | Boosting（如 GBDT） |
|---|---|---|
| 训练方式 | 并行——$n$ 个模型独立训练 | 串行——每个模型拟合前一个模型的残差 |
| 核心目标 | 降低方差 | 降低偏差 |
| 基学习器 | 强学习器（低偏差高方差，如完全生长树） | 弱学习器（高偏差低方差，如浅层树） |
| 样本权重 | 等权重 Bootstrap 采样 | 自适应加权——错分样本权重增大 |
| 模型权重 | 等权重投票 | 按模型性能加权 |
| 过拟合风险 | 低——并行平均天然正则化 | 较高——串行拟合可能过度追逐残差 |
| 标志性参数 | `n_estimators`、`max_samples` | `n_estimators`、`learning_rate`、`max_depth` |

### 理解重点

- Bagging 和 Boosting 不是"谁更好"——Bagging 在基学习器过拟合时帮它"冷静下来"（降方差），Boosting 在基学习器欠拟合时帮它"变得更准"（降偏差）。
- 当前高噪声双月牙数据 + 完全生长树是一个经典场景——单棵树严重过拟合（方差极大），Bagging 通过并行投票大幅改善。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| Bootstrap 采样 | $P(\text{未抽中}) \approx e^{-m/N}$ | `bootstrap=True`，`max_samples=0.8` |
| 基学习器数 | $n$ | `n_estimators=80` |
| 集成预测（分类） | $\hat{y} = \text{majority}\{f_b(\mathbf{x})\}_{b=1}^{n}$ | `model.predict(X)` |
| 集成概率（分类） | $\hat{p} = \frac{1}{n}\sum_b p_b$ | `model.predict_proba(X)` |
| OOB 得分 | $1 - \frac{1}{N}\sum_i \mathbb{I}[y_i \neq \hat{y}_i^{\text{OOB}}]$ | `model.oob_score_`（`oob_score=True`） |
| 基学习器 | $f_b$：完全生长决策树 | `DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1)` |
| 方差缩减 | $\text{Var}[f_{\text{bag}}] = \rho\sigma^2 + (1-\rho)\sigma^2/n$ | Bagging 核心机制 |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler`（训练集拟合/测试集变换） |
| 分层抽样 | 保持类别比例 | `train_test_split(stratify=y)` |

## 常见坑

1. 混淆 Bagging 与 Boosting 的目标——Bagging 降方差（并行投票），Boosting 降偏差（串行纠错）。
2. 把 Bagging 的基学习器设为低方差模型——方差缩减需要高方差基学习器。
3. 忽略 OOB 得分的存在——它提供了"免费的"泛化能力估计，无需额外划分验证集。
4. 把 `n_estimators` 设得过小——$n < 10$ 时方差缩减效果有限。

## 小结

- Bagging 的数学核心链：Bootstrap 采样（约 63.2% 样本被抽中）→ 并行训练 $n$ 个高方差基学习器 → 投票/平均聚合 → 方差从 $\sigma^2$ 缩减到 $\rho\sigma^2 + (1-\rho)\sigma^2/n$ → OOB 误差提供无偏估计。
- Bagging 降低方差而不降低偏差——因此需要低偏差高方差的基学习器（完全生长的决策树）。
- 当前源码 `BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=None), n_estimators=80, max_samples=0.8, bootstrap=True, oob_score=True)` 是针对高噪声双月牙数据最经典的 Bagging 配置。
