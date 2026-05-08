---
title: SVC 支持向量分类 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 SVC 的核心优化目标——最大化分类间隔，以及为什么间隔越大泛化能力越强。
2. 理解软间隔中 $C$ 对间隔宽度与误分类惩罚的权衡机制。
3. 理解 RBF 核如何通过隐式高维映射使非线性数据变得线性可分。
4. 理解 $\gamma$ 参数（`gamma`）对 RBF 核局部影响范围的控制。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 最大间隔超平面 $\mathbf{w}^T \mathbf{x} + b = 0$ | 决策面 | 寻找使两类样本到边界最小距离最大化的分离面 |
| 软间隔 $\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum\xi_i$ | 优化目标 | 允许少量样本违反间隔约束，$C$ 控制容错程度 |
| 支持向量 | 关键样本 | 落在间隔边界上或违反间隔约束的样本——唯一决定最终分类面 |
| 对偶问题 | 优化形式 | 将原问题转化为仅依赖样本内积的形式，为核技巧铺路 |
| RBF 核 $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$ | 核函数 | 当前源码默认核，通过"距离越近相似度越高"构造非线性决策能力 |
| `gamma` | 超参数 | 控制 RBF 核的局部影响半径——$\gamma$ 越大，单个支持向量的影响范围越小 |
| 决策函数 $f(\mathbf{x}) = \sum \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$ | 预测公式 | 预测时只需支持向量参与计算，非支持向量的 $\alpha_i = 0$ |

## 1. 线性可分情形：硬间隔 SVM

当数据线性可分时，SVM 寻找能将两类正确分开且间隔最大的超平面 $\mathbf{w}^T \mathbf{x} + b = 0$。

### 原问题（Primal）

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
\quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
$$

其中 $y_i \in \{-1, +1\}$。约束 $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ 要求所有样本正确分类且到边界的函数间隔 ≥ 1。最小化 $\frac{1}{2}\|\mathbf{w}\|^2$ 等价于最大化间隔（因为间隔 = $2/\|\mathbf{w}\|$）。

### 对偶问题（Dual）

引入拉格朗日乘子后，对偶形式为：

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \langle \mathbf{x}_i, \mathbf{x}_j \rangle
$$

$$
\text{s.t.} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \geq 0
$$

### 理解重点

- 对偶形式只依赖样本间的内积 $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$——这正是核技巧的入口：将内积替换为核函数即可获得非线性能力。
- KKT 条件保证只有支持向量的 $\alpha_i > 0$，其余样本的 $\alpha_i = 0$——预测时只需存储和计算支持向量。

## 2. 软间隔与参数 $C$

真实数据常有噪声或不可完全线性分离，软间隔 SVM 引入松弛变量 $\xi_i \geq 0$：

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i
$$

$$
\text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `C` | `float` | 正则化参数（误分类惩罚系数）。$C$ 越大，越不容忍误分类（间隔变窄、模型更复杂）；$C$ 越小，越强调宽间隔（允许更多违例、模型更简单）。默认 `1.0` | `0.1`、`1.0`、`10.0`、`100.0` |

### 理解重点

- $C$ 的角色与逻辑回归中的 $C$ 一致——都是正则化强度的倒数：$C \uparrow$ 等价于 $\lambda \downarrow$（正则越弱）。
- $C \to \infty$ 逼近硬间隔 SVM；$C \to 0$ 会使模型过于简单（间隔宽到几乎忽略分类准确性）。
- 当前源码默认 `C=1.0`，是一个兼顾稳定性和教学简洁度的起点。

## 3. 支持向量：谁决定边界

KKT 条件揭示了支持向量的三种角色：

- $\alpha_i = 0$：样本被正确分类且在间隔之外——不影响模型
- $0 < \alpha_i < C$：样本恰好落在间隔边界上——自由支持向量
- $\alpha_i = C$：样本在间隔内或被误分类——有界支持向量

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `n_support_` | `ndarray`，形状 `(n_classes,)` | 各类别的支持向量数量 | 当前二分类返回长度为 2 的数组 |
| `n_support_.sum()` | `int` | 支持向量总数 | 反映模型依赖的关键样本规模 |
| `support_vectors_` | `ndarray`，形状 `(n_sv, n_features)` | 支持向量的特征值 | 这些样本唯一决定分类面 |
| `dual_coef_` | `ndarray` | $\alpha_i y_i$ | 对偶系数与标签的乘积 |
| `intercept_` | `ndarray` | $b$ | 决策函数的偏置项 |

### 理解重点

- 数以百计的训练样本中，往往只有几十个是支持向量——这正是 SVM 稀疏性的体现。
- `n_support_` 的规模直接反映分类任务的难度：支持向量越多，说明两类越纠缠、边界越复杂。
- 这也是当前训练日志打印 `n_support_` 的教学意义所在。

## 4. 核函数：从线性到非线性

对偶形式中的内积 $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$ 可以替换为核函数 $K(\mathbf{x}_i, \mathbf{x}_j)$，实现隐式的高维特征映射：

$$
\phi: \mathbb{R}^d \to \mathcal{H}, \quad K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle
$$

### 参数速览

| 核函数 | 公式 $K(\mathbf{x}, \mathbf{z})$ | `kernel` 取值 | 适用场景 |
|---|---|---|---|
| 线性核 | $\mathbf{x}^T \mathbf{z}$ | `'linear'` | 高维文本、特征数远大于样本数 |
| RBF（高斯）核 | $\exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ | `'rbf'` | 通用非线性——当前默认核 |
| 多项式核 | $(\gamma \mathbf{x}^T \mathbf{z} + r)^d$ | `'poly'` | 图像等已归一化的数据 |
| Sigmoid 核 | $\tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$ | `'sigmoid'` | 近似两层神经网络 |

### 理解重点

- 核技巧的价值：无需显式构造高维特征空间 $\mathcal{H}$（可能是无穷维），只需计算低维空间中的核函数值。
- 当前源码默认 `kernel='rbf'`——这是对同心圆非线性数据的直接回应。
- RBF 核的 Mercer 条件保证了 $K(\mathbf{x}, \mathbf{z})$ 确实对应于某个高维空间的内积。

## 5. RBF 核与参数 $\gamma$

### 参数速览

适用 API：`SVC(kernel='rbf', gamma='scale')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `gamma` | `float` 或 `str` | RBF 核系数。`'scale'`（默认）时 $\gamma = 1/(n\_features \cdot X.var())$；`'auto'` 时 $\gamma = 1/n\_features$；传入 `float` 时直接使用。$\gamma$ 越大，单个支持向量的影响范围越小、边界越精细弯曲 | `'scale'`、`'auto'`、`0.1`、`1.0`、`10.0` |

### 理解重点

- $\gamma$ 控制 RBF 核的"局部性"——$\gamma$ 小意味着高斯核的宽度大，单个支持向量影响范围远，决策边界更平滑；$\gamma$ 大意味着每个支持向量只影响很近的区域，边界精细但容易过拟合。
- `gamma='scale'`（scikit-learn 0.22+ 默认）会根据特征方差自动缩放 $\gamma$——这使得标准化后的数据获得合理的默认核宽度。
- $\gamma$ 和 $C$ 共同决定模型的复杂度：高 $C$ + 高 $\gamma$ 容易过拟合，低 $C$ + 低 $\gamma$ 容易欠拟合。

## 6. 标准化为何对 SVC 至关重要

RBF 核计算样本间的欧氏距离：

$$
\|\mathbf{x} - \mathbf{z}\|^2 = \sum_{j=1}^{d} (x_j - z_j)^2
$$

如果某个特征的取值量纲远大于其他特征（如 $x_1 \in [0, 1000]$ 而 $x_2 \in [0, 1]$），则 $x_1$ 将主导距离计算，扭曲核函数的几何意义。

### 理解重点

- 标准化后每个特征的均值为 0、方差为 1，距离计算中各维度平等贡献——这是核方法的标准工程做法。
- 当前流水线统一使用 `StandardScaler` 不仅是为了工程一致性，更是 RBF 核对特征尺度的数学依赖所要求的。

## 7. 决策函数与预测

训练完成后，决策函数为：

$$
f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b
$$

预测时取 $\hat{y} = \text{sign}(f(\mathbf{x}))$。注意求和只遍历支持向量（$\alpha_i > 0$），而非全部训练样本。

### 理解重点

- 预测时只需存储支持向量及其 $\alpha_i y_i$——对于稀疏解（支持向量少），这比 KNN（需存储全部训练集）更节省内存。
- SVC 默认不直接输出概率——`predict_proba(...)` 需要额外启用 `probability=True` 并通过 Platt scaling 校准，耗时显著增加。当前流水线未使用概率输出。

## 8. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 最大间隔超平面 | $\mathbf{w}^T \mathbf{x} + b = 0$ | SVC 算法核心优化目标 |
| 软间隔原问题 | $\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum\xi_i$ | `SVC(C=1.0)` |
| 对偶问题 | $\max_{\alpha} \sum\alpha_i - \frac{1}{2}\sum\alpha_i\alpha_j y_i y_j K(\mathbf{x}_i,\mathbf{x}_j)$ | SVC 内部 `libsvm` 求解 |
| RBF 核 | $\exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$ | `kernel='rbf'` |
| 核系数 | $\gamma = 1/(d \cdot \text{Var}(X))$（`'scale'`） | `gamma='scale'` |
| 支持向量数量 | — | `model.n_support_` |
| 支持向量 | $SV = \{\mathbf{x}_i \mid \alpha_i > 0\}$ | `model.support_vectors_` |
| 对偶系数 × 标签 | $\alpha_i y_i$ | `model.dual_coef_` |
| 决策函数偏置 | $b$ | `model.intercept_` |
| 类别标签 | $\{-1, +1\}$（内部），$\{0, 1\}$（用户侧） | `model.classes_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 常见坑

1. 忽略标准化的关键性——RBF 核对特征尺度极为敏感，不标准化等于让距离计算被量纲绑架。
2. 把 $C$ 当成"越大越强"的参数——$C$ 越大越容易过拟合，$\lambda = 1/C$ 的逻辑与逻辑回归一致。
3. 忽略 $\gamma$ 与 $C$ 的联合效应——两者共同决定模型复杂度，单一调参往往效果不佳。
4. 在不需要核方法的线性数据上默认使用 RBF 核——增加了计算开销而无收益。
5. 混淆 SVC 的决策函数输出（$f(\mathbf{x})$ 的符号）与概率输出——当前流水线不使用 `predict_proba`，因为 Platt scaling 会额外引入交叉验证开销。

## 小结

- SVC 的数学核心链：最大间隔 $\min\frac{1}{2}\|\mathbf{w}\|^2$ → 软间隔 $+C\sum\xi_i$ → 对偶形式 + 内积 → RBF 核 $K(\mathbf{x},\mathbf{z}) = \exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$ → 决策函数 $f(\mathbf{x}) = \sum\alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$。
- $C$ 控制软间隔容错，$\gamma$ 控制 RBF 核局部半径——两者联合决定模型复杂度。
- 支持向量（`n_support_`）是 SVC 独有的教学视角——理解它们就是理解 SVC 行为的关键。
- 当前源码 `SVC(C=1.0, kernel='rbf', gamma='scale')` 是最经典的非线性 SVM 配置——直接回应同心圆数据的线性不可分特性。
