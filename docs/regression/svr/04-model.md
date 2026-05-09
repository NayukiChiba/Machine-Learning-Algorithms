---
title: SVR 支持向量回归 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 理解 `trainSvrRegressionModel(...)` 如何构建并训练 `SVR`——本仓库最简训练函数之一（2 行）。
2. 理解 SVR 四个超参数——`C`、`epsilon`、`kernel`、`gamma`——的默认值及其选取理由。
3. 理解 `model.support_` 和 `model.dual_coef_` 的含义——SVR 的"参数"存在于对偶空间。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainSvrRegressionModel(...)` | 函数 | 构建并训练一个 `SVR` 模型——仅 2 行，比线性回归（3 行）更简 |
| `SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')` | 类 | scikit-learn 提供的 ε-支持向量回归器 |
| `model.fit(X_train_s, y_train)` | 方法 | SMO 类算法求解对偶问题——返回支持向量集 |
| `model.support_` | 属性 | 支持向量的训练集索引——模型复杂度的直接度量 |
| `model.dual_coef_` | 属性 | $\alpha_i - \alpha_i^*$ 的值——仅支持向量有非零值 |

## 1. `trainSvrRegressionModel(...)` 的函数签名

### 参数速览

适用函数：`trainSvrRegressionModel(XTrain, yTrain)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `XTrain` | `ndarray`，形状 `(160, 10)` | 标准化后的训练特征矩阵 | `X_train_s` |
| `yTrain` | `ndarray`，形状 `(160,)` | 训练目标值——Friedman1 生成 | `y_train` |
| 返回值 | `SVR` | 已完成 `fit()` 的模型对象——含 `support_` 和 `dual_coef_` | — |

### 示例代码

```python
from sklearn.svm import SVR

def trainSvrRegressionModel(XTrain, yTrain):
    model = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    model.fit(XTrain, yTrain)
    return model
```

### 理解重点

- 这是本仓库**最短的训练函数**——仅 2 行（线性回归 3 行），因为 `SVR` 的超参数在构造器中一次性给定。
- 与 `trainLinearRegressionModel` 对比：线性回归用无参 `LinearRegression()`，SVR 用含 4 个超参数的 `SVR(...)`。
- 函数签名没有 `randomState` 参数——SVR 的 SMO 求解是确定性的（对偶问题是凸优化）。

## 2. C：惩罚系数（正则化强度的倒数）

### 参数速览

适用 API：`SVR(C=10.0)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `C` | `float` | 管道外误差的惩罚权重——正则化强度 $\|\mathbf{w}\|^2$ 的倒数 | `10.0` |
| $C \to 0$ | — | 强正则化 → 极平坦函数 → 大量样本被容忍 → 少量支持向量 → 可能欠拟合 | — |
| $C \to \infty$ | — | 弱正则化 → 极力贴合数据 → 几乎所有管道外样本都被惩罚 → 大量支持向量 → 可能过拟合 | — |

### 理解重点

- $C$ 的作用方向与其他模型的 $\alpha$（正则化强度）**相反**——$C$ 是正则化的倒数，$C$ 越大正则化越弱。
- $C=10.0$ 在 Friedman1 数据上是中等偏大的值——偏向拟合精度，在 200 样本上通常不会严重过拟合。
- 调 $C$ 时最直接的反馈是支持向量数量——$C$ 增大会使更多边界样本成为支持向量。

## 3. epsilon：不敏感管道半宽

### 参数速览

适用 API：`SVR(epsilon=0.1)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `epsilon` | `float` | ε-不敏感管道半宽——误差在 ±ε 内不计损失 | `0.1` |
| ε 增大 | — | 管道更宽 → 更多样本被容忍 → 支持向量减少 → 模型更简单 | — |
| ε 减小 | — | 管道更窄 → 更少样本被容忍 → 支持向量增多 → 模型更复杂 | — |

### 理解重点

- `epsilon=0.1` 的取值需要结合目标变量的量级——Friedman1 的 $y$ 通常在 $[0, 25]$ 范围内，ε=0.1 是相对较小的管道。
- ε 和 $C$ 有交互效应——ε 决定"什么是误差"，$C$ 决定"误差有多大代价"。
- 与 OLS 的零容忍形成对比——OLS 相当于 ε=0 且用平方惩罚所有偏离。

## 4. kernel：核函数类型

### 参数速览

适用 API：`SVR(kernel='rbf')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `kernel` | `str` | 核函数类型——决定映射到什么样的特征空间 | `'rbf'` |
| `'rbf'` | — | 高斯径向基函数——$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | 默认——当前使用 |
| `'linear'` | — | 线性核——$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$ | 退化为线性 SVR |
| `'poly'` | — | 多项式核——配合 `degree` 和 `coef0` 使用 | 需要额外调参 |
| `'sigmoid'` | — | Sigmoid 核——类似神经网络激活 | 较少使用 |

### 理解重点

- `kernel='rbf'` 是 SVR 的默认核——也是当前仓库的唯一配置。RBF 核的"无限维"特性使其能拟合任意连续函数。
- 线性核 SVR 等价于带 ε-管道损失的线性回归——失去了非线性能力，但保留了可解释性（有 `coef_`）。
- 当前仓库没有切换核函数的配置——`kernel='rbf'` 是硬编码的默认值。

## 5. gamma：RBF 核的宽度参数

### 参数速览

适用 API：`SVR(gamma='scale')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `gamma` | `float` 或 `str` | RBF 核的 $\gamma$——控制单个样本的影响半径 | `'scale'` |
| `'scale'` | — | $\gamma = 1/(d \cdot \text{Var}(X))$——scikit-learn 自动按特征方差缩放 | 默认——当前使用 |
| `'auto'` | — | $\gamma = 1/d$——仅按特征维度缩放 | — |
| 数值（如 `0.1`） | — | 手动指定——越小影响范围越大 | — |

### 理解重点

- `gamma='scale'` 是 scikit-learn 0.22+ 的默认值——比旧版 `'auto'` 更稳定，考虑了特征方差。
- γ 是 RBF 核最敏感的超参数——小变化可能导致从欠拟合到过拟合的剧烈转变。
- γ 与 $C$ 有交互效应——γ 大（局部性强）+ $C$ 大（强拟合）极易导致过拟合。

## 6. 训练后的关键属性

### 参数速览

| 属性 | 类型 | 含义 | 诊断价值 |
|---|---|---|---|
| `support_` | `ndarray(nSV,)` | 支持向量在训练集中的索引 | **直接反映模型复杂度——nSV 越多越复杂** |
| `dual_coef_` | `ndarray(1, nSV)` | $\alpha_i - \alpha_i^*$ 的值 | 支持向量的权重——正值为上界支持向量，负值为下界 |
| `intercept_` | `float` | 截距 $b$ | 核函数加权和的偏置项 |
| `n_support_` | `ndarray` | 每类支持向量数量（回归中无意义） | — |
| `shape_fit_` | `tuple` | 训练数据形状 | — |

### 示例代码

```python
n_sv = model.support_.shape[0]
print(f"支持向量数量: {n_sv}")

# 支持向量数量占训练样本的比例
sv_ratio = n_sv / len(y_train)
print(f"支持向量占比: {sv_ratio:.1%}")
```

### 理解重点

- `support_` 是 SVR 最重要的属性——不亚于线性回归的 `coef_`。它告诉你模型"用了多少个样本做决策"。
- `dual_coef_` 中的值在 $[-C, +C]$ 之间——边界上的支持向量值在 $(-C, +C)$ 内，管道外的支持向量值 = $\pm C$。
- SVR（RBF 核）没有 `coef_` 属性——因为权重存在于对偶空间的 $\alpha_i - \alpha_i^*$ 中，无法直接映射回原始特征空间。

## 7. SVR vs 线性回归 vs 正则化回归 模型构建对比

| 模型维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 模型类 | `LinearRegression` | `Lasso` / `Ridge` / `ElasticNet` | **`SVR`** |
| 训练函数 | `trainLinearRegressionModel` | `trainRegularizationModels` | **`trainSvrRegressionModel`** |
| 函数行数 | 3 行 | ~10 行 | **2 行——最简** |
| 返回值 | 单个模型 | `dict`——三个模型 | **单个模型** |
| 超参数数 | 0 | 1~2 | **4（C, ε, kernel, γ）** |
| `random_state` | 不需要 | 需要（Lasso/EN） | **不需要——凸优化确定性** |
| 核心属性 | `coef_`, `intercept_` | `coef_`, `intercept_` + 近零计数 | **`support_`, `dual_coef_`, `intercept_`** |
| 是否有 `coef_` | **是** | **是** | **否（RBF 核）——权重在对偶空间** |
| 训练方式 | SVD 闭式解 | 坐标下降 / 闭式解 | **SMO——序列最小优化** |
| 标准化 | 否 | **是** | **是** |

## 常见坑

1. 在 SVR（RBF 核）模型上访问 `model.coef_`——不存在此属性，应使用 `model.support_` 和 `model.dual_coef_`。
2. 认为 `C` 和其他模型的 `alpha` 方向一致——$C$ 是正则化倒数，增大 $C$ 意味着减弱正则化。
3. 忽略 `gamma='scale'` 不是固定值——它依赖输入数据的方差，不同数据集的 `'scale'` 值不同。
4. 将 `n_support_` 用于回归诊断——该属性为二分类 SVC 设计，回归中无参考价值。

## 小结

- `trainSvrRegressionModel(...)` 是本仓库最简训练函数——仅 2 行，但 `SVR(...)` 包含 4 个关键超参数。
- $C$（惩罚强度）、ε（管道宽度）、kernel（映射方式）、γ（核宽度）构成 SVR 的超参数体系——四者联合决定了模型行为。
- SVR（RBF 核）的模型参数不在 `coef_` 中——核心属性是 `support_`（哪些样本参与）和 `dual_coef_`（参与权重）。
- 支持向量数量是模型复杂度的最直观指标——不亚于决策树的深度和 Lasso 的近零系数数量。
