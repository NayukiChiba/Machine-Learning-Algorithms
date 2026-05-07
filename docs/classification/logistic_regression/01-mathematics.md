---
title: LogisticRegression 逻辑回归分类 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解逻辑回归为什么虽然叫"回归"，本质上却是通过 Sigmoid 做概率输出的分类模型。
2. 理解线性得分、Sigmoid、对数几率、交叉熵损失和梯度在当前实现中的数学角色。
3. 理解正则化机制与 `C` 参数（$\lambda = 1/C$）的关系——`C` 越大正则越弱。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 线性得分 $z = \mathbf{w}^T\mathbf{x} + b$ | 模型部分 | 先对输入特征做线性加权求和，是 Sigmoid 的输入 |
| Sigmoid 函数 $\sigma(z) = \frac{1}{1+e^{-z}}$ | 概率映射 | 把线性得分 $z \in (-\infty, +\infty)$ 压缩到 $(0, 1)$ 概率区间 |
| 对数几率 $\ln\frac{P}{1-P}$ | 解释方式 | 逻辑回归对对数几率建模为 $\mathbf{w}^T\mathbf{x} + b$，使得概率比取对数后呈线性 |
| 交叉熵损失 $\mathcal{L}$ | 优化目标 | 衡量预测概率与真实标签的差异，最小化等价于极大似然估计 |
| 梯度 $\nabla_{\mathbf{w}}\mathcal{L}$ | 优化信息 | 决定参数 $\mathbf{w}$ 沿哪个方向更新能最快降低损失 |
| 正则化与 `C` | 超参数机制 | $C$ 是正则化强度 $\lambda$ 的倒数——$C$ 越小正则越强，系数越收缩 |

## 1. 逻辑回归的核心思想

逻辑回归先计算一个线性得分，再把这个得分通过 Sigmoid 压缩成概率输出。因此它虽然名字里有"回归"，最终目标是做分类概率估计。

### 线性部分

$$
z = \mathbf{w}^T\mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b
$$

### Sigmoid 函数

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Sigmoid 的性质：
- $z \to +\infty$ 时 $\sigma(z) \to 1$（十分确信正类）
- $z = 0$ 时 $\sigma(z) = 0.5$（最不确定）
- $z \to -\infty$ 时 $\sigma(z) \to 0$（十分确信负类）

### 概率输出

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}
$$

### 理解重点

- 当前模型不是直接输出"正类/负类"，而是先输出正类概率。
- 这也是为什么当前流水线可以直接调用 `predict_proba(...)` 来绘制 ROC 曲线。
- 逻辑回归的核心优势：分类结果和概率解释天然结合——输出不只是标签，还有置信度。

## 2. 对数几率与线性决策边界

逻辑回归对对数几率（log-odds）建模为线性函数：

$$
\ln \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = \mathbf{w}^T\mathbf{x} + b = z
$$

当 $P(y=1) = P(y=0) = 0.5$ 时，对数几率为 0，由此得到决策边界：

$$
\mathbf{w}^T\mathbf{x} + b = 0
$$

这是一个 $d$ 维空间中的超平面，$\mathbf{w}$ 是法向量，$b$ 控制超平面的偏移。

### 理解重点

- 决策边界 $\mathbf{w}^T\mathbf{x} + b = 0$ 是一张平坦的超平面——这就是"线性"的来源。
- $w_j > 0$ 意味着特征 $x_j$ 增大时会推高正类概率；$w_j < 0$ 则压低正类概率。
- 这也是为什么 `coef_` 和 `intercept_` 在逻辑回归里很有解释价值——它们直接描述了边界的位置和方向。

## 3. 极大似然与交叉熵损失

对训练集 $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$，假设样本独立，似然函数为：

$$
L(\mathbf{w}, b) = \prod_{i=1}^{N} \hat{p}_i^{\,y_i} (1 - \hat{p}_i)^{1 - y_i}
$$

其中 $\hat{p}_i = \sigma(\mathbf{w}^T\mathbf{x}_i + b)$。取负对数并除以 $N$ 后，得到交叉熵损失（对数损失）：

$$
\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \Big]
$$

### 理解重点

- 逻辑回归不是靠最小二乘（MSE），而是靠极大似然 / 交叉熵目标来训练——这使它更适合概率建模。
- 与 MSE 不同，交叉熵对概率接近 0 或 1 时的错误预测惩罚很大（$\ln \hat{p} \to -\infty$ 当 $\hat{p} \to 0$），迫使模型给出更可信的概率估计。
- 当前数学章节应明确这一点，避免和线性回归的损失函数混淆。

## 4. 梯度推导

交叉熵损失对权重 $w_j$ 的偏导数形式非常简洁：

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i) \, x_{ij}
$$

向量形式：

$$
\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{p}} - \mathbf{y}), \quad
\nabla_b \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)
$$

### 理解重点

- 梯度形式非常直观：预测概率与真实标签的误差 $(\hat{p}_i - y_i)$，乘上对应特征值 $x_{ij}$，汇总起来更新参数。
- 可以把它理解为"当前模型在哪些方向上高估或低估了正类概率"——误差大的方向更新靠前。
- 当前源码虽然没有手写梯度下降（使用 `lbfgs` 优化器封装），但理解梯度形式有助于理解优化行为。

## 5. 正则化与 `C` 参数

当前训练代码默认使用 L2 正则化。加入 L2 正则化后的完整损失函数为：

$$
\mathcal{L}_{\text{reg}}(\mathbf{w}, b) = \mathcal{L}(\mathbf{w}, b) + \frac{1}{2C} \|\mathbf{w}\|_2^2
$$

其中 $\|\mathbf{w}\|_2^2 = \sum_{j=1}^{d} w_j^2$。

**关键关系**：$\lambda = 1/C$，其中 $\lambda$ 是传统正则化强度系数，$C$ 是 sklearn 使用的正则化强度倒数。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `penalty` | `str` 或 `None` | 正则化类型。`"l2"` 对 $\|\mathbf{w}\|_2^2$ 惩罚，系数趋于均匀收缩；`"l1"` 对 $\|\mathbf{w}\|_1$ 惩罚，产生稀疏解；`"elasticnet"` 为两者混合；`None` 不做正则化。默认为 `"l2"` | `"l2"`、`"l1"`、`"elasticnet"`、`None` |
| `C` | `float` | 正则化强度的倒数，数学上 $\lambda = 1/C$。$C$ 越大 → 正则越弱 → 模型越自由；$C$ 越小 → 正则越强 → 系数越收缩趋于 0。默认为 `1.0` | `0.01`、`1.0`、`100.0` |
| `l1_ratio` | `float` | L1 正则化在 elasticnet 中的混合比例。仅当 `penalty='elasticnet'` 时生效。$\text{penalty} = \rho \|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2$。默认为 `None` | `0.0`、`0.5`、`1.0` |

### 理解重点

- `C` 是正则化强度的倒数——这是当前逻辑回归分册最容易写反的地方，文档必须明确。
- $C \to 0$（$\lambda \to \infty$）：强正则，系数趋近于 0，模型趋近于常数预测（仅剩截距起作用）。
- $C \to \infty$（$\lambda \to 0$）：弱正则，系数自由增长，容易过拟合。
- 当前默认 `C=1.0`、`penalty='l2'`，是 sklearn 的保守默认值。

## 6. 为什么标准化会影响训练与解释

逻辑回归使用梯度优化器（`lbfgs`）最小化交叉熵损失，特征尺度差异会导致：

1. 不同维度的梯度量级差异巨大——优化器收敛困难
2. 正则化惩罚不均匀——大值特征的系数被过度惩罚
3. `coef_` 之间不可直接比较——无法判断哪个特征更重要

标准化 $x_i' = (x_i - \mu_i) / \sigma_i$ 后，所有特征均值为 0、标准差为 1，以上问题全部消除。

### 理解重点

- 标准化对逻辑回归是有实益的——不是可有可无的工程惯性。
- 标准化后 $w_j$ 的大小可以粗略反映特征 $j$ 的相对重要性（因为各特征尺度统一）。
- 这也是当前流水线必须在训练前执行 `StandardScaler` 的原因。

## 7. 多分类扩展：Softmax

逻辑回归可以自然扩展到多分类（Softmax 回归 / 多项逻辑回归）。对 $K$ 个类别，每个类别有自己的权重向量 $\mathbf{w}_k$：

$$
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}
$$

二分类退化为 Sigmoid：当 $K=2$ 时，Softmax 等价于 Sigmoid。

### 理解重点

- 当前数学章节保留此扩展以建立完整视角。
- 当前工程实现使用的是二分类数据和二分类逻辑回归，因此数据、模型、训练、评估章节都应聚焦二分类场景。
- 如果将来需要多分类，sklearn 的 `LogisticRegression` 已原生支持（通过 `multi_class='multinomial'`）。

## 8. 数学原理如何映射到当前源码

以下表格将本章涉及的数学概念与当前仓库的代码实现一一对应：

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 线性得分 | $z = \mathbf{w}^T\mathbf{x} + b$ | `model.decision_function(X)` |
| Sigmoid 概率 | $\sigma(z) = 1/(1+e^{-z})$ | `model.predict_proba(X)[:, 1]` |
| 决策边界 | $\mathbf{w}^T\mathbf{x} + b = 0$ | `model.coef_` × `X` + `model.intercept_` = 0 |
| 权重系数 | $\mathbf{w} \in \mathbb{R}^d$ | `model.coef_` |
| 截距 | $b \in \mathbb{R}$ | `model.intercept_` |
| 交叉熵损失 | $\mathcal{L} = -\frac{1}{N}\sum[y_i\ln\hat{p}_i + (1-y_i)\ln(1-\hat{p}_i)]$ | `solver='lbfgs'` 内部优化目标 |
| L2 正则化 | $\frac{1}{2C}\|\mathbf{w}\|_2^2$ | `penalty='l2'`，`C=1.0` |
| 正则化倒数 | $\lambda = 1/C$ | `C=1.0` → $\lambda = 1.0$ |
| 优化器 | — | `solver='lbfgs'` |

## 常见坑

1. 把逻辑回归误当成线性回归加阈值——本质上是线性得分 + Sigmoid 概率映射 + 交叉熵优化。
2. 忽略交叉熵与极大似然的等价关系——最小化交叉熵 = 最大化对数似然。
3. 把 `C` 的含义写反——`C` 是正则化强度的倒数，$C$ 越大正则越弱，不是"正则化系数"。
4. 忽略标准化对优化和系数解释的影响——梯度优化器对特征尺度敏感，不标准化会导致收敛困难和系数不可比。

## 小结

- 逻辑回归的核心数学链：线性得分 $z = \mathbf{w}^T\mathbf{x} + b$ → Sigmoid 概率 $\sigma(z)$ → 交叉熵损失 $\mathcal{L}$ → 梯度下降优化 → 正则化控制复杂度。
- `coef_` 与 `intercept_` 直接决定线性决策边界 $\mathbf{w}^T\mathbf{x} + b = 0$ 的位置——$w_j > 0$ 推高正类概率，$w_j < 0$ 压低正类概率。
- `C` 是 $\lambda$ 的倒数（$\lambda = 1/C$），$C$ 越大正则越弱——这个方向是当前文档必须反复强调的重点。
- 当前源码默认使用 L2 正则化 + `lbfgs` 优化器的二分类逻辑回归，与当前高维近线性可分数据高度匹配。
