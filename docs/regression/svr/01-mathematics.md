---
title: SVR 支持向量回归 — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 SVR 的 ε-不敏感损失函数——管道内误差不计，管道外线性惩罚。
2. 理解原始优化问题中的正则项 $\frac{1}{2}\|\mathbf{w}\|^2$ 与松弛变量 $\xi_i, \xi_i^*$ 的分工。
3. 理解对偶问题中核技巧的引入——内积替换为 $K(\mathbf{x}_i, \mathbf{x}_j)$ 实现非线性映射。
4. 将数学符号中的 $C$、$\epsilon$、$\gamma$ 与源码中的 `C=10.0`、`epsilon=0.1`、`gamma='scale'` 精确对应。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| ε-不敏感损失 | 损失函数 | $L_\epsilon(y, f(\mathbf{x})) = \max(0, \|y - f(\mathbf{x})\| - \epsilon)$——管道内误差为零 |
| $\frac{1}{2}\|\mathbf{w}\|^2$ | 正则项 | 控制模型平滑度——$\|\mathbf{w}\|$ 越小，函数越平坦 |
| $C$ | 超参数 | 正则化强度的倒数——越大越强调拟合，越小越强调平滑 |
| RBF 核 | 核函数 | $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$——隐式高维映射 |
| 支持向量 | 概念 | 拉格朗日乘子 $\alpha_i - \alpha_i^* \neq 0$ 的样本——仅这些样本参与预测 |

## 1. ε-不敏感损失函数

SVR 的核心创新：不惩罚管道内的误差。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $\epsilon$ | 标量 | 管道半宽——$2\epsilon$ 范围内误差不计损失 | `epsilon=0.1` |
| $L_\epsilon$ | 函数 | $\max(0, \|y - f(\mathbf{x})\| - \epsilon)$——超出管道才线性惩罚 | — |

$$
L_\epsilon(y, f(\mathbf{x})) = \max(0, |y - f(\mathbf{x})| - \epsilon)
$$

### 理解重点

- 当预测误差在 $[-\epsilon, +\epsilon]$ 内时损失为零——SVR 主动"忽视"小误差。
- ε 越大管道越宽，更多样本落入管道内——模型更平滑，支持向量更少。
- 这与 OLS（所有误差都平方惩罚）和 Lasso/Ridge（所有误差都平方惩罚 + 系数惩罚）根本不同。

## 2. 原始优化问题

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $\mathbf{w}$ | 向量 | 线性模型的系数——$\frac{1}{2}\|\mathbf{w}\|^2$ 控制平滑度 | — |
| $C$ | 标量 | 管道外误差的惩罚权重——正则化强度的倒数 | `C=10.0` |
| $\xi_i$ | 标量 | 管道上方的松弛变量——样本超出管道上界的量 | — |
| $\xi_i^*$ | 标量 | 管道下方的松弛变量——样本超出管道下界的量 | — |

$$
\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N}(\xi_i + \xi_i^*)
$$

$$
\text{s.t.} \quad
\begin{cases}
y_i - \mathbf{w}^T\mathbf{x}_i - b \leq \epsilon + \xi_i \\
\mathbf{w}^T\mathbf{x}_i + b - y_i \leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* \geq 0
\end{cases}
$$

### 理解重点

- $\frac{1}{2}\|\mathbf{w}\|^2$ 的意义：使拟合函数尽可能平坦（flatness）——这是 SVM/SVR 区别于其他回归方法的核心。
- $C\sum(\xi_i + \xi_i^*)$：管道外样本的惩罚——$C$ 越大，偏离管道的代价越高，模型越倾向于缩小管道覆盖所有样本。
- $C$ 是正则化强度的**倒数**——$C \to \infty$ 退化为硬间隔（无正则化），$C \to 0$ 趋向完全平坦。
- 样本在管道内 → $\xi_i = 0$ 且 $\xi_i^* = 0$ → 不计损失，不是支持向量。

## 3. 对偶问题与核技巧

引入拉格朗日乘子 $\alpha_i, \alpha_i^*$ 后，对偶形式将 $\mathbf{w}$ 表示为训练样本的线性组合。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $\alpha_i, \alpha_i^*$ | 标量 | 拉格朗日乘子——约束在 $[0, C]$ 之间 | — |
| $K(\mathbf{x}_i, \mathbf{x}_j)$ | 函数 | 核函数——隐式计算高维特征空间的内积 | `kernel='rbf'` |
| $\gamma$ | 标量 | RBF 核宽度——$\gamma$ 越大影响范围越局部 | `gamma='scale'` |

对偶问题：

$$
\max_{\boldsymbol{\alpha}, \boldsymbol{\alpha}^*} -\frac{1}{2}\sum_{i,j}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)K(\mathbf{x}_i, \mathbf{x}_j)
- \epsilon\sum_i(\alpha_i + \alpha_i^*) + \sum_i y_i(\alpha_i - \alpha_i^*)
$$

$$
\text{s.t.} \quad \sum_i(\alpha_i - \alpha_i^*) = 0, \quad 0 \leq \alpha_i, \alpha_i^* \leq C
$$

RBF 核：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
$$

### 理解重点

- 核技巧的本质：不显式计算 $\phi(\mathbf{x})$，而是直接计算内积 $K(\mathbf{x}_i, \mathbf{x}_j) = \langle\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$。
- RBF 核将数据映射到**无限维**空间——任何连续函数在理论上都可以被 RBF 核的 SVR 逼近。
- $\gamma$ 控制每个样本的影响半径：$\gamma$ 大 → 影响局部 → 可能过拟合；$\gamma$ 小 → 影响全局 → 可能欠拟合。
- `gamma='scale'` = $1/(d \cdot \text{Var}(X))$——scikit-learn 根据特征方差自动缩放。

## 4. 预测函数

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $f(\mathbf{x})$ | 函数 | 对新样本的预测——仅支持向量的加权核函数求和 | `model.predict(X_test_s)` |

$$
f(\mathbf{x}) = \sum_{i=1}^{N}(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b
$$

### 理解重点

- 预测不是 $\mathbf{w}^T\mathbf{x} + b$ 的矩阵乘法——而是支持向量与测试点的核函数加权和。
- $\alpha_i - \alpha_i^* = 0$ 的样本对预测**零贡献**——只有支持向量参与计算。
- 预测复杂度 $O(N_{\text{SV}} \cdot N_{\text{test}} \cdot d)$——支持向量越多预测越慢。

## 5. SMO 类优化算法

SVR 的对偶问题是一个带约束的二次规划。scikit-learn 使用 SMO（Sequential Minimal Optimization）类算法求解：

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 优化变量 | — | $\alpha_i, \alpha_i^*$——共 $2N$ 个变量 | — |
| 约束 | — | $\sum(\alpha_i - \alpha_i^*) = 0$，$0 \leq \alpha_i, \alpha_i^* \leq C$ | — |
| `tol` | `float` | 停止条件的对偶间隙容忍度 | `1e-3`（scikit-learn 默认） |
| `max_iter` | `int` | 最大迭代次数 | `-1`（无限制，scikit-learn 默认） |

### 理解重点

- SMO 每次只优化两个拉格朗日乘子——其余固定，子问题有解析解。
- 训练复杂度约 $O(N^2 \cdot d)$ 到 $O(N^3)$——样本量超过万级时训练显著变慢。
- 当前 200 样本的 SMO 求解几乎瞬时完成。

## 6. 数学概念与代码实现的映射

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 特征矩阵 | $\mathbf{X} \in \mathbb{R}^{N \times d}$ | `X_train_s`——标准化后形状 `(160, 10)` |
| 目标向量 | $\mathbf{y} \in \mathbb{R}^N$ | `y_train`——形状 `(160,)` |
| 管道半宽 | $\epsilon$ | `SVR(epsilon=0.1)` |
| 惩罚系数（正则化倒数） | $C$ | `SVR(C=10.0)` |
| 核函数 | $K(\cdot, \cdot)$ | `SVR(kernel='rbf')` |
| RBF 核宽度 | $\gamma$ | `SVR(gamma='scale')` — 即 $1/(d \cdot \text{Var}(X))$ |
| 拉格朗日乘子差值 | $\alpha_i - \alpha_i^*$ | `model.dual_coef_`——形状 `(1, nSV)` |
| 支持向量索引 | $\{i : \alpha_i - \alpha_i^* \neq 0\}$ | `model.support_` |
| 支持向量数量 | $\|\{i : \alpha_i - \alpha_i^* \neq 0\}\|$ | `model.support_.shape[0]` |
| 截距 | $b$ | `model.intercept_`——标量 |

## 7. SVR vs 线性回归 vs 正则化回归 数学对比

| 数学维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 损失函数 | 平方损失 $\|y - \hat{y}\|^2$ | 平方损失 + $\lambda R(\mathbf{w})$ | **ε-不敏感损失——管道内不计** |
| 正则化 | 无 | L1/L2 系数惩罚 | **$\frac{1}{2}\|\mathbf{w}\|^2$ 平坦性惩罚** |
| 对偶形式 | 不需要——闭式解 | 不需要（Ridge 有闭式解，Lasso 用原问题） | **需要——引入核技巧的途径** |
| 核函数 | 无 | 无 | **RBF 核——映射到无限维空间** |
| 求解方法 | SVD 闭式解 | 坐标下降 / 闭式解 | **SMO——序列最小优化** |
| 稀疏性 | 无 | Lasso: 系数稀疏 | **支持向量稀疏——仅部分样本参与预测** |
| 预测公式 | $\mathbf{X}\mathbf{w} + b$ | $\mathbf{X}\mathbf{w} + b$ | **$\sum(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b$** |
| 参数可解释性 | coef_ 直接对照 | coef_ 观察稀疏化 | **无法直接解释各特征贡献（RBF 核）** |

## 常见坑

1. 将 $C$ 理解为正则化强度——$C$ 是正则化的**倒数**，$C$ 越大正则化越弱，越容易过拟合。
2. 忘记 SVR 需要标准化——RBF 核基于欧氏距离，特征量纲不一致会导致某些维度主导核计算。
3. 将支持向量稀疏与 Lasso 系数稀疏混为一谈——前者的零是"不参与预测的样本"，后者的零是"不参与决策的特征"。
4. 期待 SVR 输出 `coef_` 做特征重要性——RBF 核的 SVR 没有 `coef_`，权重存在于对偶空间。

## 小结

- SVR 的数学核心是三层结构：ε-管道损失 + 平坦性正则化 + 核技巧非线性映射。
- ε-管道使管道内样本不计损失——它们不是支持向量，不参与预测，实现样本层面的稀疏性。
- RBF 核将数据隐式映射到无限维空间——使 SVR 能拟合任意非线性关系，但代价是可解释性下降。
- 数学公式中的 $C$、$\epsilon$、$\gamma$ 直接映射到源码中的 `C=10.0`、`epsilon=0.1`、`gamma='scale'`。
