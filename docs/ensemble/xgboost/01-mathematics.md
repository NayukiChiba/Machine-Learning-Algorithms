---
title: XGBoost — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/ensemble/xgboost.py`
>  
> 相关对象：`XGBRegressor`、`train_model(...)`

## 本章目标

1. 理解 XGBoost 为什么在 GBDT 基础上进一步引入二阶优化和正则化目标。
2. 理解叶子权重最优解和分裂增益公式是如何得到的。
3. 把这些公式和当前源码中的 `gamma`、`reg_alpha`、`reg_lambda`、`max_depth` 等参数对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 正则化目标函数 | 优化目标 | 同时考虑拟合误差与树复杂度 |
| 二阶泰勒展开 | 近似工具 | 用梯度和二阶信息近似当前轮目标 |
| 叶子权重最优解 | 闭式结果 | 给出单个叶子节点最优输出值 |
| 分裂增益 | 分裂准则 | 判断一个切分是否值得执行 |
| `gamma` / `reg_lambda` | 源码参数 | 公式中复杂度惩罚的直接映射 |

## 1. 核心思想

XGBoost（eXtreme Gradient Boosting）在 GBDT 基础上引入二阶泰勒展开来近似目标函数，并加入树结构的正则化项，使训练更快、更稳定。

### 理解重点

- 当前源码中的 `XGBRegressor(...)`，本质上是在执行这种“加法树模型 + 二阶优化 + 正则化”的训练过程。
- 和普通单棵树相比，这里更强调多轮加法建模。
- 和普通 boosting 相比，这里更强调目标函数近似和复杂度约束。

## 2. 正则化目标函数

### 参数速览（本节）

适用目标函数：XGBoost 总体目标

| 参数名 | 数学含义 | 在源码中的对应 |
|---|---|---|
| $L(y_i, \hat{y}_i)$ | 单样本损失 | 回归目标内部损失 |
| $\Omega(h_t)$ | 第 $t$ 棵树的复杂度惩罚 | `gamma`、`reg_lambda` 等 |
| $T$ | 树的轮数 | `n_estimators` |

$$
\text{Obj} = \sum_{i=1}^{N} L(y_i, \hat{y}_i) + \sum_{t=1}^{T} \Omega(h_t)
$$

其中树的正则项：

$$
\Omega(h) = \gamma \cdot |\text{叶子数}| + \frac{1}{2}\lambda \sum_{j=1}^{J} w_j^2
$$

$J$ 为叶节点数，$w_j$ 为叶节点权重。

### 理解重点

- 这个目标函数说明 XGBoost 并不是只想把训练误差压低。
- 它还要主动惩罚树太复杂或叶子权重太大。
- 当前源码中的 `gamma` 和 `reg_lambda` 就是这层数学思想最直接的工程映射。

## 3. 二阶泰勒展开推导

在第 $t$ 轮，目标函数对第 $t$ 棵树 $h_t$ 为：

$$
\text{Obj}^{(t)} = \sum_{i=1}^{N} L(y_i, \hat{y}_i^{(t-1)} + h_t(\mathbf{x}_i)) + \Omega(h_t)
$$

对 $L$ 在 $\hat{y}_i^{(t-1)}$ 处做二阶泰勒展开：

$$
L(y_i, \hat{y}_i^{(t-1)} + h_t) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i h_t(\mathbf{x}_i) + \frac{1}{2} h_i h_t^2(\mathbf{x}_i)
$$

其中：

$$
g_i = \frac{\partial L(y_i, \hat{y})}{\partial \hat{y}} \Bigg|_{\hat{y}=\hat{y}_i^{(t-1)}}, \quad
h_i = \frac{\partial^2 L(y_i, \hat{y})}{\partial \hat{y}^2} \Bigg|_{\hat{y}=\hat{y}_i^{(t-1)}}
$$

去掉与 $h_t$ 无关的常数项，目标函数化为：

$$
\tilde{\text{Obj}}^{(t)} = \sum_{i=1}^{N} \left[g_i h_t(\mathbf{x}_i) + \frac{1}{2} h_i h_t^2(\mathbf{x}_i)\right] + \Omega(h_t)
$$

### 理解重点

- 这一步是 XGBoost 和更朴素 boosting 方法的重要区别之一。
- 通过同时使用一阶梯度和二阶信息，当前轮目标可以被更稳定地近似。
- 当前数学章节里的“二阶优化”直觉，就来自这里。

## 4. 叶节点权重最优解

定义叶节点 $j$ 的样本集合 $I_j = \{i : \mathbf{x}_i \in \text{leaf}_j\}$，则 $h_t(\mathbf{x}_i) = w_j$。

代入目标函数：

$$
\tilde{\text{Obj}}^{(t)} = \sum_{j=1}^{J} \left[G_j w_j + \frac{1}{2}(H_j + \lambda)w_j^2\right] + \gamma J
$$

其中：

$$
G_j = \sum_{i \in I_j} g_i, \quad H_j = \sum_{i \in I_j} h_i
$$

对 $w_j$ 求导令其为零：

$$
\boxed{w_j^* = -\frac{G_j}{H_j + \lambda}}
$$

代回目标函数：

$$
\boxed{\tilde{\text{Obj}}^* = -\frac{1}{2}\sum_{j=1}^{J} \frac{G_j^2}{H_j + \lambda} + \gamma J}
$$

### 理解重点

- 这一结果说明叶子输出值不是随意设置的，而是由梯度统计量和正则项共同决定。
- `reg_lambda` 在这里直接参与分母，因此它会抑制叶子权重过大。
- 这也是为什么 XGBoost 的正则化不仅是“限制分裂”，还会约束叶子输出幅度。

## 5. 分裂增益

对节点分裂的增益：

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma
$$

只有当 $\text{Gain} > 0$ 时才分裂，$\gamma$ 起到预剪枝作用。

### 理解重点

- 分裂不是“只要能切就切”，而是要看切开后能不能带来足够收益。
- `gamma` 越大，分裂门槛越高，模型就越保守。
- 当前源码把 `gamma` 直接暴露成超参数，就是为了控制这种分裂谨慎程度。

## 6. XGBoost 的工程优化

- 列采样：类似随机森林的特征子集采样
- 加权分位数草图：高效近似分割点搜索
- 缓存感知访问：利用 CPU 缓存加速列遍历
- 稀疏感知：自动处理缺失值

### 理解重点

- 当前仓库训练封装没有逐条展开这些底层优化，但它们是 XGBoost 被广泛使用的重要原因。
- 从工程视角看，XGBoost 的优势不只来自数学目标函数，也来自训练实现层面的高效设计。
- 文档里需要点到这一层，但不能误写成当前仓库自己实现了这些机制。

## 7. 数学原理如何映射到当前源码

### 理解重点

- 公式中的树轮数 $T$，在工程里对应 `n_estimators`。
- 复杂度惩罚中的 $\gamma$ 和 $\lambda$，在工程里分别对应 `gamma` 和 `reg_lambda`。
- 叶子权重约束与稀疏化思想，在工程里还进一步对应到 `reg_alpha` 和 `min_child_weight` 这类参数。
- 当前训练代码没有手写梯度统计和增益搜索，而是由 `XGBRegressor.fit(...)` 内部完成。

## 常见坑

1. 把 `gamma` 误解成学习率，实际上它控制的是分裂增益门槛。
2. 只看到“很多树叠加”，却忽略 XGBoost 之所以强还在于二阶近似和正则化目标。
3. 把数学页里的工程优化误读成当前仓库手写实现了这些底层机制。

## 小结

- XGBoost 的数学核心，是在 boosting 框架下使用二阶泰勒展开和正则化目标来优化树模型。
- 叶子权重最优解和分裂增益公式，是理解其训练行为的关键。
- 当前源码中的 `gamma`、`reg_alpha`、`reg_lambda`、`n_estimators` 等参数，正是这些数学思想在工程层面的直接映射。
