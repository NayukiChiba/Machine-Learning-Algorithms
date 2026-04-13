---
title: HMM — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/probabilistic/hmm.py`
>  
> 相关对象：`train_model(...)`、`model.predict(...)`

## 本章目标

1. 理解 HMM 为什么适合描述“由隐状态序列生成观测序列”的过程。
2. 理解前向算法、Viterbi 和 Baum-Welch 分别解决什么问题。
3. 把这些数学表达和当前源码中的 `n_components`、`predict(...)`、`transmat_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| HMM 五元组 | 模型定义 | 描述状态集合、观测集合和概率参数 |
| 前向算法 | 评估算法 | 计算观测序列概率 |
| Viterbi | 解码算法 | 求最可能的隐状态路径 |
| Baum-Welch | 学习算法 | 在只给观测序列时估计 HMM 参数 |
| `transmat_` | 源码属性 | 训练后学习到的状态转移矩阵 |

## 1. 核心思想

HMM 是一种描述“由隐状态序列生成观测序列”的概率图模型。隐状态之间通过马尔可夫链连接，观测值由当前隐状态生成。

### 理解重点

- 当前源码中的 `obs` 是可见的观测序列，`state_true` 是数据生成时记录下来的参考隐状态。
- 对模型来说，真正困难的问题是只看 `obs` 时，如何学习状态结构并推断隐藏状态路径。
- 这就是 HMM 和普通独立样本模型的根本区别。

## 2. 模型定义

HMM 由五元组 $\lambda = (\mathcal{S}, \mathcal{O}, \mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$ 定义：

| 符号 | 含义 | 在当前源码中的对应 |
|------|------|---|
| $\mathcal{S} = \{s_1, \dots, s_N\}$ | $N$ 个隐状态集合 | `n_components` 对应状态数 |
| $\mathcal{O} = \{o_1, \dots, o_M\}$ | $M$ 个观测符号集合 | 离散观测 `obs` 的取值空间 |
| $\mathbf{A} = [a_{ij}]_{N \times N}$ | 状态转移概率矩阵 | `model.transmat_` |
| $\mathbf{B} = [b_j(k)]_{N \times M}$ | 发射概率矩阵 | 数据生成里的 `hmm_B`，训练后由模型内部估计 |
| $\boldsymbol{\pi} = [\pi_i]_N$ | 初始状态概率 | 数据生成里的 `hmm_pi` |

### 两个基本假设

1. 一阶马尔可夫假设：

$$
P(q_t \mid q_{t-1}, q_{t-2}, \dots) = P(q_t \mid q_{t-1})
$$

2. 观测独立假设：

$$
P(o_t \mid q_1, \dots, q_T, o_1, \dots, o_T) = P(o_t \mid q_t)
$$

### 理解重点

- 第一条假设说的是“当前状态主要由上一时刻状态决定”。
- 第二条假设说的是“当前观测只由当前隐状态决定”。
- 当前数据生成函数正是按这两层结构逐步采样出来的。

## 3. 三大基本问题

### 理解重点

HMM 经典上有三大问题：

1. 评估：给定模型和观测序列，序列出现的概率有多大。
2. 解码：给定模型和观测序列，最可能的隐状态路径是什么。
3. 学习：只给观测序列时，如何估计模型参数。

当前流水线最直接展示的是“学习 + 解码”两部分：先训练模型，再用 `predict(...)` 推断隐状态路径。

## 4. 问题一：评估（Evaluation）

给定模型 $\lambda$ 和观测序列 $\mathbf{O} = (o_1, \dots, o_T)$，计算：

$$
P(\mathbf{O} \mid \lambda)
$$

### 前向算法

定义前向变量：

$$
\alpha_t(i) = P(o_1, o_2, \dots, o_t, q_t = s_i \mid \lambda)
$$

初始化：

$$
\alpha_1(i) = \pi_i \cdot b_i(o_1), \quad i = 1, \dots, N
$$

递推：

$$
\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i) \cdot a_{ij}\right] \cdot b_j(o_{t+1})
$$

终止：

$$
P(\mathbf{O} \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i)
$$

时间复杂度从暴力枚举的 $O(N^T)$ 降到 $O(N^2 T)$。

### 理解重点

- 前向算法解决的是“这条观测序列在当前模型下有多可能”。
- 当前流水线没有显式打印这个概率，但它是 HMM 评估问题的基础。
- 理解这一层，有助于区分“序列概率评估”和“隐状态路径解码”不是同一件事。

## 5. 问题二：解码（Decoding）

给定模型和观测，找最可能的隐状态序列：

$$
\mathbf{Q}^* = \arg\max_\mathbf{Q} P(\mathbf{Q} \mid \mathbf{O}, \lambda)
$$

### Viterbi 算法

定义：

$$
\delta_t(i) = \max_{q_1, \dots, q_{t-1}} P(q_1, \dots, q_t = s_i, o_1, \dots, o_t \mid \lambda)
$$

初始化：

$$
\delta_1(i) = \pi_i \cdot b_i(o_1), \quad \psi_1(i) = 0
$$

递推：

$$
\delta_t(j) = \max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(o_t)
$$

$$
\psi_t(j) = \arg\max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}]
$$

终止：

$$
P^* = \max_{1 \leq i \leq N} \delta_T(i), \quad q_T^* = \arg\max_i \delta_T(i)
$$

回溯：

$$
q_t^* = \psi_{t+1}(q_{t+1}^*), \quad t = T-1, \dots, 1
$$

### 理解重点

- 当前流水线里的 `model.predict(X_obs, lengths)`，工程上就是在输出一条预测隐状态路径。
- 这一步最接近 HMM 里的解码问题，而不是普通分类器的单点标签预测。
- 也正因为如此，当前准确率衡量的是整条路径逐时间步的对比结果。

## 6. 问题三：学习（Learning）

给定观测序列，估计模型参数 $\lambda$。经典方法是 Baum-Welch 算法，也就是 EM 在 HMM 上的特例。

### 后向变量

$$
\beta_t(i) = P(o_{t+1}, \dots, o_T \mid q_t = s_i, \lambda)
$$

### E 步

$$
\xi_t(i, j) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(\mathbf{O} \mid \lambda)}
$$

$$
\gamma_t(i) = \sum_{j=1}^N \xi_t(i, j)
$$

### M 步

$$
\hat{\pi}_i = \gamma_1(i)
$$

$$
\hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

$$
\hat{b}_j(k) = \frac{\sum_{t=1}^{T} \gamma_t(j) \cdot \mathbb{1}(o_t = k)}{\sum_{t=1}^{T} \gamma_t(j)}
$$

### 理解重点

- 当前源码没有手写 Baum-Welch，而是把这部分交给 `hmmlearn` 内部 `fit(...)` 完成。
- 但从数学上看，当前训练函数本质上仍是在执行这一类 EM 式参数学习。
- 这也是为什么 HMM 分册和 EM 分册在“训练思想”上是相通的。

## 7. 数学原理如何映射到当前源码

### 理解重点

- 公式里的状态数 $N$，在工程里对应 `n_components`。
- 解码问题，在工程里对应 `model.predict(X_obs, lengths)`。
- 转移矩阵 $\mathbf{A}$，在工程里对应训练后打印的 `model.transmat_`。
- Baum-Welch 学习过程在工程上没有手写展开，而是由 hmmlearn 的 `fit(...)` 内部完成。

## 常见坑

1. 把 `state_true` 误当成 Baum-Welch 训练输入，实际上当前训练只依赖观测序列。
2. 只记住前向、Viterbi 和 Baum-Welch 的名字，却不把它们和 `predict(...)`、`transmat_`、`n_components` 这些源码对象对应起来。
3. 把解码问题和评估问题混为一谈，误以为“概率最大”和“路径最优”是同一件事。

## 小结

- HMM 的数学核心，是在隐状态链和观测序列之间建立概率生成关系。
- 前向算法解决评估问题，Viterbi 解决解码问题，Baum-Welch 解决学习问题。
- 当前源码中的 `n_components`、`predict(...)` 和 `transmat_`，正是这些数学思想在工程层面的直接映射。
