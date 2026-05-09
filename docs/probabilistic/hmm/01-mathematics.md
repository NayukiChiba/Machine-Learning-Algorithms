---
title: HMM — 数学原理
outline: deep
---

# 数学原理

## 本章目标

1. 理解 HMM 的概率生成过程——隐状态按马尔可夫链演化，观测由当前隐状态发射。
2. 理解三大算法的数学本质——Forward（评估，求和）、Viterbi（解码，取最大）、Baum-Welch（学习，EM 迭代）。
3. 把这些数学表达和当前源码中的 `n_components`、`predict(...)`、`transmat_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| HMM 五元组 | 模型定义 | $\lambda = (\mathcal{S}, \mathcal{O}, \mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$——完整描述离散 HMM 的概率参数 |
| 马尔可夫假设 | 核心假设 | $P(q_t \mid q_{t-1}, \dots, q_1) = P(q_t \mid q_{t-1})$——未来仅依赖当前，与历史无关 |
| Forward 算法 | 评估算法 | 计算 $P(\mathbf{O} \mid \lambda)$——观测序列在当前模型下的概率，复杂度 $O(N^2 T)$ |
| Viterbi 算法 | 解码算法 | 求全局最优隐状态路径 $\mathbf{Q}^* = \arg\max_{\mathbf{Q}} P(\mathbf{Q} \mid \mathbf{O}, \lambda)$ |
| Baum-Welch 算法 | 学习算法 | EM 在 HMM 上的特例——E 步 Forward-Backward 计算时序后验，M 步计数重估参数 |
| `transmat_` | 源码属性 | 训练后学习到的状态转移矩阵 $\mathbf{A}$（$3 \times 3$，行和为 1） |

## 1. HMM 的生成过程

HMM 描述由隐状态序列驱动观测序列的生成过程：

1. $t=1$：以概率 $\pi_i$ 选择初始隐状态 $q_1 = s_i$。
2. 从隐状态 $q_1$ 的发射分布中生成观测 $o_1$：$P(o_1 = v_k \mid q_1 = s_i) = b_i(k)$。
3. $t \ge 2$：以概率 $a_{ij}$ 从 $q_{t-1} = s_i$ 转移到 $q_t = s_j$。
4. 从 $q_t$ 的发射分布中生成 $o_t$。

两个基本假设：

**一阶马尔可夫假设**：
$$
P(q_t \mid q_{t-1}, q_{t-2}, \dots, q_1) = P(q_t \mid q_{t-1})
$$

**观测独立假设**：
$$
P(o_t \mid q_1, \dots, q_T, o_1, \dots, o_T) = P(o_t \mid q_t)
$$

### 理解重点

- 第一条假设意味着当前状态仅由上一时刻状态决定——所有历史信息被压缩到 $q_{t-1}$ 中。
- 第二条假设意味着当前观测仅由当前隐状态决定——观测之间条件独立。
- 当前数据生成函数 `ProbabilisticData.hmm()` 正是按这两层结构逐步采样：先以 $A$ 转移隐状态，再以 $B$ 发射观测。

## 2. 模型定义：五元组

HMM 由五元组 $\lambda = (\mathcal{S}, \mathcal{O}, \mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$ 定义：

| 符号 | 数学含义 | 在当前源码中的对应 |
|---|---|---|
| $\mathcal{S} = \{s_1, \dots, s_N\}$ | $N$ 个隐状态集合 | `n_components=3` 对应状态数 |
| $\mathcal{O} = \{v_1, \dots, v_M\}$ | $M$ 个离散观测符号集合 | 观测 `obs` 的取值空间 $\{0, 1, 2\}$ |
| $\mathbf{A} = [a_{ij}]_{N \times N}$ | 状态转移矩阵——$a_{ij} = P(q_{t+1}=s_j \mid q_t=s_i)$，行和为 1 | `model.transmat_` |
| $\mathbf{B} = [b_i(k)]_{N \times M}$ | 发射矩阵——$b_i(k) = P(o_t=v_k \mid q_t=s_i)$，行和为 1 | `model.emissionprob_` |
| $\boldsymbol{\pi} = [\pi_i]_{1 \times N}$ | 初始状态分布——$\pi_i = P(q_1=s_i)$，和为 1 | `model.startprob_` |

### 理解重点

- $A_{ij}$ 的物理含义是"从状态 $i$ 一步转移到状态 $j$ 的概率"——对角线越大，状态越稳定，越不容易跳变。
- 当前真实 $A$ 的对角线为 $[0.80, 0.60, 0.70]$——状态 1 最黏滞（80% 概率停留），状态 2 相对活跃（40% 概率跳走）。
- $B_i(k)$ 的物理含义是"隐状态 $i$ 产生观测符号 $k$ 的概率"——每行描述一个隐状态的"观测偏好"。

## 3. 三大基本问题

HMM 经典上有三个基本问题：

| 问题 | 英文名 | 输入 | 输出 | 对应算法 | 当前源码体现 |
|---|---|---|---|---|---|
| 评估 (Evaluation) | Likelihood | $\lambda$、$\mathbf{O}$ | $P(\mathbf{O} \mid \lambda)$ | Forward | `model.score(X, lengths)` |
| 解码 (Decoding) | Decoding | $\lambda$、$\mathbf{O}$ | $\mathbf{Q}^*$ | Viterbi | `model.predict(X, lengths)` |
| 学习 (Learning) | Training | $\mathbf{O}$ | $\lambda^*$ | Baum-Welch | `model.fit(X, lengths)` |

### 理解重点

- 三个问题的难度递增：评估只需单向递推，解码需要全局优化+回溯，学习需要迭代 EM。
- 当前流水线直接展示"学习 + 解码"——先 `fit` 训练，再 `predict` 推断路径。
- `score`（Forward 对数概率）可用于模型选择——比较不同 $K$ 下的拟合质量，但当前流水线仅打印准确率。

## 4. 问题一：评估（Forward 算法）

给定模型 $\lambda$ 和观测序列 $\mathbf{O} = (o_1, \dots, o_T)$，计算：

$$
P(\mathbf{O} \mid \lambda) = \sum_{\text{所有路径 } \mathbf{Q}} P(\mathbf{O} \mid \mathbf{Q}, \lambda) P(\mathbf{Q} \mid \lambda)
$$

暴力枚举所有 $N^T$ 条路径不可行——Forward 算法用动态规划将复杂度降为 $O(N^2 T)$。

**定义前向变量**：
$$
\alpha_t(i) = P(o_1, o_2, \dots, o_t, q_t = s_i \mid \lambda)
$$

**初始化**（$t=1$）：
$$
\alpha_1(i) = \pi_i \cdot b_i(o_1), \quad i = 1, \dots, N
$$

**递推**（$t=1 \to T-1$）：
$$
\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i) \cdot a_{ij}\right] \cdot b_j(o_{t+1})
$$

**终止**：
$$
P(\mathbf{O} \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i)
$$

### 理解重点

- 递推的核心操作是**求和**（$\sum_i$）——汇集所有到达状态 $j$ 的路径概率。
- 这反映了评估问题的本质：对"所有可能路径"的概率加权求和，而非找单条最优路径。
- 当前 `model.score(X, lengths)` 返回对数概率 $\log P(\mathbf{O} \mid \lambda)$——值越大（负得越少），模型对观测序列的解释越好。

## 5. 问题二：解码（Viterbi 算法）

给定模型和观测，找最可能的单条隐状态序列：

$$
\mathbf{Q}^* = \arg\max_{\mathbf{Q}} P(\mathbf{Q} \mid \mathbf{O}, \lambda)
$$

**定义 Viterbi 变量**：
$$
\delta_t(i) = \max_{q_1, \dots, q_{t-1}} P(q_1, \dots, q_t = s_i, o_1, \dots, o_t \mid \lambda)
$$

**初始化**（$t=1$）：
$$
\delta_1(i) = \pi_i \cdot b_i(o_1), \quad \psi_1(i) = 0
$$

**递推**（$t=2 \to T$）：
$$
\delta_t(j) = \max_{1 \le i \le N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(o_t)
$$

$$
\psi_t(j) = \arg\max_{1 \le i \le N} [\delta_{t-1}(i) \cdot a_{ij}]
$$

**终止**：
$$
P^* = \max_{1 \le i \le N} \delta_T(i), \quad q_T^* = \arg\max_i \delta_T(i)
$$

**回溯**（$t = T-1 \to 1$）：
$$
q_t^* = \psi_{t+1}(q_{t+1}^*)
$$

### 理解重点

- 递推的核心操作是**取最大**（$\max_i$）而非求和——这是与 Forward 算法的本质区别。
- $\psi_t(j)$ 记录了到达 $(t, j)$ 的最佳前驱状态——回溯时沿这条"面包屑"路径重建全局最优序列。
- Viterbi 保证路径的**全局合法性**——相邻状态间的转移概率 $a_{ij} > 0$，不会产生"不可能跳转"。
- 当前 `model.predict(X_obs, lengths)` 正是 Viterbi 解码——返回全局最优隐状态路径，与 `state_true` 逐步对比算准确率。

## 6. 问题三：学习（Baum-Welch 算法）

给定观测序列 $\mathbf{O}$，估计模型参数 $\lambda$。Baum-Welch 是 EM 在 HMM 上的特例。

**后向变量**（Backward 算法——E 步需要）：
$$
\beta_t(i) = P(o_{t+1}, \dots, o_T \mid q_t = s_i, \lambda)
$$

初始化 $\beta_T(i) = 1$，逆向递推：
$$
\beta_t(i) = \sum_{j=1}^{N} a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)
$$

**E 步：计算时序后验**

状态占有概率（单点后验）：
$$
\gamma_t(i) = P(q_t = s_i \mid \mathbf{O}, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(\mathbf{O} \mid \lambda)}
$$

状态转移概率（成对后验）：
$$
\xi_t(i, j) = P(q_t = s_i, q_{t+1} = s_j \mid \mathbf{O}, \lambda) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(\mathbf{O} \mid \lambda)}
$$

**M 步：参数重估**

初始分布：
$$
\hat{\pi}_i = \gamma_1(i)
$$

转移矩阵：
$$
\hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

发射矩阵：
$$
\hat{b}_i(k) = \frac{\sum_{t=1}^{T} \gamma_t(i) \cdot \mathbb{1}(o_t = v_k)}{\sum_{t=1}^{T} \gamma_t(i)}
$$

### 理解重点

- Baum-Welch 的 E 步需要**成对后验** $\xi_t(i,j)$——这是与普通 EM（仅需逐点后验 $\gamma_{ik}$）的根本区别。因为 HMM 的 M 步要重估转移矩阵，需要知道相邻时间步的状态联合分布。
- Forward-Backward 是计算 $\gamma_t(i)$ 和 $\xi_t(i,j)$ 的高效方法——两个方向的消息在 $t$ 处交汇，给出完整的时序后验。
- 当前源码没有手写 Baum-Welch，而是由 `hmmlearn` 的 `fit()` 内部完成——但数学本质完全一致。
- 对于 300 步 3 状态的序列，每轮 E 步复杂度 $O(300 \times 3^2) = O(2700)$——比独立样本 EM 的逐点 E 步贵。

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 隐状态数 | $N$ | `n_components=3` |
| 观测符号数 | $M$ | 观测取值空间 $\{0, 1, 2\}$ |
| 转移矩阵 | $a_{ij} = P(q_{t+1}=s_j \mid q_t=s_i)$ | `model.transmat_`（$3 \times 3$，行和为 1） |
| 发射矩阵 | $b_i(k) = P(o_t=v_k \mid q_t=s_i)$ | `model.emissionprob_`（$3 \times 3$，行和为 1） |
| 初始分布 | $\pi_i = P(q_1=s_i)$ | `model.startprob_`（长度为 3，和为 1） |
| 观测序列概率 | $P(\mathbf{O} \mid \lambda)$ | `model.score(X, lengths)`——Forward 对数概率 |
| 最优隐状态路径 | $\mathbf{Q}^* = \arg\max_{\mathbf{Q}} P(\mathbf{Q} \mid \mathbf{O}, \lambda)$ | `model.predict(X, lengths)`——Viterbi 解码 |
| 时序后验 | $\gamma_t(i)$ | Forward-Backward 内部计算——不直接暴露 |
| 最大迭代 | $T_{\max}$ | `n_iter=100` |
| 收敛阈值 | $\|\log P^{(t+1)} - \log P^{(t)}\| < \varepsilon$ | `tol=1e-3` |
| 序列长度 | $T$ | `lengths = [300]` |

## 8. HMM vs EM (GMM) 数学对比

| 维度 | EM (GMM) | HMM |
|---|---|---|
| 数据结构 | i.i.d. 样本 $\{\mathbf{x}_i\}_{i=1}^{N}$ | **序列** $\{o_t\}_{t=1}^{T}$——时间有序 |
| 隐变量 | $z_{ik}$——样本 $i$ 属于分量 $k$ | **$q_t$——时间 $t$ 的隐状态** |
| 隐变量依赖 | 各样本独立 | **马尔可夫链依赖** $q_t \to q_{t+1}$ |
| 生成过程 | $\pi_k \to z_{ik} \to \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \to \mathbf{x}_i$ | **$q_{t-1} \xrightarrow{A} q_t \xrightarrow{B} o_t$——链式生成** |
| E 步复杂度 | $O(NK)$——逐点独立计算 | **$O(T N^2)$——Forward-Backward 时间耦合** |
| E 步所需后验 | 逐点后验 $\gamma(z_{ik})$ | **成对后验 $\xi_t(i,j)$——重估转移矩阵必需** |
| M 步核心操作 | 责任加权平均 $\to \boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$ | **计数重估 $\to A$、$B$、$\pi$** |
| 参数数 | $K(\frac{d(d+1)}{2} + d + 1)$ | **$N^2 + NM + N$** |
| 预测 | 逐点 argmax $\arg\max_k \gamma(z_{ik})$ | **Viterbi 全局解码 $\arg\max_{\mathbf{Q}} P(\mathbf{Q} \mid \mathbf{O})$** |
| 收敛保证 | 对数似然单调不减 | 对数似然单调不减 |

## 常见坑

1. 把 `state_true` 误当成 Baum-Welch 训练输入——实际上当前训练只依赖观测序列，`state_true` 仅用于评估。
2. 混淆 Forward（求和）和 Viterbi（取最大）的递推公式——两者的目标不同（评估 vs 解码），操作符不同（$\sum$ vs $\max$）。
3. 以为 Baum-Welch 的 E 步和 GMM 的 E 步完全一样——HMM 需要成对后验 $\xi_t(i,j)$，因为转移矩阵的重估依赖相邻时间步的联合分布。
4. 把解码问题和评估问题混为一谈——"路径最优"（Viterbi）和"概率最大"（Forward）是两回事。

## 小结

- HMM 的数学核心链：马尔可夫假设 → 五元组定义 → 三大问题（评估/解码/学习）→ Forward（求和递推）/ Viterbi（取最大递推+回溯）/ Baum-Welch（Forward-Backward + 计数重估）。
- 与 EM (GMM) 的根本区别：HMM 的隐变量有时间依赖（马尔可夫链），E 步需成对后验 $\xi_t(i,j)$，预测需 Viterbi 全局解码——而非逐点独立计算。
- 当前源码 `CategoricalHMM(n_components=3, n_iter=100)` 将上述数学全部封装在 `fit`/`predict`/`score` 三个方法中——`transmat_`、`emissionprob_`、`startprob_` 是训练后的可直接检验的参数。
