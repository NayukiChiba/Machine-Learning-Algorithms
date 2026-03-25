# HMM 隐马尔可夫模型 (Hidden Markov Model)

## 核心思想

HMM 是一种描述**由隐状态序列生成观测序列**的概率图模型。隐状态之间通过马尔可夫链连接，观测值由当前隐状态生成。

## 模型定义

HMM 由五元组 $\lambda = (\mathcal{S}, \mathcal{O}, \mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$ 定义：

| 符号 | 含义 |
|------|------|
| $\mathcal{S} = \{s_1, \dots, s_N\}$ | $N$ 个隐状态集合 |
| $\mathcal{O} = \{o_1, \dots, o_M\}$ | $M$ 个观测符号集合 |
| $\mathbf{A} = [a_{ij}]_{N \times N}$ | 状态转移概率矩阵 |
| $\mathbf{B} = [b_j(k)]_{N \times M}$ | 发射概率矩阵 |
| $\boldsymbol{\pi} = [\pi_i]_N$ | 初始状态概率 |

### 两个基本假设

1. **一阶马尔可夫假设**：$P(q_t \mid q_{t-1}, q_{t-2}, \dots) = P(q_t \mid q_{t-1})$
2. **观测独立假设**：$P(o_t \mid q_1, \dots, q_T, o_1, \dots, o_T) = P(o_t \mid q_t)$

## 三大基本问题

### 问题一：评估 (Evaluation)

给定模型 $\lambda$ 和观测序列 $\mathbf{O} = (o_1, \dots, o_T)$，计算 $P(\mathbf{O} \mid \lambda)$。

#### 前向算法

定义前向变量：

$$
\alpha_t(i) = P(o_1, o_2, \dots, o_t, q_t = s_i \mid \lambda)
$$

**初始化**：

$$
\alpha_1(i) = \pi_i \cdot b_i(o_1), \quad i = 1, \dots, N
$$

**递推**：

$$
\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i) \cdot a_{ij}\right] \cdot b_j(o_{t+1})
$$

**终止**：

$$
P(\mathbf{O} \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i)
$$

时间复杂度从暴力的 $O(N^T)$ 降至 $O(N^2 T)$。

### 问题二：解码 (Decoding)

给定模型和观测，找最可能的隐状态序列 $\mathbf{Q}^* = \arg\max_\mathbf{Q} P(\mathbf{Q} \mid \mathbf{O}, \lambda)$。

#### Viterbi 算法

定义：

$$
\delta_t(i) = \max_{q_1, \dots, q_{t-1}} P(q_1, \dots, q_t = s_i, o_1, \dots, o_t \mid \lambda)
$$

**初始化**：

$$
\delta_1(i) = \pi_i \cdot b_i(o_1), \quad \psi_1(i) = 0
$$

**递推**：

$$
\delta_t(j) = \max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(o_t)
$$

$$
\psi_t(j) = \arg\max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}]
$$

**终止**：

$$
P^* = \max_{1 \leq i \leq N} \delta_T(i), \quad q_T^* = \arg\max_i \delta_T(i)
$$

**回溯**：

$$
q_t^* = \psi_{t+1}(q_{t+1}^*), \quad t = T-1, \dots, 1
$$

### 问题三：学习 (Learning)

给定观测序列，估计模型参数 $\lambda$。使用 **Baum-Welch 算法**（EM 算法的特殊形式）。

#### 后向变量

$$
\beta_t(i) = P(o_{t+1}, \dots, o_T \mid q_t = s_i, \lambda)
$$

#### E 步

$$
\xi_t(i, j) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(\mathbf{O} \mid \lambda)}
$$

$$
\gamma_t(i) = \sum_{j=1}^N \xi_t(i, j)
$$

#### M 步

$$
\hat{\pi}_i = \gamma_1(i)
$$

$$
\hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

$$
\hat{b}_j(k) = \frac{\sum_{t=1}^{T} \gamma_t(j) \cdot \mathbb{1}(o_t = k)}{\sum_{t=1}^{T} \gamma_t(j)}
$$

## 代码对应

```bash
python -m pipelines.probabilistic.hmm
```
