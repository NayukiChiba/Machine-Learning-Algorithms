---
title: HMM — 训练与预测
outline: deep
---

# 训练与预测

## 本章目标

1. 理解 `pipelines/probabilistic/hmm.py` 的 `run()` 流水线——序列模型的端到端流程（无标准化、无切分、无可视化）。
2. 理解 Baum-Welch 训练过程——序列数据的 EM 算法（Forward-Backward + 参数重估）。
3. 理解 Viterbi 解码的预测输出——全局最优隐状态路径及其与真实状态的对比。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 序列模型流水线编排——4 步完成数据整形、训练、Viterbi 解码和评估 |
| `model.fit(X_obs, lengths)` | 方法 | Baum-Welch 训练——迭代 E 步（Forward-Backward）+ M 步（参数重估） |
| `model.predict(X_obs, lengths)` | 方法 | Viterbi 解码——全局最优隐状态路径 |
| `model.score(X_obs, lengths)` | 方法 | Forward 算法——计算观测序列的对数概率 |
| 隐状态准确率 | 评估 | `np.mean(states_pred == y_true)`——Viterbi 路径与真实状态的逐步比较 |

## 1. 完整流水线流程

### 流程概述

```
hmm_data.copy()
    │
    ├─ ① obs = data["obs"].values.astype(int)  → reshape(-1, 1)
    ├─ ② lengths = [len(obs)]
    ├─ ③ y_true = data["state_true"].values.astype(int)
    ├─ ④ model = train_model(X_obs, lengths)
    └─ ⑤ states_pred = model.predict(X_obs, lengths) → 准确率 + 转移矩阵
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 复制数据 | `hmm_data.copy()` | 全局 `DataFrame` | 本地 `DataFrame`，`(300, 3)` | 避免修改全局变量 |
| 整形观测 | `obs.reshape(-1, 1)` | `Series`，`(300,)` | `ndarray`，`(300, 1)` | hmmlearn 要求列向量输入 |
| 序列长度 | `[len(obs)]` | — | `list[int]` | 单条 300 步序列 |
| 提取真实状态 | `data["state_true"].values` | `DataFrame` | `ndarray`，`(300,)` | 仅用于评估对比 |
| 训练 | `train_model(X_obs, lengths)` | `(ndarray, list)` | `CategoricalHMM` | Baum-Welch 迭代 |
| Viterbi 解码 | `model.predict(X_obs, lengths)` | `(ndarray, list)` | `ndarray`，`(300,)` | 全局最优隐状态路径 |
| 评估 | `np.mean(states_pred == y_true)` | `(ndarray, ndarray)` | `float` | 逐步准确率 |

### 理解重点

- 这是本仓库所有流水线中最简洁的——仅 4 步核心操作，**无标准化**（离散观测不需要）、**无切分**（单条序列）、**无可视化**（序列数据不适合散点图）。
- 每个步骤都是类型敏感的——`astype(int)` 确保 hmmlearn 识别为离散符号。
- `lengths = [len(obs)]` 虽然此处是单元素列表，但框架设计允许 `[100, 200, 150]` 等多序列批量训练。

## 2. 训练细节：Baum-Welch 算法

### 算法流程

```
初始化参数（随机或等值）
    ↓
E 步：Forward-Backward 算法
    Forward:  α_t(i) = P(o_1,...,o_t, s_t=i | λ)
    Backward: β_t(i) = P(o_{t+1},...,o_T | s_t=i, λ)
    计算后验: γ_t(i) = P(s_t=i | O, λ) = α_t(i)β_t(i) / P(O|λ)
              ξ_t(i,j) = P(s_t=i, s_{t+1}=j | O, λ)
    ↓
M 步：参数重估
    π̂_i = γ_1(i)
    Â_ij = Σ_{t=1}^{T-1} ξ_t(i,j) / Σ_{t=1}^{T-1} γ_t(i)
    B̂_ij = Σ_{t: o_t=j} γ_t(i) / Σ_{t=1}^{T} γ_t(i)
    ↓
检查收敛：|log P(O|λ_new) - log P(O|λ_old)| < tol ?
    是 → 停止
    否 → 回到 E 步
    ↓
达到 n_iter=100 → 终止
```

### 参数速览

| 参数名 | 当前取值 | 训练中的作用 |
|---|---|---|
| `n_components` | `3` | 隐状态数——决定了 $A$（3×3）、$B$（3×3）、$\pi$（3,）的维度 |
| `n_iter` | `100` | Baum-Welch 最大迭代次数 |
| `tol` | `1e-3` | 对数似然收敛阈值——连续两次变化小于此值则停止 |

### 理解重点

- Baum-Welch 在概念上是**EM 的序列版**——E 步用 Forward-Backward（而非逐点后验），M 步用计数重估（而非加权平均）。
- Forward 和 Backward 是两个互补的"消息传递"——Forward 从过去积累信息，Backward 从未来回传信息，交汇点给出每个时间步的状态后验。
- 对于 300 步 3 状态的序列，Baum-Welch 每轮 E 步的复杂度是 $O(300 \times 3^2) = O(2700)$——远比独立样本的 EM 贵。

## 3. 预测细节：Viterbi 解码

### 算法流程

```
初始化: δ_1(i) = π_i * B_{i,o_1}
递推:   δ_t(j) = max_i [δ_{t-1}(i) * A_{ij}] * B_{j,o_t}
        ψ_t(j) = argmax_i [δ_{t-1}(i) * A_{ij}]
终止:   ŝ_T = argmax_i δ_T(i)
回溯:   ŝ_t = ψ_{t+1}(ŝ_{t+1})   (t = T-1, ..., 1)
```

### 参数速览

| 方法 | 输入形状 | 输出形状 | 算法 | 输出含义 |
|---|---|---|---|---|
| `predict(X, lengths)` | `(300, 1)` + `lengths` | `(300,)` | **Viterbi** | 全局最优隐状态路径 |

### 理解重点

- Viterbi 保证路径的**全局一致性**——每一步的状态转移都是合法的（$A_{ij} > 0$），不会出现"不可能跳转"。
- 逐步 argmax（$\arg\max_i \gamma_t(i)$）可能产生 $A_{ij}=0$ 的非法转移——Viterbi 通过回溯机制避免。
- 当前流水线将 Viterbi 的预测与 `state_true` 逐步对比——准确率越高，模型越成功恢复隐状态序列。

## 4. 与 GMM（EM）训练流程的对比

| 步骤 | GMM (EM) | HMM (Baum-Welch) |
|---|---|---|
| 数据 | 独立样本矩阵 | **序列列向量 + lengths** |
| 标准化 | 有（`StandardScaler`） | **无**（离散符号不需要） |
| E 步 | 逐点后验 $\gamma(z_{ik})$ | **Forward-Backward → $\gamma_t(i)$ + $\xi_t(i,j)$** |
| M 步 | 加权更新 $\mu$、$\Sigma$、$\pi$ | **计数重估 $A$、$B$、$\pi$** |
| 复杂度 | $O(N \times K \times d^2)$ | **$O(T \times K^2)$** |
| 收敛诊断 | `lower_bound_`（对数似然） | **`monitor_`（逐次对数似然列表）** |
| 预测 | `predict`（逐点 argmax） | **`predict`（Viterbi 全局解码）** |

## 常见坑

1. 不传 `lengths` 参数——`fit(X)` 的错误调用，hmmlearn 必须知道每条序列的边界。
2. 忘记将观测 `reshape(-1, 1)`——hmmlearn 要求观测为列向量 `(n_steps, 1)`。
3. 把 `astype(int)` 漏掉——字符串或浮点观测符号可能导致 hmmlearn 无法识别。
4. 在极短序列上比较准确率——300 步中稳定迁移的比例有限，准确率波动大。

## 小结

- HMM 流水线是最简的 4 步序列流程——数据整形、Baum-Welch 训练、Viterbi 解码、准确率评估，无标准化/切分/可视化。
- `fit()` 的核心流程：Forward-Backward（E 步计算时序后验）→ 计数重估 $A$/$B$/$\pi$（M 步最大化）→ 对数似然收敛检查 → 循环。
- `predict()` 使用 Viterbi 全局解码——保证路径的转移合法性，与逐点 argmax 有本质区别。
