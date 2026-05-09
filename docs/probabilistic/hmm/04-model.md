---
title: HMM — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 HMM 模型——离散观测、序列数据、可选依赖。
2. 理解 `CategoricalHMM` 的核心构造器参数（`n_components`、`n_iter`、`tol`）及其序列含义。
3. 看清训练完成后最重要的模型属性——`transmat_`（转移矩阵）、`startprob_`（初始概率）、`emissionprob_`（发射矩阵）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 HMM 模型——含可选依赖检查（`CategoricalHMM` / `MultinomialHMM` 双备份） |
| `CategoricalHMM(...)` | 类 | hmmlearn 提供的离散 HMM——用 Baum-Welch（EM）估计参数 |
| `model.fit(X_obs, lengths)` | 方法 | Baum-Welch 训练——迭代 Forward-Backward + 参数重估 |
| `model.transmat_` | 属性 | 学习到的状态转移矩阵 $A$（3×3） |
| `model.startprob_` | 属性 | 学习到的初始状态分布 $\pi$（3,） |
| `model.emissionprob_` | 属性 | 学习到的观测发射矩阵 $B$（3×3） |
| `model.predict(X_obs, lengths)` | 方法 | Viterbi 解码——全局最优隐状态路径 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_obs, lengths, n_components=3, n_iter=100, tol=1e-3, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_obs` | `ndarray`，形状 `(300, 1)` | 观测序列列向量——离散符号 $\{0, 1, 2\}$，每行为一个时间步 | `obs.reshape(-1, 1)` |
| `lengths` | `list[int]` | 序列长度列表。`[300]`——单条 300 步的序列 | `[300]`、`[100, 200]` |
| `n_components` | `int` | 隐状态数。`3`——与真实隐状态数一致 | `2`、`3`、`5` |
| `n_iter` | `int` | Baum-Welch 最大迭代次数。`100`——比 GMM 的 EM 迭代少（序列计算更贵） | `50`、`100`、`200` |
| `tol` | `float` | 对数似然收敛阈值。`1e-3` | `1e-3`、`1e-4` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `CategoricalHMM` 或 `MultinomialHMM` | 已完成 `fit()` 的 HMM 模型 | — |

### 示例代码

```python
from model_training.probabilistic.hmm import train_model

obs = hmm_data["obs"].values.astype(int)
X_obs = obs.reshape(-1, 1)
lengths = [len(obs)]
model = train_model(X_obs, lengths)
```

### 理解重点

- `train_model(...)` 的参数签名与 GMM 完全不同——输入为序列数据（`X_obs` + `lengths`），而非独立样本矩阵。
- `lengths` 支持多条不等长序列——当前使用单条 300 步序列，但框架天然支持批量序列训练。
- 内部有双备份可选依赖处理——优先 `CategoricalHMM`，回退 `MultinomialHMM`。

## 2. `CategoricalHMM` 构造器参数

### 参数速览

适用 API：`CategoricalHMM(n_components=3, n_iter=100, tol=1e-3, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 隐状态数。`3`——HMM 的核心超参数，需预先设定 | `2`、`3`、`5` |
| `n_iter` | `int` | Baum-Welch 最大迭代次数。`100`——序列数据的 EM 通常收敛更慢 | `50`、`100`、`200` |
| `tol` | `float` | 对数似然收敛阈值。`1e-3` | `1e-3`、`1e-4` |
| `random_state` | `int` | 随机种子。`42`——保证参数初始化和结果可复现 | `42` |
| `init_params` | `str` | 参数初始化方法。默认 `"st"`（转移和发射矩阵随机初始化） | `"st"`、`""` |
| `params` | `str` | 哪些参数在训练中更新。默认 `"ste"`（startprob/transmat/emissionprob） | `"ste"`、`"st"` |
| `verbose` | `bool` | 是否打印详细日志。默认 `False` | `True`、`False` |

### 示例代码

```python
try:
    from hmmlearn.hmm import CategoricalHMM
    ModelClass = CategoricalHMM
except ImportError:
    from hmmlearn.hmm import MultinomialHMM
    ModelClass = MultinomialHMM

model = ModelClass(
    n_components=3,
    n_iter=100,
    tol=1e-3,
    random_state=42,
)
model.fit(X_obs, lengths)
```

### 理解重点

- `CategoricalHMM` 是 hmmlearn 0.3+ 的新 API——`MultinomialHMM` 是旧版本兼容（当前源码双备份）。
- `n_iter=100` 比 GMM 的 `max_iter=200` 少——因为序列计算中每次 Forward-Backward 的复杂度是 $O(T \times K^2)$，远贵于逐点的 E 步。
- 没有 `covariance_type` 参数——HMM 处理离散观测，不涉及协方差矩阵。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `transmat_` | `ndarray`，形状 `(3, 3)` | 转移矩阵 $A$ | $A_{ij} = P(s_{t+1}=j \mid s_t=i)$，行和为 1 |
| `startprob_` | `ndarray`，形状 `(3,)` | 初始分布 $\pi$ | $\pi_i = P(s_1=i)$，和为 1 |
| `emissionprob_` | `ndarray`，形状 `(3, 3)` | 发射矩阵 $B$ | $B_{ij} = P(o_t=j \mid s_t=i)$，行和为 1 |
| `monitor_` | `dict` | 训练历史 | 逐次迭代的对数似然值列表——诊断收敛 |

### 示例代码

```python
print(f"n_components: {n_components}")
print(f"n_iter: {n_iter}")
print(f"tol: {tol}")
print(f"转移矩阵:\n{model.transmat_.round(3)}")
print(f"发射矩阵:\n{model.emissionprob_.round(3)}")
print(f"初始分布: {model.startprob_.round(3)}")
```

### 理解重点

- `transmat_`（$3 \times 3$）描述隐状态的迁移行为——对角线越大（如 0.8），状态越稳定。
- `emissionprob_`（$3 \times 3$）描述每个状态的观测偏好——例如"状态 0 大概率发射符号 0"。
- `startprob_` 描述序列起始时刻的状态分布——与 GMM 的 `weights_` 概念相似，但用于序列初始而非整个序列。
- 通过对比学习到的 `transmat_` / `emissionprob_` 与真实参数，可以定量评估 HMM 的恢复能力。

## 4. `predict()` — Viterbi 解码

### 参数速览

| 方法 | 输入 | 输出 | 算法 | 说明 |
|---|---|---|---|---|
| `predict(X, lengths)` | `(n_steps, 1)` + `lengths` | `ndarray`，`(n_steps,)` | **Viterbi** | 全局最优隐状态路径——保证路径合法（不存在概率为 0 的转移） |

### 理解重点

- Viterbi 解码**不是**逐步 argmax——它全局寻找概率最大的单条路径，保证相邻状态的转移是合法的。
- 逐步 argmax（$\arg\max_i P(s_t = i \mid O, \lambda)$）可能产生"非法"的状态跳跃——Viterbi 避免了这一点。
- 在 HMM 中，`predict` 返回的是隐状态序列——用于与 `state_true` 对比计算准确率。

## 5. HMM vs GMM vs 集成模型 参数对比

| 参数/属性 | GMM | HMM | 备注 |
|---|---|---|---|
| 核心参数 | `n_components`、`covariance_type`、`max_iter` | `n_components`、`n_iter`、`tol` | 相似但 HMM 更关注迭代收敛 |
| 训练输入 | `fit(X)`——独立样本 | **`fit(X, lengths)`——序列数据** | 根本差异 |
| 模型属性 | `means_`、`covariances_`、`weights_` | **`transmat_`、`emissionprob_`、`startprob_`** | HMM 描述动态，GMM 描述分布 |
| 预测输出 | `predict(X)`、`predict_proba(X)` | **`predict(X, lengths)`（Viterbi）** | 无 `predict_proba`——但有 `score`（Forward） |
| 依赖 | sklearn 内置 | **`pip install hmmlearn`** | 可选依赖 |

## 常见坑

1. 忘记传 `lengths` 参数——`fit(X)` 会报错，HMM 必须知道每条序列的边界。
2. 混淆 `transmat_` 的行和列方向——行 $i$ 列 $j$ 表示 $P(s_{t+1}=j \mid s_t=i)$，行和为 1。
3. 在极短序列（<50 步）上训练——Baum-Welch 需要足够多的状态转移来稳定估计 $A$ 矩阵。
4. 混淆 `CategoricalHMM` 和 `MultinomialHMM`——前者是 hmmlearn 0.3+ 的新 API，语义更清晰。

## 小结

- `train_model(...)` 是本仓库 HMM 的核心训练入口——含双备份可选依赖检查，输入为序列数据而非独立样本。
- `CategoricalHMM` 的核心参数是 `n_components`（隐状态数）、`n_iter`（Baum-Welch 上限）、`tol`（收敛阈值）——结构化参数比 GMM 少但序列计算量更大。
- 训练完成后的核心属性：`transmat_` / `emissionprob_` / `startprob_`——三件套完全描述离散 HMM 的动态和观测行为。
