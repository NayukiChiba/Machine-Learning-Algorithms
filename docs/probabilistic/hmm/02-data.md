---
title: HMM — 数据构成
outline: deep
---

# 数据构成

> 对应代码：`data_generation/probabilistic.py`、`data_generation/__init__.py`、`pipelines/probabilistic/hmm.py`
>  
> 相关对象：`ProbabilisticData.hmm()`、`hmm_data`

## 本章目标

1. 明确本仓库 HMM 数据来自 `ProbabilisticData.hmm()` 的离散序列构造逻辑。
2. 明确观测序列 `obs`、真实隐状态 `state_true` 和时间步 `time` 的边界。
3. 明确当前实现如何整理序列输入，以及当前流程没有 train/test split 这一事实。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.hmm()` | 方法 | 生成 HMM 使用的离散观测序列数据 |
| `hmm_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `obs` | 列名 | 训练 HMM 的离散观测符号序列 |
| `state_true` | 列名 | 数据生成时记录的真实隐状态，仅用于训练后对比 |
| `time` | 列名 | 时间步索引 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `hmm_data`
- 生成来源：`data_generation/probabilistic.py` 中的 `ProbabilisticData.hmm()`
- 流水线使用：`pipelines/probabilistic/hmm.py` 中的 `data = hmm_data.copy()`

### 理解重点

- `hmm_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 使用 `.copy()` 的目的，是避免后续整理观测序列或调试过程意外修改原始数据对象。

## 2. 数据生成函数 `ProbabilisticData.hmm()`

### 参数速览（本节）

适用参数（本节）：

1. `hmm_n_steps`
2. `hmm_pi`
3. `hmm_A`
4. `hmm_B`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `hmm_n_steps` | `300` | 序列长度 |
| `random_state` | `42` | 随机种子，保证可复现 |
| `hmm_pi` | `[0.6, 0.3, 0.1]` | 初始状态分布 |
| `hmm_A` | `3 x 3` 矩阵 | 状态转移矩阵 |
| `hmm_B` | `3 x 3` 矩阵 | 发射矩阵，控制观测符号生成 |
| 返回值 | `DataFrame` | 含 `time`、`obs`、`state_true` 的数据表 |

### 示例代码

```python
states[0] = rng.choice(n_states, p=pi)
obs[0] = rng.choice(B.shape[1], p=B[states[0]])

for t in range(1, n_steps):
    states[t] = rng.choice(n_states, p=A[states[t - 1]])
    obs[t] = rng.choice(B.shape[1], p=B[states[t]])
```

### 理解重点

- 当前数据不是独立样本集合，而是一条按时间顺序生成的离散序列。
- 每个时间步先按转移矩阵产生新的隐状态，再按发射矩阵产生观测符号。
- 这正是 HMM 和普通聚类/回归数据最大的区别。

## 3. 三列数据各自代表什么

当前数据表结构如下：

- `time`：时间步编号
- `obs`：观测符号序列
- `state_true`：真实隐状态序列

### 参数速览（本节）

适用列组（本节）：

1. 时间列
2. 观测列
3. 隐状态列

| 列名 | 当前作用 |
|---|---|
| `time` | 标记序列中的位置 |
| `obs` | 真正参与训练的离散观测输入 |
| `state_true` | 仅用于训练后对比预测隐状态 |

### 示例代码

```python
obs = data["obs"].values.astype(int)
X_obs = obs.reshape(-1, 1)
lengths = [len(obs)]
y_true = data["state_true"].values.astype(int)
```

### 理解重点

- `obs` 是当前流水线真正用于训练 HMM 的输入。
- `state_true` 不是训练标签，而是数据生成时记录下来的参考隐状态，当前实现只在训练后用来算准确率。
- `time` 在当前流水线里没有直接送进模型，但它帮助我们理解这是一个有序序列而不是普通样本表。

## 4. 为什么需要 `reshape(-1, 1)` 和 `lengths`

### 参数速览（本节）

适用代码（分项）：

1. `obs.reshape(-1, 1)`
2. `lengths = [len(obs)]`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_obs` | `(T, 1)` 形状数组 | 符合 hmmlearn 离散观测输入要求 |
| `lengths` | `[300]` | 表示当前只有一条长度为 300 的序列 |

### 示例代码

```python
obs = data["obs"].values.astype(int)
X_obs = obs.reshape(-1, 1)
lengths = [len(obs)]
```

### 理解重点

- 当前 HMM 训练接口接收的是二维观测数组，因此一维离散观测序列需要 reshape 成 `(T, 1)`。
- `lengths` 用来告诉模型当前输入由几条序列拼接而成；当前实现里只有一条完整序列，所以它是 `[len(obs)]`。
- 这是一类序列模型特有的数据整理步骤，和前面几个分册的表格型输入完全不同。

## 5. 当前流程边界

### 参数速览（本节）

当前实现的关键边界：

1. 无 train/test split
2. 无标准化
3. 单条离散序列

| 项目 | 当前状态 |
|---|---|
| train/test split | 未使用 |
| 标准化 | 未使用 |
| 多条序列拼接训练 | 当前未展示 |

### 理解重点

- 当前 HMM 分册没有 train/test split，而是直接在整条观测序列上训练和预测。
- 当前数据是离散观测符号，不像连续特征那样需要标准化。
- 文档必须如实描述当前实现，不能把监督学习或连续特征处理习惯误套到这里。

## 常见坑

1. 把 `state_true` 当成训练标签，误以为当前 HMM 是监督学习。
2. 忽略 `reshape(-1, 1)` 和 `lengths`，把一维序列直接传给训练函数。
3. 把表格型样本思维带进来，忘记当前数据本质上是一条时间序列。

## 小结

- 当前 HMM 数据来自 `ProbabilisticData.hmm()`，底层是手工参数化生成的单条离散观测序列。
- 数据表结构清晰：`obs` 是训练输入，`state_true` 只用于训练后对比，`time` 用于表示顺序。
- 读懂数据来源、序列整理方式和标签边界，是理解后续 HMM 训练与隐状态预测的前提。
