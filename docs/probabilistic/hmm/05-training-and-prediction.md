---
title: HMM — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/probabilistic/hmm.py`、`model_training/probabilistic/hmm.py`
>  
> 运行方式：`python -m pipelines.probabilistic.hmm`

## 本章目标

1. 明确当前流水线从取数到输出隐状态预测结果的完整执行顺序。
2. 理解训练阶段、解码阶段和控制台评估分别由哪个函数负责。
3. 明确当前 HMM 实现没有 train/test split，而是直接在整条序列上训练和预测。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | HMM 端到端流水线入口 |
| `train_model(...)` | 函数 | 训练离散 HMM 模型 |
| `model.predict(X_obs, lengths)` | 方法 | 解码得到隐状态序列 |
| `model.transmat_` | 属性 | 输出学习到的状态转移矩阵 |
| `accuracy = np.mean(states_pred == y_true)` | 派生量 | 当前流水线的对比评估方式 |

## 1. 端到端入口 `run()`

### 参数速览（本节）

适用函数：`run()`

| 项目 | 当前实现 |
|---|---|
| 数据源 | `hmm_data.copy()` |
| 观测列 | `obs` |
| 对比隐状态列 | `state_true` |
| 训练入口 | `train_model(X_obs, lengths)` |
| 解码入口 | `model.predict(X_obs, lengths)` |
| 评估输出 | 控制台准确率 + 转移矩阵 |

### 示例代码

```python
def run():
    data = hmm_data.copy()
    obs = data["obs"].values.astype(int)
    X_obs = obs.reshape(-1, 1)
    lengths = [len(obs)]
    y_true = data["state_true"].values.astype(int)
```

### 理解重点

- 整个分册的运行入口就是 `pipelines/probabilistic/hmm.py` 里的 `run()`。
- 这个函数不负责实现前向、后向或 Baum-Welch 细节，而是把序列整理、训练、解码和输出串成一条完整链路。
- 当前实现重点是“整条序列上的隐状态建模”，而不是单点分类流程。

## 2. 训练前的数据准备顺序

### 参数速览（本节）

适用流程（分项）：

1. 取出观测序列
2. reshape 成二维数组
3. 构造 `lengths`
4. 取出真实隐状态作为对比列

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `obs` | 一维整数数组 | 原始观测符号序列 |
| `X_obs` | `obs.reshape(-1, 1)` | 符合 hmmlearn 输入格式 |
| `lengths` | `[len(obs)]` | 当前只有一条完整序列 |
| `y_true` | `state_true` 数组 | 仅用于训练后对比 |

### 示例代码

```python
obs = data["obs"].values.astype(int)
X_obs = obs.reshape(-1, 1)
lengths = [len(obs)]
y_true = data["state_true"].values.astype(int)
```

### 理解重点

- 当前 HMM 输入不是普通二维特征矩阵，而是经过 reshape 后的离散观测序列。
- `lengths` 在这里非常关键，因为它告诉模型输入由几条序列拼接而成。
- 当前实现没有标准化，也没有 train/test split，这些都和离散序列建模特点有关。

## 3. 训练阶段：调用 `train_model(...)`

### 参数速览（本节）

适用函数：`train_model(X_obs, lengths)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_obs` | `(T, 1)` 观测数组 | 当前直接传入 HMM 训练函数 |
| `lengths` | `[T]` | 当前序列长度列表 |
| 返回值 | `model` | 已训练好的 HMM 模型 |

### 示例代码

```python
model = train_model(X_obs, lengths)
```

### 理解重点

- 当前实现没有手写 Baum-Welch 迭代，而是把训练交给 hmmlearn 内部 `fit(...)` 完成。
- 训练阶段最重要的副产物，不只是 `model` 对象，还有控制台中的 `n_components`、`n_iter`、`tol` 和耗时信息。
- 这些日志更接近“训练设定与收敛过程”，而不是最终序列预测质量本身。

## 4. 预测阶段：解码隐状态序列

### 参数速览（本节）

适用流程（分项）：

1. `states_pred = model.predict(X_obs, lengths)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | 已训练完成 HMM 模型 | 来自 `train_model(...)` 返回值 |
| `X_obs` | 离散观测序列 | 与训练时相同的输入结构 |
| `lengths` | `[len(obs)]` | 保持与训练时一致的序列描述 |
| `states_pred` | 预测隐状态数组 | 用于与 `state_true` 做对比 |

### 示例代码

```python
states_pred = model.predict(X_obs, lengths)
```

### 理解重点

- 当前流水线的预测阶段，本质上是在做解码：给定观测序列，推断最可能的隐状态路径。
- 这一步与普通分类器的 `predict(...)` 表面上相似，但背后其实更接近 HMM 的 Viterbi 解码逻辑。
- 因此这里的“预测”应理解为“隐状态路径推断”，而不是普通标签分类。

## 5. 预测后的控制台评估输出

### 参数速览（本节）

适用输出项（分项）：

1. `accuracy = np.mean(states_pred == y_true)`
2. `model.transmat_`

| 输出项 | 当前作用 |
|---|---|
| `accuracy` | 粗略比较预测隐状态和真实隐状态是否一致 |
| `transmat_` | 查看学习到的状态转移结构 |

### 示例代码

```python
accuracy = np.mean(states_pred == y_true)
print(f"\n隐状态预测准确率: {accuracy:.4f}")
print(f"转移矩阵:\n{model.transmat_.round(3)}")
```

### 理解重点

- 当前实现最直接的评估方式，是比较解码得到的隐状态路径和数据生成时的 `state_true`。
- 同时还会打印学习得到的转移矩阵，帮助观察状态演化结构是否合理。
- 这两项输出构成了当前 HMM 分册最核心的工程结果。

## 6. 用伪代码看完整流程

### 示例代码

```python
data = hmm_data.copy()
obs = data["obs"].values.astype(int)
X_obs = obs.reshape(-1, 1)
lengths = [len(obs)]
y_true = data["state_true"].values.astype(int)

model = train_model(X_obs, lengths)
states_pred = model.predict(X_obs, lengths)

accuracy = np.mean(states_pred == y_true)
print(model.transmat_)
```

### 理解重点

- 当前 HMM 流水线的主线非常清楚：取数、整理序列、训练、解码、打印准确率和转移矩阵。
- 这条链路里最关键的中间变量是 `X_obs`、`lengths`、训练后的 `model` 和预测隐状态 `states_pred`。
- 只要把这条流程走清楚，整个 HMM 分册的工程部分就基本串起来了。

## 常见坑

1. 把监督学习里的 train/test split 惯性套进当前 HMM 流程，和源码不一致。
2. 忘记把一维观测序列 reshape 成 `(T, 1)`，导致输入格式不匹配。
3. 把 `predict(...)` 当成普通分类预测，而不是隐状态解码过程。

## 小结

- 当前流水线把序列整理、HMM 训练、隐状态解码和控制台结果输出串成了一条完整路径。
- 训练函数负责“得到 HMM 模型”，流水线函数负责“组织执行和展示结果”。
- 把这一层执行顺序读清楚，后续看评估与工程实现章节就会更顺。
