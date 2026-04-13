---
title: HMM — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/probabilistic/hmm.py`
>  
> 运行方式：`python -m model_training.probabilistic.hmm`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练当前仓库中的 HMM 模型。
2. 理解默认 `n_components`、`n_iter`、`tol`、`random_state` 在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些兼容处理和日志输出。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个离散 HMM 模型 |
| `CategoricalHMM` | 类 | hmmlearn 提供的离散观测 HMM 实现 |
| `MultinomialHMM` | 类 | 当前环境下的回退实现 |
| `model.fit(X_obs, lengths)` | 方法 | 在离散观测序列上学习 HMM 参数 |
| `model.transmat_` | 属性 | 训练后得到的状态转移矩阵 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_obs, lengths, n_components=3, n_iter=100, tol=1e-3, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_obs` | `(T, 1)` 观测数组 | 输入给 HMM 的离散观测序列 |
| `lengths` | `[len(obs)]` | 序列长度列表 |
| `n_components` | `3` | 隐状态数 |
| `n_iter` | `100` | 最大迭代次数 |
| `tol` | `1e-3` | 收敛阈值 |
| `random_state` | `42` | 随机种子，保证可复现 |
| 返回值 | HMM 模型对象 | 已训练完成的隐马尔可夫模型 |

### 示例代码

```python
from model_training.probabilistic.hmm import train_model

model = train_model(X_obs, lengths)
```

### 理解重点

- 当前训练入口返回的是单个 HMM 模型对象。
- 与 EM 分册一样，这里训练时不需要真实标签，只需要观测序列和长度信息。
- 默认参数直接来自源码，是后续理解状态数和训练迭代控制的基线。

## 2. `CategoricalHMM` 与 `MultinomialHMM` 的兼容逻辑

### 参数速览（本节）

适用逻辑（分项）：

1. 优先使用 `CategoricalHMM`
2. 回退使用 `MultinomialHMM`

| 分支 | 当前行为 | 说明 |
|---|---|---|
| `CategoricalHMM` 可用 | 优先构建它 | 更直接对应离散符号观测 |
| 否则 `MultinomialHMM` 可用 | 使用回退实现 | 保证在不同 hmmlearn 版本下仍可运行 |
| 都不可用 | 抛出 `ImportError` | 提示未安装 `hmmlearn` |

### 示例代码

```python
if CategoricalHMM is not None:
    model = CategoricalHMM(...)
else:
    model = MultinomialHMM(...)
```

### 理解重点

- 当前源码不是只写死一种 HMM 类，而是做了 hmmlearn 版本兼容处理。
- 这说明文档必须以“当前实现会优先用 `CategoricalHMM`，否则回退到 `MultinomialHMM`”来描述。
- 这是工程层的重要细节，不属于纯数学层内容。

## 3. 四个核心超参数分别控制什么

### 参数速览（本节）

适用超参数（分项）：

1. `n_components`
2. `n_iter`
3. `tol`
4. `random_state`

| 超参数 | 当前作用 | 调整时的常见影响 |
|---|---|---|
| `n_components` | 假设隐状态数量 | 设少了会混状态，设多了会拆得过细 |
| `n_iter` | 限制训练迭代上限 | 太小可能提前停止 |
| `tol` | 控制收敛阈值 | 更小通常要求更严格收敛 |
| `random_state` | 控制初始化随机性 | 影响可复现性与局部解路径 |

### 理解重点

- `n_components` 是当前 HMM 分册里最重要的建模假设之一。
- `n_iter` 和 `tol` 共同控制 Baum-Welch 训练何时停止。
- `random_state` 则保证当前实验结果更容易复现和对比。

## 4. 训练阶段的工程封装

除了 `model.fit(X_obs, lengths)` 之外，`train_model(...)` 还做了多层工程包装。

### 参数速览（本节）

适用装饰与上下文（分项）：

1. `@print_func_info`
2. `@timeit`
3. `with timer(name='模型训练耗时')`

| 包装项 | 作用 |
|---|---|
| `@print_func_info` | 打印函数调用入口 |
| `@timeit` | 打印整个函数耗时 |
| `timer(...)` | 打印模型 `fit(...)` 阶段耗时 |

### 示例代码

```python
@print_func_info
@timeit
def train_model(...):
    ...
    with timer(name="模型训练耗时"):
        model.fit(X_obs, lengths)
```

### 理解重点

- 当前训练函数会同时打印“模型训练耗时”和整个函数耗时，因此终端里会看到两层计时信息。
- 这些包装不改变 HMM 训练行为，但有助于观察训练入口和耗时表现。
- 与回归分册不同，这里更强调序列建模训练是否顺利完成，而不是系数或残差输出。

## 5. 训练完成后最直接可用的结构信息

### 参数速览（本节）

适用属性/方法（分项）：

1. `model.predict(X_obs, lengths)`
2. `model.transmat_`

| 输出项 | 含义 |
|---|---|
| `predict(...)` | 解码得到的隐状态序列 |
| `transmat_` | 学到的状态转移矩阵 |

### 理解重点

- 对当前 HMM 分册来说，训练后最重要的结果不是一个分数，而是状态路径和转移结构。
- `transmat_` 能帮助你观察模型学到了怎样的状态演化规律。
- `predict(...)` 则直接连接到流水线中的隐状态准确率计算。

## 常见坑

1. 误以为当前实现只支持 `CategoricalHMM`，忽略了回退到 `MultinomialHMM` 的兼容逻辑。
2. 只关注 `n_components`，忽略 `n_iter` 和 `tol` 对训练停止条件的影响。
3. 把训练后的重点仍放在“分数”上，而忽略 `transmat_` 和隐状态路径才是当前分册最关键的结果对象。

## 小结

- `train_model(...)` 是本仓库 HMM 的核心训练入口。
- 它本质上是对 hmmlearn 中离散 HMM 实现的薄封装，重点在于超参数传递、兼容处理和训练日志输出。
- 读懂这一层之后，再看流水线中的训练、预测和隐状态评估过程会更清晰。
