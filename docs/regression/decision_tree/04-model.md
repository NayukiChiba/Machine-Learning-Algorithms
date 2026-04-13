---
title: 决策树回归 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/regression/decision_tree.py`
>  
> 运行方式：`python -m model_training.regression.decision_tree`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `DecisionTreeRegressor`。
2. 理解各个超参数在当前源码中的默认值与作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装和日志输出。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.tree.DecisionTreeRegressor` 模型 |
| `DecisionTreeRegressor(...)` | 类 | scikit-learn 提供的决策树回归器 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上生长决策树 |
| `model.get_depth()` | 方法 | 返回当前树的最大深度 |
| `model.get_n_leaves()` | 方法 | 返回当前树的叶子节点数量 |
| `@print_func_info` / `@timeit` / `timer(...)` | 工程包装 | 打印入口信息与训练耗时 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, max_depth=6, min_samples_split=6, min_samples_leaf=3, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 训练特征数组 | 输入给 `DecisionTreeRegressor.fit(...)` 的训练特征 |
| `y_train` | 训练标签数组 | 每个样本对应的连续值目标 |
| `max_depth` | `6` | 限制树的最大深度 |
| `min_samples_split` | `6` | 一个节点继续划分所需的最小样本数 |
| `min_samples_leaf` | `3` | 叶子节点最少样本数 |
| `random_state` | `42` | 随机种子，保证可复现 |
| 返回值 | `DecisionTreeRegressor` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.regression.decision_tree import train_model

model = train_model(X_train.values, y_train.values)
```

### 理解重点

- 当前训练入口返回的是单个决策树模型，而不是多模型集合。
- `X_train` 和 `y_train` 在当前流水线里被显式转成了 NumPy 数组，这是实现细节，不是算法本身的硬性要求。
- 默认参数直接来自源码，是后续理解模型复杂度控制的基线。

## 2. `DecisionTreeRegressor(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `DecisionTreeRegressor(...)`
2. `model.fit(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `max_depth` | `6` | 控制树最多向下分裂多少层 |
| `min_samples_split` | `6` | 控制节点何时还能继续分裂 |
| `min_samples_leaf` | `3` | 避免叶子节点样本过少 |
| `random_state` | `42` | 保证分裂过程可复现 |

### 示例代码

```python
model = DecisionTreeRegressor(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=random_state,
)
model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有自己实现树的分裂搜索，而是直接调用 scikit-learn 的现成实现。
- 当前文档的重点，不是重新推导内部算法，而是理解这些超参数如何限制树的复杂度。
- 这些参数一起决定树是偏浅、偏稳，还是更深、更灵活。

## 3. 三个复杂度超参数分别控制什么

### 参数速览（本节）

适用超参数（分项）：

1. `max_depth`
2. `min_samples_split`
3. `min_samples_leaf`

| 超参数 | 当前作用 | 调大/调小的常见影响 |
|---|---|---|
| `max_depth` | 限制树的层数 | 更大更灵活，也更易过拟合 |
| `min_samples_split` | 限制是否继续分裂 | 更大更保守，树通常更浅 |
| `min_samples_leaf` | 限制叶子最少样本数 | 更大更平滑，叶子更少 |

### 理解重点

- `max_depth` 最直观，直接控制树能长多深。
- `min_samples_split` 和 `min_samples_leaf` 则更像局部约束，用来防止某些节点切得过细。
- 当前默认配置的目标，是在真实数据上给出一个相对容易观察、又不过分复杂的基础树模型。

## 4. 训练阶段的工程封装

除了 `DecisionTreeRegressor(...).fit(...)` 之外，`train_model(...)` 还做了多层工程包装。

### 参数速览（本节）

适用装饰与上下文（分项）：

1. `@print_func_info`
2. `@timeit`
3. `with timer(name="模型训练耗时")`

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
        model.fit(X_train, y_train)
```

### 理解重点

- 当前训练函数会同时打印“模型训练耗时”和“函数 train_model 耗时”，所以终端里会看到两层计时信息。
- 这类包装不改变模型行为，但有助于观察训练入口和耗时表现。
- 与线性回归分册相比，这里更强调结构复杂度和训练耗时，而不是系数解释。

## 5. 训练完成后最直接可用的结构信息

### 参数速览（本节）

适用属性/方法（分项）：

1. `model.get_depth()`
2. `model.get_n_leaves()`

| 返回值 | 含义 |
|---|---|
| `get_depth()` | 当前树的最大深度 |
| `get_n_leaves()` | 当前树的叶子节点总数 |

### 示例代码

```python
print(f"树深度: {model.get_depth()}")
print(f"叶子节点数: {model.get_n_leaves()}")
```

### 理解重点

- 这两个量是当前决策树分册里训练后最先被观察的结构性指标。
- 深度和叶子节点数越大，通常说明模型越复杂。
- 后续评估章节里对过拟合和欠拟合的讨论，也会和这两个量联系起来看。

## 常见坑

1. 只记住 `max_depth`，忽略 `min_samples_split` 和 `min_samples_leaf` 也在共同控制复杂度。
2. 把训练日志里的“树深度”和“叶子节点数”当成无关紧要的附加信息。
3. 误以为装饰器和计时上下文改变了模型训练逻辑，实际上它们只负责日志输出。

## 小结

- `train_model(...)` 是本仓库决策树回归的核心训练入口。
- 它本质上是对 `sklearn.tree.DecisionTreeRegressor` 的薄封装，重点在于超参数传递、耗时统计和结构信息打印。
- 读懂这一层之后，再看流水线中的训练、预测和评估过程会更清晰。
