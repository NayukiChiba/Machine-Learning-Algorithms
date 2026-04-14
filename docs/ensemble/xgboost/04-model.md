---
title: XGBoost — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/ensemble/xgboost.py`
>  
> 运行方式：`python -m model_training.ensemble.xgboost`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `XGBRegressor`。
2. 理解当前源码中关键超参数的默认值与作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装和错误处理。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `xgboost.XGBRegressor` 模型 |
| `XGBRegressor(...)` | 类 | XGBoost 提供的回归器实现 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上执行 boosting 训练 |
| `@print_func_info` / `@timeit` / `timer(...)` | 工程包装 | 打印入口信息与训练耗时 |
| `ImportError` 逻辑 | 错误处理 | 当未安装 `xgboost` 时给出明确报错 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.9, colsample_bytree=0.9, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 训练特征 | 输入给 `XGBRegressor.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 连续值目标 |
| `n_estimators` | `300` | boosting 轮数，也可理解为树数量 |
| `learning_rate` | `0.05` | 每轮新树贡献的缩放系数 |
| `max_depth` | `6` | 单棵树最大深度 |
| `min_child_weight` | `1` | 子节点最小样本权重和约束 |
| `subsample` | `0.9` | 行采样比例 |
| `colsample_bytree` | `0.9` | 列采样比例 |
| `gamma` | `0.0` | 分裂所需的最小损失减少 |
| `reg_alpha` | `0.0` | L1 正则化系数 |
| `reg_lambda` | `1.0` | L2 正则化系数 |
| `random_state` | `42` | 随机种子 |
| 返回值 | `XGBRegressor` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.ensemble.xgboost import train_model

model = train_model(X_train, y_train)
```

### 理解重点

- 当前训练入口返回的是单个 XGBoost 回归模型，而不是多模型集合。
- 和线性回归、决策树相比，这里的关键点是超参数更多，而且它们会共同影响 boosting 训练行为。
- 默认参数直接来自源码，是后续调参与阅读日志的基线。

## 2. `XGBRegressor(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `XGBRegressor(...)`
2. `model.fit(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `300` | 叠加 300 棵弱学习器 |
| `learning_rate` | `0.05` | 控制每轮更新步长 |
| `max_depth` | `6` | 控制单棵树复杂度 |
| `n_jobs` | `-1` | 使用全部可用 CPU 核心 |

### 示例代码

```python
model = XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    gamma=gamma,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda,
    random_state=random_state,
    n_jobs=-1,
)

model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有自己实现 boosting 过程，而是直接调用 XGBoost 官方库提供的 `XGBRegressor`。
- 当前文档的重点，不是重写内部算法，而是理解这组高层参数如何约束训练过程。
- `n_jobs=-1` 体现了当前实现偏工程化的一面：默认尽可能利用并行资源。

## 3. 三组最关键超参数分别控制什么

### 参数速览（本节）

适用分组（本节）：

1. boosting 强度参数
2. 树复杂度参数
3. 采样与正则化参数

| 分组 | 代表参数 | 当前作用 |
|---|---|---|
| boosting 强度 | `n_estimators`、`learning_rate` | 控制累计学习能力与更新步长 |
| 树复杂度 | `max_depth`、`min_child_weight`、`gamma` | 控制单棵树能切多细、多复杂 |
| 采样与正则化 | `subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda` | 控制随机性和复杂度约束 |

### 理解重点

- `n_estimators` 和 `learning_rate` 通常需要配合看，不能只盯一个参数。
- `max_depth`、`min_child_weight`、`gamma` 更偏单棵树层面的复杂度约束。
- `subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda` 则更偏泛化控制和工程稳定性。

## 4. 训练阶段的工程封装

除了 `XGBRegressor(...).fit(...)` 之外，`train_model(...)` 还做了多层工程包装。

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

### 理解重点

- 当前训练函数会同时打印“模型训练耗时”和整个函数耗时，因此终端里会看到两层计时信息。
- 这些包装不改变 XGBoost 训练行为，但有助于观察训练入口和耗时表现。
- 和线性回归、决策树相比，这里更强调训练配置回显和工程执行信息。

## 5. 依赖缺失时会发生什么

### 参数速览（本节）

适用逻辑：`ImportError` 处理

| 情况 | 当前行为 |
|---|---|
| 已安装 `xgboost` | 正常构建 `XGBRegressor` |
| 未安装 `xgboost` | 抛出明确的 `ImportError` |

### 示例代码

```python
if XGBRegressor is None:
    raise ImportError("未安装 xgboost，请先安装后再运行该模块。") from _IMPORT_ERROR
```

### 理解重点

- 当前训练代码不是默认假设依赖一定存在，而是显式处理了缺包场景。
- 这说明分册文档必须如实写清依赖边界。
- 但也要注意，当前源码只负责报错提示，并没有内置安装流程。

## 常见坑

1. 只调 `n_estimators`，忽略 `learning_rate`、`gamma`、`reg_alpha`、`reg_lambda` 等参数也在共同影响结果。
2. 把 XGBoost 当成“参数更多的单棵树”，忽略它本质是 boosting 集成模型。
3. 忽略依赖包边界，没安装 `xgboost` 就直接运行训练模块。

## 小结

- `train_model(...)` 是本仓库 XGBoost 的核心训练入口。
- 它本质上是对 `xgboost.XGBRegressor` 的薄封装，重点在于超参数传递、耗时统计、日志输出和依赖边界处理。
- 读懂这一层之后，再看流水线中的训练、预测和评估过程会更清晰。
