---
title: LightGBM — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/ensemble/lightgbm.py`
>  
> 运行方式：`python -m model_training.ensemble.lightgbm`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LGBMClassifier`。
2. 理解当前源码中关键超参数的默认值与作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装和错误处理。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `lightgbm.LGBMClassifier` 模型 |
| `LGBMClassifier(...)` | 类 | LightGBM 提供的分类器实现 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上执行 boosting 训练 |
| `@print_func_info` / `@timeit` / `timer(...)` | 工程包装 | 打印入口信息与训练耗时 |
| `ImportError` 逻辑 | 错误处理 | 当未安装 `lightgbm` 时给出明确报错 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的训练特征 | 输入给 `LGBMClassifier.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 多分类目标 |
| `n_estimators` | `300` | boosting 轮数 |
| `learning_rate` | `0.05` | 每轮新树贡献的缩放系数 |
| `num_leaves` | `31` | 单棵树最多叶子数 |
| `max_depth` | `-1` | 最大深度，`-1` 表示不限制 |
| `subsample` | `0.9` | 行采样比例 |
| `colsample_bytree` | `0.9` | 列采样比例 |
| `random_state` | `42` | 随机种子 |
| 返回值 | `LGBMClassifier` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.ensemble.lightgbm import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口返回的是单个 LightGBM 分类模型，而不是多模型集合。
- 和 XGBoost 分册相比，这里目标是分类，因此后续会配合 `predict_proba(...)` 输出概率。
- 默认参数直接来自源码，是后续理解 boosting 配置和分类评估的基线。

## 2. `LGBMClassifier(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `LGBMClassifier(...)`
2. `model.fit(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `300` | 叠加 300 轮弱学习器 |
| `learning_rate` | `0.05` | 控制每轮更新步长 |
| `num_leaves` | `31` | 控制 Leaf-wise 树结构的复杂度 |
| `max_depth` | `-1` | 当前默认不限制深度 |
| `n_jobs` | `-1` | 使用全部可用 CPU 核心 |

### 示例代码

```python
model = LGBMClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    num_leaves=num_leaves,
    max_depth=max_depth,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    random_state=random_state,
    n_jobs=-1,
)

model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有自己实现 LightGBM 的 Leaf-wise 生长、直方图算法或 GOSS / EFB，而是直接调用官方库实现。
- 当前文档的重点，不是重写内部算法，而是理解这些高层参数如何约束训练过程。
- `num_leaves` 是当前 LightGBM 分册最有代表性的结构控制参数之一。

## 3. 三组最关键超参数分别控制什么

### 参数速览（本节）

适用分组（本节）：

1. boosting 强度参数
2. 树结构参数
3. 采样参数

| 分组 | 代表参数 | 当前作用 |
|---|---|---|
| boosting 强度 | `n_estimators`、`learning_rate` | 控制累计学习能力与更新步长 |
| 树结构 | `num_leaves`、`max_depth` | 控制 Leaf-wise 生长的复杂度 |
| 采样 | `subsample`、`colsample_bytree` | 控制样本和特征采样比例 |

### 理解重点

- `n_estimators` 和 `learning_rate` 通常需要配合看，不能只盯一个参数。
- `num_leaves` 和 `max_depth` 一起决定单棵树有多复杂，尤其影响 Leaf-wise 生长是否过于激进。
- `subsample` 和 `colsample_bytree` 则更多影响泛化与稳定性。

## 4. 训练阶段的工程封装

除了 `LGBMClassifier(...).fit(...)` 之外，`train_model(...)` 还做了多层工程包装。

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
- 这些包装不改变 LightGBM 训练行为，但有助于观察训练入口和耗时表现。
- 和线性回归、决策树相比，这里更强调训练配置回显和工程执行信息。

## 5. 依赖缺失时会发生什么

### 参数速览（本节）

适用逻辑：`ImportError` 处理

| 情况 | 当前行为 |
|---|---|
| 已安装 `lightgbm` | 正常构建 `LGBMClassifier` |
| 未安装 `lightgbm` | 抛出明确的 `ImportError` |

### 示例代码

```python
if LGBMClassifier is None:
    raise ImportError(
        "未安装 lightgbm，请先安装后再运行该模块。"
    ) from _IMPORT_ERROR
```

### 理解重点

- 当前训练代码不是默认假设依赖一定存在，而是显式处理了缺包场景。
- 这说明分册文档必须如实写清依赖边界。
- 但也要注意，当前源码只负责报错提示，并没有内置安装流程。

## 常见坑

1. 只调 `n_estimators`，忽略 `learning_rate`、`num_leaves`、`max_depth` 也在共同影响模型行为。
2. 把 `num_leaves` 和 `max_depth` 完全独立看待，忽略它们共同控制 Leaf-wise 树复杂度。
3. 忽略依赖包边界，没安装 `lightgbm` 就直接运行训练模块。

## 小结

- `train_model(...)` 是本仓库 LightGBM 的核心训练入口。
- 它本质上是对 `lightgbm.LGBMClassifier` 的薄封装，重点在于超参数传递、耗时统计、日志输出和依赖边界处理。
- 读懂这一层之后，再看流水线中的训练、预测和评估过程会更清晰。
