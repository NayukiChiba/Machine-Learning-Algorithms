---
title: GBDT — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/ensemble/gbdt.py`
>  
> 运行方式：`python -m model_training.ensemble.gbdt`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `GradientBoostingClassifier`。
2. 理解当前源码中关键超参数的默认值与作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.ensemble.GradientBoostingClassifier` 模型 |
| `GradientBoostingClassifier(...)` | 类 | scikit-learn 提供的 GBDT 分类器 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上执行 boosting 训练 |
| `@print_func_info` / `@timeit` / `timer(...)` | 工程包装 | 打印入口信息与训练耗时 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的训练特征 | 输入给 `GradientBoostingClassifier.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 多分类目标 |
| `n_estimators` | `200` | boosting 轮数 |
| `learning_rate` | `0.1` | 每轮新树贡献的缩放系数 |
| `max_depth` | `3` | 基学习器树的最大深度 |
| `subsample` | `1.0` | 每轮采样比例 |
| `random_state` | `42` | 随机种子 |
| 返回值 | `GradientBoostingClassifier` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.ensemble.gbdt import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口返回的是单个 GBDT 分类模型，而不是多模型集合。
- 和 XGBoost、LightGBM 相比，这里的参数集合更少，也更适合作为 boosting 思想的基础入口。
- 默认参数直接来自源码，是后续理解 boosting 配置和分类评估的基线。

## 2. `GradientBoostingClassifier(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `GradientBoostingClassifier(...)`
2. `model.fit(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `200` | 叠加 200 轮弱学习器 |
| `learning_rate` | `0.1` | 控制每轮更新步长 |
| `max_depth` | `3` | 控制基学习器复杂度 |
| `subsample` | `1.0` | 当前默认不做样本子采样 |

### 示例代码

```python
model = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample,
    random_state=random_state,
)

model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有自己实现 boosting 过程，而是直接调用 scikit-learn 的现成实现。
- 当前文档的重点，不是重写内部算法，而是理解这组高层参数如何约束串行残差拟合过程。
- 与 XGBoost/LightGBM 相比，这里少了很多高级正则化和工程优化参数，因此更适合强调 boosting 核心思想本身。

## 3. 四个核心超参数分别控制什么

### 参数速览（本节）

适用超参数（分项）：

1. `n_estimators`
2. `learning_rate`
3. `max_depth`
4. `subsample`

| 超参数 | 当前作用 | 调整时的常见影响 |
|---|---|---|
| `n_estimators` | 控制 boosting 轮数 | 更大通常更强，但也更易过拟合 |
| `learning_rate` | 控制每轮修正步长 | 更小通常更稳，但可能需要更多轮 |
| `max_depth` | 控制基学习器复杂度 | 更深更强，也更易过拟合 |
| `subsample` | 控制每轮采样比例 | 小于 1 时更接近随机梯度提升 |

### 理解重点

- `n_estimators` 和 `learning_rate` 通常需要配合看，不能只盯一个参数。
- `max_depth` 决定每棵树有多复杂，会直接影响边界拟合能力。
- `subsample=1.0` 说明当前默认配置是全样本 boosting，而不是带随机采样的变体。

## 4. 训练阶段的工程封装

除了 `GradientBoostingClassifier(...).fit(...)` 之外，`train_model(...)` 还做了多层工程包装。

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
- 这些包装不改变 GBDT 训练行为，但有助于观察训练入口和耗时表现。
- 和线性回归、决策树相比，这里更强调 boosting 配置回显和工程执行信息。

## 5. 当前实现没有哪些复杂依赖或高级逻辑

### 理解重点

- 当前 GBDT 训练直接基于 scikit-learn，自身没有额外依赖第三方 boosting 库。
- 这与 XGBoost 和 LightGBM 分册不同，少了外部库导入失败时的额外错误处理逻辑。
- 也正因为如此，当前 GBDT 分册更适合作为理解 boosting 分类基础的过渡层。

## 常见坑

1. 只调 `n_estimators`，忽略 `learning_rate`、`max_depth`、`subsample` 也会显著影响结果。
2. 把 GBDT 当成“参数更少的 XGBoost”，忽略它的数学主轴其实是伪残差拟合。\n+3. 忘记当前模型是分类器，误把后续评估写成回归指标或残差分析。

## 小结

- `train_model(...)` 是本仓库 GBDT 的核心训练入口。
- 它本质上是对 `sklearn.ensemble.GradientBoostingClassifier` 的薄封装，重点在于超参数传递、耗时统计和日志输出。
- 读懂这一层之后，再看流水线中的训练、预测和评估过程会更清晰。
