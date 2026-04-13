---
title: 线性回归 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/regression/linear_regression.py`
>  
> 运行方式：`python -m model_training.regression.linear_regression`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LinearRegression`。
2. 理解 `coef_`、`intercept_` 和 `feature_names` 在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.linear_model.LinearRegression` 模型 |
| `LinearRegression()` | 类 | scikit-learn 提供的线性回归器 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上拟合模型 |
| `model.coef_` | 属性 | 返回各特征对应的线性系数 |
| `model.intercept_` | 属性 | 返回模型截距 |
| `@print_func_info` | 装饰器 | 打印函数调用信息，方便观察训练入口 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, feature_names=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 训练特征 | 输入给 `LinearRegression.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 每个样本对应的连续值目标 |
| `feature_names` | `None` 或特征名列表 | 用于打印各特征对应系数 |
| 返回值 | `LinearRegression` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.regression.linear_regression import train_model

model = train_model(X_train, y_train)
```

### 理解重点

- 当前训练入口非常直接，只负责训练一个 `LinearRegression` 模型。
- 与 `regularization` 分册不同，这里没有模型字典，也没有多模型并行比较。
- `feature_names` 不是模型训练必须参数，但能显著提升日志可读性。

## 2. `LinearRegression()` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `LinearRegression()`
2. `model.fit(X_train, y_train)`

| 项目 | 当前实现 | 说明 |
|---|---|---|
| 训练模型 | `LinearRegression()` | 使用 scikit-learn 默认配置 |
| 输入特征 | `X_train` | 当前流水线直接传入未标准化特征 |
| 输入标签 | `y_train` | 连续值目标 |
| 返回值 | 已训练模型 | 含 `coef_` 与 `intercept_` |

### 示例代码

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有重新实现最小二乘求解过程，而是直接调用 scikit-learn 的现成实现。
- 当前这层封装非常薄，重点不在复杂超参数，而在于把训练结果打印清楚。
- 因为数据关系透明，所以默认配置已经足以支撑教学展示。

## 3. `intercept_` 与 `coef_` 的含义

### 参数速览（本节）

适用属性/方法（分项）：

1. `model.intercept_`
2. `model.coef_`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `intercept_` | 浮点数 | 模型截距，对应常数项 |
| `coef_` | 一维数组 | 各特征对应的线性系数 |
| `names` | 列名列表 | 用于把系数和特征名一一对应打印 |

### 示例代码

```python
print(f"截距(intercept): {model.intercept_:.2f}")
print("斜率(coefficients):")
for name, coef in zip(names, model.coef_):
    print(f"  {name}: {coef:.2f}")
```

### 理解重点

- `intercept_` 对应公式中的常数项，当前数据里理论上应接近 `50`。
- `coef_` 对应各特征的线性影响，当前数据里应大致接近 `面积=2`、`房间数=10`、`房龄=-3`。
- 文档里强调这些属性，是因为它们是当前线性回归分册最核心的训练结果。

## 4. `feature_names` 的处理方式

### 参数速览（本节）

适用逻辑（分项）：

1. 使用传入特征名
2. 从 `X_train.columns` 自动获取
3. 退化为 `Feature_i`

| 情况 | 当前行为 |
|---|---|
| 显式传入 `feature_names` | 直接使用该列表 |
| `X_train` 带列名 | 自动使用 `X_train.columns` |
| 纯数组输入 | 自动生成 `Feature_0`, `Feature_1`, ... |

### 示例代码

```python
if feature_names is not None:
    names = feature_names
elif hasattr(X_train, "columns"):
    names = list(X_train.columns)
else:
    names = [f"Feature_{i}" for i in range(X_train.shape[1])]
```

### 理解重点

- 当前源码优先保证“训练日志可解释”，所以专门做了特征名处理逻辑。
- 如果直接传入 `DataFrame`，即使不手动给 `feature_names`，也能打印出真实列名。
- 对当前中文列名数据来说，这一点尤其重要，因为日志会直接显示 `面积`、`房间数`、`房龄`。

## 5. 训练阶段的工程封装

除了 `LinearRegression(...).fit(...)` 之外，`train_model(...)` 还做了一层日志包装。

### 参数速览（本节）

适用装饰与输出（分项）：

1. `@print_func_info`
2. `print("模型训练完成")`
3. 截距与系数打印

| 输出项 | 作用 |
|---|---|
| 函数调用标题 | 帮助在终端中定位训练入口 |
| `模型训练完成` | 明确训练阶段已结束 |
| 截距打印 | 快速查看基线项 |
| 系数打印 | 快速查看每个特征的线性影响 |

### 理解重点

- 当前工程封装的重点不是性能统计，而是教学型日志输出。
- 和 SVR 分册里的训练耗时、支持向量数量不同，这里最值得打印的是截距和系数。
- 这也说明当前线性回归实现强调的是“解释结果”，而不是复杂模型诊断。

## 常见坑

1. 误以为 `train_model(...)` 有很多自定义训练逻辑，实际上它只是对 `LinearRegression()` 的薄封装。
2. 只看“模型训练完成”而不看截距和各特征系数，错过本分册最核心的信息。
3. 忘记 `feature_names` 的作用，导致数组输入时日志可解释性下降。

## 小结

- `train_model(...)` 是本仓库线性回归的核心训练入口。
- 它本质上是对 `sklearn.linear_model.LinearRegression` 的薄封装，重点在于训练后把截距和系数清楚打印出来。
- 读懂这一层之后，再看流水线中的训练、预测和评估过程会更清晰。
