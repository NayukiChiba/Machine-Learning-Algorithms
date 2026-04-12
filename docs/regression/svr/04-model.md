---
title: SVR 支持向量回归 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/regression/svr.py`
>  
> 运行方式：`python -m model_training.regression.svr`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `SVR`。
2. 理解各个超参数在当前源码中的默认值与作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.svm.SVR` 模型 |
| `SVR(...)` | 类 | scikit-learn 提供的支持向量回归器 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上拟合模型 |
| `model.support_` | 属性 | 返回支持向量在训练集中的索引 |
| `timer(name='SVR 训练耗时')` | 上下文管理器 | 打印训练耗时 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, C=10.0, epsilon=0.1, kernel='rbf', gamma='scale', degree=3, coef0=0.0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的训练特征 | 输入给 `SVR.fit(...)` 的训练特征矩阵 |
| `y_train` | 训练标签 | 每个样本对应的连续值目标 |
| `C` | `10.0` | 惩罚系数，越大越强调拟合训练数据 |
| `epsilon` | `0.1` | `epsilon`-不敏感区间宽度 |
| `kernel` | `'rbf'` | 核函数类型，默认使用 RBF |
| `gamma` | `'scale'` | 核函数系数，交给 scikit-learn 自动按方差缩放 |
| `degree` | `3` | 多项式核阶数，仅 `kernel='poly'` 时起主要作用 |
| `coef0` | `0.0` | 多项式核或 sigmoid 核中的常数项 |
| 返回值 | `SVR` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.regression.svr import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 这个函数把模型创建和训练合并到了同一个入口里。
- 真实训练接口仍然是 `sklearn.svm.SVR`，本仓库只做了薄封装。
- 默认参数直接来自源码，可以作为后续调参的基线。

## 2. `SVR(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)`
2. `model.fit(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `C` | 来自 `train_model` 入参 | 传递给 `SVR(...)` 的惩罚系数 |
| `epsilon` | 来自 `train_model` 入参 | 控制不敏感区间宽度 |
| `kernel` | 来自 `train_model` 入参 | 决定线性或非线性拟合方式 |
| `gamma` | 来自 `train_model` 入参 | 控制 RBF / poly / sigmoid 核的影响范围 |
| `degree` | 来自 `train_model` 入参 | 多项式核阶数 |
| `coef0` | 来自 `train_model` 入参 | 多项式 / sigmoid 核常数项 |
| `X_train` | `X_train_s` | 标准化后的训练特征 |
| `y_train` | `y_train` | 训练标签 |

### 示例代码

```python
from sklearn.svm import SVR

model = SVR(
    C=10.0,
    epsilon=0.1,
    kernel="rbf",
    gamma="scale",
    degree=3,
    coef0=0.0,
)
model.fit(X_train_s, y_train)
```

### 理解重点

- 仓库没有重新实现 SVR 优化过程，而是直接调用 scikit-learn 的现成实现。
- 代码里默认 `kernel='rbf'`，这与当前 `svr_data` 的非线性特征相匹配。
- `degree` 和 `coef0` 虽然默认传入，但在默认 RBF 流程中不是主要调节点。

## 3. 训练阶段的工程封装

除了 `SVR(...).fit(...)` 之外，`train_model(...)` 还做了三层工程包装。

### 参数速览（本节）

适用装饰与属性（分项）：

1. `@print_func_info`
2. `with timer(name='SVR 训练耗时')`
3. `model.support_`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `name`（`timer`） | `'SVR 训练耗时'` | 控制台打印训练耗时的标签名 |
| 返回值（`model.support_`） | 索引数组 | 支持向量在训练集中的下标 |
| 派生量 `n_sv` | `model.support_.shape[0]` | 支持向量数量，用于快速判断模型复杂度 |

### 示例代码

```python
model = SVR(
    C=C,
    epsilon=epsilon,
    kernel=kernel,
    gamma=gamma,
    degree=degree,
    coef0=coef0,
)

with timer(name="SVR 训练耗时"):
    model.fit(X_train, y_train)

n_sv = model.support_.shape[0]
print(f"支持向量数量: {n_sv}")
```

### 理解重点

- `@print_func_info` 用于打印函数调用信息，方便在终端日志中定位训练入口。
- `timer(...)` 负责统计训练耗时，不改变模型行为。
- 训练后会打印 `kernel`、`C`、`epsilon`、`gamma`，并在 `kernel='poly'` 时额外打印 `degree` 与 `coef0`。

## 4. 训练完成后最直接可用的属性

### 参数速览（本节）

适用属性/方法（分项）：

1. `model.support_`
2. `model.predict(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值（`model.support_`） | 索引数组 | 对应训练集中被选为支持向量的样本下标 |
| `X` | `X_test_s` | 与训练时同分布、同预处理方式的输入特征 |
| 返回值（`model.predict(X)`） | `ndarray` | 回归预测值数组 |

### 示例代码

```python
n_sv = model.support_.shape[0]
y_pred = model.predict(X_test_s)
```

### 理解重点

- `support_` 是当前源码里训练后最先被使用的模型属性。
- 预测阶段不再重新训练，只需要对新输入调用 `predict(...)`。
- 如果输入没有沿用训练阶段的标准化方式，预测结果通常会明显变差。

## 常见坑

1. 直接把未标准化的特征传给 `train_model(...)`，会影响 RBF 核的拟合效果。
2. 误以为 `degree` 和 `coef0` 会影响默认 RBF 流程，实际上默认设置下它们不是主要因素。
3. 只看是否训练成功，不看“支持向量数量”和训练参数打印信息。

## 小结

- `train_model(...)` 是本仓库 SVR 的核心训练入口。
- 它本质上是对 `sklearn.svm.SVR` 的薄封装，重点在于参数传递、耗时统计和日志输出。
- 读懂这一层之后，再看流水线中的训练与预测过程会更清晰。
