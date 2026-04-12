---
title: SVR 支持向量回归 — 训练与预测
outline: deep
---

# 训练与预测

> 对应代码：`pipelines/regression/svr.py`
>  
> 运行方式：`python -m pipelines.regression.svr`

## 本章目标

1. 用源码视角梳理 SVR 流水线中的训练步骤。
2. 明确预测阶段的输入要求、输出形式和前置处理。
3. 理解当前实现中哪些步骤是必须的，哪些地方可以作为调参入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_test_split(...)` | 函数 | 将样本拆成训练集和测试集 |
| `StandardScaler()` | 类 | 对特征做标准化 |
| `train_model(...)` | 函数 | 训练一个 `SVR` 模型 |
| `model.predict(...)` | 方法 | 对标准化后的输入特征做回归预测 |
| `svr_data.copy()` | 表达式 | 取得当前 SVR 数据副本，避免直接改原始数据 |

## 1. 训练流程总览

本仓库 `pipelines/regression/svr.py` 中的训练主流程如下：

1. `data = svr_data.copy()`
2. `X = data.drop(columns=['price'])`
3. `y = data['price']`
4. `train_test_split(..., test_size=0.2, random_state=42)`
5. `StandardScaler().fit_transform(X_train)`
6. `StandardScaler().transform(X_test)`
7. `model = train_model(X_train_s, y_train)`

### 示例代码

```python
data = svr_data.copy()
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = train_model(X_train_s, y_train)
```

### 理解重点

- 这里的核心顺序是“先切分，再标准化，再训练”。
- `train_model(...)` 的输入不是原始特征，而是标准化后的 `X_train_s`。
- 当前流程已经是一条最小可运行的 SVR 回归基线。

## 2. 训练集/测试集切分

### 参数速览（本节）

适用 API：`train_test_split(X, y, test_size=0.2, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | `data.drop(columns=['price'])` | 全量特征矩阵 |
| `y` | `data['price']` | 全量标签向量 |
| `test_size` | `0.2` | 20% 样本划为测试集 |
| `random_state` | `42` | 固定随机切分，保证结果可复现 |
| 返回值 | `X_train, X_test, y_train, y_test` | 切分后的训练集与测试集 |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 理解重点

- 切分必须发生在标准化之前，否则会引入数据泄露。
- 固定 `random_state=42` 可以让实验结果更稳定，便于复现和比较。

## 3. 标准化处理

### 参数速览（本节）

适用 API（分项）：

1. `StandardScaler().fit_transform(X_train)`
2. `StandardScaler().transform(X_test)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 训练特征 | 在训练集上拟合均值和标准差，并完成变换 |
| 返回值（`fit_transform`） | `X_train_s` | 标准化后的训练特征 |
| `X_test` | 测试特征 | 使用训练集统计量做同样的变换 |
| 返回值（`transform`） | `X_test_s` | 标准化后的测试特征 |

### 示例代码

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- SVR 尤其是 RBF 核对特征尺度非常敏感，因此标准化几乎是必做步骤。
- `fit_transform` 只能用于训练集，测试集只能 `transform`。
- 预测新数据时也必须复用同一个 `scaler` 的统计量。

## 4. 模型训练

### 参数速览（本节）

适用函数：`train_model(X_train_s, y_train, C=10.0, epsilon=0.1, kernel='rbf', gamma='scale', degree=3, coef0=0.0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train_s` | 标准化训练集 | 训练输入特征 |
| `y_train` | 训练标签 | 模型学习目标 |
| `C` | 默认 `10.0` | 惩罚系数 |
| `epsilon` | 默认 `0.1` | 不敏感区间 |
| `kernel` | 默认 `'rbf'` | 核函数类型 |
| `gamma` | 默认 `'scale'` | 核函数参数 |
| 返回值 | `SVR` | 已训练模型 |

### 示例代码

```python
model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前流水线默认不做自动调参，而是直接使用源码中的基线参数。
- 若要优先提升效果，通常从 `C`、`epsilon`、`gamma` 和 `kernel` 开始尝试。

## 5. 预测流程

### 参数速览（本节）

适用 API：`model.predict(X_test_s)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_test_s` | 标准化测试集 | 与训练时同分布、同预处理的输入特征 |
| 返回值 | `ndarray` | 每个样本对应的预测回归值 |

### 示例代码

```python
y_pred = model.predict(X_test_s)
```

### 理解重点

- 预测阶段的关键不是代码复杂度，而是输入必须沿用训练阶段相同的标准化方式。
- `y_pred` 会被后续残差图和学习曲线相关步骤继续使用。

## 6. 从训练到预测的完整片段

### 示例代码

```python
data = svr_data.copy()
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = train_model(X_train_s, y_train)
y_pred = model.predict(X_test_s)
```

### 理解重点

- 这段代码已经覆盖了当前 SVR 流水线中训练与预测的最核心部分。
- 后续可视化评估完全建立在 `y_test` 和 `y_pred` 之上。

## 常见坑

1. 对测试集单独 `fit_transform(...)`，导致训练和预测使用了不同统计量。
2. 直接把原始 `X_test` 传给 `predict(...)`，跳过标准化。
3. 修改了训练参数，却没有同步观察残差图和学习曲线变化。

## 小结

- 当前 SVR 训练与预测流程非常短，但对数据处理顺序要求很严格。
- 真正影响结果稳定性的关键点是：正确切分、正确标准化、再用一致的特征空间做预测。
- 读完本章后，可以继续结合评估章节观察 `y_pred` 的效果。
