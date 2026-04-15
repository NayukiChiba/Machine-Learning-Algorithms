---
title: SVC 支持向量分类 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/classification/svc.py`
>
> 运行方式：`python -m model_training.classification.svc`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `SVC`。
2. 理解 `n_support_` 在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.svm.SVC` 模型 |
| `SVC(...)` | 类 | scikit-learn 提供的支持向量分类器 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上拟合分类边界 |
| `model.n_support_` | 属性 | 返回每个类别的支持向量数量 |
| `@print_func_info` / `@timeit` | 装饰器 | 打印函数信息并统计训练耗时 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的训练特征 | 输入给 `SVC.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 每个样本对应的类别标签 |
| `C` | `1.0` | 软间隔惩罚系数 |
| `kernel` | `'rbf'` | 当前默认核函数 |
| `gamma` | `'scale'` | RBF 核宽度相关参数 |
| `random_state` | `42` | 随机种子 |
| 返回值 | `SVC` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.classification.svc import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `SVC` 模型。
- 和部分对比实验型代码不同，这里没有多核函数并行比较，也没有参数搜索逻辑。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `SVC(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `SVC(...)`
2. `model.fit(X_train, y_train)`

| 项目 | 当前实现 | 说明 |
|---|---|---|
| 训练模型 | `SVC(...)` | 使用源码中显式给出的超参数 |
| 输入特征 | `X_train` | 当前流水线传入的是标准化后的训练特征 |
| 输入标签 | `y_train` | 二分类监督标签 |
| 训练方式 | `fit(X_train, y_train)` | 在监督数据上拟合分类边界 |
| 返回值 | 已训练模型 | 含支持向量相关属性与预测能力 |

### 示例代码

```python
model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state)
model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有自己实现 SMO 或对偶优化过程，而是直接调用 scikit-learn 的成熟实现。
- 当前封装的重点，不是重写支持向量机算法，而是把超参数、训练耗时和关键结果日志组织清楚。
- 这里最值得强调的是：当前默认配置是 `rbf` 核，而不是线性 SVM。

## 3. 训练完成后最重要的模型属性

### 参数速览（本节）

适用属性（分项）：

1. `model.n_support_`
2. `model.n_support_.sum()`

| 属性名 | 当前含义 | 作用 |
|---|---|---|
| `n_support_` | 每个类别对应的支持向量数量 | 用于观察两类样本对边界的贡献 |
| `n_support_.sum()` | 支持向量总数 | 用于观察模型最终依赖了多少关键样本 |

### 示例代码

```python
print(f"支持向量总数: {model.n_support_.sum()}")
print(f"各类别支持向量数: {model.n_support_.tolist()}")
```

### 理解重点

- `n_support_` 是当前 SVC 分册最值得关注的训练结果之一。
- 它直接把“支持向量决定边界”这一理论概念，映射成源码里可观察的输出。
- 相比只打印“模型训练完成”，这种日志更有教学意义。

## 4. 训练阶段的工程封装

除了 `SVC(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装。

### 参数速览（本节）

适用装饰与输出（分项）：

1. `@print_func_info`
2. `@timeit`
3. `with timer(name="模型训练耗时")`
4. 日志输出 `支持向量总数` 和 `各类别支持向量数`

| 输出项 | 作用 |
|---|---|
| 函数调用标题 | 帮助在终端中定位训练入口 |
| 训练耗时 | 观察当前模型拟合时间 |
| `模型训练完成` | 明确训练阶段已结束 |
| 支持向量总数 | 快速查看模型依赖的关键样本规模 |
| 各类别支持向量数 | 观察不同类别对边界的贡献 |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 这一层封装把“构建模型”“训练模型”“打印结果”收在一个函数里，方便文档和流水线复用。
- 从工程角度看，这样的拆分也让 `pipelines/classification/svc.py` 保持简洁。

## 常见坑

1. 误以为当前实现默认是线性核。
2. 只知道可以 `predict(...)`，却忽略 `n_support_` 才是理解 SVC 行为的重要线索。
3. 忘记当前 `X_train` 应该是标准化后的训练特征。
4. 把训练函数和后续评估逻辑混在一起理解。

## 小结

- `train_model(...)` 是本仓库 SVC 的核心训练入口。
- 它本质上是对 `sklearn.svm.SVC` 的薄封装，重点在于把超参数、训练结果和日志输出组织清楚。
- 读懂这一层之后，再看流水线中的预测、决策边界和学习曲线会更顺畅。
