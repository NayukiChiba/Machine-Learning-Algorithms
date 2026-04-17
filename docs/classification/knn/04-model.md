---
title: KNN K 近邻分类 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/classification/knn.py`
>
> 运行方式：`python -m model_training.classification.knn`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `KNeighborsClassifier`。
2. 理解 `n_neighbors`、`weights`、`metric` 在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.neighbors.KNeighborsClassifier` 模型 |
| `KNeighborsClassifier(...)` | 类 | scikit-learn 提供的 K 近邻分类器 |
| `model.fit(X_train, y_train)` | 方法 | 保存训练样本并建立近邻查询所需结构 |
| `n_neighbors` | 超参数 | 决定近邻数量 |
| `weights` | 超参数 | 决定投票时是否对近邻加权 |
| `metric` | 超参数 | 决定距离度量方式 |
| `@print_func_info` / `@timeit` | 装饰器 | 打印函数信息并统计训练耗时 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, n_neighbors=5, weights='uniform', metric='minkowski')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的训练特征 | 输入给 `KNeighborsClassifier.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 每个样本对应的类别标签 |
| `n_neighbors` | `5` | 近邻数量 `K` |
| `weights` | `'uniform'` | 投票权重方式 |
| `metric` | `'minkowski'` | 距离度量 |
| 返回值 | `KNeighborsClassifier` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.classification.knn import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `KNeighborsClassifier` 模型。
- 和部分实验型代码不同，这里没有参数搜索逻辑，也没有多模型对比。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `KNeighborsClassifier(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `KNeighborsClassifier(...)`
2. `model.fit(X_train, y_train)`

| 项目 | 当前实现 | 说明 |
|---|---|---|
| 训练模型 | `KNeighborsClassifier(...)` | 使用源码中显式给出的超参数 |
| 输入特征 | `X_train` | 当前流水线传入的是标准化后的训练特征 |
| 输入标签 | `y_train` | 二分类监督标签 |
| 训练方式 | `fit(X_train, y_train)` | 保存样本并准备近邻查询 |
| 返回值 | 已训练模型 | 含预测与概率输出能力 |

### 示例代码

```python
model = KNeighborsClassifier(
    n_neighbors=n_neighbors,
    weights=weights,
    metric=metric,
)
model.fit(X_train, y_train)
```

### 理解重点

- KNN 的“训练”与逻辑回归、SVC 很不同，它不是优化一组显式参数，而是保存样本并建立查询逻辑。
- 当前封装的重点，不是训练复杂参数，而是把超参数、训练耗时和关键配置日志组织清楚。
- 这里最值得强调的是：当前默认采用 `n_neighbors=5`、`weights='uniform'`、`metric='minkowski'`。

## 3. 训练完成后最重要的模型配置

### 参数速览（本节）

适用配置（分项）：

1. `n_neighbors`
2. `weights`
3. `metric`

| 名称 | 当前含义 | 作用 |
|---|---|---|
| `n_neighbors` | 邻居数量 | 决定局部投票范围 |
| `weights` | 投票方式 | 决定邻居是否等权 |
| `metric` | 距离度量 | 决定什么叫“最近” |

### 示例代码

```python
print(f"K: {n_neighbors}")
print(f"weights: {weights}")
print(f"metric: {metric}")
```

### 理解重点

- 对 KNN 来说，最值得解释的不是一组 learned weights，而是决定邻域关系的超参数本身。
- 这些配置会直接改变邻居集合和投票结果，因此比很多模型更“可感知”。
- 这也是当前训练日志主要打印配置项而不是参数矩阵的原因。

## 4. 训练阶段的工程封装

除了 `KNeighborsClassifier(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装。

### 参数速览（本节）

适用装饰与输出（分项）：

1. `@print_func_info`
2. `@timeit`
3. `with timer(name="模型训练耗时")`
4. 日志输出 `K`、`weights`、`metric`

| 输出项 | 作用 |
|---|---|
| 函数调用标题 | 帮助在终端中定位训练入口 |
| 训练耗时 | 观察当前模型拟合时间 |
| `模型训练完成` | 明确训练阶段已结束 |
| 超参数日志 | 快速确认当前训练配置 |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 这一层封装把“构建模型”“训练模型”“打印结果”收在一个函数里，方便文档和流水线复用。
- 从工程角度看，这样的拆分也让 `pipelines/classification/knn.py` 保持简洁。

## 常见坑

1. 把 KNN 的 `fit(...)` 理解成和参数化模型一样的“求最优参数”过程。
2. 只知道可以 `predict(...)`，却忽略 `n_neighbors`、`weights`、`metric` 才是理解 KNN 行为的重要线索。
3. 忘记当前 `X_train` 应该是标准化后的训练特征。
4. 把训练函数和后续 ROC、学习曲线等评估逻辑混在一起理解。

## 小结

- `train_model(...)` 是本仓库 KNN 的核心训练入口。
- 它本质上是对 `sklearn.neighbors.KNeighborsClassifier` 的薄封装，重点在于把超参数、训练结果和日志输出组织清楚。
- 读懂这一层之后，再看流水线中的概率输出、决策边界和学习曲线会更顺畅。
