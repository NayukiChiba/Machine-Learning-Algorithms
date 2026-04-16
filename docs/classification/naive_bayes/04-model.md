---
title: GaussianNB 高斯朴素贝叶斯 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/classification/naive_bayes.py`
>
> 运行方式：`python -m model_training.classification.naive_bayes`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `GaussianNB`。
2. 理解 `classes_`、`class_prior_` 在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.naive_bayes.GaussianNB` 模型 |
| `GaussianNB(...)` | 类 | scikit-learn 提供的高斯朴素贝叶斯分类器 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上拟合类别先验和特征分布参数 |
| `model.classes_` | 属性 | 返回模型识别到的类别列表 |
| `model.class_prior_` | 属性 | 返回各类别先验概率 |
| `@print_func_info` / `@timeit` | 装饰器 | 打印函数信息并统计训练耗时 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, var_smoothing=1e-9)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 训练特征 | 输入给 `GaussianNB.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 每个样本对应的类别标签 |
| `var_smoothing` | `1e-9` | 方差平滑项，提升数值稳定性 |
| 返回值 | `GaussianNB` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.classification.naive_bayes import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `GaussianNB` 模型。
- 和部分实验型代码不同，这里没有变体对比，也没有超参数搜索逻辑。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `GaussianNB(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `GaussianNB(...)`
2. `model.fit(X_train, y_train)`

| 项目 | 当前实现 | 说明 |
|---|---|---|
| 训练模型 | `GaussianNB(var_smoothing=var_smoothing)` | 使用源码中显式给出的超参数 |
| 输入特征 | `X_train` | 当前流水线传入的是标准化后的训练特征 |
| 输入标签 | `y_train` | 多分类监督标签 |
| 训练方式 | `fit(X_train, y_train)` | 在监督数据上估计类别先验和高斯参数 |
| 返回值 | 已训练模型 | 含类别与先验相关属性、预测能力 |

### 示例代码

```python
model = GaussianNB(var_smoothing=var_smoothing)
model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有自己实现概率估计公式，而是直接调用 scikit-learn 的成熟实现。
- 当前封装的重点，不是重写朴素贝叶斯算法，而是把超参数、训练耗时和关键结果日志组织清楚。
- 这里最值得强调的是：当前默认模型是 `GaussianNB`，不是其他朴素贝叶斯变体。

## 3. 训练完成后最重要的模型属性

### 参数速览（本节）

适用属性（分项）：

1. `model.classes_`
2. `model.class_prior_`

| 属性名 | 当前含义 | 作用 |
|---|---|---|
| `classes_` | 模型识别到的类别列表 | 用于确认当前分类任务类别集合 |
| `class_prior_` | 各类别先验概率 | 用于观察训练集中各类别的基础比例 |

### 示例代码

```python
print(f"类别: {model.classes_.tolist()}")
print(f"类别先验: {model.class_prior_.round(4)}")
```

### 理解重点

- `class_prior_` 是当前 Naive Bayes 分册最值得关注的训练结果之一。
- 它把“先验概率”这一理论概念，映射成源码里可观察的输出。
- 相比只打印“模型训练完成”，这种日志更有教学意义。

## 4. 训练阶段的工程封装

除了 `GaussianNB(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装。

### 参数速览（本节）

适用装饰与输出（分项）：

1. `@print_func_info`
2. `@timeit`
3. `with timer(name="模型训练耗时")`
4. 日志输出 `var_smoothing`、`类别`、`类别先验`

| 输出项 | 作用 |
|---|---|
| 函数调用标题 | 帮助在终端中定位训练入口 |
| 训练耗时 | 观察当前模型拟合时间 |
| `模型训练完成` | 明确训练阶段已结束 |
| `var_smoothing` | 确认当前平滑参数配置 |
| `类别` / `类别先验` | 观察当前多分类结构与先验比例 |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 这一层封装把“构建模型”“训练模型”“打印结果”收在一个函数里，方便文档和流水线复用。
- 从工程角度看，这样的拆分也让 `pipelines/classification/naive_bayes.py` 保持简洁。

## 常见坑

1. 误以为当前实现使用的是所有朴素贝叶斯的通用封装，而不是专门的 `GaussianNB`。
2. 只知道可以 `predict(...)`，却忽略 `class_prior_` 才是理解概率分类的重要线索。
3. 忘记当前 `X_train` 应该是标准化后的训练特征。
4. 把训练函数和后续 ROC、学习曲线等评估逻辑混在一起理解。

## 小结

- `train_model(...)` 是本仓库 Naive Bayes 的核心训练入口。
- 它本质上是对 `sklearn.naive_bayes.GaussianNB` 的薄封装，重点在于把超参数、训练结果和日志输出组织清楚。
- 读懂这一层之后，再看流水线中的预测概率、ROC 曲线和学习曲线会更顺畅。
