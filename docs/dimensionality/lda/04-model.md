---
title: LDA — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/dimensionality/lda.py`
>  
> 运行方式：`python -m model_training.dimensionality.lda`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LinearDiscriminantAnalysis`。
2. 理解 `n_components`、`solver` 和解释方差相关属性在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装和日志输出。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 模型 |
| `LinearDiscriminantAnalysis(...)` | 类 | scikit-learn 提供的 LDA 实现 |
| `model.fit(X_train, y_train)` | 方法 | 在标准化后的特征和标签上学习判别方向 |
| `model.explained_variance_ratio_` | 属性 | 若存在，返回各判别方向的解释比例 |
| `model.transform(X)` | 方法 | 将原始高维数据投影到判别子空间 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, y_train, n_components=2, solver='svd')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的特征 | 输入给 `LinearDiscriminantAnalysis.fit(...)` 的训练矩阵 |
| `y_train` | 训练标签 | 当前 LDA 训练必须使用的监督信息 |
| `n_components` | `2` | 保留的判别方向数量 |
| `solver` | `'svd'` | 求解器 |
| 返回值 | `LinearDiscriminantAnalysis` | 已训练完成的 LDA 模型 |

### 示例代码

```python
from model_training.dimensionality.lda import train_model

model = train_model(X_scaled, y, n_components=2)
```

### 理解重点

- 当前训练入口返回的是单个 LDA 模型对象。
- 与 PCA 不同，这里训练时必须同时提供 `X_train` 和 `y_train`。
- 默认参数直接来自源码，是后续理解 2D 判别投影流程的基线。

## 2. `LinearDiscriminantAnalysis(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `LinearDiscriminantAnalysis(...)`
2. `model.fit(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_components` | `2` | 当前保留两个判别方向 |
| `solver` | `'svd'` | 当前使用的默认求解器 |

### 示例代码

```python
model = LinearDiscriminantAnalysis(
    n_components=n_components,
    solver=solver,
)

model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有自己实现类间散度和类内散度的广义特征值求解，而是直接调用 scikit-learn 的现成 LDA。
- 当前文档的重点，不是重写底层线性代数实现，而是理解这几个高层参数如何决定输出判别空间。
- `solver='svd'` 是当前实现的默认路径，需要在文档里明确写清。

## 3. 两个核心参数分别控制什么

### 参数速览（本节）

适用参数（分项）：

1. `n_components`
2. `solver`

| 参数 | 当前作用 | 调整时的常见影响 |
|---|---|---|
| `n_components` | 决定保留多少个判别方向 | 影响降维后维度与类别分离表达能力 |
| `solver` | 控制求解方式 | 影响数值求解路径与可用属性 |

### 理解重点

- `n_components` 最直观，决定最终投影到几维。
- 当前 Wine 数据是 3 类，因此 `n_components=2` 已经达到理论上限。
- `solver` 则更多决定内部求解方式，以及某些属性是否可直接获得。

## 4. 为什么 `explained_variance_ratio_` 是“若存在才打印”

### 参数速览（本节）

适用逻辑：`hasattr(model, "explained_variance_ratio_")`

| 条件 | 当前含义 |
|---|---|
| 属性存在 | 打印解释方差比和累计解释方差 |
| 属性不存在 | 当前实现不强行输出该信息 |

### 示例代码

```python
if hasattr(model, "explained_variance_ratio_"):
    print(f"解释方差比: {model.explained_variance_ratio_.round(4)}")
    print(f"累计解释方差: {model.explained_variance_ratio_.sum():.4f}")
```

### 理解重点

- 当前源码没有假设所有求解器都一定提供 `explained_variance_ratio_`。
- 因此它用 `hasattr(...)` 做保护式输出，这属于很典型的工程边界处理。
- 文档必须把这点如实写清楚，不能把该属性写成“总是必然存在”。

## 5. 训练阶段的工程封装

除了 `LinearDiscriminantAnalysis(...).fit(...)` 之外，`train_model(...)` 还做了多层工程包装。

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
- 这些包装不改变 LDA 训练行为，但有助于观察训练入口和执行耗时。
- 和监督分类分册相比，这里的日志重点是判别方向与解释比例，而不是分类精度。

## 6. `transform(...)` 为什么是关键步骤

### 参数速览（本节）

适用方法：`model.transform(X)`

| 方法 | 当前作用 |
|---|---|
| `fit(...)` | 学习判别方向 |
| `transform(...)` | 把原始高维数据投影到判别子空间 |

### 理解重点

- 当前训练并不会自动给出最终低维坐标结果，它先学习判别方向，再由 `transform(...)` 生成投影坐标。
- 当前 2D 图的输入，正是这一步之后得到的结果。
- 这使 `transform(...)` 成为连接“模型训练”和“可视化输出”的核心桥梁。

## 常见坑

1. 把 `fit(...)` 和 `transform(...)` 混为一步，忽略投影结果需要单独生成。
2. 把 `explained_variance_ratio_` 误写成一定存在的属性。
3. 忘记当前 LDA 分册里标签是训练输入，而不是只用于着色的辅助信息。

## 小结

- `train_model(...)` 是本仓库 LDA 的核心训练入口。
- 它本质上是对 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 的薄封装，重点在于判别维度设定、解释比例输出和训练耗时日志。
- 读懂这一层之后，再看流水线中的投影和可视化过程会更清晰。
