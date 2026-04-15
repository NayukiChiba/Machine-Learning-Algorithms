---
title: PCA — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/dimensionality/pca.py`
>  
> 运行方式：`python -m model_training.dimensionality.pca`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `PCA`。
2. 理解 `n_components`、`svd_solver`、`random_state` 和解释方差相关属性在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装和日志输出。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.decomposition.PCA` 模型 |
| `PCA(...)` | 类 | scikit-learn 提供的主成分分析实现 |
| `model.fit(X_train)` | 方法 | 在标准化后的特征矩阵上学习主成分方向 |
| `model.explained_variance_ratio_` | 属性 | 返回各主成分的解释方差比 |
| `model.transform(X)` | 方法 | 将原始高维数据投影到主成分空间 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, n_components=2, svd_solver='auto', random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的特征 | 输入给 `PCA.fit(...)` 的训练矩阵 |
| `n_components` | `2` | 保留的主成分数量 |
| `svd_solver` | `'auto'` | 主成分求解器 |
| `random_state` | `42` | 随机种子 |
| 返回值 | `PCA` | 已训练完成的 PCA 模型 |

### 示例代码

```python
from model_training.dimensionality.pca import train_model

model = train_model(X_scaled)
```

### 理解重点

- 当前训练入口返回的是单个 PCA 模型对象。\n+- 与分类或回归模型不同，这里训练目标不是预测标签，而是学习主成分方向。\n+- 默认参数直接来自源码，是后续理解 2D / 3D 降维流程的基线。

## 2. `PCA(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `PCA(...)`
2. `model.fit(X_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_components` | `2` 或 `3` | 当前分册会分别训练 2D 和 3D PCA |
| `svd_solver` | `'auto'` | 让 sklearn 自动选择合适求解方式 |
| `random_state` | `42` | 保证部分求解流程可复现 |

### 示例代码

```python
model = PCA(
    n_components=n_components,
    svd_solver=svd_solver,
    random_state=random_state,
)

model.fit(X_train)
```

### 理解重点

- 仓库没有自己实现特征值分解或 SVD，而是直接调用 scikit-learn 的现成 PCA。\n+- 当前文档的重点，不是重写底层线性代数算法，而是理解这几个高层参数如何影响降维输出。\n+- `n_components` 是当前分册最关键的建模选择之一。

## 3. 三个核心参数分别控制什么

### 参数速览（本节）

适用参数（分项）：

1. `n_components`
2. `svd_solver`
3. `random_state`

| 参数 | 当前作用 | 调整时的常见影响 |
|---|---|---|
| `n_components` | 决定保留多少个主成分 | 影响降维后维度和信息保留量 |
| `svd_solver` | 控制求解方式 | 影响数值求解策略 |
| `random_state` | 控制部分随机过程 | 影响可复现性 |

### 理解重点

- `n_components` 最直观，决定最终投影到几维。\n+- `svd_solver='auto'` 表示当前实现把求解器选择交给 sklearn，而不是手工强行指定。\n+- `random_state` 在 PCA 里不像树模型那样总是核心，但保留它有利于流程可复现。

## 4. 解释方差相关属性在当前实现里表示什么

### 参数速览（本节）

适用属性（分项）：

1. `explained_variance_ratio_`
2. `explained_variance_ratio_.sum()`

| 属性名 | 当前含义 |
|---|---|
| `explained_variance_ratio_` | 每个主成分分别解释了多少方差 |
| 累计解释方差 | 当前保留主成分总共解释了多少方差 |

### 示例代码

```python
print(f"解释方差比: {model.explained_variance_ratio_.round(4)}")
print(f"累计解释方差: {model.explained_variance_ratio_.sum():.4f}")
```

### 理解重点

- 当前日志中最有价值的输出，不是精度，而是解释方差比和累计解释方差。\n+- 它们帮助你判断当前降维压缩是否保留了足够多的信息。\n+- 这也是 PCA 分册最核心的定量解释线索之一。

## 5. 训练阶段的工程封装

除了 `PCA(...).fit(...)` 之外，`train_model(...)` 还做了多层工程包装。

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

- 当前训练函数会同时打印“模型训练耗时”和整个函数耗时，因此终端里会看到两层计时信息。\n+- 这些包装不改变 PCA 训练行为，但有助于观察训练入口和执行耗时。\n+- 和监督学习分册相比，这里的日志重点是解释方差，而不是预测性能。

## 6. `transform(...)` 为什么是关键步骤

### 参数速览（本节）

适用方法：`model.transform(X)`

| 方法 | 当前作用 |
|---|---|
| `fit(...)` | 学主成分方向 |
| `transform(...)` | 把原始数据投影到主成分空间 |

### 理解重点

- PCA 训练并不会自动给出降维后的坐标结果，它先学习方向，再由 `transform(...)` 做投影。\n+- 当前流水线中的 2D 图和 3D 图，都是建立在这一步之后的结果上。\n+- 这使 `transform(...)` 成为连接“模型训练”和“可视化结果”的核心桥梁。

## 常见坑

1. 只看 `fit(...)`，忽略 `transform(...)` 才是真正得到降维结果的步骤。\n+2. 把解释方差比误当成监督学习精度指标。\n+3. 忘记当前 PCA 分册会分别训练 2D 和 3D 两个模型，而不是一个模型同时输出两种结果。

## 小结

- `train_model(...)` 是本仓库 PCA 的核心训练入口。\n+- 它本质上是对 `sklearn.decomposition.PCA` 的薄封装，重点在于主成分数量设定、解释方差输出和训练耗时日志。\n+- 读懂这一层之后，再看流水线中的投影和可视化过程会更清晰。
