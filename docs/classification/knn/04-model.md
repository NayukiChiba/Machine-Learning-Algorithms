---
title: KNN K 近邻分类 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `KNeighborsClassifier`。
2. 理解每个构造器参数的数学含义与调参方向。
3. 理解 KNN 的 `fit()` 与其他参数化模型的本质区别。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练 `KNeighborsClassifier`，返回已训练模型 |
| `KNeighborsClassifier(...)` | 构造器 | 创建 K 近邻分类器，通过超参数控制邻域定义与投票规则 |
| `model.fit(X_train, y_train)` | 方法 | 保存训练样本并建立近邻查询结构（KD-Tree 或 Ball-Tree） |
| `n_neighbors` | 超参数 | 控制近邻数量 $k$ |
| `weights` | 超参数 | 控制投票权重策略 |
| `metric` | 超参数 | 控制距离度量方式 |

## 1. `train_model(...)` 的函数签名

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，形状 `(n_samples, n_features)`。传入 `model.fit()`。每行为一个样本，每列为一个特征 | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，形状 `(n_samples,)`。二分类标签取值为 $\{0, 1\}$ | `y_train` |
| `n_neighbors` | `int` | 近邻数量 $k$。$k$ 越小偏差越低方差越高；$k$ 越大偏差越高方差越低。当前默认 `5` | `3`、`5`、`15` |
| `weights` | `str` | 投票权重策略。`"uniform"` 等权投票 $\hat{y} = \arg\max_c \sum \mathbb{1}(y_i=c)$；`"distance"` 距离倒数加权 $\hat{y} = \arg\max_c \sum \frac{\mathbb{1}(y_i=c)}{d(\mathbf{x}, \mathbf{x}_i)}$。当前默认 `"uniform"` | `"uniform"`、`"distance"` |
| `metric` | `str` | 距离度量方式。`"minkowski"` 对应 $d_p(\mathbf{x}, \mathbf{y}) = (\sum \vert x_i - y_i\vert^p)^{1/p}$，配合 `p` 参数使用。当前默认 `"minkowski"` | `"minkowski"`、`"euclidean"`、`"manhattan"` |
| 返回值 | `KNeighborsClassifier` | 已完成 `fit()` 的模型对象，含 `_fit_X`、`_fit_y` 等内部属性，可立即调用 `predict()` 和 `predict_proba()` | — |

### 示例代码

```python
from model_training.classification.knn import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `KNeighborsClassifier` 模型。
- 和部分实验型代码不同，这里没有参数搜索逻辑，也没有多模型对比。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `KNeighborsClassifier(...)` 的完整参数

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_neighbors` | `int` | 近邻数量 $k$。决定了投票邻域的大小，是 KNN 最核心的超参数。默认为 `5` | `1`、`5`、`15`、`50` |
| `weights` | `str` 或 `callable` | 投票权重函数。`"uniform"` 等权，$w_i = 1$；`"distance"` 按距离倒数加权，$w_i = 1/d(\mathbf{x}, \mathbf{x}_i)$。默认为 `"uniform"` | `"uniform"`、`"distance"` |
| `algorithm` | `str` | 近邻搜索算法。`"auto"` 自动选择（当前默认）；`"ball_tree"` 球树；`"kd_tree"` KD 树；`"brute"` 暴力搜索。低维数据 KD-Tree 通常最快。默认为 `"auto"` | `"auto"`、`"kd_tree"`、`"ball_tree"`、`"brute"` |
| `leaf_size` | `int` | Ball-Tree 或 KD-Tree 的叶节点大小。影响构建速度和查询速度，小值 → 构建慢查询快，大值 → 构建快查询慢。仅当 `algorithm` 为 `"ball_tree"` 或 `"kd_tree"` 时生效。默认为 `30` | `20`、`30`、`50` |
| `p` | `int` | 闵可夫斯基距离的幂参数。$p=1$ 曼哈顿距离 $d_1 = \sum \vert x_i - y_i \vert$；$p=2$ 欧几里得距离 $d_2 = \sqrt{\sum (x_i - y_i)^2}$。仅当 `metric='minkowski'` 时生效。默认为 `2` | `1`、`2` |
| `metric` | `str` 或 `callable` | 距离度量方式。默认为 `"minkowski"`（配合 `p` 参数），也可直接设为 `"euclidean"`、`"manhattan"`、`"chebyshev"` 等 | `"minkowski"`、`"euclidean"`、`"manhattan"` |
| `metric_params` | `dict` 或 `None` | 距离度量的额外关键字参数。如对某些度量传入额外配置。默认为 `None` | `None`、`{}` |
| `n_jobs` | `int` 或 `None` | 并行作业数。`-1` 用全部核心，`None` 为单核。加速近邻搜索的并行计算。默认为 `None` | `None`、`-1`、`4` |

### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    algorithm="auto",
    metric="minkowski",
    p=2,
)
model.fit(X_train_s, y_train)
```

### 理解重点

- KNN 的"训练"与逻辑回归、SVC 很不同——`fit()` 不是优化损失函数，而是存储训练样本并建立近邻查询所需的数据结构（如 KD-Tree）。
- 因此 KNN 的 `fit()` 非常快（几乎无计算），但 `predict()` 较重（需要扫描训练集找近邻）。
- 当前封装的重点不是重写算法，而是把超参数、训练耗时和关键结果日志组织清楚。
- 最值得关注的三参数：`n_neighbors`、`weights`、`metric`/`p`——它们共同定义了"邻居是谁"和"怎么投票"。

## 3. 训练完成后最重要的模型属性

### 属性表

| 属性 | 类型 | 数学含义 |
|---|---|---|
| `classes_` | `ndarray` | 模型学到的类别标签数组，形状 `(n_classes,)`。当前二分类为 `[0, 1]` |
| `n_features_in_` | `int` | 训练时的特征维度 $d$。当前为 `2` |
| `effective_metric_` | `str` | 实际使用的距离度量名称。例如 `metric='minkowski'` 且 `p=2` 时返回 `'euclidean'` |
| `effective_metric_params_` | `dict` | 实际使用的距离度量参数。例如 `{'p': 2}` |
| `n_samples_fit_` | `int` | 训练样本数 $n_{\text{train}}$。当前为 $400 \times 0.8 = 320$ |
| `outputs_2d_` | `bool` | 输出是否为二维。用于内部判断 `predict_proba` 行为 |

### 示例代码

```python
print(f"实际度量: {model.effective_metric_}")
print(f"训练样本数: {model.n_samples_fit_}")
print(f"类别: {model.classes_}")
```

### 理解重点

- KNN 没有显式的"参数矩阵"（如逻辑回归的 $\mathbf{w}$），因此属性集中在配置信息和数据统计上。
- `effective_metric_` 和 `effective_metric_params_` 反映了 sklearn 内部的度量解析结果——你传 `'minkowski'` + `p=2`，它内部解析为 `'euclidean'` + `{'p': 2}`。
- `n_samples_fit_` 是 KNN 特有的属性，因为 KNN 的"知识"就是全体训练样本。

## 4. 训练阶段的工程封装

除了 `KNeighborsClassifier(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| 函数调用标题（`@print_func_info`） | 帮助在终端中定位训练入口 |
| 训练耗时（`@timeit`） | 观察 KNN `fit()` 的执行时间——通常非常快 |
| 超参数日志（`K`、`weights`、`metric`） | 确认当前训练使用的配置 |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 这一层封装把"构建模型""训练模型""打印结果"收在一个函数里，方便文档和流水线复用。
- 从工程角度看，这样的拆分也让 `pipelines/classification/knn.py` 保持简洁。

## 常见坑

1. 把 KNN 的 `fit(...)` 理解成和参数化模型一样的"求最优参数"过程——它是存储数据，不是优化。
2. 只知道可以 `predict(...)`，却忽略 `n_neighbors`、`weights`、`metric`/`p` 才是理解 KNN 行为的重要线索。
3. 忘记当前 `X_train` 应该是标准化后的特征——原始特征会让距离关系失真。
4. 忽略 `algorithm` 参数对大规模数据的影响——数据量大时暴力搜索会很慢。

## 小结

- `train_model(...)` 是本仓库 KNN 的核心训练入口，本质上是对 `sklearn.neighbors.KNeighborsClassifier` 的薄封装。
- `KNeighborsClassifier` 的 8 个构造器参数中，`n_neighbors`、`weights`、`metric`/`p` 是最核心的四个——它们决定"谁是你的邻居"和"怎么请邻居投票"。
- KNN 的 `fit()` 不学习参数，只存储数据 + 建立索引结构，这是它与所有参数化模型最根本的区别。
- 训练后属性 `effective_metric_`、`n_samples_fit_` 等反映了模型的实际底层配置。
