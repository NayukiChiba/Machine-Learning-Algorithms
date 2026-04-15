---
title: KMeans K 均值聚类 — 模型构建
outline: deep
---

# 模型构建

> 对应代码：`model_training/clustering/kmeans.py`
>
> 运行方式：`python -m model_training.clustering.kmeans`

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `KMeans`。
2. 理解 `labels_`、`cluster_centers_` 和 `inertia_` 在当前源码中的作用。
3. 看清训练函数除了 `fit(...)` 之外还做了哪些工程封装。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.cluster.KMeans` 模型 |
| `KMeans(...)` | 类 | scikit-learn 提供的 K 均值聚类器 |
| `model.fit(X_train)` | 方法 | 在训练数据上拟合聚类中心 |
| `model.labels_` | 属性 | 返回训练样本的簇分配结果 |
| `model.cluster_centers_` | 属性 | 返回最终学习到的簇中心坐标 |
| `model.inertia_` | 属性 | 返回簇内平方和，用于衡量紧凑度 |
| `@print_func_info` / `@timeit` | 装饰器 | 打印函数信息并统计训练耗时 |

## 1. `train_model(...)` 的函数签名

### 参数速览（本节）

适用函数：`train_model(X_train, n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` | 标准化后的特征 | 输入给 `KMeans.fit(...)` 的训练矩阵 |
| `n_clusters` | `4` | 期望聚出的簇数量 |
| `init` | `'k-means++'` | 初始中心选择策略 |
| `n_init` | `10` | 随机初始化并重复运行的次数 |
| `max_iter` | `300` | 单次运行的最大迭代次数 |
| `random_state` | `42` | 随机种子，保证结果可复现 |
| 返回值 | `KMeans` | 已训练完成的模型对象 |

### 示例代码

```python
from model_training.clustering.kmeans import train_model

model = train_model(X_scaled)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `KMeans` 模型。
- 和部分监督学习分册不同，这里没有训练集/测试集拆分，也没有多模型对比字典。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `KMeans(...)` 的实际构建方式

### 参数速览（本节）

适用 API（分项）：

1. `KMeans(...)`
2. `model.fit(X_train)`

| 项目 | 当前实现 | 说明 |
|---|---|---|
| 训练模型 | `KMeans(...)` | 使用源码中显式给出的超参数 |
| 输入特征 | `X_train` | 当前流水线传入的是标准化后的二维特征 |
| 训练方式 | `fit(X_train)` | 无监督拟合，不需要监督标签 |
| 返回值 | 已训练模型 | 含 `labels_`、`cluster_centers_`、`inertia_` |

### 示例代码

```python
model = KMeans(
    n_clusters=n_clusters,
    init=init,
    n_init=n_init,
    max_iter=max_iter,
    random_state=random_state,
)
model.fit(X_train)
```

### 理解重点

- 仓库没有自己实现 KMeans 迭代求解，而是直接调用 scikit-learn 的成熟实现。
- 当前封装的重点，不是重新发明算法，而是把超参数、训练耗时和结果日志组织清楚。
- `fit(X_train)` 只接收特征，不接收标签，这一点和监督学习的 `.fit(X, y)` 有本质差异。

## 3. 训练完成后最重要的模型属性

### 参数速览（本节）

适用属性（分项）：

1. `model.labels_`
2. `model.cluster_centers_`
3. `model.inertia_`

| 属性名 | 当前含义 | 作用 |
|---|---|---|
| `labels_` | 每个训练样本所属簇编号 | 用于绘制预测簇标签图 |
| `cluster_centers_` | 每个簇最终中心坐标 | 用于显示红色中心点 |
| `inertia_` | 簇内平方和 | 用于日志观察聚类紧凑程度 |

### 示例代码

```python
print(f"inertia: {model.inertia_:.4f}")

plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    centers=model.cluster_centers_,
)
```

### 理解重点

- `labels_` 是当前训练样本的聚类结果，因此它在本仓库里承担了“训练后簇分配输出”的角色。
- `cluster_centers_` 是 KMeans 真正学到的核心结果，相当于每个簇的代表位置。
- `inertia_` 很重要，但它只能说明簇内是否紧凑，不能单独证明聚类是否合理。

## 4. 训练阶段的工程封装

除了 `KMeans(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装。

### 参数速览（本节）

适用装饰与输出（分项）：

1. `@print_func_info`
2. `@timeit`
3. `with timer(name="模型训练耗时")`
4. 日志输出 `n_clusters` 和 `inertia`

| 输出项 | 作用 |
|---|---|
| 函数调用标题 | 帮助在终端中定位训练入口 |
| 训练耗时 | 观察当前模型拟合时间 |
| `模型训练完成` | 明确训练阶段已结束 |
| `n_clusters` | 确认当前簇数配置 |
| `inertia` | 快速查看当前簇内平方和 |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 这一层封装把“构建模型”“训练模型”“打印结果”收在一个函数里，方便文档和流水线复用。
- 从工程角度看，这样的拆分也让 `pipelines/clustering/kmeans.py` 保持简洁。

## 常见坑

1. 误以为 `train_model(...)` 需要传入 `y_train`。
2. 只关注 `labels_`，却忽略 `cluster_centers_` 和 `inertia_` 才是理解 KMeans 的关键属性。
3. 把 `inertia_` 当成越小越绝对正确，而不结合可视化与数据形状一起判断。
4. 忘记当前 `X_train` 应该是标准化后的特征。

## 小结

- `train_model(...)` 是本仓库 KMeans 的核心训练入口。
- 它本质上是对 `sklearn.cluster.KMeans` 的薄封装，重点在于把超参数、训练结果和日志输出组织清楚。
- 读懂这一层之后，再看流水线中的数据准备、可视化和评估过程会更顺畅。
