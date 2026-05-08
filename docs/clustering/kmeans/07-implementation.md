---
title: KMeans K 均值聚类 — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 从工程角度看清 KMeans 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和聚类可视化分别负责什么。
3. 理解 KMeans 工程实现与 DBSCAN 在架构上的关键差异——有 `cluster_centers_`、有 `inertia_`、有 `predict()`。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/clustering.py` | `ClusteringData.kmeans()` 生成球形 blob 聚类数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `kmeans_data` |
| 训练封装 | `model_training/clustering/kmeans.py` | 构建并训练 `KMeans`，打印 `inertia_` 日志 |
| 流水线入口 | `pipelines/clustering/kmeans.py` | 组织数据拆分、标准化、训练与聚类可视化 |
| 聚类结果可视化 | `result_visualization/cluster_plot.py` | 绘制预测簇标签与真实标签的左右对照散点图（含质心标记） |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.clustering.kmeans
```

### 理解重点

- 这个命令串起当前 KMeans 分册中最核心的工程流程。
- 依次完成：数据复制 → 剥离 `true_label` → 全量标准化 → KMeans `fit()`（分配-更新交替迭代）→ `inertia_` 日志 → 对照散点图（含质心标记）。
- 对大多数读者来说，`pipelines/clustering/kmeans.py` 是理解工程实现的最佳起点——代码量少、流程清晰。

## 2. `run()` 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征与对照标签
    data = kmeans_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])

    # 2. 全量标准化——无切分（无监督聚类不需要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. KMeans 训练——fit() 执行分配-更新交替迭代
    model = train_model(X_scaled)

    # 4. 单一可视化（左右对照散点图 + 质心标记）
    plot_clusters(
        X_scaled,
        labels_pred=model.labels_,
        labels_true=y_true,
        centers=model.cluster_centers_,
        title="KMeans 聚类分布",
        dataset_name=DATASET,
        model_name=MODEL,
    )
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——真正的分配-更新迭代在 `KMeans.fit()` 中。
- 数据流是单向的：数据 → 标准化 → KMeans 迭代优化 → `labels_` + `cluster_centers_` + `inertia_` → 对照散点图（含质心标记）。
- 与分类流水线的核心差异：
  - **无 `train_test_split`**——无监督聚类不划分训练/测试集
  - **无 `predict()` 调用**——流水线直接使用 `model.labels_`（虽然 KMeans 支持 `predict()`，但教学流水线不演示）
  - **无 `predict_proba`**——KMeans 不产生概率
  - **单一可视化**（`plot_clusters`）而非四类（混淆矩阵+ROC+决策边界+学习曲线）
- 与 DBSCAN 流水线的差异：
  - `plot_clusters` 传入了 `centers=model.cluster_centers_`——DBSCAN 不传此参数
  - 训练日志打印 `inertia_`——DBSCAN 打印 `n_clusters` 和 `n_noise`

## 3. 训练模块负责什么

`model_training/clustering/kmeans.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)` 实例
2. 调用 `model.fit(X_train)`——分配-更新交替迭代（仅传特征，不传标签）
3. 打印 `n_clusters` 和 `inertia_` 日志
4. 返回训练完成的模型对象

### 参数速览

适用函数：`train_model(X_train, n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的全量特征矩阵，传入 `KMeans.fit()` | `X_scaled` |
| `n_clusters` | `int` | 预设簇数 $K$。默认 `4`，与 `make_blobs(centers=4)` 一致 | `3`、`4`、`5` |
| `init` | `str` | 质心初始化策略。`'k-means++'` 加权随机采样 | `'k-means++'`、`'random'` |
| `n_init` | `int` | 不同初始质心下独立运行的次数，返回 `inertia_` 最小的结果。默认 `10` | `1`、`10`、`20` |
| `max_iter` | `int` | 单次运行的最大迭代次数。默认 `300` | `100`、`300` |
| `random_state` | `int` | 随机种子，保证可复现。默认 `42` | `42` |
| 返回值 | `KMeans` | 已完成 `fit()` 的模型对象，含 `cluster_centers_`、`labels_`、`inertia_` | — |

### 理解重点

- KMeans 的 `fit()` 是无监督的——不接收标签参数。这与分类分册中所有 `train_model` 都有 `y_train` 参数形成鲜明对比。
- 训练日志中的 `inertia_` 是 KMeans 独有的统计输出——它直接反映聚类紧密度，DBSCAN 没有对应物。
- `@print_func_info` 和 `@timeit` 装饰器提供函数标题和耗时——增强了教学型仓库的可读性。

## 4. 可视化模块负责什么

### 模块职责

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 聚类对照图 | `plot_clusters(...)` | `X_scaled`、`labels_pred`（`model.labels_`）、`labels_true`（`y_true`）、`centers`（`model.cluster_centers_`） | 左右对照散点图（PNG），含红色 `X` 质心标记 |

### 理解重点

- `plot_clusters(...)` 是当前 KMeans 流水线中**唯一**的可视化模块——与分类分册的 4 种评估函数形成鲜明对比。
- `centers` 参数是 KMeans 调用的特有参数——红色 `X` 标记直观展示每个簇的中心位置。DBSCAN 调用时不传此参数。
- 左右对照布局：左侧为 KMeans 聚类结果（含质心标记），右侧为真实标签——这种设计在教学上非常直观。
- 不涉及 PCA 降维——原始数据本身就是二维的，可以直接用作散点图坐标。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `kmeans_data` | `data_generation/clustering.py` | `pipelines/clustering/kmeans.py` |
| `y_true` | `data["true_label"]` 提取 | `plot_clusters`（仅对照用） |
| `X_scaled` | `StandardScaler` | `train_model`、`plot_clusters` |
| `model`（含 `labels_`、`cluster_centers_`、`inertia_`） | `train_model(...)` | `plot_clusters`、终端日志 |
| 图片产物 | `plot_clusters(...)` | `outputs/kmeans/` 目录 |

### 理解重点

- 数据依赖关系极为简洁——只有 5 个节点，单向流动无循环依赖。
- 比分类流水线少了 `train_test_split`、`predict`、`predict_proba`、PCA、ROC 评估、学习曲线等 6+ 个节点。
- `cluster_centers_` 的流向是 KMeans 数据流独有的——它从 `train_model` 产出，流入 `plot_clusters` 作为红色 `X` 标记。
- `y_true` 的流向是单向的——从数据到可视化，不经过模型训练。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `KMeans 聚类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、`n_clusters`、`inertia_`（4 位小数） | 查看迭代优化耗时和聚类紧密度 |
| 聚类对照图 | `outputs/kmeans/cluster_plot.png` | 左右对照：KMeans 聚类结果（含红色 `X` 质心） vs 真实 4 簇标签 |

### 理解重点

- 输出比分类分册少得多——只有 2 类（日志 + 1 张图），而非 5 类（日志 + 4 张图）。
- `inertia_` 是 KMeans 独有的日志输出——它在 DBSCAN 的训练日志中不存在（DBSCAN 打印 `n_clusters` 和 `n_noise`）。
- 聚类对照图（含质心 `X` 标记）是最核心的教学产出——它直接展示了中心式聚类的效果和与真实结构的吻合度。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/clustering/kmeans.py` — 入口，代码量少，流程清晰
2. 再看 `model_training/clustering/kmeans.py` — 训练封装，理解无监督 `fit()` 和 `inertia_` 日志输出
3. 再看 `result_visualization/cluster_plot.py` — 聚类对照散点图绘制逻辑（含 `centers` 参数处理）
4. 最后回到 `data_generation/clustering.py` — 理解 `make_blobs` 球形高斯簇数据生成

### 理解重点

- 从入口看整体流程，再下钻到训练和可视化细节，阅读成本最低。
- KMeans 的调用链比分类分册短得多——这本身就是无监督聚类简洁性的体现。

## 运行结果

![运行结果展示](../../../outputs/kmeans/result_display.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的分配-更新迭代在 `KMeans.fit()` 中。
2. 期待当前流水线有 `train_test_split`——无监督聚类不需要。
3. 忽略 `inertia_` 的日志输出——它是理解聚类紧密度的最直接依据。
4. 把 `true_label` 当成参与训练的数据流——它的流向是"数据 → 可视化"，从未进入模型。
5. 忘记 `centers` 参数是 KMeans 调用 `plot_clusters` 的特有参数——DBSCAN 不传此参数。

## 小结

- 当前 KMeans 工程实现采用极简的模块分层：数据生成 → 训练封装（无监督）→ 流水线编排 → 单一可视化（对照散点图 + 质心标记）。
- `run()` 负责串联，`train_model(...)` 负责交替迭代优化（仅 `fit(X)`），`plot_clusters(...)` 负责视觉对照（含 `centers` 参数）。
- KMeans 在工程上最不同于 DBSCAN 的地方：有 `cluster_centers_`（传入 `plot_clusters` 的 `centers`）、有 `inertia_`（训练日志输出）、有 `predict()`（虽然流水线未演示）、无噪声点概念。
- KMeans 在工程上最不同于分类算法的地方：无切分、无监督 `fit()`、单一可视化（对照散点图）——这是由无监督聚类本质决定的。
