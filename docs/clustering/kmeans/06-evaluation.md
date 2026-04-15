---
title: KMeans K 均值聚类 — 评估与诊断
outline: deep
---

# 评估与诊断

> 对应代码：`pipelines/clustering/kmeans.py`、`model_training/clustering/kmeans.py`、`result_visualization/cluster_plot.py`
>
> 相关对象：`model.inertia_`、`plot_clusters(...)`

## 本章目标

1. 明确当前仓库 KMeans 实现实际上是如何做结果诊断的。
2. 理解 `inertia_` 能说明什么，不能说明什么。
3. 理解为什么当前分册把可视化对照作为主要评估入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `model.inertia_` | 属性 | 反映簇内平方和，衡量簇内紧凑程度 |
| `plot_clusters(...)` | 函数 | 绘制预测标签图和真实标签图 |
| `labels_pred=model.labels_` | 参数 | 提供预测簇标签 |
| `labels_true=y_true` | 参数 | 提供真实标签用于对照 |
| `centers=model.cluster_centers_` | 参数 | 提供聚类中心用于标注 |

## 1. 当前仓库的评估入口

当前 KMeans 流水线里的主要结果诊断手段有两个：

1. 终端日志里打印 `inertia_`
2. 调用 `plot_clusters(...)` 绘制聚类分布图

### 示例代码

```python
print(f"inertia: {model.inertia_:.4f}")

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

- 当前实现没有引入复杂的聚类指标面板，而是优先提供最直观的数值与图像反馈。
- 对教学型仓库来说，这样的设计非常务实：读者可以先看簇是否分开，再理解指标意味着什么。
- 由于数据本身是二维的，图像对照在这里尤其有效。

## 2. `inertia_` 表示什么

### 参数速览（本节）

适用属性：`model.inertia_`

| 属性名 | 当前含义 | 说明 |
|---|---|---|
| `inertia_` | 簇内平方和 | 所有样本到所属簇中心的平方距离总和 |

### 理解重点

- `inertia_` 越小，说明样本总体上离各自簇中心越近。
- 在同一数据集上比较不同 `K` 或不同初始化结果时，它可以作为一个有用参考。
- 但它不能单独回答“这个聚类是否符合真实结构”这个问题。

## 3. `inertia_` 不能单独证明聚类合理

### 理解重点

- 随着 `n_clusters` 增大，`inertia_` 往往会继续下降，因此它天然偏向更多的簇。
- 即使 `inertia_` 很小，也不代表聚类一定有好的业务意义。
- 对当前仓库来说，`inertia_` 更适合被当作“紧凑度日志”，而不是最终裁判。

## 4. 可视化对照图能观察什么

`plot_clusters(...)` 会生成一张包含两幅子图的图片：

- 左图：预测簇标签
- 右图：真实标签

如果传入 `centers=model.cluster_centers_`，左图中还会显示红色中心点。

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    centers=model.cluster_centers_,
)
```

### 理解重点

- 这种对照图最适合回答一个直观问题：模型分出来的簇，是否和数据生成时的真实结构大致一致。
- 观察重点不在“颜色编号是否一一对应”，而在簇的边界和整体分布是否接近。
- 因为聚类标签编号本身没有固定语义，所以 `0` 不一定要对应真实标签的 `0`。

## 5. 当前实现中尚未纳入的量化指标

在一般聚类任务中，还常见以下指标：

- 轮廓系数（Silhouette Score）
- 调整兰德指数（ARI）
- 归一化互信息（NMI）

### 理解重点

- 当前仓库并没有在 KMeans 流水线中实现这些指标。
- 文档可以提到它们是常见扩展方向，但不能写成“当前源码已经在计算”。
- 现阶段最准确的表述是：当前实现以 `inertia_` 和二维可视化为主要诊断手段。

## 常见坑

1. 只看 `inertia_`，不看图像分布。
2. 看到预测标签编号和真实标签编号不一致，就误判模型失败。
3. 把当前仓库未实现的 ARI、NMI、轮廓系数写成现有流程的一部分。
4. 忽略 `inertia_` 与 `n_clusters` 之间天然相关这一事实。

## 小结

- 当前仓库对 KMeans 的评估方式很明确：数值上看 `inertia_`，图像上看聚类分布对照图。
- `inertia_` 能帮助判断簇是否紧凑，但不能单独判定聚类是否合理。
- 对当前二维教学数据而言，可视化是最直观、也最符合本仓库目标的诊断方式。
