---
title: DBSCAN 密度聚类 — 工程实现
outline: deep
---

# 工程实现

> 对应代码：`pipelines/clustering/dbscan.py`、`model_training/clustering/dbscan.py`、`data_generation/clustering.py`
>
> 运行方式：`python -m pipelines.clustering.dbscan`

## 本章目标

1. 从工程角度看清 DBSCAN 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和结果可视化分别负责什么。
3. 理解为什么当前实现要把训练逻辑与流水线逻辑拆开。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/clustering.py` | 生成 `dbscan_data` |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `dbscan_data` |
| 训练封装 | `model_training/clustering/dbscan.py` | 构建并训练 `DBSCAN` |
| 流水线入口 | `pipelines/clustering/dbscan.py` | 组织标准化、训练与可视化 |
| 结果可视化 | `result_visualization/cluster_plot.py` | 保存聚类分布图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.clustering.dbscan
```

### 理解重点

- 对大多数读者来说，这个命令是理解当前 DBSCAN 工程实现的最佳入口。
- 它会依次完成数据读取、特征准备、模型训练和结果绘图。
- 如果只读一个文件，建议先读 `pipelines/clustering/dbscan.py`。

## 2. `run()` 串起了整个流程

当前流水线的核心函数是：

```python
def run():
    data = dbscan_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = train_model(X_scaled)

    plot_clusters(
        X_scaled,
        labels_pred=model.labels_,
        labels_true=y_true,
        title="DBSCAN 聚类分布",
        dataset_name=DATASET,
        model_name=MODEL,
    )
```

### 理解重点

- `run()` 本身没有复杂算法，它的职责是把不同模块串起来。
- 这类文件更像“编排层”，重点是流程顺序正确、调用关系清楚。
- 文档要帮助读者看到：真正训练模型的是 `train_model(...)`，真正画图的是 `plot_clusters(...)`。

## 3. 训练模块负责什么

`model_training/clustering/dbscan.py` 里的 `train_model(...)` 主要负责：

1. 创建 `DBSCAN(...)`
2. 调用 `fit(X_train)`
3. 统计簇数量和噪声点数量
4. 打印训练日志
5. 返回训练完成的模型对象

### 理解重点

- 这层抽离让“模型训练逻辑”和“业务流程编排逻辑”分开。
- 这样写的好处是，训练函数既可以被流水线调用，也可以单独运行做局部验证。
- 这也是当前仓库多个算法分册共享的组织方式。

## 4. 可视化模块负责什么

`result_visualization/cluster_plot.py` 里的 `plot_clusters(...)` 主要负责：

- 校验输入是否为二维特征
- 绘制预测簇标签图
- 在提供 `labels_true` 时绘制真实标签对照图
- 将结果保存到图像文件

### 理解重点

- 当前 DBSCAN 文档必须明确：可视化不是训练的一部分，而是训练完成后的结果呈现步骤。
- `plot_clusters(...)` 对二维输入有限制，这正好和当前 `make_moons` 二维数据相匹配。
- DBSCAN 流水线没有传入 `centers`，因为当前模型没有簇中心这一结果对象。

## 5. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 |
|---|---|
| 终端标题 | `DBSCAN 聚类流水线` |
| 训练日志 | 训练耗时、`eps`、`min_samples`、`簇数量`、`噪声点数量` |
| 图像文件 | 保存的聚类分布图 |

### 理解重点

- 运行结果并不只是一个模型对象，还包括面向阅读者的日志和图像产物。
- 对教学仓库而言，这种“代码 + 日志 + 图像”的组合比单纯返回数值更易理解。

## 6. 推荐的源码阅读顺序

1. 先看 `pipelines/clustering/dbscan.py`
2. 再看 `model_training/clustering/dbscan.py`
3. 再看 `result_visualization/cluster_plot.py`
4. 最后回到 `data_generation/clustering.py`

### 理解重点

- 先从入口看整体流程，再下钻到训练与可视化细节，阅读成本最低。
- 如果一开始就直接看数据生成器或绘图函数，容易看见局部却看不见整体。

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体。
2. 不区分“训练模块”和“可视化模块”的职责边界。
3. 忽略 `plot_clusters(...)` 仅支持二维特征这一条件。
4. 只看单个文件，不顺着调用链理解整体执行流程。

## 小结

- 当前 DBSCAN 工程实现采用了清晰的模块分层：数据生成、训练封装、流水线编排、结果可视化。
- `run()` 负责串联流程，`train_model(...)` 负责训练，`plot_clusters(...)` 负责结果展示。
- 这种拆分方式既便于教学讲解，也便于后续为其他聚类算法复用同类结构。
