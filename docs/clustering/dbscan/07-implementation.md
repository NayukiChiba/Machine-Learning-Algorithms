---
title: DBSCAN 密度聚类 — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 从工程角度看清 DBSCAN 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和聚类可视化分别负责什么。
3. 理解 DBSCAN 工程实现与分类算法在架构上的关键差异——无切分、无 `predict`、单一可视化。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/clustering.py` | `ClusteringData.dbscan()` 生成双月牙聚类数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `dbscan_data` |
| 训练封装 | `model_training/clustering/dbscan.py` | 构建并训练 `DBSCAN`，打印聚类统计日志 |
| 流水线入口 | `pipelines/clustering/dbscan.py` | 组织数据拆分、标准化、训练与聚类可视化 |
| 聚类结果可视化 | `result_visualization/cluster_plot.py` | 绘制预测簇标签与真实标签的左右对照散点图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.clustering.dbscan
```

### 理解重点

- 这个命令串起当前 DBSCAN 分册中最核心的工程流程。
- 依次完成：数据复制 → 剥离 `true_label` → 全量标准化 → DBSCAN `fit()`（密度扩展）→ 聚类统计 → 对照散点图。
- 对大多数读者来说，`pipelines/clustering/dbscan.py` 是理解工程实现的最佳起点——代码量少、流程清晰。

## 2. `run()` 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征与对照标签
    data = dbscan_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])

    # 2. 全量标准化——无切分（无监督聚类不需要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 密度聚类——fit() 即得到全部结果（无 predict）
    model = train_model(X_scaled)

    # 4. 单一可视化（左右对照散点图）
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

- `run()` 的职责是编排，不是算法实现——真正的密度扩展在 `DBSCAN.fit()` 中。
- 数据流是单向的：数据 → 标准化 → 密度扩展 → `labels_` → 对照散点图。
- 与分类流水线的核心差异：
  - **无 `train_test_split`**——无监督聚类不划分训练/测试集
  - **无 `predict()` 调用**——`fit()` 即输出 `labels_`
  - **无 `predict_proba`**——DBSCAN 不产生概率
  - **单一可视化**（`plot_clusters`）而非四类（混淆矩阵+ROC+决策边界+学习曲线）

## 3. 训练模块负责什么

`model_training/clustering/dbscan.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `DBSCAN(eps=0.3, min_samples=5, metric='euclidean')` 实例
2. 调用 `model.fit(X_train)`——密度聚类（仅传特征，不传标签）
3. 从 `labels_` 推导 `n_clusters` 和 `n_noise` 并打印日志
4. 返回训练完成的模型对象

### 参数速览

适用函数：`train_model(X_train, eps=0.3, min_samples=5, metric='euclidean')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的全量特征矩阵，传入 `DBSCAN.fit()` | `X_scaled` |
| `eps` | `float` | $\epsilon$ 邻域半径。默认 `0.3` | `0.2`、`0.3`、`0.5` |
| `min_samples` | `int` | 核心点阈值。默认 `5` | `3`、`5`、`10` |
| `metric` | `str` | 距离度量。默认 `'euclidean'` | `'euclidean'`、`'manhattan'` |
| 返回值 | `DBSCAN` | 已完成 `fit()` 的模型对象，含 `labels_`、`core_sample_indices_` | — |

### 理解重点

- DBSCAN 的 `fit()` 是无监督的——不接收标签参数。这与分类分册中所有 `train_model` 都有 `y_train` 参数形成鲜明对比。
- 训练日志中的 `n_clusters` 和 `n_noise` 是 DBSCAN 独有的统计输出——它们直接反映密度参数配置的合理性。

## 4. 可视化模块负责什么

### 模块职责

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 聚类对照图 | `plot_clusters(...)` | `X_scaled`、`labels_pred`（`model.labels_`）、`labels_true`（`y_true`） | 左右对照散点图（PNG） |

### 理解重点

- `plot_clusters(...)` 是当前 DBSCAN 流水线中**唯一**的可视化模块——与分类分册的 4 种评估函数形成鲜明对比。
- 左右对照布局：左侧为算法聚类结果（噪声点特殊着色），右侧为真实标签——这种设计在教学上非常直观。
- 不涉及 PCA 降维——原始数据本身就是二维的，可以直接用作散点图坐标。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `dbscan_data` | `data_generation/clustering.py` | `pipelines/clustering/dbscan.py` |
| `y_true` | `data["true_label"]` 提取 | `plot_clusters`（仅对照用） |
| `X_scaled` | `StandardScaler` | `train_model`、`plot_clusters` |
| `model`（含 `labels_`） | `train_model(...)` | `plot_clusters` |
| 图片产物 | `plot_clusters(...)` | `outputs/dbscan/` 目录 |

### 理解重点

- 数据依赖关系极为简洁——只有 5 个节点，单向流动无循环依赖。
- 比分类流水线少了 `train_test_split`、`predict`、`predict_proba`、PCA、ROC 评估、学习曲线等 6+ 个节点。
- `y_true` 的流向是单向的——从数据到可视化，不经过模型训练。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `DBSCAN 聚类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、`eps`、`min_samples`、`簇数量`、`噪声点数量` | 查看密度扩展耗时、参数配置和聚类统计量 |
| 聚类对照图 | `outputs/dbscan/cluster_plot.png` | 左右对照：DBSCAN 聚类结果 vs 真实双月牙标签 |

### 理解重点

- 输出比分类分册少得多——只有 2 类（日志 + 1 张图），而非 5 类（日志 + 4 张图）。
- `簇数量` 和 `噪声点数量` 是 DBSCAN 独有的日志输出——它们在其他算法的训练日志中不存在。
- 聚类对照图是最核心的教学产出——它直接展示了密度聚类的效果和与真实结构的吻合度。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/clustering/dbscan.py` — 入口，代码量少，流程清晰
2. 再看 `model_training/clustering/dbscan.py` — 训练封装，理解无监督 `fit()` 和日志输出
3. 再看 `result_visualization/cluster_plot.py` — 聚类对照散点图绘制逻辑
4. 最后回到 `data_generation/clustering.py` — 理解 `make_moons` 双月牙数据生成

### 理解重点

- 从入口看整体流程，再下钻到训练和可视化细节，阅读成本最低。
- DBSCAN 的调用链比分类分册短得多——这本身就是密度聚类简洁性的体现。

## 运行结果

![运行结果展示](../../../outputs/dbscan/result_display.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的密度扩展在 `DBSCAN.fit()` 中。
2. 期待当前流水线有 `train_test_split` 或 `predict()` 调用——无监督聚类不需要这些。
3. 忽略 `n_clusters` 和 `n_noise` 的日志输出——它们是理解参数配置是否合理的最直接依据。
4. 把 `true_label` 当成参与训练的数据流——它的流向是"数据 → 可视化"，从未进入模型。

## 小结

- 当前 DBSCAN 工程实现采用极简的模块分层：数据生成 → 训练封装（无监督）→ 流水线编排 → 单一可视化（对照散点图）。
- `run()` 负责串联，`train_model(...)` 负责密度扩展（仅 `fit(X)`），`plot_clusters(...)` 负责视觉对照。
- DBSCAN 在工程上最不同于分类算法的地方：无切分、无监督 `fit()`、无 `predict()`、单一可视化（对照散点图）——这是由无监督聚类和 sklearn 的 DBSCAN 实现特性共同决定的。
