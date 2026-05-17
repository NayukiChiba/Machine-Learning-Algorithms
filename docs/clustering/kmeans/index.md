---
title: KMeans K 均值聚类 — 总览
outline: deep
---

# KMeans K 均值聚类

## 本章目标

1. 明确本分册对应的 KMeans 源码入口与运行方式。
2. 理解当前 KMeans 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到可视化评估的整体阅读路线——注意这是无监督聚类，`true_label` 仅用于结果对照。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/clustering.py` | `ClusteringData.kmeans()` 生成球形 blob 聚类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `kmeans_data` |
| 训练封装 | `model_training/clustering/kmeans.py` | `train_model(...)` 封装 `sklearn.cluster.KMeans` 训练 |
| 端到端流水线 | `pipelines/clustering/kmeans.py` | 完成数据拆分、标准化、训练与聚类结果可视化 |
| 聚类结果可视化 | `result_visualization/cluster_plot.py` | 绘制预测簇标签、真实标签和聚类中心对照图 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)` |
| 数据来源 | `make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)` |
| 特征预处理 | `StandardScaler().fit_transform(X)`——使各特征同等贡献于到中心的距离计算 |
| 训练方式 | `model.fit(X_scaled)`——无监督，不传入标签 |
| 评估呈现 | 聚类散点图（含聚类中心标记）+ `true_label` 对照 + `inertia_` 日志 |

## 阅读路线

1. [数学原理](/clustering/kmeans/01-mathematics)
2. [数据构成](/clustering/kmeans/02-data)
3. [思路与直觉](/clustering/kmeans/03-intuition)
4. [模型构建](/clustering/kmeans/04-model)
5. [训练与预测](/clustering/kmeans/05-training-and-prediction)
6. [评估与诊断](/clustering/kmeans/06-evaluation)
7. [工程实现](/clustering/kmeans/07-implementation)
8. [练习与参考文献](/clustering/kmeans/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.clustering.kmeans
```

### 理解重点

- 这个命令会串起当前 KMeans 分册中最核心的工程流程。
- 运行后会训练一个 KMeans 模型（迭代优化质心位置以最小化簇内平方和），并输出含聚类中心标记的对照散点图。
- 当前流程是无监督聚类——`true_label` 仅用于结果对照，不参与 `fit()`。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [项目架构](/appendix/)

## 小结

- 本分册严格对应当前仓库中的 KMeans 源码实现。
- KMeans 的核心特点：中心式聚类 + 需预设 $k$ + 迭代优化簇内平方和 + 偏好各向同性球形簇——与 DBSCAN（密度聚类、无需预设 $k$、能发现任意形状簇、天然识别噪声点）在建模思路上有本质区别。
- 当前使用 `make_blobs` 构造的球形高斯簇数据 + `KMeans(n_clusters=4, init='k-means++')`，是展示中心式聚类最经典的教学配置。
