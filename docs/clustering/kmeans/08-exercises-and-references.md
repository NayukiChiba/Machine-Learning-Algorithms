---
title: KMeans K 均值聚类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`docs/clustering/kmeans/`、`data_generation/clustering.py`、`model_training/clustering/kmeans.py`

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 KMeans 实现。
2. 给出继续深入阅读 KMeans 与相关数据生成工具的可靠入口。

## 自检题

1. 为什么 `pipelines/clustering/kmeans.py` 在训练前要先删除 `true_label` 列？
2. 为什么 KMeans 往往需要先做标准化？如果不做，结果可能受到什么影响？
3. 当前 `train_model(...)` 中的 `n_clusters`、`n_init`、`max_iter` 分别控制什么？
4. 为什么 `inertia_` 变小，不一定就表示聚类结果一定更合理？
5. 在当前仓库里，为什么 `make_blobs(...)` 更适合 KMeans，而 `make_moons(...)` 更适合 DBSCAN？
6. 为什么 `model.labels_` 和 `true_label` 的编号不需要严格一一对应？

## 练习方向

### 1. 改动 `n_clusters`

- 把 `n_clusters=4` 改成 `3` 或 `5`
- 观察 `inertia_` 与聚类分布图的变化
- 思考“数值变小”和“结构更合理”是否总是同一件事

### 2. 改动数据分布

- 调整 `ClusteringData.kmeans()` 中的 `kmeans_cluster_std`
- 观察簇内离散程度变大后，中心位置与簇边界会如何变化

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`
- 对比绘图结果，体会距离型算法对特征尺度的敏感性

### 4. 与 DBSCAN 对比

- 对照阅读 `docs/clustering/dbscan/`
- 比较两种算法对数据形状假设、超参数设置和结果解释方式的不同

## 参考文献

1. scikit-learn 官方文档：`KMeans`
   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
2. scikit-learn 官方文档：`make_blobs`
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
3. scikit-learn 用户指南：Clustering
   https://scikit-learn.org/stable/modules/clustering.html
4. Arthur, D. and Vassilvitskii, S. (2007).
   *k-means++: The Advantages of Careful Seeding*.

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释 `true_label` 的角色、标准化的必要性以及 `inertia_` 的边界，说明已经掌握了当前 KMeans 分册的核心内容。
