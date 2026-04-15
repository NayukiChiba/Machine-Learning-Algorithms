---
title: DBSCAN 密度聚类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`docs/clustering/dbscan/`、`data_generation/clustering.py`、`model_training/clustering/dbscan.py`

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 DBSCAN 实现。
2. 给出继续深入阅读 DBSCAN 与相关数据生成工具的可靠入口。

## 自检题

1. 为什么 `pipelines/clustering/dbscan.py` 在训练前要先删除 `true_label` 列？
2. 为什么 DBSCAN 往往也需要先做标准化？如果不做，`eps` 会受到什么影响？
3. 当前 `train_model(...)` 中的 `eps`、`min_samples`、`metric` 分别控制什么？
4. 为什么噪声点数量变多，不一定就表示聚类结果一定更差？
5. 在当前仓库里，为什么 `make_moons(...)` 更适合 DBSCAN，而 `make_blobs(...)` 更适合 KMeans？
6. 为什么 `-1` 在 DBSCAN 的 `labels_` 中具有特殊含义？

## 练习方向

### 1. 改动 `eps`

- 把 `eps=0.3` 改成更小或更大的值
- 观察簇数量、噪声点数量和聚类分布图的变化
- 思考“分得更细”和“分得更合理”是否总是同一件事

### 2. 改动 `min_samples`

- 调整 `min_samples`
- 观察算法对核心点判定和噪声点数量的影响

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`
- 对比绘图结果，体会密度型算法对距离尺度的敏感性

### 4. 与 KMeans 对比

- 对照阅读 `docs/clustering/kmeans/`
- 比较两种算法对数据形状假设、超参数设置和结果解释方式的不同

## 参考文献

1. scikit-learn 官方文档：`DBSCAN`
   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
2. scikit-learn 官方文档：`make_moons`
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
3. scikit-learn 用户指南：Clustering
   https://scikit-learn.org/stable/modules/clustering.html
4. Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996).
   *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*.

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释 `true_label` 的角色、标准化的必要性、`-1` 的含义以及簇数量/噪声点数量的边界，说明已经掌握了当前 DBSCAN 分册的核心内容。
