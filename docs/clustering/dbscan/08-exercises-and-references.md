---
title: DBSCAN 密度聚类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 DBSCAN 实现。
2. 给出继续深入阅读密度聚类与相关数据工具的可靠入口。

## 自检题

1. 为什么 DBSCAN 流水线没有 `train_test_split`？无监督聚类为什么不需要训练/测试切分？
2. 为什么 `eps=0.3` 和 `min_samples=5` 在当前标准化后的双月牙数据上能正确分离两个月牙？如果 `eps` 增大到 `0.6` 会怎样？减小到 `0.1` 会怎样？
3. DBSCAN 的 `fit()` 与分类模型的 `fit(X, y)` 在参数签名和输出上有何本质差异？为什么 sklearn 的 DBSCAN 没有 `predict()` 方法？
4. 为什么 `model.labels_` 中会出现 $-1$？它与分类模型中的误分类样本有何本质区别？
5. 为什么 DBSCAN 的评估不包含混淆矩阵、ROC 曲线和准确率？聚类评估与分类评估的根本差异是什么？
6. 为什么 `true_label` 在当前流水线中不传入 `model.fit()`？它在流程中扮演什么角色？
7. 为什么标准化对 DBSCAN 是必要的？如果去掉标准化，`eps=0.3` 的几何意义会发生什么变化？

## 练习方向

### 1. 改变 `eps`

- 把 `eps=0.3` 改成 `0.1`、`0.2`、`0.3`、`0.5`、`0.7`
- 观察变化：
  - $n\_clusters$——`eps` 过大时两个月牙被错误合并为一个簇；`eps` 过小时每个高密度微区域都变成独立簇
  - $n\_noise$——`eps` 越小噪声点越多，直到几乎所有点都变成噪声
  - 聚类对照图——左右的视觉差异直接反映参数是否合理
- 核心理解：`eps` 是最需要精心调整的参数——它决定了"多大范围内的邻居算同一密度区域"

### 2. 改变 `min_samples`

- 把 `min_samples=5` 改成 `2`、`5`、`10`、`20`、`50`
- 观察变化：
  - 核心点的比例——`min_samples` 越大，越少的点满足核心条件
  - 簇的数量和噪声点数量——高 `min_samples` 时更多点被判定为噪声
  - 边界效应——对于 `min_samples=2`，几乎所有点都成为核心点
- 核心理解：`min_samples` 与 `eps` 联动——`eps` 增大时一般需要相应增大 `min_samples` 以避免合并过度

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用原始 `X` 训练
- 观察 `eps=0.3` 下的聚类结果变化
- 对比：原始数据坐标与标准化数据坐标的实际数值范围差异
- 体会：`eps` 是绝对数值——不标准化的数据让邻域判定在不同维度上含义不等同

### 4. 改变噪声水平

- 修改 `make_moons(noise=...)` 的 `noise` 参数（`0.0`、`0.05`、`0.08`、`0.15`、`0.25`）
- 观察变化：
  - 无噪声（`0.0`）——两个月牙完美分离，几乎没有噪声点
  - 高噪声（`0.25`）——两个月牙之间的间隙被部分填充，密度扩展可能跨过间隙
  - `n_noise` 随噪声增大而变化的趋势
- 核心理解：DBSCAN 的密度假设在低噪声数据上最有效，但适当噪声（`0.08`）下的表现展示了算法的鲁棒性

### 5. 对比 KMeans

- 用 `KMeans(n_clusters=2)` 在同一双月牙数据上聚类
- 对比变化：
  - 簇边界的形状——KMeans 以直线（Voronoi 边界）划分，将弯月切分；DBSCAN 沿月牙弧形密度扩展
  - 噪声处理——KMeans 强制分配每个点，DBSCAN 噪声点单独标记
  - 对非球形簇的适应性——这是两者最根本的差异
- 核心理解：聚类算法没有万能方案——算法选择必须匹配数据的结构特征

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`DBSCAN` | 完整构造器参数（`eps`、`min_samples`、`metric`、`algorithm`、`leaf_size`、`p`、`n_jobs`）、属性（`labels_`、`core_sample_indices_`）与方法说明 |
| 2 | scikit-learn 官方文档：`make_moons` | 双月牙数据生成器的 `n_samples`、`noise`、`shuffle`、`random_state` 等参数说明 |
| 3 | scikit-learn 用户指南：Clustering → DBSCAN | 密度聚类原理、`eps`/`min_samples` 选参方法、不同密度数据上的局限性与与其他聚类算法的使用场景对比 |
| 4 | Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*. KDD-96. | DBSCAN 原始论文——$\epsilon$ 邻域、核心点/边界点/噪声点概念、密度可达/密度相连关系和算法伪代码的源头 |

- scikit-learn `DBSCAN`：https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- scikit-learn `make_moons`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
- scikit-learn 用户指南 Clustering：https://scikit-learn.org/stable/modules/clustering.html#dbscan

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 DBSCAN 分册的核心内容：
  - 无监督聚类不需要训练/测试切分——`fit()` 在全量数据上执行，`labels_` 直接输出
  - `eps` 和 `min_samples` 联动决定核心点/边界点/噪声点的划分——改变其中一个通常需要调整另一个
  - DBSCAN 的 `fit()` 不接收标签——与分类分册的 `fit(X, y)` 有本质区别
  - sklearn 的 DBSCAN 没有 `predict()` 方法——它只能对训练数据本身做标记，不能预测新样本
  - 噪声点（$-1$）是 DBSCAN 的核心输出而非错误——噪声过多或过少才暗示参数不当
  - 聚类评估不同于分类评估——没有 accuracy/混淆矩阵/ROC，依赖散点图视觉对照和 `n_clusters`/`n_noise` 统计量
  - 标准化对基于距离的 `eps` 邻域判定是硬性要求——`eps` 是绝对数值，其几何意义依赖各维度尺度一致
  - DBSCAN 与 KMeans 不是"谁更好"的关系——球形数据选 KMeans，弯曲不规则数据选 DBSCAN
