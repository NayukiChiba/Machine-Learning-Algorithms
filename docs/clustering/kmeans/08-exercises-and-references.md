---
title: KMeans K 均值聚类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 KMeans 实现。
2. 给出继续深入阅读 K 均值聚类与相关数据工具的可靠入口。

## 自检题

1. 为什么 KMeans 流水线没有 `train_test_split`？无监督聚类为什么不需要训练/测试切分？
2. 为什么 `n_clusters` 必须在训练前预设？如果把 `n_clusters=4` 改成 `3` 或 `5`，聚类结果会发生什么变化？
3. `k-means++` 初始化与纯随机初始化有什么本质区别？为什么 `k-means++` 能显著减少不良局部最优的风险？
4. 为什么 `inertia_` 随 $K$ 增大单调递减？为什么不能直接用 `inertia_` 选择最优 $K$？肘部法则的原理是什么？
5. 为什么 KMeans 在 `make_blobs` 球形高斯簇数据上表现极好，但在 `make_moons` 弯月数据上会失败？Voronoi 划分与数据形态不匹配的后果是什么？
6. 为什么标准化对 KMeans 是硬性要求？如果去掉标准化，欧氏距离计算会被哪个特征主导？
7. 为什么 KMeans 的评估不包含混淆矩阵、ROC 曲线和准确率？聚类评估与分类评估的根本差异是什么？

## 练习方向

### 1. 改变 `n_clusters`

- 把 `n_clusters=4` 改成 `2`、`3`、`4`、`5`、`8`
- 观察变化：
  - $K=2$——哪些真实簇被合并了？Voronoi 边界如何错误地穿过真实簇？
  - $K=5$——哪个真实簇被错误分裂了？新增的质心落在什么位置？
  - $K=8$——簇过度分裂的视觉特征是什么？`inertia_` 是否显著下降？
  - 聚类对照图——左右两侧的视觉差异是否随 $K$ 偏离真实值而急剧增大？
- 核心理解：$K$ 是 KMeans 最重要的参数——选错 $K$ 意味着预设的分组方式与数据真实结构不匹配

### 2. 改变 `init` 初始化策略

- 把 `init='k-means++'` 改成 `init='random'`
- 同时可以去掉 `random_state` 固定，多次运行观察结果的波动
- 观察变化：
  - `inertia_`——`random` 初始化是否经常得到更高的 `inertia_`（更差的局部最优）？
  - `n_iter_`——`random` 初始化是否需要更多迭代才能收敛？
  - 质心位置——是否有时出现两个质心挤在同一真实簇内的情况？
- 核心理解：`k-means++` 是 KMeans 从"频繁得到差结果"到"实践中稳定可靠"的关键改进

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用原始 `X` 训练
- 观察聚类结果的变化——当 `x1` 和 `x2` 的数值范围不同时，聚类是否被尺度更大的特征主导？
- 对比：原始数据坐标与标准化数据坐标的实际数值范围差异
- 体会：标准化后各特征平等贡献于距离计算——质心的位置和簇的边界在几何上才有意义

### 4. 改变 `cluster_std`

- 修改 `make_blobs(cluster_std=...)` 的 `cluster_std` 参数（`0.3`、`0.8`、`1.5`、`2.5`）
- 观察变化：
  - 低标准差（`0.3`）——簇极度紧凑，聚类几乎完美，`inertia_` 极小
  - 高标准差（`2.5`）——簇间边界模糊，部分点跨越 Voronoi 边界被错误分配
  - `inertia_` 随 `cluster_std` 增大而增大的趋势——簇越松散，紧密度越差
- 核心理解：KMeans 假设簇内方差相近——当 `cluster_std` 过大导致簇间重叠时，Voronoi 边界不再准确

### 5. 对比 DBSCAN

- 用 DBSCAN（`eps=0.3, min_samples=5`）在同一 `make_blobs` 数据上聚类，观察密度聚类在球形数据上的表现
- 用 KMeans（`n_clusters=2`）在 DBSCAN 的 `make_moons` 弯月数据上聚类，观察 Voronoi 直线边界如何沿月牙弧形错误切分
- 对比变化：
  - KMeans 在 `make_blobs` 上更简洁高效——这是它的理想数据
  - DBSCAN 在 `make_moons` 上能沿弯月密度扩展——这是 KMeans 做不到的
  - 两种算法的输出属性差异——KMeans 有 `cluster_centers_` + `inertia_`，DBSCAN 有噪声标签 $-1$
- 核心理解：聚类算法没有万能方案——算法选择必须匹配数据的结构特征。KMeans 和 DBSCAN 不是"谁更强"，而是分别适合形状截然不同的数据

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`KMeans` | 完整构造器参数（`n_clusters`、`init`、`n_init`、`max_iter`、`tol`、`algorithm`、`random_state`）、属性（`cluster_centers_`、`labels_`、`inertia_`、`n_iter_`）与方法（`fit`、`predict`、`fit_predict`、`fit_transform`、`transform`）说明 |
| 2 | scikit-learn 官方文档：`make_blobs` | 各向同性高斯簇数据生成器的 `n_samples`、`centers`、`cluster_std`、`n_features`、`shuffle`、`random_state` 等参数说明 |
| 3 | scikit-learn 用户指南：Clustering → K-means | KMeans 算法原理、`k-means++` 初始化、肘部法则选 $K$、不同数据形态上的局限性与与其他聚类算法的使用场景对比 |
| 4 | Arthur, D. and Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding*. SODA 2007. | k-means++ 原始论文——加权随机采样初始化策略的理论分析、近似比证明（$\Theta(\log K)$ 竞争比）和实验验证 |

- scikit-learn `KMeans`：https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- scikit-learn `make_blobs`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
- scikit-learn 用户指南 Clustering：https://scikit-learn.org/stable/modules/clustering.html#k-means

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 KMeans 分册的核心内容：
  - 无监督聚类不需要训练/测试切分——`fit()` 在全量数据上执行，`labels_` 直接输出
  - $K$ 必须预设——这是 KMeans 最刚性的约束，选错 $K$ 意味着强制拆分或合并真实簇
  - `k-means++` 通过加权随机采样使初始质心分散——显著降低收敛到不良局部最优的概率
  - `inertia_` 随 $K$ 单调递减——不能直接用于选 $K$，需配合肘部法则
  - KMeans 偏好球形簇——Voronoi 划分的直线边界在弯月等非凸结构上必然出错
  - 标准化对基于欧氏距离的 KMeans 是硬性要求——距离计算不能被特征量纲绑架
  - KMeans 的 `fit()` 不接收标签——与分类分册的 `fit(X, y)` 有本质区别
  - KMeans 有 `predict()`——新样本只需找最近质心，DBSCAN 没有这个能力
  - `cluster_centers_` 是 KMeans 区别于 DBSCAN 的标志性属性——中心式聚类有显式质心
  - 聚类评估不同于分类评估——没有 accuracy/混淆矩阵/ROC，依赖散点图视觉对照和 `inertia_` 定量诊断
  - KMeans 与 DBSCAN 不是"谁更好"的关系——球形数据选 KMeans，弯曲不规则数据选 DBSCAN
