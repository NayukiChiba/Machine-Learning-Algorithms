---
title: DBSCAN 密度聚类 — 总览
outline: deep
---

# DBSCAN 密度聚类

> 对应代码：`pipelines/clustering/dbscan.py`、`model_training/clustering/dbscan.py`
>
> 运行方式：`python -m pipelines.clustering.dbscan`

## 本章目标

1. 明确本分册对应的 DBSCAN 源码入口与运行方式。
2. 理解当前 DBSCAN 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到可视化评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/clustering.py` | `ClusteringData.dbscan()` 生成双月牙聚类数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `dbscan_data` |
| 训练封装 | `model_training/clustering/dbscan.py` | `train_model(...)` 封装 `sklearn.cluster.DBSCAN` 训练 |
| 端到端流水线 | `pipelines/clustering/dbscan.py` | 完成数据拆分、标准化、训练与聚类结果可视化 |
| 聚类结果可视化 | `result_visualization/cluster_plot.py` | 绘制预测簇标签与真实标签对照图 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `DBSCAN(eps=0.3, min_samples=5, metric='euclidean')` |
| 数据来源 | `ClusteringData.dbscan()` 调用 `make_moons(...)` |
| 特征预处理 | `StandardScaler().fit_transform(X)` |
| 训练输入 | 去掉 `true_label` 后的二维特征 |
| 评估呈现 | 聚类散点图 + `true_label` 对照 + 簇数量/噪声点数量日志 |

## 阅读路线

1. [数学原理](/clustering/dbscan/01-mathematics)
2. [数据构成](/clustering/dbscan/02-data)
3. [思路与直觉](/clustering/dbscan/03-intuition)
4. [模型构建](/clustering/dbscan/04-model)
5. [训练与预测](/clustering/dbscan/05-training-and-prediction)
6. [评估与诊断](/clustering/dbscan/06-evaluation)
7. [工程实现](/clustering/dbscan/07-implementation)
8. [练习与参考文献](/clustering/dbscan/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.clustering.dbscan
```

### 理解重点

- 这个命令会串起当前 DBSCAN 分册中最核心的工程流程。
- 运行后会训练一个 DBSCAN 模型，并输出聚类分布图。
- 当前流程是无监督聚类，因此 `true_label` 仅用于结果对照，不参与拟合。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 DBSCAN 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/clustering/dbscan.py` 和 `model_training/clustering/dbscan.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“数据构成”“模型构建”或“训练与预测”章节开始阅读。
