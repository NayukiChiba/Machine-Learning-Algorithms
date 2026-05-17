---
title: EM 与 GMM — 总览
outline: deep
---

# EM 与 GMM

## 本章目标

1. 明确本分册对应的 EM（GMM）源码入口与运行方式——注意这是无监督聚类，与集成学习的分类/回归任务不同。
2. 理解当前 EM 文档各章节分别负责解释什么内容。
3. 建立从概率模型、EM 迭代、参数估计到聚类评估的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/probabilistic.py` | `ProbabilisticData.em()` 手动合成 3 分量高斯混合数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `em_data` |
| 训练封装 | `model_training/probabilistic/em.py` | `train_model(...)` 封装 `sklearn.mixture.GaussianMixture` 训练 |
| 端到端流水线 | `pipelines/probabilistic/em.py` | 完成标准化、EM 训练、聚类预测和可视化评估 |
| 聚类可视化 | `result_visualization/cluster_plot.py` | 绘制聚类分布图（预测标签 vs 真实标签双面板对比） |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `GaussianMixture(n_components=3, covariance_type="full", max_iter=200, random_state=42)` |
| 数据来源 | 手动合成 GMM 数据——3 个分量，均值 $\{[0,0], [4,4], [-3,4]\}$，标准差不同，权重 $[0.5, 0.3, 0.2]$ |
| 特征预处理 | `StandardScaler().fit_transform(X)`——全量数据标准化（无训练/测试切分） |
| 数据拆分 | **无**——聚类评估在全量数据上进行 |
| 评估呈现 | 聚类分布图（预测标签 + 真实标签对比）+ 对数似然下界日志 |

## EM 与本仓库其他算法的定位对比

| 配置项 | KMeans | DBSCAN | EM (GMM) |
|---|---|---|---|
| 任务类型 | 聚类 | 聚类 | 聚类 |
| 算法范式 | 质心迭代 | 密度连接 | **概率生成模型** |
| 赋值方式 | 硬赋值（每个点一个簇） | 硬赋值 + 噪声点 | **软赋值（每个点对每个分量有归属概率）** |
| 簇形状 | 球形 | 任意 | **椭圆形（full covariance）** |
| 训练输入 | 有 `y` — `true_label` 仅用于评估 | 有 `y` — `true_label` 仅用于评估 | 有 `y` — `true_label` 仅用于评估 |
| 标准化 | 有 | 有 | 有 |
| 评估 | 聚类图 + inertia_ | 聚类图 + 噪声点分析 | 聚类图 + log-likelihood |
| 核心输出 | `labels_`、`cluster_centers_` | `labels_` | **`predict_proba()`（软赋值）、`means_`、`covariances_`** |

## 阅读路线

1. [数学原理](/probabilistic/em/01-mathematics)
2. [数据构成](/probabilistic/em/02-data)
3. [思路与直觉](/probabilistic/em/03-intuition)
4. [模型构建](/probabilistic/em/04-model)
5. [训练与预测](/probabilistic/em/05-training-and-prediction)
6. [评估与诊断](/probabilistic/em/06-evaluation)
7. [工程实现](/probabilistic/em/07-implementation)
8. [练习与参考文献](/probabilistic/em/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.probabilistic.em
```

### 理解重点

- 这个命令会运行 GMM 的 EM 算法训练——拟合一个 3 分量全协方差高斯混合模型。
- 当前流程是**无监督聚类**——`true_label` 仅在评估时用于对比真实分量归属，**不参与模型训练**。
- EM 算法的输出是软聚类（每个样本对每个分量有一个概率归属）——这是与 KMeans 硬聚类的根本区别。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [KMeans 聚类](/clustering/kmeans/)
- [DBSCAN 聚类](/clustering/dbscan/)
- [项目架构](/appendix/)

## 小结

- 本分册严格对应当前仓库中的 EM（GMM）源码实现。
- EM 的核心特点：概率生成模型 + 软赋值 + E 步（计算期望）+ M 步（最大化参数）+ 全协方差椭圆形簇——与 KMeans 的硬赋值球形簇形成根本差异。
- 当前使用手动合成的 3 分量非球形 GMM 数据 + `GaussianMixture(covariance_type="full")`，是展示 GMM 对椭圆形簇建模能力最经典的教学配置。
