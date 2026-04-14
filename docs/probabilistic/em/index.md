---
title: EM 与 GMM — 总览
outline: deep
---

# EM 与 GMM

> 对应代码：`pipelines/probabilistic/em.py`、`model_training/probabilistic/em.py`
>  
> 运行方式：`python -m pipelines.probabilistic.em`

## 本章目标

1. 明确本分册对应的 EM / GMM 源码入口与运行方式。
2. 理解当前 EM 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到聚类可视化的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/probabilistic.py` | `ProbabilisticData.em()` 生成二维高斯混合数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `em_data` |
| 训练封装 | `model_training/probabilistic/em.py` | `train_model(...)` 封装 `sklearn.mixture.GaussianMixture` 训练 |
| 端到端流水线 | `pipelines/probabilistic/em.py` | 完成标签拆分、标准化、训练、预测和聚类分布图输出 |
| 聚类可视化 | `result_visualization/cluster_plot.py` | 绘制预测标签与真实标签对比散点图 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `GaussianMixture(n_components=3, covariance_type='full', max_iter=200, random_state=42)` |
| 数据预处理 | `StandardScaler().fit_transform(X)` |
| 数据形态 | 二维特征 `x1`、`x2` + 对比标签 `true_label` |
| 训练方式 | 全量数据直接训练，无 train/test split |
| 评估方式 | 聚类分布图（预测标签 vs 真实标签） |

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

- 这个命令会串起当前 EM / GMM 分册中最核心的工程流程。
- 运行后会生成聚类分布对比图，并在控制台打印分量数、协方差类型和 `log-likelihood` 日志。
- 当前实现重点在于展示 EM 如何在二维混合高斯数据上完成软聚类建模。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 EM / GMM 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/probabilistic/em.py` 和 `model_training/probabilistic/em.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
