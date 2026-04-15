---
title: PCA — 总览
outline: deep
---

# PCA

> 对应代码：`pipelines/dimensionality/pca.py`、`model_training/dimensionality/pca.py`
>  
> 运行方式：`python -m pipelines.dimensionality.pca`

## 本章目标

1. 明确本分册对应的 PCA 源码入口与运行方式。
2. 理解当前 PCA 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到降维可视化的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/dimensionality.py` | `DimensionalityData.pca()` 生成低秩高维数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `pca_data` |
| 训练封装 | `model_training/dimensionality/pca.py` | `train_model(...)` 封装 `sklearn.decomposition.PCA` 训练 |
| 端到端流水线 | `pipelines/dimensionality/pca.py` | 完成标准化、2D/3D PCA 训练、投影与可视化 |
| 降维可视化 | `result_visualization/dimensionality_plot.py` | 绘制 2D 或 3D 降维散点图 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `PCA(n_components=2, svd_solver='auto', random_state=42)` |
| 特征预处理 | `StandardScaler().fit_transform(X)` |
| 数据来源 | 手工构造的 10 维低秩高维数据 |
| 可视化输出 | 2D PCA 图 + 3D PCA 图 |
| 训练方式 | 全量数据直接训练，无 train/test split |

## 阅读路线

1. [数学原理](/dimensionality/pca/01-mathematics)
2. [数据构成](/dimensionality/pca/02-data)
3. [思路与直觉](/dimensionality/pca/03-intuition)
4. [模型构建](/dimensionality/pca/04-model)
5. [训练与预测](/dimensionality/pca/05-training-and-prediction)
6. [评估与诊断](/dimensionality/pca/06-evaluation)
7. [工程实现](/dimensionality/pca/07-implementation)
8. [练习与参考文献](/dimensionality/pca/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.dimensionality.pca
```

### 理解重点

- 这个命令会串起当前 PCA 分册中最核心的工程流程。
- 运行后会分别训练 2D 和 3D 两个 PCA 模型，并生成对应的降维可视化图。
- 当前实现重点在于展示解释方差比、累计解释方差和主成分投影后的结构变化。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 PCA 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/dimensionality/pca.py` 和 `model_training/dimensionality/pca.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
