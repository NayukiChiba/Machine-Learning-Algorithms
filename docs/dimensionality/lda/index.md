---
title: LDA — 总览
outline: deep
---

# LDA

> 对应代码：`pipelines/dimensionality/lda.py`、`model_training/dimensionality/lda.py`
>  
> 运行方式：`python -m pipelines.dimensionality.lda`

## 本章目标

1. 明确本分册对应的 LDA 源码入口与运行方式。
2. 理解当前 LDA 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到降维可视化的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/dimensionality.py` | `DimensionalityData.lda()` 加载 Wine 真实数据集 |
| 数据导出 | `data_generation/__init__.py` | 导出 `lda_data` |
| 训练封装 | `model_training/dimensionality/lda.py` | `train_model(...)` 封装 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 训练 |
| 端到端流水线 | `pipelines/dimensionality/lda.py` | 完成标准化、LDA 训练、投影和 2D 可视化 |
| 降维可视化 | `result_visualization/dimensionality_plot.py` | 绘制降维后的 2D 散点图 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `LinearDiscriminantAnalysis(n_components=2, solver='svd')` |
| 特征预处理 | `StandardScaler().fit_transform(X)` |
| 数据来源 | `load_wine(as_frame=True)`，标签列重命名为 `label` |
| 可视化输出 | 2D LDA 图 |
| 训练方式 | 全量数据直接训练，无 train/test split |

## 阅读路线

1. [数学原理](/dimensionality/lda/01-mathematics)
2. [数据构成](/dimensionality/lda/02-data)
3. [思路与直觉](/dimensionality/lda/03-intuition)
4. [模型构建](/dimensionality/lda/04-model)
5. [训练与预测](/dimensionality/lda/05-training-and-prediction)
6. [评估与诊断](/dimensionality/lda/06-evaluation)
7. [工程实现](/dimensionality/lda/07-implementation)
8. [练习与参考文献](/dimensionality/lda/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.dimensionality.lda
```

### 理解重点

- 这个命令会串起当前 LDA 分册中最核心的工程流程。
- 运行后会训练一个 2D LDA 模型，并生成带类别着色的判别投影图。
- 当前实现重点在于展示类间可分性如何通过监督降维方向被强化。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 LDA 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/dimensionality/lda.py` 和 `model_training/dimensionality/lda.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
