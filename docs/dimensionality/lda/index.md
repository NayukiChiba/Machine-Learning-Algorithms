---
title: LDA 线性判别分析 — 总览
outline: deep
---

# LDA 线性判别分析

## 本章目标

1. 明确本分册对应的 LDA 源码入口与运行方式。
2. 理解当前 LDA 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到降维可视化的整体阅读路线——注意这是有监督降维，`label` 参与训练而非仅用于着色。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/dimensionality.py` | `DimensionalityData.lda()` 加载 Wine 真实数据集 |
| 数据导出 | `data_generation/__init__.py` | 导出 `lda_data` |
| 训练封装 | `model_training/dimensionality/lda.py` | `train_model(...)` 封装 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 训练 |
| 端到端流水线 | `pipelines/dimensionality/lda.py` | 完成标准化、LDA 训练、投影和 2D 判别可视化 |
| 降维可视化 | `result_visualization/dimensionality_plot.py` | 绘制降维后的 2D 散点图（按类别着色） |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `LinearDiscriminantAnalysis(n_components=2, solver='svd')` |
| 数据来源 | `load_wine(as_frame=True)`，标签列重命名为 `label`，178 样本 × 13 特征 × 3 类别 |
| 特征预处理 | `StandardScaler().fit_transform(X)`——使各特征同等贡献于散度矩阵计算 |
| 训练方式 | `model.fit(X_scaled, y)`——有监督，标签参与判别方向学习 |
| 投影方式 | `model.transform(X_scaled)`——将 13 维特征投影到 2 维判别子空间 |
| 评估呈现 | 2D 判别散点图（按类别着色）+ `explained_variance_ratio_` 日志（若求解器支持） |

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
- 运行后会训练一个 2D LDA 模型（学习最大化类间/类内散度比的判别方向），并生成按类别着色的判别投影图。
- 当前流程是有监督降维——`label` 既参与训练（定义类间/类内散度），也用于可视化着色。这与 PCA 无监督降维有本质区别。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 LDA 源码实现。
- LDA 的核心特点：有监督降维 + 最大化类间/类内散度比 + 判别方向最多 $K-1$ 个 + 广义特征值求解——与 PCA（无监督、最大化方差、维数无类别限制）在建模目标上有本质区别。
- 当前使用 Wine 真实数据集（3 类 13 特征）+ `LinearDiscriminantAnalysis(n_components=2, solver='svd')`，是展示有监督判别式降维最经典的教学配置。
