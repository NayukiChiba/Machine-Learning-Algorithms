---
title: PCA 主成分分析 — 总览
outline: deep
---

# PCA 主成分分析

## 本章目标

1. 明确本分册对应的 PCA 源码入口与运行方式。
2. 理解当前 PCA 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到降维可视化的整体阅读路线——注意这是无监督降维，`label` 仅用于着色对照。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/dimensionality.py` | `DimensionalityData.pca()` 生成低秩高维合成数据 |
| 数据导出 | `data_generation/__init__.py` | 导出 `pca_data` |
| 训练封装 | `model_training/dimensionality/pca.py` | `train_model(...)` 封装 `sklearn.decomposition.PCA` 训练 |
| 端到端流水线 | `pipelines/dimensionality/pca.py` | 完成标准化、2D/3D PCA 训练、投影和降维可视化 |
| 降维可视化 | `result_visualization/dimensionality_plot.py` | 绘制降维后的 2D/3D 散点图（按类别着色，轴标注解释占比） |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `PCA(n_components=2, svd_solver='auto', random_state=42)` |
| 数据来源 | 低秩合成数据：`base`（400×3）@ `projection`（3×10）+ 高斯噪声（$\sigma=0.5$），含伪标签 `label` |
| 特征预处理 | `StandardScaler().fit_transform(X)`——使各特征同等贡献于协方差矩阵 |
| 训练方式 | `model.fit(X_scaled)`——无监督，不传入标签。先后训练 2D 和 3D 两个独立模型 |
| 投影方式 | `model.transform(X_scaled)`——将 10 维特征投影到低维主成分空间 |
| 评估呈现 | 2D/3D 降维散点图（按伪标签着色）+ `explained_variance_ratio_` 日志（各方向 + 累计） |

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
- 运行后会先后训练 2D PCA 和 3D PCA 两个独立模型，分别生成 2D 和 3D 降维散点图。
- 当前流程是无监督降维——`label` 是数据生成时构造的伪标签，仅用于可视化着色，不参与 `fit()`。这与 LDA（有监督降维）有本质区别。
- 同时训练 2D 和 3D 两个模型的做法在所有算法分册中独一无二——旨在展示不同降维程度下的结构保留对比。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [预处理](/foundations/sklearn/02-preprocessing)
- [术语表](/appendix/glossary)

## 小结

- 本分册严格对应当前仓库中的 PCA 源码实现。
- PCA 的核心特点：无监督降维 + 最大化投影方差 + 维数无类别限制 + SVD 数值求解——与 LDA（有监督、最大化类间/类内散度比、$K-1$ 维上限）在建模目标上有本质区别。
- 当前使用低秩合成数据（3 个真实方向隐藏在 10 维特征中）+ `PCA(n_components=2)`，是展示无监督方差压缩最经典的教学配置。
