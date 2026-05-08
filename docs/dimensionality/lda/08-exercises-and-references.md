---
title: LDA 线性判别分析 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 LDA 实现。
2. 给出继续深入阅读线性判别分析与相关数据工具的可靠入口。

## 自检题

1. 为什么 LDA 的 `fit()` 必须接收 `y` 参数，而 PCA 不需要？`y` 在散度矩阵 $\mathbf{S}_W$ 和 $\mathbf{S}_B$ 的构造中分别扮演什么角色？
2. 为什么 Wine 数据（$K=3$）的 LDA 最多只能降到 2 维？这个 $K-1$ 上限从 $\mathbf{S}_B$ 的秩如何推导出来？
3. `solver='svd'` 与 `solver='eigen'` 和 `solver='lsqr'` 有什么关键差异？为什么当前源码用 `hasattr` 检查 `explained_variance_ratio_` 是否存在？
4. 为什么 LDA 的 `explained_variance_ratio_` 在 Wine 数据上，当 `n_components=2` 时累计值必然为 100%？这与 PCA 的同名属性在含义上有何本质区别？
5. 为什么 LDA 2D 投影图上的类别分离效果通常优于 PCA 2D 投影图？两者的优化目标分别是什么？
6. 为什么标准化对 LDA 是硬性要求？Wine 数据中哪个特征的量纲范围最极端？不标准化会有什么后果？
7. LDA 既可以降维也可以分类——当前仓库将它定位为哪种？`transform()` 和 `predict()` 分别做什么？

## 练习方向

### 1. 改变 `solver`

- 把 `solver='svd'` 改成 `'eigen'` 和 `'lsqr'`
- 观察变化：
  - `explained_variance_ratio_`——在 `'lsqr'` 下是否消失？`'eigen'` 下的值与 `'svd'` 是否一致？
  - 2D 判别图——不同求解器下散点分布是否相同？（应基本一致，因优化目标相同）
  - 训练耗时——三种求解器的速度差异
- 核心理解：不同求解器的求解路径不同，但目标相同——`explained_variance_ratio_` 的可用性是选择求解器时需考虑的关键工程因素

### 2. 改变 `n_components`

- 把 `n_components=2` 改成 `1`
- 观察变化：
  - 2D 图变为 1D 图——丢失了多少类别分离信息？
  - `explained_variance_ratio_`——第一个判别方向占了多大比例？
  - 尝试设 `n_components=3`——是否会报错？错误信息说明了什么？
- 核心理解：$K-1$ 是刚性约束——`n_components=2` 是 3 类数据下的最优配置

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用原始 `X` 训练
- 观察 2D 判别图的变化——类别分离效果是否显著变差或偏斜？
- 对比：原始数据中各特征的数值范围差异（`proline` ~278-1680 vs `nonflavanoid_phenols` ~0.13-0.66）
- 体会：标准化后各特征对散度矩阵贡献均等——判别方向才反映真实的类别可分性结构

### 4. 使用不同特征子集

- 在流水线中临时只保留部分特征列，例如只保留 3-4 个特征，再重新训练
- 观察变化：
  - 类别在 2D 判别空间中的分离效果是否明显变化
  - `explained_variance_ratio_` 的分布是否变化
- 体会：哪些化学特征对葡萄品种的区分更重要——这能帮助建立对 Wine 数据结构的直观认识

### 5. 对比 PCA 在相同 Wine 数据上的效果

- 用 PCA（`n_components=2`）在相同 Wine 数据上降维并生成 2D 散点图
- 对比变化：
  - 类别分离度——LDA 的类间分离通常显著优于 PCA
  - 坐标轴含义——`LD1/LD2`（判别方向）vs `PC1/PC2`（主成分方向）
  - `explained_variance_ratio_` 含义——"判别能力占比" vs "方差占比"
  - 累计解释方差——LDA 在 `n_components=2` 时必为 100%，PCA 通常 < 100%
- 核心理解：LDA 和 PCA 不是"谁更好"——同一份数据在两种优化目标下给出不同的低维表示。分类预处理选 LDA，数据探索选 PCA

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`LinearDiscriminantAnalysis` | 完整构造器参数（`n_components`、`solver`、`priors`、`shrinkage`、`tol`、`covariance_estimator`）、属性（`explained_variance_ratio_`、`scalings_`、`means_`、`priors_`、`classes_`、`xbar_`）与方法（`fit`、`transform`、`predict`、`predict_proba`、`predict_log_proba`、`decision_function`）说明 |
| 2 | scikit-learn 官方文档：`load_wine` | Wine 数据集的加载方式、特征说明和类别信息——`as_frame` 参数、`return_X_y` 参数 |
| 3 | scikit-learn 用户指南：LDA and QDA | 线性判别分析和二次判别分析的原理、`solver` 选择指南、收缩（shrinkage）正则化、与 PCA 的使用场景对比 |
| 4 | Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic Problems*. Annals of Eugenics. | LDA 原始论文——Fisher 判别准则、类间/类内散度、Iris 数据上的经典应用——奠定线性判别分析数学基础的里程碑工作 |

- scikit-learn `LinearDiscriminantAnalysis`：https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
- scikit-learn `load_wine`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
- scikit-learn 用户指南 LDA and QDA：https://scikit-learn.org/stable/modules/lda_qda.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 LDA 分册的核心内容：
  - LDA 是有监督降维——`fit()` 必须接收 `y`，标签用于定义类间/类内散度结构
  - Fisher 判别准则最大化类间散度与类内散度之比——这是 LDA 与 PCA（最大化方差）最根本的数学差异
  - $K-1$ 维上限来自 $\text{rank}(\mathbf{S}_B) \leq K-1$——Wine 数据 $K=3$，最多 2 个判别方向
  - `explained_variance_ratio_` 在 LDA 中表示"判别能力占比"，在 PCA 中表示"方差占比"——同名异义
  - `solver='svd'` 是默认选择——数值稳定性好，且支持 `explained_variance_ratio_`
  - 标准化对基于散度矩阵的 LDA 是硬性要求——Wine 数据特征量纲差异达数千倍
  - LDA 既有 `transform()`（降维投影）也有 `predict()`（分类预测）——当前仓库定位为降维工具
  - LDA 与 PCA 不是"谁更好"的关系——分类场景选 LDA，数据压缩/探索选 PCA
