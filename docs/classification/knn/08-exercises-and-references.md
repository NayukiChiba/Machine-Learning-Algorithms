---
title: KNN K 近邻分类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 KNN 实现。
2. 给出继续深入阅读 KNN 与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/knn.py` 要先做训练/测试切分，再做标准化？如果在切分前标准化会有什么问题？
2. 为什么当前 `make_moons(n_samples=400, noise=0.1)` 数据适合 KNN 的局部邻域思路？如果用 `make_blobs` 会怎样？
3. 当前 `train_model(...)` 中的 `n_neighbors`、`weights`、`metric`/`p` 分别控制什么？各自的数学含义是什么？
4. 为什么标准化对 KNN 特别重要？如果不标准化，距离 $d_p(\mathbf{x}, \mathbf{y}) = (\sum \vert x_i - y_i \vert^p)^{1/p}$ 会出现什么问题？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？KNN 的概率值为什么只能是离散的？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它和主模型 `model` 在什么特征空间上训练？
7. KNN 的 `fit()` 和逻辑回归的 `fit()` 有什么本质区别？为什么 KNN 被称为懒惰学习？

## 练习方向

### 1. 改动 n_neighbors

- 把 `n_neighbors=5` 改成 `1`、`3`、`15`、`50`、`100`
- 观察变化：
  - 混淆矩阵中各类别正确/错误分布
  - ROC 曲线 AUC 值的变化
  - 决策边界图的弯曲程度（$k$ 小 → 锯齿状边界，$k$ 大 → 平滑边界）
  - 学习曲线中训练得分与验证得分的差距
- 思考：$k$ 值与偏差-方差权衡的关系——小 $k$ 低偏差高方差，大 $k$ 高偏差低方差

### 2. 改动 weights

- 把 `weights='uniform'` 改成 `weights='distance'`
- 对比变化：
  - 决策边界形状——加权投票通常产生更精细的边界
  - ROC 曲线 AUC 值——
  - 对噪声的敏感性——加权投票对噪声更敏感
- 理解：加权投票的数学公式 $\hat{y} = \arg\max_c \sum \frac{\mathbb{1}(y_i=c)}{d(\mathbf{x}, \mathbf{x}_i)}$，邻居越近权重越大

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用 `X_train`、`X_test`
- 对比模型训练结果和可视化输出
- 体会：距离计算中量纲大的特征如何主导 $d_p(\mathbf{x}, \mathbf{y})$，量纲小的特征几乎形同虚设

### 4. 改动 metric 与 p

- 尝试 `metric='manhattan'`（$p=1$）与 `metric='minkowski'`（$p=2$，默认）
- 对比决策边界的变化——曼哈顿距离倾向于产生菱形边界，欧几里得距离倾向于产生圆弧边界
- 尝试调整 `noise` 参数（如 `0.0`、`0.3`），观察边界复杂度随噪声的变化

### 5. 与 Logistic Regression 和 Decision Tree 对比

- 对照阅读 `docs/classification/logistic_regression/` 和 `docs/classification/decision_tree/`
- 比较要点：
  - KNN 的局部投票 vs 逻辑回归的全局线性边界 $\mathbf{w}^T\mathbf{x} + b = 0$ vs 决策树的轴对齐切分
  - 是否需要标准化：KNN 需要，逻辑回归需要，决策树不需要
  - 评估方式差异：KNN 和逻辑回归都没有特征重要性图，决策树有
- 分别在同一数据（`make_moons`）上运行三个流水线，对比混淆矩阵和 ROC 曲线

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`KNeighborsClassifier` | 完整构造器参数列表、属性与方法说明 |
| 2 | scikit-learn 官方文档：`make_moons` | 双月牙数据生成器的参数与使用说明 |
| 3 | scikit-learn 用户指南：Nearest Neighbors | KNN 算法原理、距离度量选择与搜索算法的详细讲解 |
| 4 | Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*. | 第 13 章：Prototype Methods and Nearest-Neighbors，涵盖 KNN、偏差-方差分析、距离度量选择 |

- scikit-learn `KNeighborsClassifier`：https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- scikit-learn `make_moons`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
- scikit-learn 用户指南 Nearest Neighbors：https://scikit-learn.org/stable/modules/neighbors.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 KNN 分册的核心内容：
  - 标准化必须在切分后执行（防止数据泄露），KNN 比决策树更需要标准化（距离度量依赖特征尺度）
  - 局部邻域投票思路——近邻是谁就投谁，$k$ 控制局部范围大小
  - $k$ 值与偏差-方差权衡——小 $k$ 过拟合、大 $k$ 欠拟合
  - `predict_proba(...)` 的概率来自邻域频率，取值离散（分母为 $k$）
  - `model`、`model_2d` 和学习曲线实例分别在标准化空间、PCA 空间和交叉验证循环中运行
