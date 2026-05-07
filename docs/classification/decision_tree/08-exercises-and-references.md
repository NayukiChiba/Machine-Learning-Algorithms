---
title: DecisionTreeClassifier 决策树分类 — 练习与参考文献
outline: deep
---

# 练习与参考文献


## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Decision Tree 实现。
2. 给出继续深入阅读决策树与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/decision_tree.py` 的主流程里没有显式标准化步骤？
2. 为什么当前 `make_blobs(n_samples=400, centers=4, cluster_std=1.0)` 数据适合决策树的区域切分方式？
3. 当前 `train_model(...)` 中的 `max_depth`、`min_samples_split`、`min_samples_leaf`、`criterion` 分别控制什么？各自的数学含义是什么？
4. 为什么 `model.get_depth()` 与 `model.get_n_leaves()` 对理解树复杂度很重要？两者的关系是什么（理论最大叶节点数 $2^d$）？
5. 为什么特征重要性图对理解树模型有帮助？特征重要性的数学公式是什么？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它和主模型 `model` 在什么特征空间上训练？
7. `predict(...)` 和 `predict_proba(...)` 的返回值有什么区别？各自服务于哪种评估方式？

## 练习方向

### 1. 改动 max_depth

- 把 `max_depth=6` 改成 `None`、`2`、`3`、`10`、`20`
- 观察变化：
  - `model.get_depth()` 的实际深度
  - `model.get_n_leaves()` 的叶节点数（注意与 $2^d$ 的关系）
  - 混淆矩阵中各类别正确/错误分布
  - 学习曲线中训练得分与验证得分的差距
- 思考：树深与泛化误差之间的关系

### 2. 改动 criterion

- 尝试 `criterion="gini"` 与 `criterion="entropy"`
- 对比变化：
  - 树结构（深度、叶节点数）是否不同
  - ROC 曲线 AUC 值的变化
  - 特征重要性的分布变化
- 理解：两种不纯度度量公式不同（$\text{Gini}(D) = 1 - \sum p_k^2$ vs $H(D) = -\sum p_k \log_2 p_k$），但在当前小规模数据上表现通常接近

### 3. 观察 feature_importances_

- 同时查看训练日志（`get_depth()`、`get_n_leaves()`）与特征重要性图
- 对比 `x1` 和 `x2` 在当前树中的贡献差异
- 修改 `make_blobs` 的 `cluster_std`（如从 `1.0` 改为 `2.0`），观察特征重要性的变化

### 4. 与 Logistic Regression 对比

- 对照阅读 `docs/classification/logistic_regression/`
- 比较要点：
  - 决策树的局部规则切分（轴对齐 $x_j \leq \text{threshold}$）vs 逻辑回归的全局线性边界（$\mathbf{w}^T\mathbf{x} + b = 0$）
  - 是否需要标准化：树不需要，逻辑回归需要
  - 评估方式差异：树有特征重要性图，逻辑回归有权重系数解释
- 分别在同一数据上运行两个流水线，对比混淆矩阵和 ROC 曲线

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`DecisionTreeClassifier` | 完整构造器参数列表、属性与方法说明 |
| 2 | scikit-learn 官方文档：`make_blobs` | 数据生成器的参数与使用说明 |
| 3 | scikit-learn 用户指南：Decision Trees | CART 算法原理、复杂度控制与剪枝策略的详细讲解 |
| 4 | Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*. | 第 9 章：Tree-Based Methods，涵盖 CART、信息增益、代价复杂度剪枝的数学推导 |

- scikit-learn `DecisionTreeClassifier`：https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- scikit-learn `make_blobs`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
- scikit-learn 用户指南 Decision Trees：https://scikit-learn.org/stable/modules/tree.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 Decision Tree 分册的核心内容：
  - 为什么主流程不强调标准化（树基于阈值切分，不依赖距离尺度）
  - 树深和叶节点数的意义与数学关系（$d$、$\vert T\vert$、$2^d$）
  - 特征重要性的解释边界（基于不纯度下降加权，不等于因果关系）
  - `model`、`model_2d` 和学习曲线实例的职责差异
