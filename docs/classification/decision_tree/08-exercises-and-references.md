---
title: DecisionTreeClassifier 决策树分类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`docs/classification/decision_tree/`、`data_generation/classification.py`、`model_training/classification/decision_tree.py`

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Decision Tree 实现。
2. 给出继续深入阅读决策树与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/decision_tree.py` 的主流程里没有显式标准化步骤？
2. 为什么当前 `make_blobs(...)` 数据适合决策树的区域切分方式？
3. 当前 `train_model(...)` 中的 `max_depth`、`min_samples_split`、`min_samples_leaf`、`criterion` 分别控制什么？
4. 为什么 `model.get_depth()` 与 `model.get_n_leaves()` 对理解树复杂度很重要？
5. 为什么特征重要性图对理解树模型有帮助？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？

## 练习方向

### 1. 改动 `max_depth`

- 把 `max_depth=6` 改成更小或更大的值
- 观察树深、叶子节点数、混淆矩阵和学习曲线的变化
- 思考树复杂度与泛化表现之间的关系

### 2. 改动 `criterion`

- 尝试不同划分标准
- 对比树结构、ROC 曲线和特征重要性变化

### 3. 观察 `feature_importances_`

- 同时查看训练日志与特征重要性图
- 对比不同特征在当前树中的贡献差异

### 4. 与 Logistic Regression 对比

- 对照阅读 `docs/classification/logistic_regression/`
- 比较决策树的局部规则切分思路与逻辑回归的全局线性概率边界在训练输出和评估方式上的不同

## 参考文献

1. scikit-learn 官方文档：`DecisionTreeClassifier`
   https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
2. scikit-learn 官方文档：`make_blobs`
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
3. scikit-learn 用户指南：Decision Trees
   https://scikit-learn.org/stable/modules/tree.html
4. Hastie, T., Tibshirani, R., and Friedman, J. (2009).
   *The Elements of Statistical Learning*.

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释为什么当前主流程不强调标准化、树深和叶节点数的意义、特征重要性的解释边界以及 `model_2d` 的角色，说明已经掌握了当前 Decision Tree 分册的核心内容。
