---
title: KNN K 近邻分类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`docs/classification/knn/`、`data_generation/classification.py`、`model_training/classification/knn.py`

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 KNN 实现。
2. 给出继续深入阅读 KNN 与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/knn.py` 要先做训练/测试切分，再做标准化？
2. 为什么当前 `make_moons(...)` 数据适合 KNN 的局部邻域思路？
3. 当前 `train_model(...)` 中的 `n_neighbors`、`weights`、`metric` 分别控制什么？
4. 为什么标准化对 KNN 特别重要？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？

## 练习方向

### 1. 改动 `n_neighbors`

- 把 `n_neighbors=5` 改成更小或更大的值
- 观察混淆矩阵、ROC 曲线、决策边界和学习曲线的变化
- 思考局部性与平滑程度之间的关系

### 2. 改动 `weights`

- 把 `weights='uniform'` 改成 `weights='distance'`
- 对比投票机制变化对分类边界的影响

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`
- 对比模型训练结果和可视化输出，体会距离失真对近邻关系的影响

### 4. 与 Logistic Regression 对比

- 对照阅读 `docs/classification/logistic_regression/`
- 比较 KNN 的局部投票思路与逻辑回归的全局线性概率边界在训练输出和评估方式上的不同

## 参考文献

1. scikit-learn 官方文档：`KNeighborsClassifier`
   https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
2. scikit-learn 官方文档：`make_moons`
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
3. scikit-learn 用户指南：Nearest Neighbors
   https://scikit-learn.org/stable/modules/neighbors.html
4. Hastie, T., Tibshirani, R., and Friedman, J. (2009).
   *The Elements of Statistical Learning*.

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释标准化顺序、局部邻域思路、`n_neighbors` 的意义、`predict_proba(...)` 的作用以及 `model_2d` 的角色，说明已经掌握了当前 KNN 分册的核心内容。
