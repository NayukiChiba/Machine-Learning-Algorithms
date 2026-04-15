---
title: SVC 支持向量分类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`docs/classification/svc/`、`data_generation/classification.py`、`model_training/classification/svc.py`

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 SVC 实现。
2. 给出继续深入阅读支持向量机与相关数据生成工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/svc.py` 要先做训练/测试切分，再做标准化？
2. 为什么当前同心圆数据更适合 `rbf` 核，而不是简单的线性分类边界？
3. 当前 `train_model(...)` 中的 `C`、`kernel`、`gamma` 分别控制什么？
4. 为什么 `model.n_support_` 对理解 SVC 很重要？
5. 为什么决策边界图里需要额外训练一个 `model_2d`？
6. 为什么学习曲线函数里传入的是新的 `SVC_Model(...)` 实例，而不是直接复用 `model`？

## 练习方向

### 1. 改动 `C`

- 把 `C=1.0` 改成更小或更大的值
- 观察支持向量数量、混淆矩阵和决策边界的变化
- 思考“边界更复杂”和“泛化更好”是否总是同一件事

### 2. 改动 `gamma`

- 调整 `gamma`
- 观察 RBF 核边界的弯曲程度变化

### 3. 改用线性核

- 暂时把 `kernel='rbf'` 改成 `kernel='linear'`
- 对比同心圆数据上的分类表现与边界形状

### 4. 去掉标准化

- 暂时去掉 `StandardScaler()`
- 对比模型训练结果和可视化输出，体会核方法对尺度的敏感性

## 参考文献

1. scikit-learn 官方文档：`SVC`
   https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
2. scikit-learn 官方文档：`make_circles`
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
3. scikit-learn 用户指南：SVM
   https://scikit-learn.org/stable/modules/svm.html
4. Cortes, C. and Vapnik, V. (1995).
   *Support-Vector Networks*.

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释标准化顺序、RBF 核的必要性、`n_support_` 的意义以及 `model_2d` 的角色，说明已经掌握了当前 SVC 分册的核心内容。
