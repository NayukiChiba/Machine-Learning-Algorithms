---
title: GaussianNB 高斯朴素贝叶斯 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`docs/classification/naive_bayes/`、`data_generation/classification.py`、`model_training/classification/naive_bayes.py`

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Naive Bayes 实现。
2. 给出继续深入阅读高斯朴素贝叶斯与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/naive_bayes.py` 要先做训练/测试切分，再做标准化？
2. 为什么当前 iris 连续特征数据更适合 `GaussianNB`，而不是文本分类常见的多项式朴素贝叶斯？
3. 当前 `train_model(...)` 中的 `var_smoothing` 控制什么？
4. 为什么 `model.class_prior_` 对理解朴素贝叶斯很重要？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？

## 练习方向

### 1. 改动 `var_smoothing`

- 把 `var_smoothing=1e-9` 改成更大或更小的值
- 观察混淆矩阵、ROC 曲线和学习曲线的变化
- 思考数值稳定性与分类表现之间的关系

### 2. 去掉标准化

- 暂时去掉 `StandardScaler()`
- 对比模型训练结果和可视化输出，体会统一预处理流程对工程稳定性的影响

### 3. 观察 `predict` 与 `predict_proba`

- 同时输出类别预测结果与概率输出
- 对比它们在混淆矩阵和 ROC 曲线中的不同用途

### 4. 与 SVC 对比

- 对照阅读 `docs/classification/svc/`
- 比较生成式分类和判别式分类在建模思路、训练输出和评估方式上的不同

## 参考文献

1. scikit-learn 官方文档：`GaussianNB`
   https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
2. scikit-learn 官方文档：`load_iris`
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
3. scikit-learn 用户指南：Naive Bayes
   https://scikit-learn.org/stable/modules/naive_bayes.html
4. Murphy, K. P. (2012).
   *Machine Learning: A Probabilistic Perspective*.

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释标准化顺序、`GaussianNB` 的连续特征假设、`class_prior_` 的意义、`predict_proba(...)` 的作用以及 `model_2d` 的角色，说明已经掌握了当前 Naive Bayes 分册的核心内容。
