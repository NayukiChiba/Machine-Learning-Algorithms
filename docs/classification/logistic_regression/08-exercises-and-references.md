---
title: LogisticRegression 逻辑回归分类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`docs/classification/logistic_regression/`、`data_generation/classification.py`、`model_training/classification/logistic_regression.py`

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Logistic Regression 实现。
2. 给出继续深入阅读逻辑回归与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/logistic_regression.py` 要先做训练/测试切分，再做标准化？
2. 为什么当前 `make_classification(...)` 数据适合逻辑回归的线性边界假设？
3. 当前 `train_model(...)` 中的 `penalty`、`C`、`solver` 分别控制什么？
4. 为什么 `model.coef_` 与 `model.intercept_` 对理解逻辑回归很重要？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？

## 练习方向

### 1. 改动 `C`

- 把 `C=1.0` 改成更小或更大的值
- 观察系数、混淆矩阵、ROC 曲线和学习曲线的变化
- 思考正则化强度与泛化表现之间的关系

### 2. 改动 `penalty`

- 尝试不同正则化配置
- 观察对系数稀疏性和训练稳定性的影响

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`
- 对比模型训练结果和可视化输出，体会标准化对优化与系数解释的影响

### 4. 与 SVC 对比

- 对照阅读 `docs/classification/svc/`
- 比较逻辑回归的概率输出思路与 SVC 的最大间隔思路在训练输出和评估方式上的不同

## 参考文献

1. scikit-learn 官方文档：`LogisticRegression`
   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
2. scikit-learn 官方文档：`make_classification`
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
3. scikit-learn 用户指南：Linear Models
   https://scikit-learn.org/stable/modules/linear_model.html
4. Hastie, T., Tibshirani, R., and Friedman, J. (2009).
   *The Elements of Statistical Learning*.

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释标准化顺序、线性边界假设、`coef_`/`intercept_` 的意义、`predict_proba(...)` 的作用以及 `model_2d` 的角色，说明已经掌握了当前 Logistic Regression 分册的核心内容。
