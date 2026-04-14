---
title: GBDT — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/ensemble.py`、`model_training/ensemble/gbdt.py`、`pipelines/ensemble/gbdt.py`
>  
> 相关对象：`EnsembleData.gbdt()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察 GBDT 行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用中等难度多分类数据来讲 GBDT？
2. 在本项目里，`n_estimators`、`learning_rate`、`max_depth` 分别在控制什么？
3. 为什么当前流水线既要调用 `predict(...)`，又要调用 `predict_proba(...)`？
4. 混淆矩阵、ROC 曲线、特征重要性图和学习曲线分别更适合回答哪类问题？
5. 当前数学章节里提到的“伪残差”，在分类任务里应该如何理解？

## 动手练习

### 1. 调整 `n_estimators`

修改 `model_training/ensemble/gbdt.py` 中的默认参数：

```python
n_estimators: int = 200
```

观察重点：

- boosting 轮数增大或减小时，混淆矩阵是否明显变化。
- ROC 曲线是否出现更明显的提升或波动。
- 学习曲线中的训练/验证走势是否发生变化。

### 2. 调整 `learning_rate`

修改以下默认参数：

```python
learning_rate: float = 0.1
```

观察重点：

- 学习率更大时，模型是否更激进。
- 学习率更小时，是否需要更多树数才能维持类似表现。
- 分类结果和学习曲线是否明显变化。

### 3. 调整 `max_depth`

尝试修改：

```python
max_depth: int = 3
```

观察重点：

- 基学习器更深时，是否更容易过拟合。
- 混淆矩阵中哪些类别受到影响最大。
- 特征重要性分布是否变化明显。

### 4. 调整 `subsample`

尝试调小以下参数：

```python
subsample: float = 1.0
```

观察重点：

- 当 `subsample < 1.0` 时，模型是否更接近随机梯度提升风格。
- 训练稳定性和学习曲线是否变化。
- 分类结果是否更稳或更波动。

### 5. 修改数据生成难度

在 `data_generation/ensemble.py` 中调整以下参数之一：

```python
gbdt_class_sep
gbdt_n_informative
gbdt_n_redundant
```

观察重点：

- 类间间隔更小后，混淆矩阵是否变得更复杂。
- 有效特征数变化后，重要性分布是否更集中。
- ROC 曲线和学习曲线是否明显受到影响。

## 阅读建议

1. 先运行一次默认源码，记录训练日志、混淆矩阵、ROC 曲线、特征重要性图和学习曲线。
2. 每次只改一个参数，例如只改 `n_estimators` 或只改 `learning_rate`，避免多个变量同时变化。
3. 观察时优先对比四条线索：分类结果变化、概率区分能力变化、特征重要性变化、学习曲线变化。

## 参考文献

- scikit-learn `GradientBoostingClassifier` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html`
- scikit-learn Ensemble 用户指南：`https://scikit-learn.org/stable/modules/ensemble.html`
- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Boosting and Additive Trees.
- Geron, *Hands-On Machine Learning*, Gradient Boosting 相关章节。

## 小结

- 这部分练习最重要的目标，不是死记所有参数，而是亲手观察树数、学习率、树深和采样比例如何一起影响分类结果。
- 当前源码已经提供了多分类数据、训练日志和四类评估图，因此很适合做基础 GBDT 实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
