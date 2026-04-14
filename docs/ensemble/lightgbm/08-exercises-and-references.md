---
title: LightGBM — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/ensemble.py`、`model_training/ensemble/lightgbm.py`、`pipelines/ensemble/lightgbm.py`
>  
> 相关对象：`EnsembleData.lightgbm()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察 LightGBM 行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用高维多分类数据来讲 LightGBM？
2. 在本项目里，`num_leaves`、`max_depth`、`learning_rate` 分别在控制什么？
3. 为什么当前流水线既要调用 `predict(...)`，又要调用 `predict_proba(...)`？
4. 混淆矩阵、ROC 曲线和特征重要性图分别更适合回答哪类问题？
5. 当前数学章节里提到的 Leaf-wise、GOSS、EFB，和这份高维数据有什么关系？

## 动手练习

### 1. 调整 `num_leaves`

修改 `model_training/ensemble/lightgbm.py` 中的默认参数：

```python
num_leaves: int = 31
```

观察重点：

- 叶子数增大或减小时，混淆矩阵是否明显变化。
- ROC 曲线是否出现更明显的提升或波动。
- 特征重要性分布是否更集中或更分散。

### 2. 调整 `max_depth`

修改以下默认参数：

```python
max_depth: int = -1
```

观察重点：

- 限制深度后，模型是否更保守。
- 混淆矩阵中是否出现更明显的欠拟合迹象。
- 由此理解 `num_leaves` 和 `max_depth` 的配合关系。

### 3. 调整 `n_estimators` 与 `learning_rate`

尝试分别修改：

```python
n_estimators
learning_rate
```

观察重点：

- boosting 轮数和步长变化后，分类结果是否明显变化。
- ROC 曲线的整体形态是否变化。
- 不同参数组合下，模型是更激进还是更保守。

### 4. 调整 `subsample`、`colsample_bytree`

尝试调小或调大以下参数：

```python
subsample
colsample_bytree
```

观察重点：

- 样本和特征采样比例变化后，分类稳定性是否变化。
- 特征重要性图是否明显改变。
- 由此理解采样机制对泛化能力的影响。

### 5. 修改数据生成难度

在 `data_generation/ensemble.py` 中调整以下参数之一：

```python
lgbm_class_sep
lgbm_n_features
lgbm_n_informative
```

观察重点：

- 类间间隔更小后，混淆矩阵是否变得更复杂。
- 高维程度和有效特征数变化后，重要性分布是否更集中。
- ROC 曲线是否明显受到影响。

## 阅读建议

1. 先运行一次默认源码，记录训练日志、混淆矩阵、ROC 曲线和特征重要性图。
2. 每次只改一个参数，例如只改 `num_leaves` 或只改 `learning_rate`，避免多个变量同时变化。
3. 观察时优先对比三条线索：分类结果变化、概率区分能力变化、特征重要性变化。

## 参考文献

- LightGBM 官方文档：`https://lightgbm.readthedocs.io/`
- LightGBM Python API 文档：`https://lightgbm.readthedocs.io/en/latest/Python-API.html`
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Boosting and Tree Methods.
- Geron, *Hands-On Machine Learning*, Gradient Boosting and LightGBM 相关章节。

## 小结

- 这部分练习最重要的目标，不是死记所有参数，而是亲手观察叶子数、深度、采样比例和数据难度如何一起影响分类结果。
- 当前源码已经提供了高维多分类数据、训练日志和三类评估图，因此很适合做基础 LightGBM 实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
