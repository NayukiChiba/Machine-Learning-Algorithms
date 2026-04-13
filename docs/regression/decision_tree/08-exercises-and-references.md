---
title: 决策树回归 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/regression.py`、`model_training/regression/decision_tree.py`、`pipelines/regression/decision_tree.py`
>  
> 相关对象：`RegressionData.decision_tree()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察决策树行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用 California Housing 真实数据集来讲决策树回归？
2. 在本项目里，`max_depth`、`min_samples_split`、`min_samples_leaf` 分别在控制什么？
3. 为什么当前训练日志里更值得关注“树深度”和“叶子节点数”，而不是像线性回归那样关注系数？
4. 特征重要性图和残差图分别更适合回答哪类问题？
5. 为什么学习曲线要传入一个新的 `DecisionTreeRegressor(...)`，而不是复用已经训练好的 `model`？

## 动手练习

### 1. 调小或调大 `max_depth`

修改 `model_training/regression/decision_tree.py` 中的默认参数：

```python
max_depth: int = 6
```

观察重点：

- `max_depth` 变小时，树深度和叶子节点数是否明显减少。
- 残差图是否更容易出现欠拟合迹象。
- 学习曲线中训练得分和验证得分的差距是否变化。

### 2. 修改 `min_samples_split` 或 `min_samples_leaf`

修改以下默认参数：

```python
min_samples_split: int = 6
min_samples_leaf: int = 3
```

观察重点：

- 树结构是否变得更保守或更复杂。
- 叶子节点数是否明显变化。
- 残差图和学习曲线是否表现出不同的过拟合/欠拟合趋势。

### 3. 对比特征重要性图变化

分别在不同 `max_depth` 设置下运行流水线。

观察重点：

- 哪些特征在不同树深度下始终重要。
- 重要性是否集中在少数特征上。
- 模型复杂度变化是否会影响特征重要性的分布。

### 4. 尝试去掉 `.values`

把 `pipelines/regression/decision_tree.py` 中的训练和预测输入从数组改为原始 `DataFrame` / `Series` 形式，重新运行。

示意位置：

```python
model = train_model(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)
```

观察重点：

- 结果是否保持一致。
- 日志和图像输出是否仍然正常。
- 由此理解 `.values` 是当前实现细节，而不是算法硬性要求。

### 5. 补一个数值指标

在 `pipelines/regression/decision_tree.py` 中为模型增加 `R^2` 或 `MSE` 打印。

观察重点：

- 数值指标和残差图是否给出一致结论。
- 学习曲线上的走势是否和单次测试指标相匹配。
- 特征重要性高的模型是否一定得到更好的数值指标。

## 阅读建议

1. 先运行一次默认源码，记录树深度、叶子节点数、残差图、特征重要性图和学习曲线。
2. 每次只改一个超参数，例如只改 `max_depth` 或只改 `min_samples_leaf`，避免多个变量同时变化。
3. 观察时优先对比四条线索：树结构日志、残差图、特征重要性图、学习曲线。

## 参考文献

- scikit-learn `DecisionTreeRegressor` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html`
- scikit-learn Tree 用户指南：`https://scikit-learn.org/stable/modules/tree.html`
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Tree-Based Methods.
- James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning*, Tree-Based Methods.
- Breiman, Friedman, Olshen, Stone, *Classification and Regression Trees*.

## 小结

- 这部分练习最重要的目标，不是死记分裂公式，而是亲手观察树深度、叶子节点数、特征重要性和误差图如何一起变化。
- 当前源码已经提供了真实数据、结构日志和三类图像输出，因此很适合做基础树模型实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
