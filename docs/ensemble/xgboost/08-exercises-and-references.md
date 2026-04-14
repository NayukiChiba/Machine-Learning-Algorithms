---
title: XGBoost — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/ensemble.py`、`model_training/ensemble/xgboost.py`、`pipelines/ensemble/xgboost.py`
>  
> 相关对象：`EnsembleData.xgboost()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察 XGBoost 行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用 California Housing 真实数据来讲 XGBoost 回归？
2. 在本项目里，`n_estimators`、`learning_rate`、`max_depth` 分别在控制什么？
3. `gamma`、`reg_alpha`、`reg_lambda` 为什么能看作正则化相关参数？
4. 特征重要性图和残差图分别更适合回答哪类问题？
5. 为什么当前分册虽然是树模型，但不能简单按“单棵决策树”的方式去理解它？

## 动手练习

### 1. 调小或调大 `n_estimators`

修改 `model_training/ensemble/xgboost.py` 中的默认参数：

```python
n_estimators: int = 300
```

观察重点：

- 树数量减少或增加后，残差图是否明显变化。
- 训练耗时是否明显变化。
- 当前特征重要性分布是否变得更集中或更分散。

### 2. 调整 `learning_rate`

修改以下默认参数：

```python
learning_rate: float = 0.05
```

观察重点：

- 学习率更大时，模型是否更激进。
- 学习率更小时，是否需要更多树数才能维持类似表现。
- 残差图和重要性图是否出现明显差异。

### 3. 修改 `max_depth`、`subsample`、`colsample_bytree`

尝试调整以下参数之一：

```python
max_depth
subsample
colsample_bytree
```

观察重点：

- 单棵树更深时，是否更容易过拟合。
- 采样比例下降时，模型是否更保守或更稳定。
- 特征重要性分布是否变化明显。

### 4. 修改 `gamma`、`reg_alpha`、`reg_lambda`

尝试调大这些正则化相关参数。

观察重点：

- 分裂是否变得更谨慎。
- 残差图中的误差分布是否更平滑或更保守。
- 重要性图是否更集中到少数关键特征。

### 5. 补一个数值指标

在 `pipelines/ensemble/xgboost.py` 中增加 `R^2` 或 `MSE` 打印。

观察重点：

- 数值指标和残差图是否给出一致结论。
- 特征重要性高的模型是否一定得到更好的数值结果。
- 由此区分“模型关注什么”和“模型预测得怎样”这两件事。

## 阅读建议

1. 先运行一次默认源码，记录训练日志、残差图和特征重要性图。
2. 每次只改一个参数，例如只改 `n_estimators` 或只改 `learning_rate`，避免多个变量同时变化。
3. 观察时优先对比三条线索：超参数变化、残差图变化、特征重要性图变化。

## 参考文献

- XGBoost 官方文档：`https://xgboost.readthedocs.io/`
- XGBoost Python API 文档：`https://xgboost.readthedocs.io/en/stable/python/python_api.html`
- Chen, Guestrin, *XGBoost: A Scalable Tree Boosting System*.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Boosting and Additive Trees.
- Geron, *Hands-On Machine Learning*, Gradient Boosting and XGBoost 相关章节。

## 小结

- 这部分练习最重要的目标，不是死记所有超参数，而是亲手观察 boosting 强度、树复杂度和正则化如何一起影响结果。
- 当前源码已经提供了真实表格回归数据、训练日志和两类结果图，因此很适合做基础 XGBoost 实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
