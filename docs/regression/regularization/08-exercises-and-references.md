---
title: 正则化回归 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/regression.py`、`model_training/regression/regularization.py`、`pipelines/regression/regularization.py`
>  
> 相关对象：`RegressionData.regularization()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和诊断方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察正则化行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要在 diabetes 数据集基础上额外添加 `bmi_corr`、`bp_corr`、`s5_corr` 和 `noise_1` ~ `noise_8`？
2. 在本项目里，Ridge、Lasso、ElasticNet 三者最值得对比的“行为差异”分别是什么？
3. `train_model(...)` 打印的“接近 0 的系数数量”为什么对正则化模型特别有价值？
4. 为什么当前流水线必须先切分，再对 `X_train` 做 `fit_transform(...)`，最后只对 `X_test` 做 `transform(...)`？
5. 如果一个模型的残差图看起来更分散，但近零系数更多，应该如何解读这种现象？

## 动手练习

### 1. 调大或调小 `alpha`

修改 `model_training/regression/regularization.py` 中的默认 `alphas`：

```python
alphas = {"lasso": 0.15, "ridge": 2.0, "elasticnet": 0.2}
```

观察重点：

- `alpha` 变大后，系数是否整体更小。
- Lasso 和 ElasticNet 的 `near_zero` 是否明显增多。
- 残差图是否出现更明显的欠拟合迹象。

### 2. 对比不同 `l1_ratio`

修改 `train_model(...)` 的 `l1_ratio`，例如改成 `0.2`、`0.8`。

观察重点：

- 当 `l1_ratio` 更接近 `0` 时，ElasticNet 是否更像 Ridge。
- 当 `l1_ratio` 更接近 `1` 时，ElasticNet 是否更像 Lasso。
- `noise_*` 和 `*_corr` 两类特征的系数变化是否同步。

### 3. 关闭相关特征构造

修改 `data_generation/regression.py` 中 `RegressionData` 的参数：

```python
reg_add_corr_features: bool = True
```

尝试将其改为 `False` 后重新运行流水线。

观察重点：

- 没有 `bmi_corr`、`bp_corr`、`s5_corr` 后，三种模型在共线性处理上的差异是否变弱。
- 系数打印中，是否更难看出“同组特征如何分配权重”这一现象。

### 4. 关闭噪声特征构造

修改 `data_generation/regression.py` 中的参数：

```python
reg_add_noise_features: int = 8
```

尝试把它改成 `0` 后重新运行。

观察重点：

- Lasso 的稀疏化优势是否变得不那么明显。
- `near_zero` 的对比是否失去一部分教学意义。
- 残差图是否变化不大，但系数结构显著不同。

### 5. 补一个数值指标

在 `pipelines/regression/regularization.py` 中为每个模型增加 `R^2` 或 `MSE` 打印。

观察重点：

- 数值指标和残差图是否给出一致结论。
- “系数更稀疏”与“指标更高”是否总是同时发生。

## 阅读建议

1. 先运行一次默认源码，记录三种模型的系数与残差图。
2. 每次只改一个超参数或一个数据构造开关，避免多个变量同时变化。
3. 观察时优先对比 `noise_*`、`*_corr`、`near_zero` 和残差图四条线索。

## 参考文献

- scikit-learn Linear Models 用户指南：`https://scikit-learn.org/stable/modules/linear_model.html`
- scikit-learn `Ridge` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html`
- scikit-learn `Lasso` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html`
- scikit-learn `ElasticNet` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html`
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Chapter 3.
- James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning*, Linear Model Selection and Regularization.

## 小结

- 这部分练习最重要的目标，不是背结论，而是亲手观察正则化如何改变系数结构和残差表现。
- 当前源码已经提供了很适合做实验的数据构造和日志输出，因此非常适合做小步调参练习。
- 把这些练习做完，再回头看数学原理和模型构建章节，理解通常会更扎实。
