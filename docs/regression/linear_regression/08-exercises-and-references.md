---
title: 线性回归 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/regression.py`、`model_training/regression/linear_regression.py`、`pipelines/regression/linear_regression.py`
>  
> 相关对象：`RegressionData.linear_regression()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察线性回归行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用手工合成的 `面积`、`房间数`、`房龄` 数据来讲线性回归？
2. 在本项目里，`coef_` 和 `intercept_` 分别应该如何解释？
3. 为什么训练结果通常只会接近真实公式中的 `2`、`10`、`-3`、`50`，而不是完全相等？
4. 当前流水线为什么能直接使用 `X_test` 做预测，而不像其他分册那样先做标准化？
5. 残差图和学习曲线分别更适合帮助你发现哪类问题？

## 动手练习

### 1. 调大或调小噪声 `lr_noise`

修改 `data_generation/regression.py` 中的默认参数：

```python
lr_noise: float = 10.0
```

观察重点：

- 噪声变大后，训练得到的系数是否更偏离真实公式。
- 残差图是否变得更分散。
- 学习曲线的验证得分是否更不稳定。

### 2. 修改样本数 `n_samples`

修改 `RegressionData` 中的参数：

```python
n_samples: int = 200
```

观察重点：

- 样本数减少时，训练日志中的系数波动是否更明显。
- 学习曲线是否更容易出现波动。
- 样本数增大后，模型参数是否更接近真实生成公式。

### 3. 对照真实公式检查训练结果

运行默认流水线后，把控制台输出的截距和系数与真实关系对照：

```python
price = 2 * 面积 + 10 * 房间数 - 3 * 房龄 + noise + 50
```

观察重点：

- `面积` 的系数是否接近 `2`
- `房间数` 的系数是否接近 `10`
- `房龄` 的系数是否接近 `-3`
- 截距是否接近 `50`

### 4. 手动新增一个无关特征

在 `RegressionData.linear_regression()` 中增加一个随机噪声特征，再重新训练。

示意代码：

```python
noise_feature = rng.normal(size=self.n_samples)
```

观察重点：

- 新特征的系数是否接近 0。
- 原始三个特征的系数是否发生波动。
- 残差图和学习曲线是否明显变化。

### 5. 补一个数值指标

在 `pipelines/regression/linear_regression.py` 中为模型增加 `R^2` 或 `MSE` 打印。

观察重点：

- 数值指标和残差图是否给出一致结论。
- 学习曲线上的趋势是否和单次测试指标相匹配。

## 阅读建议

1. 先运行一次默认源码，记录系数、截距、残差图和学习曲线。
2. 每次只改一个参数，例如只改 `lr_noise` 或只改 `n_samples`，避免多个变量同时变化。
3. 观察时优先对比四条线索：真实公式、训练日志、残差图、学习曲线。

## 参考文献

- scikit-learn `LinearRegression` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html`
- scikit-learn Linear Models 用户指南：`https://scikit-learn.org/stable/modules/linear_model.html`
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Chapter 3.
- James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning*, Linear Regression.
- Montgomery, Peck, Vining, *Introduction to Linear Regression Analysis*.

## 小结

- 这部分练习最重要的目标，不是死记公式，而是亲手观察线性关系、噪声和样本量如何影响训练结果。
- 当前源码已经提供了非常透明的数据生成公式和简单的工程流程，因此特别适合做基础回归实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
