---
title: Bagging 与随机森林 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/ensemble.py`、`model_training/ensemble/bagging.py`、`pipelines/ensemble/bagging.py`
>  
> 相关对象：`EnsembleData.bagging()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察 Bagging 行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用高噪声双月牙数据来讲 Bagging？
2. 在本项目里，`n_estimators`、`max_samples`、`bootstrap` 分别在控制什么？
3. 为什么当前训练日志里的 `OOB 得分` 对 Bagging 特别有代表性？
4. 为什么 ROC 曲线在当前实现里是条件性输出，而混淆矩阵是固定输出？
5. 当前数学章节中的 Bootstrap 采样和方差缩减，与这份数据场景有什么关系？

## 动手练习

### 1. 调整 `n_estimators`

修改 `model_training/ensemble/bagging.py` 中的默认参数：

```python
n_estimators: int = 80
```

观察重点：

- 基学习器数量增大或减小时，混淆矩阵是否明显变化。
- `OOB 得分` 是否更稳定。
- 分类结果是否更平滑或更波动。

### 2. 调整 `max_samples`

修改以下默认参数：

```python
max_samples: float = 0.8
```

观察重点：

- 每棵树看到更少或更多样本时，`OOB 得分` 是否变化。
- 测试集混淆矩阵是否明显变化。
- 由此理解 Bootstrap 采样强度和集成差异性的关系。

### 3. 调整 `max_features`

尝试修改：

```python
max_features: float = 1.0
```

观察重点：

- 当每棵基学习器只看部分特征时，结果是否更像随机森林式的额外随机化。
- 当前二分类结果是否更稳定或更波动。
- 由此理解 Bagging 和随机森林之间的联系与差别。

### 4. 调整 `bootstrap` 与 `oob_score`

尝试修改以下参数之一：

```python
bootstrap
oob_score
```

观察重点：

- 关闭 `bootstrap` 后，`OOB 得分` 是否还能成立。
- 关闭 `oob_score` 后，训练日志会少什么信息。
- 由此理解 OOB 的前提条件。

### 5. 调整数据噪声 `bagging_noise`

在 `data_generation/ensemble.py` 中调整：

```python
bagging_noise: float = 0.35
```

观察重点：

- 噪声更大后，混淆矩阵是否更差。
- Bagging 相比单棵树的稳定性优势是否更明显。
- 由此理解当前数据为什么适合讲“降方差”。

## 阅读建议

1. 先运行一次默认源码，记录训练日志、`OOB 得分`、混淆矩阵和 ROC 曲线。
2. 每次只改一个参数，例如只改 `n_estimators` 或只改 `max_samples`，避免多个变量同时变化。
3. 观察时优先对比三条线索：`OOB 得分` 变化、分类结果变化、噪声敏感性变化。

## 参考文献

- scikit-learn `BaggingClassifier` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html`
- scikit-learn Ensemble 用户指南：`https://scikit-learn.org/stable/modules/ensemble.html`
- Breiman, *Bagging Predictors*.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Bagging and Random Forests.
- Geron, *Hands-On Machine Learning*, Bagging and Pasting 相关章节。

## 小结

- 这部分练习最重要的目标，不是死记参数，而是亲手观察采样比例、基学习器数量和噪声强度如何一起影响模型稳定性。
- 当前源码已经提供了双月牙高噪声数据、`OOB 得分` 和分类评估图，因此很适合做基础 Bagging 实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
