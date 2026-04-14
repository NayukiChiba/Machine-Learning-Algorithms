---
title: EM 与 GMM — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/probabilistic.py`、`model_training/probabilistic/em.py`、`pipelines/probabilistic/em.py`
>  
> 相关对象：`ProbabilisticData.em()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察 EM / GMM 行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用手工合成的二维混合高斯数据来讲 EM 与 GMM？
2. 在本项目里，`true_label` 为什么只用于训练后对比，而不参与训练？
3. `n_components`、`covariance_type`、`max_iter` 分别在控制什么？
4. `model.predict(...)` 输出的簇标签和 EM 训练中的责任度是什么关系？
5. 为什么当前日志里的 `log-likelihood` 不能直接当成最终聚类质量指标？

## 动手练习

### 1. 修改 `n_components`

修改 `model_training/probabilistic/em.py` 中的默认参数：

```python
n_components: int = 3
```

观察重点：

- 当分量数改成 `2` 或 `4` 后，聚类图是否出现明显合簇或拆簇现象。
- `log-likelihood` 是否发生变化。
- 预测标签和真实分量的空间结构是否更接近或更偏离。

### 2. 修改 `covariance_type`

修改以下默认参数：

```python
covariance_type: str = "full"
```

尝试改成 `diag`、`spherical` 或 `tied`。

观察重点：

- 不同协方差建模方式下，GMM 对簇形状的表达能力是否变化。
- 当前椭圆形簇数据是否更适合 `full`。
- 聚类分布图是否出现明显形状失配。

### 3. 修改合成数据的均值、方差或权重

修改 `data_generation/probabilistic.py` 中以下参数之一：

```python
em_means
em_stds
em_weights
```

观察重点：

- 分量之间更接近或更分散时，聚类图会怎么变化。
- 某个分量权重变小时，小簇是否更难被稳定识别。
- 不同簇形状下，当前协方差类型是否仍然合适。

### 4. 对比标准化前后的输入效果

在 `pipelines/probabilistic/em.py` 中尝试绕过标准化，再重新运行。

示意位置：

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

观察重点：

- 聚类分布图是否发生变化。
- `log-likelihood` 是否发生变化。
- 由此理解当前实现为什么把标准化作为固定步骤。

### 5. 补一个聚类指标

在 `pipelines/probabilistic/em.py` 中增加 `ARI` 或 `NMI` 打印。

观察重点：

- 数值指标和聚类分布图是否给出一致结论。
- `log-likelihood` 较高时，外部聚类指标是否一定更好。
- 由此区分“训练收敛好”与“聚类结果更贴近真实分量”这两件事。

## 阅读建议

1. 先运行一次默认源码，记录 `log-likelihood` 和聚类对比图。
2. 每次只改一个参数，例如只改 `n_components` 或只改 `covariance_type`，避免多个变量同时变化。
3. 观察时优先对比三条线索：数据分布变化、训练日志变化、聚类对比图变化。

## 参考文献

- scikit-learn `GaussianMixture` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html`
- scikit-learn Mixture Models 用户指南：`https://scikit-learn.org/stable/modules/mixture.html`
- Bishop, *Pattern Recognition and Machine Learning*, Chapter 9.
- Murphy, *Machine Learning: A Probabilistic Perspective*, Mixture Models and EM.
- Dempster, Laird, Rubin, *Maximum Likelihood from Incomplete Data via the EM Algorithm*.

## 小结

- 这部分练习最重要的目标，不是死记 E 步和 M 步公式，而是亲手观察分量数、协方差结构和数据分布如何改变聚类结果。
- 当前源码已经提供了非常透明的二维混合高斯数据和直观的聚类对比图，因此很适合做基础 EM / GMM 实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
