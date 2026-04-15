---
title: PCA — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/dimensionality.py`、`model_training/dimensionality/pca.py`、`pipelines/dimensionality/pca.py`
>  
> 相关对象：`DimensionalityData.pca()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察 PCA 行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用“低秩结构映射到高维空间”的数据来讲 PCA？
2. 在本项目里，`n_components`、`explained_variance_ratio_`、累计解释方差分别在说明什么？
3. 为什么当前流水线既要训练 2D PCA，又要训练 3D PCA？
4. 为什么当前 `label` 不参与训练，却仍然出现在降维图里？
5. PCA 和 LDA 的核心区别是什么？

## 动手练习

### 1. 调整 `n_components`

修改 `model_training/dimensionality/pca.py` 中的参数，或在流水线中尝试不同主成分数量。

示意：

```python
model = train_model(X_scaled, n_components=2)
```

观察重点：

- 当保留更多主成分时，累计解释方差如何变化。\n+- 降维图结构是否明显更清晰。\n+- 2D 和 3D 之间的差异是否足够大到值得保留第三个主成分。

### 2. 调整原始特征维度 `pca_n_features`

在 `data_generation/dimensionality.py` 中修改：

```python
pca_n_features: int = 10
```

观察重点：

- 原始维度增大后，前几个主成分是否仍然能解释大部分方差。\n+- 解释方差比是否更分散。\n+- 由此理解“高维不等于高信息量”。

### 3. 调整真实信息维度 `pca_n_informative`

修改以下参数：

```python
pca_n_informative: int = 3
```

观察重点：

- 当真正有信息的方向变多时，前 2 个或前 3 个主成分是否还足够。\n+- 累计解释方差是否下降。\n+- 由此理解 PCA 对低秩结构强弱的敏感性。

### 4. 调整噪声强度 `pca_noise_std`

修改以下参数：

```python
pca_noise_std: float = 0.5
```

观察重点：

- 噪声更大后，解释方差比是否更分散。\n+- 2D / 3D 图是否变得更混乱。\n+- 由此理解 PCA 为什么对噪声水平敏感。

### 5. 对比标准化前后结果

在 `pipelines/dimensionality/pca.py` 中尝试绕过标准化，再重新运行。

示意位置：

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

观察重点：

- 主成分方向和解释方差比是否明显变化。\n+- 由此理解为什么当前 PCA 流水线把标准化作为固定步骤。\n+- 对比有无标准化时的可视化结构差异。

## 阅读建议

1. 先运行一次默认源码，记录 2D / 3D 的解释方差比、累计解释方差和降维图。\n+2. 每次只改一个参数，例如只改 `pca_noise_std` 或只改 `pca_n_informative`，避免多个变量同时变化。\n+3. 观察时优先对比三条线索：解释方差变化、累计解释方差变化、投影图结构变化。

## 参考文献

- scikit-learn PCA API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html`
- scikit-learn PCA 用户指南：`https://scikit-learn.org/stable/modules/decomposition.html#pca`
- Jolliffe, Cadima, *Principal Component Analysis: A Review and Recent Developments*.\n+- Bishop, *Pattern Recognition and Machine Learning*, Principal Component Analysis.\n+- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Unsupervised Learning and PCA.

## 小结

- 这部分练习最重要的目标，不是死记特征值推导，而是亲手观察主成分数量、噪声水平和真实信息维度如何一起影响降维结果。\n+- 当前源码已经提供了低秩高维数据、解释方差输出和 2D / 3D 可视化，因此很适合做基础 PCA 实验。\n+- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
