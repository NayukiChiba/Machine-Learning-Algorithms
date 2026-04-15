---
title: LDA — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/dimensionality.py`、`model_training/dimensionality/lda.py`、`pipelines/dimensionality/lda.py`
>  
> 相关对象：`DimensionalityData.lda()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察 LDA 行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用 Wine 真实数据集来讲 LDA？
2. 在本项目里，`label`、`n_components`、`solver` 分别在控制什么？
3. 为什么当前数据有 3 个类别时，LDA 最多只能降到 2 维？
4. 为什么当前 `label` 在 LDA 中既参与训练，又参与图像着色？
5. LDA 与 PCA 的核心区别是什么？

## 动手练习

### 1. 调整 `solver`

修改 `model_training/dimensionality/lda.py` 中的默认参数：

```python
solver: str = "svd"
```

观察重点：

- 不同求解器下，是否还能获得 `explained_variance_ratio_`。
- 2D 判别图是否明显变化。
- 由此理解当前代码为什么对解释比例输出做了条件判断。

### 2. 调整 `n_components`

尝试修改：

```python
n_components: int = 2
```

观察重点：

- 当改成 `1` 时，二维结构会损失多少判别信息。
- 为什么不能超过 `2`。
- 由此理解 `K-1` 上限在当前数据中的实际约束。

### 3. 对比标准化前后结果

在 `pipelines/dimensionality/lda.py` 中尝试绕过标准化，再重新运行。

示意位置：

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

观察重点：

- 投影结果是否明显变化。
- 类别分离效果是否变差或偏斜。
- 由此理解当前 LDA 流水线为什么把标准化作为固定步骤。

### 4. 使用不同特征子集做观察

在流水线中临时只保留部分特征，例如只保留少量化学成分列，再重新训练。

观察重点：

- 类别分离效果是否明显变化。
- 由此理解哪些特征对判别方向更重要。
- 这也能帮助建立对 Wine 数据结构的直观认识。

### 5. 补一个下游分类比较

在 `pipelines/dimensionality/lda.py` 中增加“LDA 降维后再接一个简单分类器”的对比实验。

观察重点：

- 判别子空间是否真的更适合分类。
- 当前 2D LDA 投影是否已经保留了足够多的判别信息。
- 由此区分“投影图好看”和“下游可用性强”这两件事。

## 阅读建议

1. 先运行一次默认源码，记录解释比例信息和 2D 判别图。
2. 每次只改一个参数，例如只改 `solver` 或只改 `n_components`，避免多个变量同时变化。
3. 观察时优先对比三条线索：解释比例变化、类别分离图变化、监督信息对结果的影响。

## 参考文献

- scikit-learn `LinearDiscriminantAnalysis` API 文档：`https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html`
- scikit-learn Discriminant Analysis 用户指南：`https://scikit-learn.org/stable/modules/lda_qda.html`
- Fisher, *The Use of Multiple Measurements in Taxonomic Problems*.
- Bishop, *Pattern Recognition and Machine Learning*, Linear Discriminants.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Linear Methods for Classification.

## 小结

- 这部分练习最重要的目标，不是死记散度矩阵公式，而是亲手观察标签参与训练后，投影空间如何更偏向类别可分性。
- 当前源码已经提供了 Wine 真实数据、解释比例输出和 2D 判别图，因此很适合做基础 LDA 实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
