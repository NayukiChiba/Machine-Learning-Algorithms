---
title: Bagging 集成学习 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 Bagging 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 Bagging 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 Bootstrap、方差缩减、OOB、Bagging vs Boosting 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 Bagging 行为变化——建立参数-效果的直觉 |
| 参考文献 | 入口 | 提供 Bagging 原始论文、教材章节和 scikit-learn 官方文档 |

## 1. 自检问题

1. Bootstrap 采样中，当 $m = N$ 时，单个样本未被抽中的概率约等于多少？这个概率对应的常数名称是什么？

2. Bagging 降低的是偏差还是方差？为什么必须选择完全生长的决策树（`max_depth=None`）作为基学习器？

3. OOB 得分的数学含义是什么？它为什么能作为泛化能力的无偏估计——无需额外划分验证集？

4. `n_estimators` 从 10 增加到 80 时，方差缩减的效果如何变化？从 80 增加到 200 呢？（提示：边际递减效应）

5. `max_samples=0.8` 和 `max_samples=1.0` 的区别是什么？哪个配置下子集间的多样性更大？为什么？

6. Bagging 和 Boosting 在训练方式（并行 vs 串行）、基学习器选择（强学习器 vs 弱学习器）、核心目标（降方差 vs 降偏差）上有哪些本质区别？

7. 当前 Bagging 流水线中，`hasattr(model, "predict_proba")` 条件检查的目的是什么？对 `BaggingClassifier` 而言，这个条件是否始终为 `True`？

## 2. 动手练习

### 练习 1：改变基学习器数量 `n_estimators`

将 `n_estimators` 分别设为 `1`、`5`、`10`、`50`、`80`、`200`，观察 OOB 得分和混淆矩阵的变化。

```python
# 修改 train_model 调用
model = train_model(X_train_s, y_train, n_estimators=5)   # 试试不同值
```

回答：从多少棵树开始，OOB 得分趋于稳定？单棵树（`n_estimators=1`）和 80 棵树的混淆矩阵有什么肉眼可见的差异？

### 练习 2：改变采样比例 `max_samples`

将 `max_samples` 分别设为 `0.3`、`0.5`、`0.8`、`1.0`，观察 OOB 得分的变化。

```python
model = train_model(X_train_s, y_train, max_samples=0.3)
```

回答：`max_samples` 越小，每个基学习器看到的样本越少——这对模型多样性和单个基学习器的偏差分别有什么影响？是否存在一个"最佳"采样比例？

### 练习 3：改变特征采样比例 `max_features`

将 `max_features` 从 `1.0` 改为 `0.5`（即每个子集只随机使用 1 个特征），观察效果变化。

```python
model = train_model(X_train_s, y_train, max_features=0.5)
```

回答：在仅有 2 个特征的情况下，`max_features=0.5`（随机使用 1 个特征）对模型性能有什么影响？如果在高维数据（如 100 个特征）上，`max_features < 1.0` 的意义是什么？

### 练习 4：关闭 Bootstrap 和 OOB

分别设置 `bootstrap=False` 和 `oob_score=False`，观察效果变化。

```python
# 关闭 Bootstrap——每棵树使用全部训练数据
model = train_model(X_train_s, y_train, bootstrap=False)

# 关闭 OOB 得分——训练后无法获取 oob_score_
model = train_model(X_train_s, y_train, oob_score=False)
```

回答：`bootstrap=False`（每棵树看到完全相同的训练数据）时，Bagging 还能降低方差吗？为什么？OOB 得分关闭后是否还能通过其他方式估计泛化能力？

### 练习 5：改变数据噪声水平

修改 `data_generation/ensemble.py` 中的 `bagging_noise` 参数（分别设为 `0.05`、`0.2`、`0.35`、`0.5`），重新运行流水线。

```python
# 在 data_generation/ensemble.py 中
class EnsembleData:
    bagging_noise: float = 0.05  # 试试 0.05, 0.2, 0.35, 0.5
```

回答：噪声为 `0.05`（极低）时，单棵决策树大约已经表现良好——此时 Bagging 的改善有多大？噪声为 `0.5`（极高）时，Bagging 是否仍然能有效平滑边界？这说明了什么？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Breiman, L. (1996). *Bagging Predictors*. Machine Learning, 24(2), 123-140. | Bagging 的原始论文——Bootstrap 聚合的理论基础和实验验证 |
| 2 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 8.7, 15. | ESL 教材——Bagging 和随机森林的数学推导与偏差-方差分解 |
| 3 | scikit-learn 官方文档 — [BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) | API 参考——全部参数、属性和方法的详细说明 |
| 4 | Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly. Chapter 7. | 实战教材——Bagging 和随机森林的实现与调参指南 |

## 常见坑

1. 把 `n_estimators=1` 当成"Bagging 的简化版"——此时没有集成、没有投票、没有方差缩减，就是一棵单决策树。
2. 认为 `max_samples` 越小越好——太小会导致每个基学习器偏差增大（样本太少无法学到完整模式），抵消方差缩减收益。
3. 忘记在修改 `bagging_noise` 后重新导入数据——`bagging_data` 是模块级变量，修改 `EnsembleData` 类的默认参数后需要重新实例化并调用 `bagging()`。
4. 混淆 `max_samples`（样本采样比例）和 `max_features`（特征采样比例）——两者共同决定子集与原始数据的差异度。

## 小结

- 7 个自检问题覆盖 Bagging 的核心概念：Bootstrap 概率、方差缩减、OOB 估计、参数选择、与 Boosting 对比。
- 5 个动手练习从不同角度探索 Bagging 的行为——改变基学习器数量、采样比例、特征比例、Bootstrap 开关、数据噪声。
- 4 篇参考文献从原始论文 → 教材 → 官方文档 → 实战指南构成完整的阅读路线。
