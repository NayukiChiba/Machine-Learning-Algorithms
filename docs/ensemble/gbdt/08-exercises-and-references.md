---
title: GBDT 梯度提升树 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 GBDT 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 GBDT 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对加法模型、负梯度、学习率收缩、GBDT vs Bagging 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 GBDT 行为变化——建立参数-效果的直觉 |
| 参考文献 | 入口 | 提供 GBDT 原始论文、教材章节和 scikit-learn 官方文档 |

## 1. 自检问题

1. GBDT 的加法模型 $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu h_m(\mathbf{x})$ 与 Bagging 的投票平均 $f_{\text{bag}} = \frac{1}{n}\sum f_b$ 在结构上有何本质区别？

2. 为什么 GBDT 的每棵树拟合的是"负梯度"而不是原始标签 $y$？负梯度的直观含义是什么？

3. GBDT 降低的是偏差还是方差？为什么必须选择浅层决策树（`max_depth=3`）作为基学习器？

4. 学习率 $\nu=0.1$ 与 $\nu=1.0$ 的区别是什么？$\nu=0.1$ 时为什么需要更多的树（`n_estimators=200`）？

5. `subsample=0.8` 和 `subsample=1.0` 的区别是什么？引入随机子采样能带来什么额外收益？

6. GBDT 和 Bagging 在训练方式（串行 vs 并行）、基学习器选择（弱学习器 vs 强学习器）、核心目标（降偏差 vs 降方差）、独有诊断工具（特征重要性 + 学习曲线 vs OOB 得分）上有哪些本质区别？

7. 为什么 GBDT 的 `estimators_` 内部是回归树而非分类树？三分类场景下 `estimators_` 包含多少棵树？

## 2. 动手练习

### 练习 1：改变学习率 `learning_rate`

将 `learning_rate` 分别设为 `0.01`、`0.05`、`0.1`、`0.5`、`1.0`，同时调整 `n_estimators` 使总修正量（$\nu \times M$）大致相同，观察效果。

```python
# 保持总修正量约为 20
# ν=0.01 → n_estimators=2000
# ν=0.05 → n_estimators=400
# ν=0.1  → n_estimators=200
# ν=0.5  → n_estimators=40
# ν=1.0  → n_estimators=20

model = train_model(X_train_s, y_train, n_estimators=40, learning_rate=0.5)
```

回答：相同的总修正量下，大学习率 + 少树 vs 小学习率 + 多树，哪种组合的泛化效果更好？为什么？

### 练习 2：改变基学习器深度 `max_depth`

将 `max_depth` 分别设为 `1`、`3`、`5`、`10`、`None`，观察特征重要性和混淆矩阵的变化。

```python
model = train_model(X_train_s, y_train, max_depth=5)
```

回答：`max_depth` 增大后，模型是更倾向于 Bagging 还是保持了 GBDT 的特性？过深的基学习器对 GBDT 有什么危害？

### 练习 3：改变弱学习器数量 `n_estimators`

保持 `learning_rate=0.1`，将 `n_estimators` 分别设为 `10`、`50`、`100`、`200`、`500`，观察学习曲线的变化。

```python
model = train_model(X_train_s, y_train, n_estimators=10)
```

回答：从多少棵树开始测试集准确率趋于平稳？继续增加树数量是否会过拟合？

### 练习 4：启用随机梯度提升

将 `subsample` 从 `1.0` 改为 `0.8`，观察混淆矩阵和特征重要性的变化。

```python
model = train_model(X_train_s, y_train, subsample=0.8)
```

回答：`subsample=0.8` 对训练耗时和泛化能力分别有什么影响？这种影响与 Bagging 的 `max_samples=0.8` 有何异同？

### 练习 5：改变数据难度

修改 `data_generation/ensemble.py` 中的 `gbdt_class_sep` 参数（分别设为 `0.3`、`0.7`、`1.5`），重新运行流水线。

```python
# 在 data_generation/ensemble.py 中
class EnsembleData:
    gbdt_class_sep: float = 0.3  # 试试 0.3, 0.7, 1.5
```

回答：类别间隔越小，GBDT 相对于单棵决策树的优势是增大还是减小？为什么？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189-1232. | GBDT 的原始论文——梯度提升框架的数学推导和泛化分析 |
| 2 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 10. | ESL 教材——Boosting 和加法模型的完整数学推导 |
| 3 | scikit-learn 官方文档 — [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) | API 参考——全部参数、属性和方法的详细说明 |
| 4 | Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly. Chapter 7. | 实战教材——GBDT 的实现、调参与与 XGBoost 的对比 |

## 常见坑

1. 把 `n_estimators=10` 当成合理的 GBDT 配置——`learning_rate=0.1` 时 10 棵树的总修正量仅为 1.0 倍步长，远不足以收敛。
2. 认为 `max_depth` 越大越好——GBDT 的基学习器应是弱学习器，深度过大会破坏偏差缩减的串行机制。
3. 忘记 `learning_rate` 与 `n_estimators` 的耦合——调整一个必须同步调整另一个。
4. 把特征重要性当成因果关系的证明——高重要性只是"使用频率高"的统计描述。

## 小结

- 7 个自检问题覆盖 GBDT 的核心概念：加法模型、负梯度、偏差缩减、学习率收缩、随机梯度提升、与 Bagging 对比、回归树基学习器。
- 5 个动手练习从不同角度探索 GBDT 的行为——改变学习率、基学习器深度、树数量、子采样、数据难度。
- 4 篇参考文献从原始论文（Friedman 2001）→ 教材 → 官方文档 → 实战指南构成完整的阅读路线。
