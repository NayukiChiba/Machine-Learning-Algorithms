---
title: 决策树回归 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对决策树回归核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索回归树的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对平方误差分裂、局部常数预测、复杂度约束、树 vs 线性回归等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数和对比模型配置观察回归树行为——建立树模型直觉 |
| 参考文献 | 入口 | 提供决策树回归经典教材和 scikit-learn 官方文档 |

## 1. 自检问题

1. 决策树回归的分裂准则是什么？为什么用平方误差而非基尼系数或信息增益？

2. 叶子节点的预测值为什么取区域内目标值的均值？从平方损失最小化的角度解释。

3. `max_depth`、`min_samples_split`、`min_samples_leaf` 分别从什么角度限制树的生长？如果三者同时设得很宽松（如 `max_depth=None, min_samples_split=2, min_samples_leaf=1`），会发生什么？

4. 决策树回归的预测函数为什么是分段常数形态？这种形态在处理房价这种连续值目标时有什么优缺点？

5. 为什么决策树回归不需要标准化？从分裂准则的数学形式给出解释。

6. `feature_importances_` 衡量的是什么？与线性回归的 `coef_` 在含义上有何根本区别？

7. 决策树回归和线性回归在处理非线性和特征交互方面各有什么优势和劣势？什么场景下应优先选树模型？

## 2. 动手练习

### 练习 1：改变 `max_depth`

将 `max_depth` 分别设为 `2`、`4`、`6`、`10`、`None`，观察树结构、残差图和学习曲线的变化。

```python
# 在 trainDecisionTreeRegressionModel 中修改
model = DecisionTreeRegressor(
    max_depth=2,  # 试试 2, 4, 6, 10, None
    min_samples_split=6,
    min_samples_leaf=3,
    random_state=42,
)
```

回答：`max_depth=2` 时树是否欠拟合（训练和验证 R² 都低）？`max_depth=None` 时是否明显过拟合（训练 R² 远高于验证 R²）？叶子节点数随深度如何变化？

### 练习 2：改变 `min_samples_leaf`

将 `min_samples_leaf` 分别设为 `1`、`3`、`10`、`50`，保持其他参数不变。

```python
model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=6,
    min_samples_leaf=1,  # 试试 1, 3, 10, 50
    random_state=42,
)
```

回答：`min_samples_leaf=1` 时叶子节点数是否大幅增加？残差图中是否出现了更多极端预测？`min_samples_leaf=50` 时树是否变得过于保守？

### 练习 3：对比特征重要性在不同深度下的变化

分别记录 `max_depth=3` 和 `max_depth=10` 时的特征重要性排名。

```python
importances = model.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")
```

回答：哪些特征在两个深度下都排在最前面？`max_depth` 增大后是否出现了新的重要特征？为什么深度会影响特征重要性的分布？

### 练习 4：对比决策树回归与线性回归在 California Housing 上的表现

在同一数据上分别训练线性回归和决策树回归，对比残差图。

```python
from sklearn.linear_model import LinearRegression

# 注意：线性回归需要标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 对比决策树
dt = DecisionTreeRegressor(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

回答：残差图的表现有何不同？线性回归的预测值 vs 真实值图是否呈现更连续的分布？决策树的预测值是否呈现离散的分段特征？哪种模型的 R² 更高？

### 练习 5：手动计算一个叶子的预测值

用 `model.tree_` 的底层属性找到任意一个叶子节点，提取该叶子内的训练样本索引，验证该叶子的预测值是否等于这些样本目标值的均值。

```python
tree = model.tree_
# 找到叶子节点（children_left[i] == children_right[i] == -1）
leaf_indices = [i for i in range(tree.node_count) 
                if tree.children_left[i] == -1]
# 选择一个叶子
leaf_id = leaf_indices[0]
# 获取该叶子的预测值
leaf_value = tree.value[leaf_id].flatten()[0]
print(f"叶子 {leaf_id} 的预测值: {leaf_value:.4f}")
```

回答：叶子的预测值是否确实等于落到该叶子的训练样本的 `y` 均值？如果偏差较大，可能是什么原因？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth. | CART 算法的原始专著——分类树与回归树的完整理论体系和算法推导 |
| 2 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 9. | 教材——树模型的偏差-方差分析、剪枝策略和与集成方法的衔接 |
| 3 | scikit-learn 官方文档 — [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) | scikit-learn 的 API 参考——所有构造器参数、属性和方法的详细说明 |
| 4 | James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 8. | 入门教材——树模型的基础直觉、R/Python 实现和与线性模型的对比 |

## 常见坑

1. 把回归树的分裂准则与分类树混淆——回归用平方误差（MSE），分类用基尼系数或熵。
2. 在未设 `random_state` 的情况下对比不同实验——树的分裂可能因随机性而不同，实验结果不可复现。
3. 只用 R² 评估模型——树模型的残差图和结构图能揭示数值指标无法反映的局部拟合问题。
4. 把 `feature_importances_` 解读为"特征对目标的正负影响方向和大小"——重要性只看分裂贡献，不表示方向也不等效于线性系数。

## 小结

- 7 个自检问题覆盖决策树回归的核心概念：平方误差分裂、局部常数预测、三重复杂度约束、无标准化原因、特征重要性含义、与线性回归对比。
- 5 个动手练习从不同角度探索回归树的行为——改变深度和叶子约束、对比特征重要性、与线性回归横向对比、验证叶子预测值的数学本质。
- 4 篇参考文献覆盖 CART 原始专著（Breiman 1984）、两本经典教材和 scikit-learn 官方文档——构成完整的回归树学习路线。
