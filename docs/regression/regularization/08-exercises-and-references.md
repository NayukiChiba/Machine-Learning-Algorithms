---
title: 正则化回归 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对正则化回归核心概念的理解程度。
2. 通过动手练习在代码层面验证 Lasso、Ridge、ElasticNet 的行为差异。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 L1/L2 惩罚、稀疏性、标准化必要性、共线性处理等核心概念的理解 |
| 动手练习 | 实践 | 修改 α、l1_ratio、关闭共线/噪声特征——观察三种正则化模型的行为变化 |
| 参考文献 | 入口 | 提供正则化回归经典教材和 scikit-learn 官方文档 |

## 1. 自检问题

1. L1 正则化和 L2 正则化的数学惩罚项分别是什么？为什么 L1 能产生稀疏解而 L2 不能？

2. `alpha` 和 `l1_ratio` 分别控制什么？`l1_ratio=0` 和 `l1_ratio=1` 时 ElasticNet 分别退化成什么模型？

3. 为什么正则化回归**必须**标准化而线性回归和决策树回归不需要？从惩罚项对尺度的敏感性解释。

4. 当前数据为什么刻意构造 `bmi_corr`/`bp_corr`/`s5_corr` 和 `noise_1`~`noise_8`？每层特征分别测试正则化的什么能力？

5. `np.sum(np.abs(coef) < 1e-3)` 为什么用 `< 1e-3` 而非 `== 0`？Lasso、Ridge、ElasticNet 的 `near_zero` 预期分别是多少？

6. Lasso 和 Ridge 在面对 `bmi` 和 `bmi_corr` 这对高度相关特征时，系数分配策略有何不同？哪种策略更稳定？

7. 当前正则化回归流水线为什么没有学习曲线？这与 `PipelineSpec` 中的哪个配置字段对应？

## 2. 动手练习

### 练习 1：改变 Lasso 的 alpha

修改 `trainRegularizationModels` 中 Lasso 的 `alpha` 值，分别设为 `0.01`、`0.15`（默认）、`1.0`、`5.0`。

```python
# 在 trainRegularizationModels 中修改
models = {
    "lasso": Lasso(alpha=0.01, max_iter=10000, random_state=randomState),
    # 试试 0.01, 0.15, 1.0, 5.0
}
```

回答：`alpha=0.01` 时近零系数数量是否接近 0？`alpha=5.0` 时是否几乎所有系数都被清零？`noise_*` 系数在哪个 α 值开始被清零？

### 练习 2：改变 ElasticNet 的 l1_ratio

将 `l1_ratio` 分别设为 `0.1`、`0.5`（默认）、`0.9`。

```python
# 在 trainRegularizationModels 中修改
"elasticnet": ElasticNet(alpha=0.2, l1_ratio=0.1, max_iter=10000, random_state=randomState),
# 试试 0.1, 0.5, 0.9
```

回答：`l1_ratio=0.1` 时 ElasticNet 的系数分布是否接近 Ridge？`l1_ratio=0.9` 时是否接近 Lasso？`bmi` 和 `bmi_corr` 的系数分配随 `l1_ratio` 如何变化？

### 练习 3：关闭噪声特征

修改 `loadRegularizationDataset` 中噪声特征的数量为 `0`。

```python
# 在 loadRegularizationDataset 中修改循环终止条件
for index in range(0):  # 原为 8——改为 0
    data[f"noise_{index + 1}"] = rng.normal(size=len(data))
```

回答：没有 `noise_*` 特征后，Lasso 的稀疏化优势是否变得不明显？三种模型的近零系数数量是否趋同？残差图是否受影响？

### 练习 4：关闭共线特征

修改 `loadRegularizationDataset` 中共线特征的构造逻辑——注释掉 `bmi_corr`/`bp_corr`/`s5_corr` 的追加。

```python
# 注释掉以下三行
# data["bmi_corr"] = data["bmi"] * 0.9 + rng.normal(scale=0.02, size=len(data))
# data["bp_corr"] = data["bp"] * 0.9 + rng.normal(scale=0.02, size=len(data))
# data["s5_corr"] = data["s5"] * 0.9 + rng.normal(scale=0.02, size=len(data))
```

回答：没有共线特征后，Ridge 和 Lasso 的系数差异是否变小？哪种模型的行为变化最明显？

### 练习 5：手动计算 R² 并就系数图对比

在流水线预测循环中手动计算并打印 R²：

```python
from sklearn.metrics import r2_score, mean_squared_error

for name, model in models.items():
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} - R²: {r2:.4f}, MSE: {mse:.4f}")
```

回答：三个模型的 R² 差距是否显著？"系数更稀疏"和"R² 更高"是否同时发生？数值指标与残差图的视觉判断是否一致？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 3. | 经典教材——线性回归、岭回归与 Lasso 的完整数学推导，含 LAR 算法 |
| 2 | James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 6. | 入门教材——线性模型选择与正则化的基础直觉和 R/Python 实现 |
| 3 | Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society, Series B, 58(1), 267-288. | Lasso 原始论文——L1 正则化产生稀疏解的理论基础和算法 |
| 4 | Zou, H., & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net*. Journal of the Royal Statistical Society, Series B, 67(2), 301-320. | ElasticNet 原始论文——L1+L2 混合正则化的提出与理论分析 |
| 5 | scikit-learn 官方文档 — [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) / [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) / [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) | scikit-learn 的 API 参考——构造器参数、属性和使用示例 |

## 常见坑

1. 修改 `alpha` 后只运行不看 `near_zero`——α 变化的核心效果体现在近零系数数量，而非残差图。
2. 调参时同时改多个参数——每次只改一个变量，才能确定是哪个参数导致的行为变化。
3. 关闭噪声/共线特征后忘记恢复——建议在修改前用 `git stash` 保存原始状态。
4. 只看一个模型的系数不看三模型对比——正则化回归的诊断价值在于"对比"，单独看一个模型意义有限。
5. 在未标准化的数据上开启正则化——若跳过了 `StandardScaler`，系数形态会完全不可预期。

## 小结

- 7 个自检问题覆盖正则化回归的核心概念：L1/L2 惩罚、稀疏性几何直觉、标准化必要性、三层数据结构、近零系数、共线性处理和 PipelineSpec 配置。
- 5 个动手练习从不同角度探索正则化行为——调 α、调 l1_ratio、关闭噪声、关闭共线、计算数值指标。
- 5 篇参考文献覆盖 ESL、ISLR 两本经典教材、Lasso 和 ElasticNet 两篇原始论文、scikit-learn 官方 API 文档——构成完整的正则化回归学习路线。
