---
title: SVR 支持向量回归 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 SVR 核心概念的理解程度。
2. 通过动手练习在代码层面观察 C、ε、γ 和支持向量数量的联动关系。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 ε-管道、RBF 核、支持向量稀疏性、C/ε/γ 角色等核心概念的理解 |
| 动手练习 | 实践 | 修改 C、ε、kernel、γ 并观察 nSV、残差图、学习曲线的变化 |
| 参考文献 | 入口 | 提供 SVR 经典论文和 scikit-learn 官方文档 |

## 1. 自检问题

1. SVR 的 ε-不敏感损失函数与 OLS 的平方损失有什么根本区别？为什么管道内的样本不参与预测？

2. C 在 SVR 中扮演什么角色？为什么说 C 是正则化强度的**倒数**？C=0.1 和 C=100 时模型行为有何差异？

3. RBF 核的作用是什么？当 `kernel='linear'` 时，SVR 退化为什么？gamma 控制什么？

4. 为什么 SVR（RBF 核）没有 `coef_` 属性？支持向量的权重存储在哪里？

5. SVR 为什么必须标准化？这与 RBF 核的数学公式有何关系？

6. 支持向量数量反映了什么？如果 nSV/160 > 90% 或 < 10%，分别意味着什么？

7. 当前 SVR 的 `PipelineSpec` 中训练后诊断列表为什么是 `[]`？这对 SVR 的评估策略有什么影响？

## 2. 动手练习

### 练习 1：改变 C 的值

修改 `trainSvrRegressionModel` 中的 `C`，分别设为 `0.1`、`10.0`（默认）、`100`。

```python
# 在 trainSvrRegressionModel 中修改
model = SVR(C=0.1, epsilon=0.1, kernel="rbf", gamma="scale")
# 试试 0.1, 1.0, 10.0, 100
```

回答：C=0.1 时支持向量数量是否显著减少？C=100 时训练 R² 是否接近 1.0 但验证 R² 下降？残差图在三个 C 值下有何不同？

### 练习 2：改变 epsilon 的值

将 `epsilon` 分别设为 `0.01`、`0.1`（默认）、`1.0`。

```python
# 在 trainSvrRegressionModel 中修改
model = SVR(C=10.0, epsilon=0.01, kernel="rbf", gamma="scale")
# 试试 0.01, 0.1, 0.5, 1.0
```

回答：ε=0.01 时支持向量数量是否急剧增加？ε=1.0 时是否几乎所有样本都在管道内（nSV 极小）？残差图的散点分布随 ε 如何变化？

### 练习 3：对比线性核与 RBF 核

将 `kernel` 从 `'rbf'` 改为 `'linear'`。

```python
# 在 trainSvrRegressionModel 中修改
model = SVR(C=10.0, epsilon=0.1, kernel="linear")
```

回答：线性核在 Friedman1 上的残差图是否显著劣于 RBF 核？线性核的 SVR 现在是否有 `coef_` 属性？支持向量数量有何变化？

### 练习 4：改变 gamma 的值

将 `gamma` 从 `'scale'` 改为具体数值——`0.01`、`0.1`、`1.0`。

```python
# 在 trainSvrRegressionModel 中修改
model = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma=0.01)
# 试试 'scale', 0.01, 0.1, 1.0
```

回答：γ=0.01 时学习曲线的训练/验证间隙是否缩小（更平滑）？γ=1.0 时训练 R² 是否极高但验证 R² 暴跌（严重过拟合）？nSV 随 γ 如何变化？

### 练习 5：手动加入 R² 和 MSE 计算

在流水线预测后手动计算并打印 R² 和 MSE：

```python
from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_test_s)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"测试集 R²: {r2:.4f}")
print(f"测试集 MSE: {mse:.4f}")
print(f"支持向量数量: {model.support_.shape[0]}/{len(y_train)}")
```

回答：R² 是否与学习曲线中的验证得分一致？"R² 更高"和"nSV 更少"是否同时发生？数值指标与残差图的视觉判断是否吻合？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14(3), 199-222. | SVR 经典教程——从 ε-不敏感损失到对偶问题的完整推导 |
| 2 | Drucker, H., Burges, C. J., Kaufman, L., Smola, A., & Vapnik, V. (1997). Support vector regression machines. *Advances in Neural Information Processing Systems*, 9. | SVR 原始论文——ε-SVR 的提出与算法实现 |
| 3 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 12. | 经典教材——支持向量机和核方法的完整数学推导 |
| 4 | scikit-learn 官方文档 — [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) | scikit-learn 的 SVR API 参考——所有构造器参数、属性和方法 |
| 5 | scikit-learn 官方文档 — [SVM 回归用户指南](https://scikit-learn.org/stable/modules/svm.html#svm-regression) | scikit-learn 的 SVR 使用指南——核函数选择和调参建议 |

## 常见坑

1. 修改 C 时忘记 C 是正则化倒数——增大 C = 减弱正则化，与 Lasso 的 α 方向相反。
2. 改 gamma 只看残差图不关注 nSV——gamma 的局部性变化最直接体现为 nSV 的剧烈变化。
3. 对比线性核和 RBF 核时忘记线性核不需要标准化（但当前流水线仍会标准化）——统一预处理保持对比公平。
4. 只改训练函数的参数忘记学习曲线中的 `SVR(...)` 也要同步——两者参数不一致会导致学习曲线的诊断无效。
5. 在 Friedman1 上期待线性核与 RBF 核表现相近——Friedman1 的目标函数高度非线性，线性核必然显著劣于 RBF 核。

## 小结

- 7 个自检问题覆盖 SVR 的核心概念：ε-管道损失、C 的角色、RBF 核、参数空间、标准化必要性、支持向量数量、PipelineSpec 配置。
- 5 个动手练习从不同角度探索 SVR 行为——调 C、调 ε、切换核函数、调 γ、加入数值指标。
- 5 篇参考文献覆盖 SVR 经典教程、原始论文、ESL 教材和 scikit-learn 文档——构成完整的 SVR 学习路线。
