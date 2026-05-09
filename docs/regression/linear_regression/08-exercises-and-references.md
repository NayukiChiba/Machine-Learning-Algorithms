---
title: 线性回归 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对线性回归核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 OLS 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 OLS、正规方程、MLE、系数解释、残差图等核心概念的理解 |
| 动手练习 | 实践 | 修改噪声、样本量、新增无关特征观察 OLS 行为——建立线性模型直觉 |
| 参考文献 | 入口 | 提供线性回归经典教材和 scikit-learn 官方文档 |

## 1. 自检问题

1. 线性回归的模型形式是什么？`coef_` 和 `intercept_` 分别对应数学公式中的什么？

2. OLS 的损失函数是什么？为什么选择平方误差而非绝对误差？从高斯噪声假设的 MLE 视角解释。

3. 正规方程 $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ 在什么条件下会失效？scikit-learn 实际使用什么方法来避免这个问题？

4. 为什么当前线性回归流水线没有标准化？在什么场景下线性回归需要标准化？

5. 当前手工合成数据的真实公式是什么？如果训练后 `房龄` 的系数变成了正数，可能是什么原因？

6. 残差图和学习曲线分别更适合诊断什么问题？为什么两者需要结合来看？

7. 线性回归和决策树回归在处理特征交互方面有什么区别？"多一个房间且面积大一倍"这种交互效应在线性回归中如何表达？

## 2. 动手练习

### 练习 1：改变噪声水平

修改 `RegressionDatasetFactory` 中数据生成的噪声标准差（当前为 10），分别设为 `1`、`10`、`50`。

```python
# 在 loadLinearRegressionDataset 中修改 noise 的标准差
noise = rng.normal(0, 1, size=200)   # 低噪声——试试 1, 10, 50
```

回答：噪声 $\sigma=1$ 时系数是否几乎精确等于真实值 `[2, 10, -3]`？噪声 $\sigma=50$ 时系数偏移多少？残差图在三种噪声水平下有何不同？

### 练习 2：改变样本量

将 `nSamples` 分别设为 `20`、`100`、`500`、`2000`。

```python
# 在 RegressionDatasetFactory 中修改
nSamples: int = 20  # 试试 20, 100, 500, 2000
```

回答：`nSamples=20` 时系数估计是否极不稳定？学习曲线的验证得分波动是否随样本量增大而减小？$N=2000$ 时系数是否几乎完美恢复真实值？

### 练习 3：对照真实公式验证训练结果

运行默认流水线后，将控制台输出的系数与真实公式对照：

```python
# 真实公式: price = 2*面积 + 10*房间数 - 3*房龄 + noise + 50
# 训练输出:
#   面积: X.XX  (真实: 2)
#   房间数: X.XX (真实: 10)
#   房龄: X.XX  (真实: -3)
#   截距: X.XX  (真实: 50)
```

回答：三项系数的正负号是否全部正确？数值偏差最大的特征是什么？截距偏差了多少？

### 练习 4：新增一个无关特征

在数据生成中增加一个完全随机的噪声列，观察 OLS 如何处理无关特征。

```python
noise_feature = rng.normal(0, 5, size=200)
# 添加到 DataFrame 中，重新训练
```

回答：无关特征的系数是否接近 0？加入无关特征后，原有三个特征的系数是否发生明显变化？学习曲线是否受影响？

### 练习 5：手动计算 R² 并与残差图对照

在流水线中手动计算并打印测试集 $R^2$：

```python
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"测试集 R²: {r2:.4f}")
print(f"测试集 MSE: {mse:.4f}")
```

回答：$R^2$ 是否接近 1.0？MSE 是否与噪声方差 $\sigma^2=100$ 量级一致？数值指标与残差图的视觉判断是否一致？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 3. | 经典教材——线性回归的完整理论：OLS、子集选择、岭回归与 Lasso 的数学推导 |
| 2 | James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 3. | 入门教材——线性回归的基础直觉、R/Python 实现和与 KNN 的对比 |
| 3 | Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). *Introduction to Linear Regression Analysis*. Wiley. | 线性回归专著——从一元到多元、诊断、影响分析和共线性处理的全面覆盖 |
| 4 | scikit-learn 官方文档 — [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) | scikit-learn 的 API 参考——所有构造器参数、属性和方法的详细说明 |

## 常见坑

1. 在合成数据上期待系数精确等于真实值——噪声 $\sigma=10$ 和有限样本 200 意味着系数必然存在统计波动，这是 OLS 的固有属性而非 bug。
2. 只改噪声不改样本量——噪声和样本量对系数估计精度的联合影响才完整反映 OLS 行为。
3. 只看系数不看残差图——系数正负正确但残差图有系统偏差，可能意味着模型设定本身有问题。
4. 手动修改源码后忘记还原——建议在修改前用 `git stash` 保存原始状态，便于对比。

## 小结

- 7 个自检问题覆盖线性回归的核心概念：OLS、正规方程、MLE、标准化场景、系数验证、残差与学习曲线、特征交互。
- 5 个动手练习从不同角度探索 OLS 行为——改变噪声、改变样本量、对照真实公式、加入无关特征、计算数值指标。
- 4 篇参考文献覆盖两本经典教材（ESL、ISLR）、线性回归专著和 scikit-learn 官方文档——构成完整的线性回归学习路线。
