# 结果可视化（visualize_results.py）

这一模块把模型预测效果直观画出来，是理解模型好坏的关键一步。

---

## 1. 输出位置

图像保存到：

```
outputs/LinearRegression/
```

对应文件名：

- `04_Prediction_effect.png`
- `05_Residual_analysis.png`
- `06_Single_feature_regression.png`

---

## 2. 图 1：预测值 vs 真实值

```python
axes[0].scatter(y_train, y_train_pred)
axes[1].scatter(y_test, y_test_pred)
```

解释：
- 横轴是真实值
- 纵轴是预测值
- 越接近对角线越好

![04_prediction_effect](images/linear_regression/04_prediction_effect.png)

---

## 3. 图 2：残差分析

残差定义：

\[
\text{残差} = y - \hat{y}
\]

### 3.1 残差分布直方图

- 理想情况下残差以 0 为中心
- 分布越对称越好

### 3.2 残差 vs 预测值

- 理想：散点均匀分布在 0 附近
- 若呈现结构化趋势，说明模型可能遗漏了非线性关系

![05_residual_analysis](images/linear_regression/05_residual_analysis.png)

---

## 4. 图 3：单特征回归效果

```python
axes[i].scatter(X_test_original.iloc[:, i], y_test)
axes[i].scatter(X_test_original.iloc[:, i], y_test_pred)
```

意义：
- 查看单个特征与目标的关系
- 对比预测值与真实值在该特征上的分布

![06_single_feature_regression](images/linear_regression/06_single_feature_regression.png)

---

## 5. 为什么一定要看残差

- 残差能暴露模型假设是否成立
- 若残差呈现曲线趋势，说明模型可能需要非线性特征

---

## 6. 小结

- 这一步帮助你从“视觉层面”判断模型是否合理
- 预测效果好 ≠ 模型正确，残差结构很关键
- 如果残差结构明显，考虑增加特征或改用非线性模型
