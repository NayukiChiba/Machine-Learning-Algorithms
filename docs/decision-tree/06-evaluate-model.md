# 模型评估（evaluate_model.py）

这一模块评估决策树回归模型的效果，包含训练集与测试集指标。

---

## 1. 评估指标

### MSE
$$
\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2
$$

### RMSE
$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

### MAE
$$
\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|
$$

### R²
$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

---

## 2. 预测

```python
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

---

## 3. 训练集与测试集指标

两者都需要看：

- 训练集好、测试集差 → 过拟合
- 训练集差、测试集差 → 欠拟合

---

## 4. 过拟合检查

代码比较训练集与测试集 R² 差值：

```python
r2_diff = train_r2 - test_r2
```

- `< 0.05`：泛化良好
- `< 0.1`：轻微过拟合
- `>= 0.1`：可能过拟合

---

## 5. 小结

- 指标越好越不一定可信，要结合训练/测试表现
- 过拟合是决策树常见问题
