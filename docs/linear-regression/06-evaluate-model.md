# 模型评估（evaluate_model.py）

这一模块对训练好的模型进行性能评估，包含 **训练集** 和 **测试集** 两部分。

---

## 1. 为什么要评估

评估的目的：

- 看模型是否学到了规律
- 判断泛化能力（是否过拟合）
- 给出可量化的指标

---

## 2. 核心函数

```python
@print_func_info
def evaluate_model(model, X_train, X_test, y_train, y_test):
    ...
```

输入：
- 训练好的 `model`
- 训练集 / 测试集特征与标签

输出：
- 训练集预测值 `y_train_pred`
- 测试集预测值 `y_test_pred`

---

## 3. 预测

```python
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

- `predict` 只是把特征代入公式得到 \(\hat{y}\)

---

## 4. 评估指标

### 4.1 MSE / RMSE

\[
\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2
\]
\[
\text{RMSE} = \sqrt{\text{MSE}}
\]

- RMSE 与原始单位一致，更直观

### 4.2 MAE

\[
\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|
\]

- 对异常值更“温和”

### 4.3 R²

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

- R² 越接近 1 越好
- 负值表示模型很差（甚至不如直接预测均值）

---

## 5. 过拟合检测逻辑

代码里比较训练集与测试集 R² 的差值：

```python
r2_diff = train_r2 - test_r2
```

判断标准：
- `< 0.05`：泛化良好
- `< 0.1`：轻微过拟合
- `>= 0.1`：可能过拟合

---

## 6. 输出示例

```text
训练集性能:
  R^2 Score:  0.92
  RMSE:      3.12
  MAE:       2.35

测试集性能:
  R^2 Score:  0.89
  RMSE:      3.56
  MAE:       2.72
```

---

## 7. 小结

- 评估模块给出定量指标
- 你可以快速判断模型是否可靠
- 训练集和测试集都要看，不能只看训练集
