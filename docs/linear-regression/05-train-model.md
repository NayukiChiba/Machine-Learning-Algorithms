# 模型训练（train_model.py）

这一模块负责训练线性回归模型，并输出模型参数（截距与系数）。

---

## 1. 训练目标

通过最小二乘法拟合参数：

$$
\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

在代码中，`sklearn` 会用数值稳定的方法（如 SVD）实现这个过程。

---

## 2. 核心函数

```python
@print_func_info
def train_model(X_train, y_train, feature_names=None):
    model = LinearRegression()
    model.fit(X_train, y_train)
    ...
```

- `X_train`：训练特征（通常是标准化后的数据）
- `y_train`：训练标签
- `feature_names`：可选，用来打印系数时显示真实特征名

---

## 3. 训练过程

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

- `.fit()` 会计算参数 $\beta$
- 训练结束后可用 `coef_` 和 `intercept_` 访问参数

---

## 4. 输出参数含义

```python
print(f"截距: {model.intercept_:.2f}")
print("斜率(coefficients):")
for name, coef in zip(features_names, model.coef_):
    print(f"{name}: {coef:.2f}")
```

解释：
- `intercept_` 是 $\beta_0$
- `coef_` 是 $\beta_1 ... \beta_p$

正负号含义：
- **正数**：特征值增大，预测值增大
- **负数**：特征值增大，预测值减小

---

## 5. 特征名处理逻辑

代码中有一段逻辑，用于处理特征名：

```python
if feature_names is not None:
    features_names = feature_names
elif hasattr(X_train, 'columns'):
    features_names = list(X_train.columns)
else:
    features_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
```

这样做的目的：
- 如果传入了特征名就用它
- 如果是 DataFrame 就自动取列名
- 如果是 NumPy 数组就生成默认名

---

## 6. 小结

- 训练模块输出模型参数，方便解释模型
- 线性回归是可解释性很强的模型
- 你可以通过系数判断特征的重要性和方向
