# 数据预处理（preprocess_data.py）

这一模块负责 **划分训练集/测试集** 并进行 **特征标准化**。

---

## 1. 为什么要预处理

- 防止“训练集和测试集混用”，保证评估公正
- 特征尺度差异较大时，标准化可以让优化更稳定

---

## 2. 分离特征与目标

```python
features = data.drop("价格", axis=1)
price = data["价格"]
```

- `features` 是输入 $X$
- `price` 是输出 $y$

---

## 3. 划分训练集 / 测试集

```python
X_train, X_test, y_train, y_test = train_test_split(
    features, price, test_size=test_size, random_state=random_state)
```

- `test_size=0.2`：20% 作为测试集
- `random_state`：确保划分结果可复现

划分后打印：

- 训练集样本数
- 测试集样本数
- 训练/测试占比

---

## 4. 标准化（StandardScaler）

标准化公式：

$$
 x' = \frac{x - \mu}{\sigma}
$$

代码：

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

注意：
- **只能在训练集上 fit**
- 测试集只能 transform，否则会泄露信息

---

## 5. 返回值说明

函数返回：

- `X_train_scaled`：标准化后的训练集
- `X_test_scaled`：标准化后的测试集
- `y_train, y_test`：标签
- `scaler`：标准化器（后续可逆变换）
- `X_train, X_test`：原始未标准化数据

为什么要保留原始数据？
- 结果可视化时需要原始尺度
- 解释模型时更直观

---

## 6. 小结

- 本模块不改变标签，只处理特征
- 训练集/测试集分离是机器学习的基本原则
- 标准化让训练更稳定，但别忘了保留原始数据
