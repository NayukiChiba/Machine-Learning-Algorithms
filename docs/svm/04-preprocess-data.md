# 数据预处理（preprocess_data.py）

本模块负责划分训练/测试集并进行标准化。

---

## 1. 为什么要标准化

SVM 对特征尺度敏感：

- 不同量纲会影响距离计算
- 影响支持向量与决策边界

因此必须进行标准化。

---

## 2. 关键代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=test_size,
    random_state=random_state,
    stratify=target,
)
```

- `stratify=target` 保证训练集和测试集的类别比例一致

---

## 3. 标准化流程

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

要点：

- 只在训练集上 `fit`
- 测试集只做 `transform`，避免数据泄漏

---

## 4. 返回值说明

函数返回：

- `X_train_scaled`：标准化后的训练特征
- `X_test_scaled`：标准化后的测试特征
- `y_train`：训练标签
- `y_test`：测试标签
- `scaler`：标准化器（后续绘图要用）
- `X_train`：原始训练特征
- `X_test`：原始测试特征

---

## 5. 小结

- 预处理是 SVM 的关键步骤
- 标准化正确与否会直接影响模型效果
