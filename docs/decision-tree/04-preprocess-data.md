# 数据预处理（preprocess_data.py）

这一模块负责划分训练集/测试集。决策树不需要标准化，但必须保证评估公平。

---

## 1. 分离特征与目标

```python
features = data.drop("price", axis=1)
target = data["price"]
```

- `features` 是输入 X
- `target` 是输出 y

---

## 2. 划分训练集/测试集

```python
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=test_size, random_state=random_state
)
```

- `test_size=0.2`：20% 用作测试集
- `random_state`：保证结果可复现

---

## 3. 为什么不做标准化

决策树只根据“大小比较”做切分，不依赖距离计算：

- 特征缩放不会改变排序关系
- 因此标准化不是必须的

---

## 4. 返回值说明

```text
X_train, X_test, y_train, y_test, features, target
```

- `features` 和 `target` 方便后续可视化或分析

---

## 5. 小结

- 决策树模型训练不需要标准化
- 训练/测试划分是必须的基本步骤
