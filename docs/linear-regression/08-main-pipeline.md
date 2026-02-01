# 一键运行完整流程（main.py）

`main.py` 是整个线性回归项目的“入口”，按照顺序执行所有步骤。

---

## 1. 执行顺序

```text
1. 生成数据
2. 数据探索
3. 数据可视化
4. 数据预处理
5. 模型训练
6. 模型评估
7. 结果可视化
```

---

## 2. 代码结构

```python
from generate_data import generate_data
from explore_data import explore_data
from visualize_data import visualize_data
from preprocess_data import preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model
from visualize_results import visualize_results
```

这些模块刚好对应前面每一个文档章节。

---

## 3. 步骤讲解

### 3.1 生成数据

```python
df = generate_data(n_samples=200, noise=10, random_state=42)
```

你可以在这里控制样本数和噪声大小。

---

### 3.2 数据探索

```python
correlation = explore_data(df)
```

输出统计信息与相关系数，帮助确认数据结构。

---

### 3.3 数据可视化

```python
visualize_data(df)
```

生成分布图、相关性热力图、散点图。

---

### 3.4 数据预处理

```python
X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = preprocess_data(df)
```

- `X_train, X_test` 是标准化后的数据
- `X_train_orig, X_test_orig` 是原始数据（用于可视化）

---

### 3.5 模型训练

```python
model = train_model(X_train, y_train, feature_names=X_train_orig.columns.tolist())
```

模型用标准化数据训练，但系数输出仍使用原始特征名。

---

### 3.6 模型评估

```python
y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
```

得到训练集/测试集预测结果并打印指标。

---

### 3.7 结果可视化

```python
visualize_results(
    y_train, y_train_pred,
    y_test, y_test_pred,
    X_test_orig, X_train_orig.columns
)
```

生成预测效果图、残差分析图、单特征回归效果图。

---

## 4. 常用修改点

- **改变噪声**：`generate_data` 中 `noise`
- **改变训练/测试比例**：`preprocess_data` 中 `test_size`
- **改变训练特征**：修改 `generate_data` 和 `preprocess_data` 中特征列

---

## 5. 小结

`main.py` 的目的就是：

> 把所有模块连起来，让你一键跑完整个线性回归项目。

当你理解了每个模块的原理，这个脚本就是你验证理解的“总开关”。
