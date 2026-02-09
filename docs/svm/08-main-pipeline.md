# 一键运行完整流程（main.py）

`main.py` 是 SVC 项目的入口文件，按顺序调用所有模块。

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

## 2. 关键代码

```python
df = generate_data(n_samples=400, noise=0.2, random_state=42)
class_ratio = explore_data(df)
visualize_data(df)
X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = preprocess_data(df)
model = train_model(X_train, y_train, C=1.5, kernel="rbf", gamma="scale")
y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
visualize_results(model, scaler, X_train_orig, X_test_orig, y_train, y_test, y_train_pred, y_test_pred)
```

---

## 3. 常用可调参数

- `n_samples`：样本数量
- `noise`：噪声强度
- `test_size`：训练/测试划分比例
- `C`：惩罚系数
- `kernel`：核函数类型
- `gamma`：核参数

---

## 4. 小结

`main.py` 的目的就是：

> 一键跑完整个 SVC 二分类流程。
