# 一键运行完整流程（main.py）

`main.py` 是决策树项目的入口文件，顺序调用所有模块。

---

## 1. 执行顺序

```text
1. 加载数据
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
df = generate_data()
correlation = explore_data(df)
visualize_data(df)
X_train, X_test, y_train, y_test, X, y = preprocess_data(df)
model = train_model(X_train, y_train, max_depth=6, min_samples_split=6, min_samples_leaf=3)
y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
visualize_results(y_train, y_train_pred, y_test, y_test_pred, model, X.columns.tolist())
```

---

## 3. 可调参数

你可以通过以下参数控制模型复杂度：

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

修改这些参数后，观察评估指标和树结构的变化。

---

## 4. 小结

这个脚本的作用就是：

> 用最少的代码，一键跑完整个决策树回归流程。
