# 结果可视化（visualize_results.py）

本模块输出混淆矩阵和决策边界图，帮助直观理解模型效果。

---

## 1. 输出目录

```
outputs/SVC/
```

输出文件：

- `04_confusion_matrix.png`
- `05_decision_boundary.png`

---

## 2. 混淆矩阵

```python
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)
```

作用：

- 观察分类错误的类型
- 比如“把 0 判成 1”的数量

---

## 3. 决策边界

核心流程：

```python
grid = DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["x1", "x2"])
grid_scaled = scaler.transform(grid)
zz = model.predict(grid_scaled).reshape(xx.shape)
```

作用：

- 背景颜色表示预测区域
- 支持向量用空心圆标出
- 训练集/测试集分开绘制

---

## 4. 小结

- 混淆矩阵体现分类错误类型
- 决策边界帮助理解 SVC 的分割方式
