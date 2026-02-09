# 模型评估（evaluate_model.py）

本模块计算分类任务常用指标，并输出分类报告。

---

## 1. 使用指标

- Accuracy：整体正确率
- Precision：预测为正样本中正确的比例
- Recall：真实正样本被找回的比例
- F1-Score：Precision 与 Recall 的调和平均

---

## 2. 关键代码

```python
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

---

## 3. 评估指标计算

```python
train_acc = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(...)
train_recall = recall_score(...)
train_f1 = f1_score(...)
```

同样对测试集计算指标，并打印对比。

---

## 4. 泛化检查

代码里用训练/测试准确率差异做简单判断：

- 差异很小：模型泛化较好
- 差异较大：可能过拟合

---

## 5. 分类报告

```python
print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))
```

报告包含每个类别的 Precision / Recall / F1 / Support。

---

## 6. 小结

- 评估模块帮助判断模型是否可靠
- 分类任务不只看 Accuracy，尤其在类别不平衡时
