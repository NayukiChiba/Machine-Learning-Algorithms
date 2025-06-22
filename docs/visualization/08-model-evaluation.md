# 模型性能评估可视化

> 对应代码: [08_model_evaluation.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Visualization/08_model_evaluation.py)

## 混淆矩阵

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

## ROC 曲线

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
ax.plot([0, 1], [0, 1], 'r--')
```

## 学习曲线

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(clf, X, y)

ax.plot(train_sizes, train_scores.mean(axis=1), label='Training')
ax.plot(train_sizes, test_scores.mean(axis=1), label='Validation')
```

## 练习

```bash
python Basic/Visualization/08_model_evaluation.py
```
