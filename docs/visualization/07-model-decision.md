# 模型决策过程可视化

> 对应代码: [07_model_decision.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Visualization/07_model_decision.py)

## 决策边界

```python
# 创建网格
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# 预测并绘制
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3)
ax.scatter(X[:, 0], X[:, 1], c=y)
```

## 决策树可视化

```python
from sklearn.tree import plot_tree

plot_tree(clf, filled=True, feature_names=names)
```

## 特征重要性

```python
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
ax.barh(range(len(importances)), importances[indices])
```

## 练习

```bash
python Basic/Visualization/07_model_decision.py
```
