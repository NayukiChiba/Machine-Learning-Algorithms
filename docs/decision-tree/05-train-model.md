# 模型训练（train_model.py）

这一模块训练决策树回归模型，并输出树的深度和叶子节点数量。

---

## 1. 核心模型

```python
from sklearn.tree import DecisionTreeRegressor
```

这是 sklearn 提供的 CART 回归树实现。

---

## 2. 关键参数

```python
DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=6,
    min_samples_leaf=3,
    random_state=42
)
```

参数解释：
- `max_depth`：限制树的最大深度，防止过拟合
- `min_samples_split`：节点继续分裂需要的最小样本数
- `min_samples_leaf`：叶子节点最少样本数
- `random_state`：结果可复现

---

## 3. 训练过程

```python
model.fit(X_train, y_train)
```

训练后可以输出：
- 树深度：`model.get_depth()`
- 叶子节点数：`model.get_n_leaves()`

---

## 4. 装饰器与计时

代码中用了两个工具：

- `@print_func_info`：打印函数调用
- `@timeit` + `timer`：记录训练耗时

这能帮助你观察模型训练速度。

---

## 5. 小结

- 决策树回归训练非常快
- 参数越大，树越复杂，越容易过拟合
- 你可以调节参数观察指标变化
