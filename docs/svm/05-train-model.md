# 模型训练（train_model.py）

本模块负责训练 SVM 分类器，并输出支持向量信息。

---

## 1. 使用的模型

```python
from sklearn.svm import SVC
```

`SVC` 是 sklearn 的支持向量机分类器实现。

---

## 2. 关键参数

```python
model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state)
```

参数说明：

- `C`：惩罚系数，控制间隔与误差的权衡
- `kernel`：核函数类型（`linear` / `rbf` / `poly` / `sigmoid`）
- `gamma`：核函数系数（常用 `scale`）

---

## 3. 训练过程

```python
model.fit(X_train, y_train)
```

训练完成后可以查看：

- `model.n_support_`：每个类别的支持向量数量
- 支持向量位置用于决定分割边界

---

## 4. 装饰器与计时

- `@print_func_info`：打印函数调用信息
- `@timeit` 与 `timer`：统计训练耗时

---

## 5. 小结

- SVM 的效果高度依赖 `C` 与 `gamma`
- 支持向量数量越多，模型越复杂
- 训练后可在可视化模块直观看到决策边界
