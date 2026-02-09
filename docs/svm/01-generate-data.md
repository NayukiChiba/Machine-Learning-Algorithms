# 生成数据（generate_data.py）

本模块负责生成**双月牙二分类数据集**，用于演示 SVM 的非线性分类能力。

---

## 1. 核心目标

- 使用 `make_moons` 生成二维特征 + 二分类标签
- 加入噪声，让数据更接近真实场景
- 返回 `pandas.DataFrame` 供后续模块使用

---

## 2. 代码结构

```python
from utils.decorate import print_func_info

@print_func_info
def generate_data(n_samples=400, noise=0.2, random_state=42) -> DataFrame:
    ...
```

- 使用装饰器 `print_func_info`：调用函数时会输出提示

---

## 3. 关键参数

- `n_samples`：样本数量
- `noise`：噪声大小（越大越难分类）
- `random_state`：随机种子，保证结果可复现

---

## 4. 数据生成

```python
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
```

- `X` 是二维特征
- `y` 是 0/1 的类别标签

---

## 5. 输出 DataFrame

```python
data = DataFrame({
    "x1": X[:, 0],
    "x2": X[:, 1],
    "label": y
})
```

输出结构：

| x1  | x2  | label |
|-----|-----|-------|
| ... | ... | 0/1   |

---

## 6. 你可以尝试的修改

1. 改大 `noise`，观察分类难度变化
2. 改变 `n_samples`，观察决策边界稳定性
3. 换成其他数据集（如 `make_circles`）

---

## 7. 小结

- 这个模块只负责“造数据”
- 后面的探索、训练、评估全部基于这里的输出
