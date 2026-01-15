# 连接与分割

> 对应代码: [09_concat_split.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/09_concat_split.py)

## 学习目标

- 掌握 NumPy 数组的连接方法
- 学会分割数组
- 理解不同轴的连接和分割

## 连接函数

| 函数               | 说明                  |
| ------------------ | --------------------- |
| `np.concatenate()` | 沿指定轴连接          |
| `np.vstack()`      | 垂直堆叠（沿 axis=0） |
| `np.hstack()`      | 水平堆叠（沿 axis=1） |
| `np.stack()`       | 沿新轴堆叠            |
| `np.dstack()`      | 沿深度方向堆叠        |

## 分割函数

| 函数               | 说明           |
| ------------------ | -------------- |
| `np.split()`       | 沿指定轴分割   |
| `np.vsplit()`      | 垂直分割       |
| `np.hsplit()`      | 水平分割       |
| `np.array_split()` | 允许不均匀分割 |

## 连接 vs 堆叠

| 对比     | concatenate/vstack/hstack | stack/dstack         |
| -------- | ------------------------- | -------------------- |
| 维度变化 | 不增加维度                | 增加一个维度         |
| 形状要求 | 除连接轴外维度相同        | 所有数组形状完全相同 |

## concatenate 连接

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 沿 axis=0 连接（垂直）
np.concatenate([A, B], axis=0)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]
# 形状: (4, 2)

# 沿 axis=1 连接（水平）
np.concatenate([A, B], axis=1)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]
# 形状: (2, 4)
```

## vstack 和 hstack

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)

# vstack: 垂直堆叠
B = np.array([[7, 8, 9]])  # (1, 3)
np.vstack([A, B])  # (3, 3)

# hstack: 水平堆叠
C = np.array([[10], [20]])  # (2, 1)
np.hstack([A, C])  # (2, 4)
```

## stack 沿新轴堆叠

```python
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[5, 6], [7, 8]])  # (2, 2)

# 沿 axis=0 堆叠
np.stack([A, B], axis=0).shape  # (2, 2, 2)

# 沿 axis=2 堆叠
np.stack([A, B], axis=2).shape  # (2, 2, 2)
```

> [!NOTE]
> `stack` 会增加一个新维度，所有数组形状必须完全相同。

## dstack 深度堆叠

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

np.dstack([A, B]).shape  # (2, 2, 2)
# 在第三个轴（深度）堆叠
```

## split 分割

```python
arr = np.arange(12).reshape(4, 3)
# [[ 0,  1,  2],
#  [ 3,  4,  5],
#  [ 6,  7,  8],
#  [ 9, 10, 11]]

# 沿 axis=0 分成 2 份
parts = np.split(arr, 2, axis=0)
# 返回 2 个 (2, 3) 数组

# 沿 axis=1 分成 3 份
parts = np.split(arr, 3, axis=1)
# 返回 3 个 (4, 1) 数组

# 指定分割位置
parts = np.split(arr, [1, 3], axis=0)
# 分成 3 份: [:1], [1:3], [3:]
```

## vsplit 和 hsplit

```python
arr = np.arange(12).reshape(4, 3)

# vsplit: 垂直分割
np.vsplit(arr, 2)  # 分成 2 个 (2, 3) 数组

# hsplit: 水平分割
np.hsplit(arr, 3)  # 分成 3 个 (4, 1) 数组
```

## array_split 不均匀分割

```python
arr = np.arange(10)  # [0, 1, 2, ..., 9]

# 分成 3 份（不均匀）
parts = np.array_split(arr, 3)
# [array([0, 1, 2, 3]),  # 4 个元素
#  array([4, 5, 6]),     # 3 个元素
#  array([7, 8, 9])]     # 3 个元素
```

> [!TIP]
> `split()` 要求均匀分割，`array_split()` 允许不均匀分割。

## 常用场景

| 场景            | 方法                              |
| --------------- | --------------------------------- |
| 垂直拼接表格    | `vstack` 或 `concatenate(axis=0)` |
| 水平拼接特征    | `hstack` 或 `concatenate(axis=1)` |
| 添加批次维度    | `stack(axis=0)`                   |
| 分割训练/测试集 | `split` 或 `array_split`          |

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/09_concat_split.py
```
