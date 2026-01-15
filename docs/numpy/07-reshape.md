# 数组变形

> 对应代码: [07_reshape.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/07_reshape.py)

## 学习目标

- 掌握 NumPy 数组的变形方法
- 学会改变数组的形状和维度
- 理解数组转置和轴变换

## 核心方法

| 方法              | 说明             | 返回类型         |
| ----------------- | ---------------- | ---------------- |
| `reshape(shape)`  | 改变数组形状     | 视图（如果可能） |
| `flatten()`       | 展平为一维       | 副本             |
| `ravel()`         | 展平为一维       | 视图（如果可能） |
| `T`               | 转置             | 视图             |
| `transpose(axes)` | 按指定轴顺序转置 | 视图             |

## 视图 vs 副本

| 类型            | 说明             | 修改是否影响原数组 |
| --------------- | ---------------- | ------------------ |
| **视图 (View)** | 与原数组共享数据 | 是                 |
| **副本 (Copy)** | 独立的数据拷贝   | 否                 |

```python
# flatten() 返回副本
flat = arr.flatten()
flat[0] = 999  # 不影响原数组

# ravel() 返回视图
rav = arr.ravel()
rav[0] = 999  # 会影响原数组！
```

## reshape 变形

```python
arr = np.arange(1, 13)  # [1, 2, 3, ..., 12]

# 变形为 3x4
arr.reshape(3, 4)
# [[ 1,  2,  3,  4],
#  [ 5,  6,  7,  8],
#  [ 9, 10, 11, 12]]

# 变形为 4x3
arr.reshape(4, 3)

# 使用 -1 自动计算维度
arr.reshape(2, -1)   # 2x6
arr.reshape(-1, 6)   # 2x6
arr.reshape(3, -1)   # 3x4
```

> [!TIP]
> 使用 `-1` 可以让 NumPy 自动计算该维度的大小，但只能有一个 `-1`。

## flatten 和 ravel 展平

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# flatten: 返回副本
arr.flatten()     # [1, 2, 3, 4, 5, 6]
arr.flatten('F')  # [1, 4, 2, 5, 3, 6]（按列展平）

# ravel: 返回视图（如果可能）
arr.ravel()       # [1, 2, 3, 4, 5, 6]
```

## 转置

### 二维数组转置

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3

arr.T  # 3x2
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

### 三维数组转置

```python
arr_3d = np.arange(24).reshape(2, 3, 4)  # (2, 3, 4)

arr_3d.T.shape  # (4, 3, 2) - 完全反转

# 指定轴顺序
np.transpose(arr_3d, axes=(1, 0, 2)).shape  # (3, 2, 4)
```

## squeeze 和 expand_dims

### squeeze: 移除长度为 1 的维度

```python
arr = np.array([[[1, 2, 3]]])  # (1, 1, 3)

np.squeeze(arr).shape  # (3,)
```

### expand_dims: 增加维度

```python
arr = np.array([1, 2, 3])  # (3,)

np.expand_dims(arr, axis=0).shape  # (1, 3)
np.expand_dims(arr, axis=1).shape  # (3, 1)
```

## np.newaxis 增加维度

```python
arr = np.array([1, 2, 3, 4, 5])  # (5,)

# 增加行维度
arr[np.newaxis, :].shape  # (1, 5)

# 增加列维度
arr[:, np.newaxis].shape  # (5, 1)
```

## resize 调整大小

```python
arr = np.array([1, 2, 3, 4])

# 大小不匹配时会重复元素
np.resize(arr, (2, 4))
# [[1, 2, 3, 4],
#  [1, 2, 3, 4]]

np.resize(arr, (3, 3))
# [[1, 2, 3],
#  [4, 1, 2],
#  [3, 4, 1]]
```

## 常用场景

| 场景         | 方法                                       |
| ------------ | ------------------------------------------ |
| 改变形状     | `reshape()`                                |
| 二维转一维   | `flatten()` 或 `ravel()`                   |
| 行列转换     | `T` 或 `transpose()`                       |
| 添加批次维度 | `expand_dims(axis=0)` 或 `[np.newaxis, :]` |
| 移除单维度   | `squeeze()`                                |

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/07_reshape.py
```
