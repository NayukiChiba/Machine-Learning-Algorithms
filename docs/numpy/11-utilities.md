# 实用函数

> 对应代码: [11_utilities.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/11_utilities.py)

## 学习目标

- 掌握 NumPy 的实用函数
- 学会排序和搜索操作
- 理解集合操作

## 排序函数

| 函数              | 说明             |
| ----------------- | ---------------- |
| `np.sort(arr)`    | 返回排序后的副本 |
| `arr.sort()`      | 原地排序         |
| `np.argsort(arr)` | 返回排序后的索引 |

### sort 排序

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# 返回副本，原数组不变
sorted_arr = np.sort(arr)  # [1, 1, 2, 3, 4, 5, 6, 9]

# 原地排序
arr.sort()  # arr 被修改
```

### argsort 排序索引

```python
arr = np.array([3, 1, 4, 1, 5])

indices = np.argsort(arr)  # [1, 3, 0, 2, 4]

# 使用索引重建排序后的数组
arr[indices]  # [1, 1, 3, 4, 5]
```

### 二维数组排序

```python
arr = np.array([[3, 1, 2], [6, 4, 5]])

np.sort(arr, axis=1)  # 每行排序
np.sort(arr, axis=0)  # 每列排序
```

## 唯一值

```python
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

# 获取唯一值
np.unique(arr)  # [1, 2, 3, 4]

# 获取唯一值和首次出现的索引
unique, indices = np.unique(arr, return_index=True)

# 获取唯一值和计数
unique, counts = np.unique(arr, return_counts=True)
# unique: [1, 2, 3, 4]
# counts: [1, 2, 3, 4]
```

## 集合操作

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])

# 交集
np.intersect1d(a, b)  # [3, 4, 5]

# 并集
np.union1d(a, b)  # [1, 2, 3, 4, 5, 6, 7]

# 差集 (在 a 不在 b)
np.setdiff1d(a, b)  # [1, 2]

# 对称差集
np.setxor1d(a, b)  # [1, 2, 6, 7]

# 成员检测
np.in1d(a, [2, 4])  # [False, True, False, True, False]
```

## 搜索函数

| 函数                  | 说明               |
| --------------------- | ------------------ |
| `np.where(condition)` | 返回满足条件的索引 |
| `np.argmax(arr)`      | 最大值的索引       |
| `np.argmin(arr)`      | 最小值的索引       |
| `np.nonzero(arr)`     | 非零元素的索引     |

```python
arr = np.array([1, 5, 2, 8, 3, 9, 4, 7])

np.argmax(arr)  # 5 (最大值 9 的索引)
np.argmin(arr)  # 0 (最小值 1 的索引)

np.where(arr > 5)  # (array([1, 3, 5, 7]),)
```

## 裁剪和取整

### clip 裁剪

```python
arr = np.array([1, 5, 10, 15, 20])

np.clip(arr, 5, 15)  # [5, 5, 10, 15, 15]
```

### 取整函数

```python
arr = np.array([1.2, 2.5, 3.7, -1.2, -2.5])

np.floor(arr)  # [ 1.,  2.,  3., -2., -3.] 向下取整
np.ceil(arr)   # [ 2.,  3.,  4., -1., -2.] 向上取整
np.round(arr)  # [ 1.,  2.,  4., -1., -2.] 四舍五入
np.trunc(arr)  # [ 1.,  2.,  3., -1., -2.] 截断取整
```

## 复制操作

| 方式            | 类型   | 修改是否影响原数组 |
| --------------- | ------ | ------------------ |
| `arr_ref = arr` | 引用   | 是                 |
| `arr.view()`    | 视图   | 是                 |
| `arr.copy()`    | 深拷贝 | 否                 |

```python
arr = np.array([1, 2, 3, 4, 5])

# 引用（指向同一对象）
arr_ref = arr
arr_ref[0] = 100  # arr 也会变

# 视图（共享数据）
arr_view = arr.view()
arr_view[1] = 200  # arr 也会变

# 副本（独立数据）
arr_copy = arr.copy()
arr_copy[2] = 300  # arr 不变
```

## 其他实用函数

| 函数                   | 说明                       |
| ---------------------- | -------------------------- |
| `np.array_equal(a, b)` | 判断两数组是否相同         |
| `np.allclose(a, b)`    | 判断两数组是否接近（浮点） |
| `np.any(arr)`          | 是否有任意 True            |
| `np.all(arr)`          | 是否全为 True              |

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/11_utilities.py
```
