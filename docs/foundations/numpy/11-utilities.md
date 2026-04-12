---
title: NumPy 实用函数
outline: deep
---

# NumPy 实用函数

> 对应脚本：`Basic/Numpy/11_utilities.py`  
> 运行方式：`python Basic/Numpy/11_utilities.py`

## 本章目标

1. 掌握排序、唯一值、集合操作。
2. 掌握索引搜索、裁剪、取整与复制语义。
3. 理解 `copy` / `view` / 引用赋值的区别。

## 重点方法速览

| 分类 | 方法 |
|---|---|
| 排序 | `np.sort`、`np.argsort` |
| 去重统计 | `np.unique(..., return_counts=True)` |
| 集合 | `np.intersect1d`、`np.union1d`、`np.setdiff1d`、`np.setxor1d` |
| 搜索 | `np.argmax`、`np.argmin`、`np.where`、`np.nonzero` |
| 数值处理 | `np.clip`、`np.floor`、`np.ceil`、`np.round`、`np.trunc` |
| 复制 | 赋值引用、`view()`、`copy()` |

## 1. 排序：`sort` 与 `argsort`

### 参数速览（本节）

适用 API（分项）：

1. `np.sort(a, axis=-1, kind=None, order=None, stable=None)`
2. `np.argsort(a, axis=-1, kind=None, order=None, stable=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a`（`sort`） | `arr=[3,1,4,1,5,9,2,6]` | 返回排序后的值 |
| `axis`（`sort`） | `-1`（默认） | 沿最后一个轴排序 |
| `kind` / `stable`（`sort`） | `None` / `None` | 指定排序算法与稳定性策略 |
| `a`（`argsort`） | `arr` | 返回排序后对应索引 |
### 示例代码

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.sort(arr))
idx = np.argsort(arr)
print(idx)
print(arr[idx])
```

### 结果输出

```text
[1 1 2 3 4 5 6 9]
----------------
[1 3 6 0 2 4 7 5]
----------------
[1 1 2 3 4 5 6 9]
```

### 理解重点

- `sort` 返回排序后的值。
- `argsort` 返回“排序后的下标”，常用于对齐多个数组。

## 2. 唯一值：`unique`

### 参数速览（本节）

适用 API：`np.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, equal_nan=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `ar` | `[1,2,2,3,3,3,4,4,4,4]` | 待去重数组 |
| `return_index` | `True`（第二个示例） | 返回每个唯一值首次出现位置 |
| `return_counts` | `True`（第三个示例） | 返回每个唯一值出现次数 |
| `axis` | `None` | 在扁平化视角下去重 |
### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print(np.unique(arr))
print(np.unique(arr, return_index=True))
print(np.unique(arr, return_counts=True))
```

### 结果输出

```text
[1 2 3 4]
----------------
(array([1, 2, 3, 4]), array([0, 1, 3, 6]))
----------------
(array([1, 2, 3, 4]), array([1, 2, 3, 4]))
```

## 3. 集合操作

### 参数速览（本节）

适用 API（分项）：

1. `np.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)`
2. `np.union1d(ar1, ar2)`
3. `np.setdiff1d(ar1, ar2, assume_unique=False)`
4. `np.setxor1d(ar1, ar2, assume_unique=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `ar1` / `ar2`（`intersect1d`） | `a` / `b` | 返回交集 |
| `ar1` / `ar2`（`union1d`） | `a` / `b` | 返回并集并排序 |
| `ar1` / `ar2`（`setdiff1d`） | `a` / `b`、`b` / `a` | 返回差集，方向不同结果不同 |
| `ar1` / `ar2`（`setxor1d`） | `a` / `b` | 返回对称差集 |
### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])

print(np.intersect1d(a, b))
print(np.union1d(a, b))
print(np.setdiff1d(a, b))
print(np.setdiff1d(b, a))
print(np.setxor1d(a, b))
```

### 结果输出

```text
[3 4 5]
----------------
[1 2 3 4 5 6 7]
----------------
[1 2]
----------------
[6 7]
----------------
[1 2 6 7]
```

> 注：脚本中使用了 `np.in1d`，当前 NumPy 已提示弃用，建议新代码改为 `np.isin`。

## 4. 搜索函数

### 参数速览（本节）

适用 API（分项）：

1. `np.argmax(a, axis=None, out=None, keepdims=False)`
2. `np.argmin(a, axis=None, out=None, keepdims=False)`
3. `np.where(condition, x=None, y=None)`
4. `np.nonzero(a)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` / `axis`（`argmax`） | `arr` / `None` | 返回全局最大值索引 |
| `a` / `axis`（`argmin`） | `arr` / `None` | 返回全局最小值索引 |
| `condition`（`where`） | `arr > 5` | 仅传条件时返回满足条件的索引 |
| `a`（`nonzero`） | `arr2=[0,1,0,2,0,3]` | 返回非零元素索引元组 |
### 示例代码

```python
import numpy as np

arr = np.array([1, 5, 2, 8, 3, 9, 4, 7])
print(np.argmax(arr), np.argmin(arr))

idx = np.where(arr > 5)
print(idx[0])
print(arr[idx])

arr2 = np.array([0, 1, 0, 2, 0, 3])
print(np.nonzero(arr2)[0])
```

### 结果输出

```text
5 0
----------------
[3 5 7]
----------------
[8 9 7]
----------------
[1 3 5]
```

## 5. `clip` 与取整系列

### 参数速览（本节）

适用 API（分项）：

1. `np.clip(a, a_min, a_max, out=None)`
2. `np.floor(a)`
3. `np.ceil(a)`
4. `np.trunc(a)`
5. `np.round(a, decimals=0, out=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a_min` / `a_max`（`clip`） | `5` / `15` | 把值限制在区间 `[5, 15]` |
| `a`（`floor/ceil/trunc`） | `arr_float` | 分别向下取整、向上取整、向零截断 |
| `decimals`（`round`） | `0`（默认） | 控制四舍五入保留位数 |
### 示例代码

```python
import numpy as np

arr = np.array([1, 5, 10, 15, 20])
print(np.clip(arr, 5, 15))

arr_float = np.array([1.2, 2.5, 3.7, -1.2, -2.5, -3.7])
print(np.floor(arr_float))
print(np.ceil(arr_float))
print(np.round(arr_float))
print(np.trunc(arr_float))
```

### 结果输出

```text
[ 5  5 10 15 15]
----------------
[ 1.  2.  3. -2. -3. -4.]
----------------
[ 2.  3.  4. -1. -2. -3.]
----------------
[ 1.  2.  4. -1. -2. -4.]
----------------
[ 1.  2.  3. -1. -2. -3.]
```

## 6. 引用、视图、拷贝

### 参数速览（本节）

适用语法/API（分项）：

1. `arr_ref = arr`
2. `arr.view(dtype=None, type=None)`
3. `arr.copy(order='C')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值（`arr_ref = arr`） | `ndarray` 引用 | 仅创建新引用，指向同一底层数据 |
| `dtype` / `type`（`view`） | `None` / `None` | 创建视图，共享底层内存 |
| `order`（`copy`） | `'C'`（默认） | 创建独立副本，不共享底层数据 |
### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

arr_ref = arr
arr_ref[0] = 100
print(arr)

arr[0] = 1
arr_view = arr.view()
arr_view[1] = 200
print(arr)

arr[1] = 2
arr_copy = arr.copy()
arr_copy[2] = 300
print(arr)
```

### 结果输出

```text
[100   2   3   4   5]
----------------
[  1 200   3   4   5]
----------------
[1 2 3 4 5]
```

### 理解重点

- `arr_ref = arr`：同一对象，改一处全改。
- `view()`：共享数据缓冲区，常会联动。
- `copy()`：独立数据，互不影响。

## 常见坑

1. 把 `view` 当成 `copy` 用，会出现“神秘联动修改”。
2. 对象很大时盲目 `copy` 会导致额外内存开销。
3. `argsort` 结果是索引，不是排序后的值本身。

## 小结

- 本章方法属于“高频工具箱”，日常数据处理会反复使用。
- 熟悉这些函数可以显著减少手写循环和条件分支。
