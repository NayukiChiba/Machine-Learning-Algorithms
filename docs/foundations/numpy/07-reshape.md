---
title: NumPy 变形
outline: deep
---

# NumPy 变形

> 对应脚本：`Basic/Numpy/07_reshape.py`  
> 运行方式：`python Basic/Numpy/07_reshape.py`

## 本章目标

1. 掌握 `reshape`、`flatten`、`ravel` 的区别。
2. 掌握维度调整工具：`transpose`、`squeeze`、`expand_dims`、`newaxis`。
3. 理解“视图 vs 副本”对数据修改的影响。

## 重点方法速览

| 方法 | 作用 |
|---|---|
| `arr.reshape` | 改变形状（元素总数不变） |
| `arr.flatten` | 展平为一维，返回副本 |
| `arr.ravel` | 展平为一维，尽量返回视图 |
| `arr.T` / `np.transpose` | 维度重排/转置 |
| `np.squeeze` | 去掉长度为 1 的维度 |
| `np.expand_dims` / `np.newaxis` | 增加长度为 1 的维度 |
| `np.resize` | 调整大小，不足会循环填充 |

## 1. `reshape` 基础

### 参数速览（本节）

适用 API：`arr.reshape(shape, order='C')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `shape` | `(3,4)`、`(4,3)`、`(2,-1)`、`(-1,6)` | 指定目标形状，`-1` 由 NumPy 自动推导 |
| `order` | `'C'`（默认） | 按行优先顺序解释内存 |
| 元素总数约束 | `12 -> 12` | 变形前后元素总数必须一致，且 `-1` 最多一次 |
### 示例代码

```python
import numpy as np

arr = np.arange(1, 13)
print(arr.reshape(3, 4))
print(arr.reshape(4, 3))
print(arr.reshape(2, -1))
print(arr.reshape(-1, 6))
```

### 结果输出

```text
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
----------------
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
----------------
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
----------------
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
```

### 理解重点

- 仅允许一个 `-1`，由 NumPy 自动推导该维长度。
- 变形前后元素总数必须一致。

## 2. `flatten` 与 `ravel`

### 参数速览（本节）

适用 API（分项）：

1. `arr.flatten(order='C')`
2. `arr.ravel(order='C')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `order`（`flatten`） | `'C'`（默认） | 展平为一维并返回副本 |
| `order`（`ravel`） | `'C'`（默认） | 展平为一维，尽量返回视图 |
| 内存共享语义 | `flatten: 否`、`ravel: 通常是` | 决定修改结果是否影响原数组 |
### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
flat = arr.flatten()
rav = arr.ravel()

flat[0] = 999
rav[1] = 888
print(arr)
```

### 结果输出

```text
[[  1 888   3]
 [  4   5   6]]
```

### 理解重点

- `flatten` 修改不影响原数组（副本）。
- `ravel` 修改通常会影响原数组（视图，若内存连续）。

## 3. 转置与轴重排

### 参数速览（本节）

适用 API/属性（分项）：

1. `arr.T`
2. `np.transpose(a, axes=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值（`arr.T`） | `ndarray` | 返回转置后的数组视图 |
| `a` | `arr_3d` 形状 `(2,3,4)` | 对多维数组做轴重排 |
| `axes` | `None`、`(1,0,2)` | `None` 时反转轴顺序；元组时按指定顺序重排 |
### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.T)

arr_3d = np.arange(24).reshape(2, 3, 4)
print(arr_3d.T.shape)
print(np.transpose(arr_3d, axes=(1, 0, 2)).shape)
```

### 结果输出

```text
[[1 4]
 [2 5]
 [3 6]]
----------------
(4, 3, 2)
----------------
(3, 2, 4)
```

## 4. `squeeze` 与 `expand_dims`

### 参数速览（本节）

适用 API（分项）：

1. `np.squeeze(a, axis=None)`
2. `np.expand_dims(a, axis)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a`（`squeeze`） | 形状 `(1,1,3)` 的数组 | 删除长度为 1 的轴 |
| `axis`（`squeeze`） | `None`（默认） | 删除所有长度为 1 的轴 |
| `a`（`expand_dims`） | `v=[1,2,3]` | 在指定位置插入长度为 1 的维度 |
| `axis`（`expand_dims`） | `0`、`1` | 生成 `(1,3)` 与 `(3,1)` 两种形状 |
### 示例代码

```python
import numpy as np

arr = np.array([[[1, 2, 3]]])
print(arr.shape)
print(np.squeeze(arr).shape)

v = np.array([1, 2, 3])
print(np.expand_dims(v, axis=0).shape)
print(np.expand_dims(v, axis=1).shape)
```

### 结果输出

```text
(1, 1, 3)
----------------
(3,)
----------------
(1, 3)
----------------
(3, 1)
```

## 5. `np.resize`

### 参数速览（本节）

适用 API：`np.resize(a, new_shape)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` | `[1,2,3,4]` | 输入数组 |
| `new_shape` | `(2,4)`、`(3,3)` | 指定目标形状，可改变总元素个数 |
| 填充规则 | 元素循环重复 | 元素不足时按原顺序重复填充 |
### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(np.resize(arr, (2, 4)))
print(np.resize(arr, (3, 3)))
```

### 结果输出

```text
[[1 2 3 4]
 [1 2 3 4]]
----------------
[[1 2 3]
 [4 1 2]
 [3 4 1]]
```

### 理解重点

- 与 `reshape` 不同，`resize` 可改变元素总数。
- 不够时会重复原数组元素填充。

## 6. `np.newaxis`

### 参数速览（本节）

适用语法（分项）：

1. `arr[np.newaxis, :]`
2. `arr[:, np.newaxis]`
3. `np.newaxis`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `np.newaxis` 插入位置（前置） | 第 0 轴 | 把 `(n,)` 扩展为 `(1,n)` |
| `np.newaxis` 插入位置（后置） | 第 1 轴 | 把 `(n,)` 扩展为 `(n,1)` |
| 等价对象 | `None` | 语义上用于显式增加长度为 1 的维度 |
### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr[np.newaxis, :].shape)
print(arr[:, np.newaxis].shape)
```

### 结果输出

```text
(1, 5)
----------------
(5, 1)
```

## 常见坑

1. `reshape` 失败通常是元素总数不匹配。
2. `ravel` 可能返回拷贝，也可能返回视图，依赖内存布局。
3. 写模型输入时，列向量 `(n, 1)` 与行向量 `(1, n)` 语义不同。

## 小结

- 变形本质是“组织数据”的能力。
- 熟练使用维度操作，能显著简化后续广播和矩阵计算。
