---
title: NumPy 属性与 dtype
outline: deep
---

# NumPy 属性与 dtype

> 对应脚本：`Basic/Numpy/03_attributes.py`  
> 运行方式：`python Basic/Numpy/03_attributes.py`

## 本章目标

1. 理解数组结构属性：`shape`、`ndim`、`size`。
2. 掌握类型与内存属性：`dtype`、`itemsize`、`nbytes`。
3. 掌握 `astype` 的安全用法和布尔数组的筛选能力。

## 重点方法与属性速览

| 名称 | 类型 | 含义 |
|---|---|---|
| `arr.shape` | 属性 | 每个维度长度组成的元组 |
| `arr.ndim` | 属性 | 维度数量 |
| `arr.size` | 属性 | 元素总数 |
| `arr.dtype` | 属性 | 元素类型 |
| `arr.itemsize` | 属性 | 每个元素占用字节 |
| `arr.nbytes` | 属性 | 总字节数，等于 `size * itemsize` |
| `arr.astype(dtype)` | 方法 | 返回转换后新数组 |
| `arr > x` | 表达式 | 生成布尔数组 |

## 1. 结构属性：`shape`、`ndim`、`size`

### 参数速览（本节）

适用 API/属性（分项）：

1. `np.random.random(size)`
2. `arr.shape`
3. `arr.ndim`
4. `arr.size`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `size` | `(3, 4)` | 创建 3x4 的随机数组 |
| 返回值（`arr.shape`） | `tuple` | 返回数组形状元组 |
| 返回值（`arr.ndim`） | `int` | 返回数组维度数量 |
| 返回值（`arr.size`） | `int` | 返回元素总数 |
### 示例代码

```python
import numpy as np

arr = np.random.random((3, 4))
print(arr)
print(arr.shape)
print(arr.ndim)
print(arr.size)
```

### 结果输出（示例）

```text
[[0.86395484 0.55333229 0.49186088 0.65651355]
 [0.65818868 0.01198379 0.0954384  0.54282681]
 [0.3904872  0.28345003 0.64304407 0.45011224]]
----------------
(3, 4)
----------------
2
----------------
12
```

### 理解重点

- `shape=(3, 4)` 表示 3 行 4 列。
- `ndim=2` 表示二维数组。
- `size=12` 表示一共 12 个元素。

## 2. 内存属性：`dtype`、`itemsize`、`nbytes`

### 参数速览（本节）

适用属性/表达式（分项）：

1. `arr.dtype`
2. `arr.itemsize`
3. `arr.nbytes`
4. `arr.size * arr.itemsize`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值（`arr.dtype`） | `dtype('float64')`（示例） | 返回元素类型对象 |
| 返回值（`arr.itemsize`） | `8`（示例） | 返回单个元素字节数 |
| 返回值（`arr.nbytes`） | `96`（示例） | 返回总字节数，用于估算数组内存占用 |
| 计算表达式 | - | 验证 `nbytes = size * itemsize` |
### 示例代码

```python
import numpy as np

arr = np.random.random((3, 4))
print(arr.dtype)
print(arr.itemsize)
print(arr.nbytes)
print(arr.size * arr.itemsize)
```

### 结果输出（示例）

```text
float64
----------------
8
----------------
96
----------------
96
```

### 理解重点

- `float64` 每个元素占 8 字节。
- 总内存开销可快速估算：

$$
\text{nbytes} = \text{size} \times \text{itemsize}
$$

## 3. 常见 dtype 范围与精度

脚本中演示了：

- 整数：`int8`、`int16`、`int32`、`int64`
- 浮点：`float16`、`float32`、`float64`
- 其他：`bool`、`complex64`、`complex128`

学习建议：

1. 存储大量数据时，可优先考虑 `float32` 节省内存。
2. 需要更高数值稳定性时，使用 `float64`。

## 4. 类型转换：`astype`

### 参数速览（本节）

适用 API：`arr.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `dtype` | `np.int32`、`str` | 指定目标数据类型 |
| `casting` | `'unsafe'`（默认） | 允许更宽松的类型转换规则 |
| `copy` | `True`（默认） | 通常返回新数组，不改原数组 |
### 示例代码

```python
import numpy as np

arr_float = np.array([1.5, 2.7, 3.2, 4.8])
arr_int = arr_float.astype(np.int32)
arr_str = arr_float.astype(str)

print(arr_float, arr_float.dtype)
print(arr_int, arr_int.dtype)
print(arr_str, arr_str.dtype)
```

### 结果输出

```text
[1.5 2.7 3.2 4.8] float64
----------------
[1 2 3 4] int32
----------------
['1.5' '2.7' '3.2' '4.8'] <U32
```

### 理解重点

- 浮点转整数会截断小数部分（不是四舍五入）。
- `astype` 返回新数组，不会原地修改原数组。

## 5. 布尔数组与条件筛选

### 参数速览（本节）

适用表达式（分项）：

1. `arr > 5`
2. `arr[mask]`
3. `mask.sum()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 阈值 | `5` | 逐元素比较，返回布尔数组 |
| `mask` | `arr > 5` 的结果 | 布尔索引需与原数组形状可对齐 |
| 返回值（`mask.sum()`） | `int` | 统计 `True` 个数，按 `True=1`、`False=0` 计算 |
### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = arr > 5
print(mask)
print(arr[mask])
print(mask.sum())
```

### 结果输出

```text
[False False False False False  True  True  True  True  True]
----------------
[ 6  7  8  9 10]
----------------
5
```

### 理解重点

- 比较表达式返回同形状布尔数组。
- 布尔数组可直接作为索引，完成高效过滤。

## 常见坑

1. 混合整数与浮点时，NumPy 可能自动提升为浮点类型。
2. `astype` 频繁调用会产生拷贝，影响性能。
3. 大数组建议先算内存，再决定 dtype，避免 OOM。

## 小结

- 本章是“理解数组本体”的关键章节。
- 后续所有运算和索引，本质都建立在这些属性之上。
