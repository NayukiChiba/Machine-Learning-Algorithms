---
title: NumPy 属性与 dtype
outline: deep
---

# NumPy 属性与 dtype

## 本章目标

1. 理解数组结构属性 `shape` / `ndim` / `size`。
2. 掌握类型与内存属性 `dtype` / `itemsize` / `nbytes`。
3. 熟悉常用 `dtype` 的范围与精度。
4. 掌握 `astype` 的类型转换用法。
5. 掌握布尔数组与条件筛选的基础模式。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `arr.shape` | 属性 | 各维度长度组成的元组 |
| `arr.ndim` | 属性 | 维度数量 |
| `arr.size` | 属性 | 元素总数 |
| `arr.dtype` | 属性 | 元素数据类型 |
| `arr.itemsize` | 属性 | 每个元素占用字节数 |
| `arr.nbytes` | 属性 | 总字节数 = `size × itemsize` |
| `arr.astype(...)` | 方法 | 返回转换类型后的新数组 |
| `np.iinfo(...)` | 函数 | 查询整数类型的取值范围 |
| `np.finfo(...)` | 函数 | 查询浮点类型的精度参数 |
| `arr > x` 等 | 表达式 | 生成布尔数组（比较运算） |
| `arr[mask]` | 表达式 | 用布尔数组进行索引筛选 |

## 数组结构属性

### 属性速览

| 属性       | 返回数据类型       | 含义                       |
| ---------- | ------------------ | -------------------------- |
| `arr.shape` | `tuple[int, ...]` | 各维度长度组成的元组       |
| `arr.ndim`  | `int`             | 维度数量                   |
| `arr.size`  | `int`             | 元素总数                   |

### 示例代码

```python
import numpy as np

arr = np.random.random((3, 4))
print(arr)
print(f"shape: {arr.shape}")
print(f"ndim: {arr.ndim}")
print(f"size: {arr.size}")
print(f"行数: {arr.shape[0]}")
print(f"列数: {arr.shape[1]}")
```

### 输出

```text
[[0.86395484 0.55333229 0.49186088 0.65651355]
 [0.65818868 0.01198379 0.0954384  0.54282681]
 [0.3904872  0.28345003 0.64304407 0.45011224]]
shape: (3, 4)
ndim: 2
size: 12
行数: 3
列数: 4
```

### 理解重点

- `shape=(3, 4)` 表示 3 行 4 列。
- `ndim == len(shape)`。
- `size == shape[0] × shape[1] × ...`。

## 内存与数据类型属性

### 属性速览

| 属性         | 返回数据类型    | 含义                                   |
| ------------ | --------------- | -------------------------------------- |
| `arr.dtype`    | `numpy.dtype` | 元素数据类型对象                       |
| `arr.itemsize` | `int`         | 单个元素占用字节数                     |
| `arr.nbytes`   | `int`         | 数组总字节数，等于 `size × itemsize`   |

### 示例代码

```python
import numpy as np

arr = np.random.random((3, 4))
print(f"dtype: {arr.dtype}")
print(f"itemsize: {arr.itemsize}")
print(f"nbytes: {arr.nbytes}")
print(f"验证 size × itemsize: {arr.size * arr.itemsize}")
```

### 输出

```text
dtype: float64
itemsize: 8
nbytes: 96
验证 size × itemsize: 96
```

### 理解重点

- `float64` 每个元素占 8 字节。
- 大数组可先估算内存：`arr.nbytes / 1024**2` 得到 MB 数。
- 存储大量数据时可考虑 `float32` 换取一半内存。

## 常见数据类型

### 常见 dtype 一览

| dtype 名称    | 类别         | 每元素字节数 | 典型取值范围 / 精度            |
| ------------- | ------------ | ------------ | ------------------------------ |
| `bool_`       | 布尔         | 1            | `True` / `False`               |
| `int8`        | 有符号整数   | 1            | `[-128, 127]`                  |
| `int16`       | 有符号整数   | 2            | `[-32768, 32767]`              |
| `int32`       | 有符号整数   | 4            | `[-2³¹, 2³¹−1]`                |
| `int64`       | 有符号整数   | 8            | `[-2⁶³, 2⁶³−1]`                |
| `uint8`       | 无符号整数   | 1            | `[0, 255]`                     |
| `uint16`      | 无符号整数   | 2            | `[0, 65535]`                   |
| `float16`     | 半精度浮点   | 2            | 约 3 位有效数字                |
| `float32`     | 单精度浮点   | 4            | 约 6~7 位有效数字              |
| `float64`     | 双精度浮点   | 8            | 约 15 位有效数字               |
| `complex64`   | 复数         | 8            | 实部 + 虚部各 `float32`        |
| `complex128`  | 复数         | 16           | 实部 + 虚部各 `float64`        |

### `np.iinfo`

#### 作用

查询整数类型的取值范围与位数等元信息。

#### 重点方法

```python
np.iinfo(int_type)
```

#### 参数

| 参数名     | 本例取值      | 说明                                   |
| ---------- | ------------- | -------------------------------------- |
| `int_type` | `np.int32` 等 | 任意 NumPy 整数 dtype 或整数数组的类型 |

#### 返回内容

| 属性     | 类型    | 含义             |
| -------- | ------- | ---------------- |
| `.min`   | `int`   | 类型最小值       |
| `.max`   | `int`   | 类型最大值       |
| `.bits`  | `int`   | 占用位数         |
| `.dtype` | `dtype` | 对应 dtype 对象  |

#### 示例代码

```python
import numpy as np

for dtype in [np.int8, np.int16, np.int32, np.int64]:
    info = np.iinfo(dtype)
    print(f"{dtype.__name__}: [{info.min}, {info.max}]")
```

#### 输出

```text
int8: [-128, 127]
int16: [-32768, 32767]
int32: [-2147483648, 2147483647]
int64: [-9223372036854775808, 9223372036854775807]
```

### `np.finfo`

#### 作用

查询浮点类型的精度参数（机器精度、有效位数、最小/最大值等）。

#### 重点方法

```python
np.finfo(dtype)
```

#### 参数

| 参数名  | 本例取值        | 说明                              |
| ------- | --------------- | --------------------------------- |
| `dtype` | `np.float32` 等 | 任意 NumPy 浮点 dtype 或浮点数组  |

#### 返回内容

| 属性          | 类型    | 含义                                   |
| ------------- | ------- | -------------------------------------- |
| `.eps`        | `float` | 机器精度，使 `1 + eps > 1` 的最小正数  |
| `.min`        | `float` | 类型可表示的最小（负）值               |
| `.max`        | `float` | 类型可表示的最大（正）值               |
| `.precision`  | `int`   | 十进制有效位数                         |
| `.bits`       | `int`   | 占用位数                               |
| `.resolution` | `float` | 该类型近似分辨率                       |

#### 示例代码

```python
import numpy as np

for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"{dtype.__name__}: 精度 {info.precision} 位, eps={info.eps}")
```

#### 输出

```text
float16: 精度 3 位, eps=0.000977
float32: 精度 6 位, eps=1.1920929e-07
float64: 精度 15 位, eps=2.220446049250313e-16
```

## 类型转换

### `np.ndarray.astype`

#### 作用

将数组元素转换为另一种 `dtype`，返回新数组（通常不修改原数组）。

#### 重点方法

```python
arr.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
```

#### 参数

| 参数名    | 本例取值          | 说明                                                                                       |
| --------- | ----------------- | ------------------------------------------------------------------------------------------ |
| `dtype`   | `np.int32`、`str` | 目标数据类型                                                                               |
| `order`   | `'K'`（默认）     | 内存布局：`'K'` 保持原样、`'C'` 行优先、`'F'` 列优先、`'A'` 任意                           |
| `casting` | `'unsafe'`（默认）| 类型转换策略：`'no'` / `'equiv'` / `'safe'` / `'same_kind'` / `'unsafe'`，从严格到宽松     |
| `subok`   | `True`（默认）    | `True` 时保留子类类型；`False` 强制返回基础 `ndarray`                                      |
| `copy`    | `True`（默认）    | `True` 总是复制；`False` 时，若类型与内存布局都满足则返回原数组视图                        |

#### 示例代码

```python
import numpy as np

arr_float = np.array([1.5, 2.7, 3.2, 4.8])
arr_int = arr_float.astype(np.int32)
arr_str = arr_float.astype(str)

print(f"原数组: {arr_float}, dtype={arr_float.dtype}")
print(f"转 int32: {arr_int}, dtype={arr_int.dtype}")
print(f"转 str: {arr_str}, dtype={arr_str.dtype}")
```

#### 输出

```text
原数组: [1.5 2.7 3.2 4.8], dtype=float64
转 int32: [1 2 3 4], dtype=int32
转 str: ['1.5' '2.7' '3.2' '4.8'], dtype=<U32
```

#### 理解重点

- 浮点转整数是**截断**（向零取整），不是四舍五入。
- `astype` 返回**新数组**，不原地修改。
- 频繁 `astype` 会产生拷贝，大数组场景注意性能。

## 布尔数组与条件筛选

### 作用

通过比较运算（`>`、`<`、`==`、`!=` 等）生成与原数组同形状的布尔数组，再将其作为索引完成条件过滤。这是 NumPy 最常用、也是最高效的筛选模式。

### 常用比较与逻辑运算

| 表达式                          | 含义                   | 返回       |
| ------------------------------- | ---------------------- | ---------- |
| `arr > x`                       | 逐元素大于             | 布尔数组   |
| `arr < x`                       | 逐元素小于             | 布尔数组   |
| `arr == x`                      | 逐元素等于             | 布尔数组   |
| `arr != x`                      | 逐元素不等             | 布尔数组   |
| `(arr > a) & (arr < b)`         | 逻辑与（括号必加）     | 布尔数组   |
| `(arr < a) \| (arr > b)`        | 逻辑或（括号必加）     | 布尔数组   |
| `~mask`                         | 逻辑非                 | 布尔数组   |

### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = arr > 5

print(f"原数组: {arr}")
print(f"mask (arr > 5): {mask}")
print(f"mask.dtype: {mask.dtype}")
print(f"筛选结果 arr[mask]: {arr[mask]}")
print(f"大于 5 的元素个数: {mask.sum()}")
```

### 输出

```text
原数组: [ 1  2  3  4  5  6  7  8  9 10]
mask (arr > 5): [False False False False False  True  True  True  True  True]
mask.dtype: bool
筛选结果 arr[mask]: [ 6  7  8  9 10]
大于 5 的元素个数: 5
```

### 理解重点

- 比较表达式返回**同形状**的布尔数组。
- 布尔数组可直接作为索引，完成无显式循环的过滤。
- `mask.sum()` 利用 `True=1`、`False=0` 统计命中个数。
- 多条件组合必须用 `&` / `|` / `~`（而不是 Python 的 `and` / `or` / `not`），且每个条件都要加括号，避免运算符优先级出错。

## 常见坑

1. 整数与浮点混合运算时，NumPy 会自动提升为浮点，可能不符预期。
2. `astype` 每次都拷贝，大数组频繁调用会成为性能瓶颈。
3. 大数组先用 `nbytes` 估算内存，再决定 dtype，避免 OOM。
4. 布尔组合用 `&` / `|` 时忘加括号，会因运算符优先级报 `TypeError` 或得到错结果。
5. `arr.astype(int)` 会截断小数，若需四舍五入应先 `np.round`。

## 小结

- 本章是"理解数组本体"的关键章节。
- `shape` 与 `dtype` 决定了数据的几何形状和数值能力。
- 后续所有运算、索引、广播本质上都建立在这些属性之上。
- 布尔数组是 NumPy 过滤的第一选择，优先于显式 `for` 循环。
