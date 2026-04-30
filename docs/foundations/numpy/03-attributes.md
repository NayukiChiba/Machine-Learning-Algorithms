---
title: NumPy 属性与 dtype
outline: deep
---

# NumPy 属性与 dtype

## 本章目标

1. 掌握数组结构属性 `shape`、`ndim`、`size` 及其关系
2. 掌握类型与内存属性 `dtype`、`itemsize`、`nbytes`
3. 会用 `np.iinfo` / `np.finfo` 查询整数与浮点类型的取值范围和精度
4. 掌握 `astype` 的类型转换用法
5. 掌握布尔数组与条件筛选的基础模式

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `arr.shape` | 属性 | 各维度长度组成的元组 |
| `arr.ndim` | 属性 | 维度数量，等于 `len(shape)` |
| `arr.size` | 属性 | 元素总数，等于各维度长度乘积 |
| `arr.dtype` | 属性 | 元素数据类型对象 |
| `arr.itemsize` | 属性 | 每个元素占用字节数 |
| `arr.nbytes` | 属性 | 数组总字节数，等于 `size × itemsize` |
| `arr.astype(...)` | 方法 | 返回类型转换后的新数组 |
| `np.iinfo(...)` | 函数 | 查询整数类型的取值范围和位数 |
| `np.finfo(...)` | 函数 | 查询浮点类型的精度参数和范围 |
| `arr > x` 等比较运算符 | 表达式 | 生成布尔数组 |
| `arr[mask]` | 表达式 | 用布尔数组进行索引筛选 |

## 1. 数组结构属性

### `arr.shape`

#### 作用

返回各维度长度组成的元组。例如 `(3, 4)` 表示 3 行 4 列，一维向量为 `(n, )`。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `tuple[int, ...]` | 各维度长度，`ndim` 等于元组长度 |

### `arr.ndim`

#### 作用

返回数组的维度数量（轴数），等于 `len(arr.shape)`。标量为 `0`，向量为 `1`，矩阵为 `2`。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 维度数量 |

### `arr.size`

#### 作用

返回数组中所有元素的总数，等于各维度长度的乘积：$\prod \text{shape}[i]$。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 元素总数 |

### 示例代码

```python
import numpy as np

arr = np.random.random((3, 4))
print(arr)
print(f"shape: {arr.shape}")
print(f"ndim: {arr.ndim}")
print(f"size: {arr.size}")
print(f"行数: {arr.shape[0]}, 列数: {arr.shape[1]}")
```

### 输出

```text
[[0.86395484 0.55333229 0.49186088 0.65651355]
 [0.65818868 0.01198379 0.0954384  0.54282681]
 [0.3904872  0.28345003 0.64304407 0.45011224]]
shape: (3, 4)
ndim: 2
size: 12
行数: 3, 列数: 4
```

### 理解重点

- `ndim == len(shape)`，始终成立
- `size == shape[0] × shape[1] × ...`，始终成立
- `shape` 是后续索引、广播、变形的基础——拿到数组先看 `shape`

## 2. 内存与数据类型属性

### `arr.dtype`

#### 作用

返回数组元素的数据类型对象，决定了每个元素的存储方式和取值范围。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `numpy.dtype` | 数据类型描述对象，如 `float64`、`int32` |

### `arr.itemsize`

#### 作用

返回每个元素占用的字节数。`float64` 为 8 字节，`float32` 为 4 字节。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 单个元素的字节数 |

### `arr.nbytes`

#### 作用

返回数组占用的总字节数，计算公式：$\text{nbytes} = \text{size} \times \text{itemsize}$。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 数组总字节数 |

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

- `float64` 每元素 8 字节，`float32` 每元素 4 字节——内存翻倍
- 大数组先估算内存：`arr.nbytes / 1024**2` 得 MB 数
- 内存紧张时可考虑 `float32` 替换 `float64`

## 3. 常见数据类型

### 常见 dtype 一览

| dtype 名称 | 类别 | 每元素字节数 | 典型取值范围 / 精度 |
|---|---|---|---|
| `bool_` | 布尔 | 1 | `True` / `False` |
| `int8` | 有符号整数 | 1 | $[-128, 127]$ |
| `int16` | 有符号整数 | 2 | $[-32768, 32767]$ |
| `int32` | 有符号整数 | 4 | $[-2^{31}, 2^{31}-1]$ |
| `int64` | 有符号整数 | 8 | $[-2^{63}, 2^{63}-1]$ |
| `uint8` | 无符号整数 | 1 | $[0, 255]$ |
| `float16` | 半精度浮点 | 2 | 约 3 位有效数字 |
| `float32` | 单精度浮点 | 4 | 约 6~7 位有效数字 |
| `float64` | 双精度浮点 | 8 | 约 15 位有效数字 |
| `complex64` | 复数 | 8 | 实部 + 虚部各 `float32` |
| `complex128` | 复数 | 16 | 实部 + 虚部各 `float64` |

### `np.iinfo`

#### 作用

查询整数类型的元信息：最小值、最大值、位数。返回一个包含这些属性值的对象。

#### 重点方法

```python
np.iinfo(int_type)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `int_type` | `dtype` | 任意 NumPy 整数 dtype 或整数数组的类型 | `np.int32` |

#### 返回内容

| 属性 | 类型 | 含义 |
|---|---|---|
| `.min` | `int` | 该类型可表示的最小值 |
| `.max` | `int` | 该类型可表示的最大值 |
| `.bits` | `int` | 占用的二进制位数 |
| `.dtype` | `dtype` | 对应的 dtype 对象 |

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

查询浮点类型的精度参数：机器精度、有效位数、最小/最大值等。

#### 重点方法

```python
np.finfo(dtype)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `dtype` | `dtype` | 任意 NumPy 浮点 dtype 或浮点数组的类型 | `np.float32` |

#### 返回内容

| 属性 | 类型 | 含义 |
|---|---|---|
| `.eps` | `float` | 机器精度，使 $1 + \varepsilon > 1$ 的最小正数 |
| `.min` | `float` | 该类型可表示的最小值（最负） |
| `.max` | `float` | 该类型可表示的最大值（最正） |
| `.precision` | `int` | 十进制有效位数 |
| `.bits` | `int` | 占用位数 |
| `.resolution` | `float` | 该类型的近似分辨率 |

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

## 4. 类型转换

### `ndarray.astype`

#### 作用

将数组元素转换为目标 `dtype`，返回新数组。浮点转整数是**截断**（向零取整），不是四舍五入。

#### 重点方法

```python
arr.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `dtype` | `dtype` 或 `str` | 目标数据类型 | `np.int32`、`str` |
| `order` | `str` | 内存布局：`'K'` 保持原样 / `'C'` 行优先 / `'F'` 列优先 / `'A'` 任意，默认为 `'K'` | `'C'` |
| `casting` | `str` | 类型转换策略：`'no'` / `'equiv'` / `'safe'` / `'same_kind'` / `'unsafe'`，从严格到宽松，默认为 `'unsafe'` | `'safe'` |
| `subok` | `bool` | `True` 保留子类类型，默认为 `True` | `False` |
| `copy` | `bool` | `True` 总是复制，默认为 `True` | `False` |

#### 示例代码

```python
import numpy as np

arrFloat = np.array([1.5, 2.7, 3.2, 4.8])
arrInt = arrFloat.astype(np.int32)
arrStr = arrFloat.astype(str)

print(f"原数组: {arrFloat}, dtype={arrFloat.dtype}")
print(f"转 int32: {arrInt}, dtype={arrInt.dtype}")
print(f"转 str: {arrStr}, dtype={arrStr.dtype}")
```

#### 输出

```text
原数组: [1.5 2.7 3.2 4.8], dtype=float64
转 int32: [1 2 3 4], dtype=int32
转 str: ['1.5' '2.7' '3.2' '4.8'], dtype=<U32
```

#### 理解重点

- 浮点转整数是**截断**（向零取整），`1.9` → `1`，`-1.9` → `-1`——如需四舍五入先用 `np.round`
- `astype` 返回新数组，不修改原数组——每次调用都有拷贝开销
- 大数组频繁类型转换是性能瓶颈，应在创建时指定正确的 `dtype`

## 5. 布尔数组与条件筛选

#### 作用

通过比较运算符（`>`、`<`、`==`、`!=` 等）生成与原数组同形状的布尔数组，再用布尔数组作为索引完成条件过滤。这是 NumPy 最常用、最高效的筛选模式。

### 运算符一览

| 运算符 | 含义 | 返回 |
|---|---|---|
| `>` | 逐元素大于 | 布尔数组 |
| `<` | 逐元素小于 | 布尔数组 |
| `>=` | 逐元素大于等于 | 布尔数组 |
| `<=` | 逐元素小于等于 | 布尔数组 |
| `==` | 逐元素等于 | 布尔数组 |
| `!=` | 逐元素不等 | 布尔数组 |

组合条件使用 `&`（与）、`|`（或）、`~`（非），**不能**使用 Python 的 `and`/`or`/`not`，每个条件必须加括号。

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

- 比较表达式返回同形状的布尔数组
- 布尔数组可直接作为索引完成无显式循环的过滤
- `mask.sum()` 利用 $True=1$、$False=0$ 统计命中个数
- 多条件必须用 `&` / `|` / `~`，每个条件必须加括号——`(arr >= 3) & (arr <= 7)` 而非 `arr >= 3 & arr <= 7`

## 常见坑

1. 整数与浮点混合运算时 NumPy 自动提升为浮点，结果类型可能与预期不同
2. `astype` 每次都拷贝，大数组频繁调用是性能瓶颈
3. 大数组先用 `nbytes` 估算内存占用量，再决定用哪种 `dtype`
4. 布尔组合用 `&` / `|` 忘记加括号，会因运算符优先级报 `TypeError` 或得出错误结果
5. `arr.astype(int)` 是截断不是四舍五入——`1.9` → `1`

## 小结

- `shape` 与 `dtype` 是理解数组的两个核心维度：形状决定了数据组织，类型决定了数值能力
- 后续所有运算、索引、广播本质上都建立在属性之上
- 布尔数组是 NumPy 过滤的第一选择，优先于显式 `for` 循环
- 养成拿到数组先查 `shape` + `dtype` 的习惯
