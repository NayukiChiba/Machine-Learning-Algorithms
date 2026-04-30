---
title: NumPy 拼接与拆分
outline: deep
---

# NumPy 拼接与拆分

## 本章目标

1. 掌握数组拼接：`concatenate`、`vstack`、`hstack`、`stack`、`dstack`
2. 掌握数组拆分：`split`、`vsplit`、`hsplit`、`array_split`
3. 理解"沿现有轴拼接"与"沿新轴堆叠"的本质区别
4. 掌握等分与不等分拆分的正确 API 选择

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.concatenate(...)` | 函数 | 沿现有轴拼接多个数组 |
| `np.vstack(...)` | 函数 | 垂直拼接（相当于 `axis=0`） |
| `np.hstack(...)` | 函数 | 水平拼接（二维时相当于 `axis=1`） |
| `np.stack(...)` | 函数 | 沿新轴堆叠，结果维度 +1 |
| `np.dstack(...)` | 函数 | 沿第三轴（depth）堆叠 |
| `np.split(...)` | 函数 | 等分拆分，必须整除 |
| `np.vsplit(...)` / `np.hsplit(...)` | 函数 | 垂直 / 水平拆分 |
| `np.array_split(...)` | 函数 | 不均匀拆分，允许不能整除 |

## 1. 沿现有轴拼接

### `np.concatenate`

#### 作用

沿已有的某个轴将多个数组拼接到一起。非拼接轴的长度必须一致，拼接后维度不变。

#### 重点方法

```python
np.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting='same_kind')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arrays` | `sequence of array_like` | 待拼接的数组序列，元素维度必须相同 | `[A, B]` |
| `axis` | `int` | 沿哪个现有轴拼接，默认为 `0`；`None` 时先展平再拼接 | `1` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |
| `dtype` | `dtype` 或 `None` | 结果 dtype | —— |
| `casting` | `str` | 类型转换策略，默认为 `'same_kind'` | `'safe'` |

#### 示例代码

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"axis=0 垂直拼接:\n{np.concatenate([A, B], axis=0)}")
print(f"axis=1 水平拼接:\n{np.concatenate([A, B], axis=1)}")
```

#### 输出

```text
axis=0 垂直拼接:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
axis=1 水平拼接:
[[1 2 5 6]
 [3 4 7 8]]
```

#### 理解重点

- `axis=0`：行方向增长（叠行），**列数必须一致**
- `axis=1`：列方向增长（拼列），**行数必须一致**
- 一维数组拼接 `axis` 只能是 `0`

### `np.vstack`

#### 作用

垂直（纵向）堆叠。对二维数组等价于 `concatenate(axis=0)`；对一维数组会先视作行向量再堆叠。

#### 重点方法

```python
np.vstack(tup, *, dtype=None, casting='same_kind')
```

### `np.hstack`

#### 作用

水平（横向）堆叠。对二维数组等价于 `concatenate(axis=1)`；对一维数组等价于 `concatenate(axis=0)`。

#### 重点方法

```python
np.hstack(tup, *, dtype=None, casting='same_kind')
```

### 综合示例

#### 示例代码

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9]])           # (1, 3)
C = np.array([[10], [20]])          # (2, 1)

print(f"vstack([A, B]):\n{np.vstack([A, B])}")
print(f"hstack([A, C]):\n{np.hstack([A, C])}")
```

#### 输出

```text
vstack([A, B]):
[[1 2 3]
 [4 5 6]
 [7 8 9]]
hstack([A, C]):
[[ 1  2  3 10]
 [ 4  5  6 20]]
```

## 2. 沿新轴堆叠

### `np.stack`

#### 作用

在新插入的轴上堆叠多个形状完全一致的数组，结果维度比输入多 1。经典场景：把 $N$ 张 $(H, W)$ 图像堆成 $(N, H, W)$ 的 batch。

#### 重点方法

```python
np.stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arrays` | `sequence of array_like` | 待堆叠的数组序列，**形状必须完全一致** | `[A, B]` |
| `axis` | `int` | 新轴插入位置，范围 $[0, ndim]$（可为负），默认为 `0` | `2` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |
| `dtype` | `dtype` 或 `None` | 结果 dtype | —— |
| `casting` | `str` | 类型转换策略，默认为 `'same_kind'` | —— |

#### 示例代码

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

s0 = np.stack([A, B], axis=0)
s2 = np.stack([A, B], axis=2)

print(f"stack axis=0 形状: {s0.shape}")
print(s0)
print(f"\nstack axis=2 形状: {s2.shape}")
print(s2)
```

#### 输出

```text
stack axis=0 形状: (2, 2, 2)
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

stack axis=2 形状: (2, 2, 2)
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
```

#### 理解重点

- `concatenate` 不改变维度；`stack` 维度 +1——这是两者最核心的区别
- `axis=2` 时相当于把两张矩阵按像素位置"摞"在一起——类似图像通道堆叠

### `np.dstack`

#### 作用

沿第三轴（depth）堆叠。对一维数组视作 `(1, N, 1)`，对二维数组视作 `(M, N, 1)` 后拼接。常用于图像 RGB 三通道堆叠。

#### 重点方法

```python
np.dstack(tup)
```

## 3. 拆分

### `np.split`

#### 作用

将数组沿指定轴等分拆分。传整数 `n` 时要求该轴长度能被 `n` 整除；也可传索引列表精确控制分割点。

#### 重点方法

```python
np.split(ary, indices_or_sections, axis=0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `ary` | `ndarray` | 待拆分数组 | `arr(4x3)` |
| `indices_or_sections` | `int` 或 `list[int]` | 整数为等分块数（必须整除）；列表为分割点索引 | `2`、`[2, 5]` |
| `axis` | `int` | 沿哪个轴拆分，默认为 `0` | `1` |

### `np.vsplit` / `np.hsplit`

#### 作用

- `vsplit` 等价于 `split(axis=0)`，沿行方向拆分
- `hsplit` 等价于 `split(axis=1)`，沿列方向拆分

#### 重点方法

```python
np.vsplit(ary, indices_or_sections)
np.hsplit(ary, indices_or_sections)
```

### `np.array_split`

#### 作用

与 `np.split` 类似，但**允许整除不均**。无法整除时，`size % n` 块多一个元素。机器学习交叉验证、batch 划分常用。

#### 重点方法

```python
np.array_split(ary, indices_or_sections, axis=0)
```

### 综合示例

#### 示例代码

```python
import numpy as np

# 等分拆分
arr = np.arange(12).reshape(4, 3)
print(f"原数组:\n{arr}")

partsRow = np.split(arr, 2, axis=0)
print(f"\nsplit(axis=0) 分 2 块:")
for i, p in enumerate(partsRow):
    print(f"第 {i+1} 块:\n{p}")

# 不等分拆分
arr1d = np.arange(10)
parts = np.array_split(arr1d, 3)
for i, p in enumerate(parts):
    print(f"第 {i+1} 块(大小 {len(p)}): {p}")
```

#### 输出

```text
原数组:
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

split(axis=0) 分 2 块:
第 1 块:
[[0 1 2]
 [3 4 5]]
第 2 块:
[[ 6  7  8]
 [ 9 10 11]]
第 1 块(大小 4): [0 1 2 3]
第 2 块(大小 3): [4 5 6]
第 3 块(大小 3): [7 8 9]
```

#### 理解重点

- 10 个元素分 3 份 → `4, 3, 3`：前面 `size % n` 块各多一个元素
- 等分优先 `split`（明确意图），不整除用 `array_split`（容忍不齐）

## 常见坑

1. `np.split(arr, n)` 不能整除会抛 `ValueError`——不确定整除时用 `np.array_split`
2. `np.stack` 要求所有输入形状完全相同；不同形状应先 `reshape` / `pad` 对齐
3. `concatenate` 的 `axis` 容易写反——拼接前先 `print(a.shape, b.shape)` 排查
4. `hstack` 对一维与二维行为不同：一维等价 `axis=0`，二维等价 `axis=1`
5. `stack` 和 `concatenate` 不可互换：前者加维度，后者不加维度

## 小结

- 拼接与拆分是数据批处理、窗口构造、特征组合的核心操作
- 选 API 的思路：**沿现有轴** → `concatenate` / `vstack` / `hstack`；**沿新轴** → `stack` / `dstack`
- 拆分首选 `split`（严格），不整除退化到 `array_split`
- 先想清楚"要沿哪个轴变化"，再选具体 API
