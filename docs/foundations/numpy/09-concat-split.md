---
title: NumPy 拼接与拆分
outline: deep
---

# NumPy 拼接与拆分

## 本章目标

1. 掌握数组拼接：`concatenate`、`vstack`、`hstack`、`stack`、`dstack`。
2. 掌握数组拆分：`split`、`vsplit`、`hsplit`、`array_split`。
3. 理解"沿现有轴拼接"与"沿新轴堆叠"的区别。
4. 掌握等分与不等分拆分的正确 API 选择。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.concatenate(...)` | 函数 | 沿**现有轴**拼接多个数组 |
| `np.vstack(...)` | 函数 | 垂直拼接（相当于 `axis=0`） |
| `np.hstack(...)` | 函数 | 水平拼接（二维时相当于 `axis=1`） |
| `np.stack(...)` | 函数 | 沿**新轴**堆叠，结果维度 +1 |
| `np.dstack(...)` | 函数 | 沿第三轴（depth）堆叠 |
| `np.split(...)` | 函数 | 等分拆分，必须整除 |
| `np.vsplit(...)` | 函数 | 垂直拆分（`axis=0`） |
| `np.hsplit(...)` | 函数 | 水平拆分（`axis=1`） |
| `np.array_split(...)` | 函数 | 不均匀拆分，允许不能整除 |

## 沿现有轴拼接

### `np.concatenate`

#### 作用

沿**已有**的某个轴将多个数组拼接到一起。非拼接轴的长度必须一致，拼接后维度不变。

#### 重点方法

```python
np.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting='same_kind')
```

#### 参数

| 参数名    | 本例取值             | 说明                                                  |
| --------- | -------------------- | ----------------------------------------------------- |
| `arrays`  | `[A, B]`             | 待拼接的数组序列（列表或元组），元素维度必须相同      |
| `axis`    | `0`、`1`             | 沿哪个现有轴拼接；`None` 时先展平再拼接               |
| `out`     | `None`（默认）       | 写入结果的目标数组                                    |
| `dtype`   | `None`（默认）       | 结果 dtype；`None` 时由输入推断                       |
| `casting` | `'same_kind'`（默认）| 类型转换策略                                          |

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

- `axis=0`：**行方向**增长（叠行），列数必须一致。
- `axis=1`：**列方向**增长（拼列），行数必须一致。
- 一维数组拼接 `axis` 只能是 `0`。

### `np.vstack`

#### 作用

垂直（纵向）拼接。对二维数组等价于 `concatenate(axis=0)`；对一维数组会先视作行向量再堆叠。

#### 重点方法

```python
np.vstack(tup, *, dtype=None, casting='same_kind')
```

#### 参数

| 参数名    | 本例取值              | 说明                                 |
| --------- | --------------------- | ------------------------------------ |
| `tup`     | `[A, B]`              | 待堆叠的数组序列                     |
| `dtype`   | `None`（默认）        | 结果 dtype                           |
| `casting` | `'same_kind'`（默认） | 类型转换策略                         |

### `np.hstack`

#### 作用

水平（横向）拼接。对二维数组等价于 `concatenate(axis=1)`；对一维数组等价于 `concatenate(axis=0)`（沿唯一的轴）。

#### 重点方法

```python
np.hstack(tup, *, dtype=None, casting='same_kind')
```

#### 参数

| 参数名    | 本例取值              | 说明                                 |
| --------- | --------------------- | ------------------------------------ |
| `tup`     | `[A, C]`              | 待堆叠的数组序列                     |
| `dtype`   | `None`（默认）        | 结果 dtype                           |
| `casting` | `'same_kind'`（默认） | 类型转换策略                         |

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

## 沿新轴堆叠

### `np.stack`

#### 作用

在**新插入**的轴上堆叠多个形状完全一致的数组。结果维度比输入**多 1**。

#### 重点方法

```python
np.stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')
```

#### 参数

| 参数名    | 本例取值              | 说明                                                   |
| --------- | --------------------- | ------------------------------------------------------ |
| `arrays`  | `[A, B]`              | 待堆叠的数组序列，**形状必须完全一致**                 |
| `axis`    | `0`、`2`              | 新轴插入位置，范围是 `[0, ndim]`（可为负）             |
| `out`     | `None`（默认）        | 目标数组                                               |
| `dtype`   | `None`（默认）        | 结果 dtype                                             |
| `casting` | `'same_kind'`（默认） | 类型转换策略                                           |

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

- `concatenate` **不改变维数**；`stack` **维数 +1**。
- 选择 `stack` 的经典场景：把 `N` 张 `(H, W)` 图像堆成 `(N, H, W)` 的 batch。

### `np.dstack`

#### 作用

沿**第三轴**（depth）堆叠。对一维数组视作 `(1, N, 1)`，对二维数组视作 `(M, N, 1)` 后拼接。常用于图像通道堆叠（RGB）。

#### 重点方法

```python
np.dstack(tup)
```

#### 参数

| 参数名 | 本例取值 | 说明                 |
| ------ | -------- | -------------------- |
| `tup`  | `[A, B]` | 待堆叠的数组序列     |

#### 示例代码

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"dstack 形状: {np.dstack([A, B]).shape}")
print(np.dstack([A, B]))
```

#### 输出

```text
dstack 形状: (2, 2, 2)
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
```

## 拆分

### `np.split`

#### 作用

将数组沿指定轴**等分**。若传整数 `n`，要求该轴长度能被 `n` **整除**；否则需传索引列表。

#### 重点方法

```python
np.split(ary, indices_or_sections, axis=0)
```

#### 参数

| 参数名                | 本例取值     | 说明                                                              |
| --------------------- | ------------ | ----------------------------------------------------------------- |
| `ary`                 | `arr(4x3)`   | 待拆分数组                                                        |
| `indices_or_sections` | `2`、`3`、`[2, 5]` | 整数表示等分块数（必须整除）；列表表示分割点位置            |
| `axis`                | `0`、`1`     | 沿哪个轴拆分                                                      |

#### 示例代码

```python
import numpy as np

arr = np.arange(12).reshape(4, 3)
print(f"原数组:\n{arr}")

parts_row = np.split(arr, 2, axis=0)
print(f"\nsplit(arr, 2, axis=0):")
for i, p in enumerate(parts_row):
    print(f"第 {i+1} 块:\n{p}")

parts_col = np.split(arr, 3, axis=1)
print(f"\nsplit(arr, 3, axis=1):")
for i, p in enumerate(parts_col):
    print(f"第 {i+1} 块: {p.flatten()}")
```

#### 输出

```text
原数组:
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

split(arr, 2, axis=0):
第 1 块:
[[0 1 2]
 [3 4 5]]
第 2 块:
[[ 6  7  8]
 [ 9 10 11]]

split(arr, 3, axis=1):
第 1 块: [0 3 6 9]
第 2 块: [ 1  4  7 10]
第 3 块: [ 2  5  8 11]
```

### `np.vsplit` / `np.hsplit`

#### 作用

- `vsplit` 等价于 `split(axis=0)`，沿**行方向**拆分。
- `hsplit` 等价于 `split(axis=1)`，沿**列方向**拆分。

#### 重点方法

```python
np.vsplit(ary, indices_or_sections)
np.hsplit(ary, indices_or_sections)
```

#### 参数

| 参数名                | 本例取值 | 说明                                 |
| --------------------- | -------- | ------------------------------------ |
| `ary`                 | 二维数组 | 待拆分数组                           |
| `indices_or_sections` | `2`、`3` | 等分块数或分割点索引列表             |

### `np.array_split`

#### 作用

与 `np.split` 类似，但**允许整除不均**。无法整除时，前面的块多一个元素。

#### 重点方法

```python
np.array_split(ary, indices_or_sections, axis=0)
```

#### 参数

| 参数名                | 本例取值      | 说明                                              |
| --------------------- | ------------- | ------------------------------------------------- |
| `ary`                 | `np.arange(10)` | 待拆分数组                                      |
| `indices_or_sections` | `3`           | 整数表示块数（可不整除）；列表表示分割点          |
| `axis`                | `0`（默认）   | 沿哪个轴拆分                                      |

#### 示例代码

```python
import numpy as np

arr = np.arange(10)
parts = np.array_split(arr, 3)
for i, p in enumerate(parts):
    print(f"第 {i+1} 块（大小 {len(p)}）: {p}")
```

#### 输出

```text
第 1 块（大小 4）: [0 1 2 3]
第 2 块（大小 3）: [4 5 6]
第 3 块（大小 3）: [7 8 9]
```

#### 理解重点

- 10 个元素分 3 份 → `4, 3, 3`。前 `size % n` 块多一个元素。
- 机器学习交叉验证、batch 划分常用 `array_split`。

## 常见坑

1. `np.split(arr, n)` 不能整除会抛 `ValueError`；不确定整除时用 `np.array_split`。
2. `np.stack` 要求所有输入形状完全相同；不同形状应先 `reshape` / `pad` 对齐。
3. `concatenate` 的 `axis` 容易写反，拼接前先 `print(a.shape, b.shape)` 排查。
4. `hstack` 对一维数组与二维数组行为不同：一维时等价 `axis=0`，二维时等价 `axis=1`，注意区分。
5. `stack` 和 `concatenate` 不可互换：前者**加维度**，后者**不加维度**。

## 小结

- 拼接与拆分是数据批处理、窗口构造、特征组合的核心操作。
- 选择 API 的思路：**沿现有轴** → `concatenate` / `vstack` / `hstack`；**沿新轴** → `stack` / `dstack`。
- 拆分首选 `split`（严格），不整除时退化到 `array_split`。
- 先想清楚"要沿哪个轴变化"，再选具体 API。
