---
title: NumPy 拼接与拆分
outline: deep
---

# NumPy 拼接与拆分

> 对应脚本：`Basic/Numpy/09_concat_split.py`  
> 运行方式：`python Basic/Numpy/09_concat_split.py`

## 本章目标

1. 掌握数组拼接：`concatenate`、`vstack`、`hstack`、`stack`、`dstack`。
2. 掌握数组拆分：`split`、`vsplit`、`hsplit`、`array_split`。
3. 理解“沿现有轴拼接”与“沿新轴堆叠”的区别。

## 重点方法速览

| 方法 | 核心用途 |
|---|---|
| `np.concatenate` | 沿现有轴拼接 |
| `np.vstack` / `np.hstack` | 垂直/水平拼接快捷方式 |
| `np.stack` | 沿新轴堆叠 |
| `np.dstack` | 沿深度轴（第三轴）堆叠 |
| `np.split` | 等分拆分（必须整除） |
| `np.vsplit` / `np.hsplit` | 垂直/水平拆分快捷方式 |
| `np.array_split` | 不均匀拆分（可不整除） |

## 1. `concatenate`：沿现有轴拼接

### 参数速览（本节）

适用 API：`np.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting='same_kind')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `arrays` | `[A, B]` | 输入数组序列 |
| `axis` | `0`、`1` | 沿现有轴拼接：`0` 叠行，`1` 拼列 |
| `dtype` / `casting` | `None` / `'same_kind'` | 控制输出类型与类型转换策略 |
### 示例代码

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.concatenate([A, B], axis=0))
print(np.concatenate([A, B], axis=1))
```

### 结果输出

```text
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
----------------
[[1 2 5 6]
 [3 4 7 8]]
```

### 理解重点

- `axis=0`：行方向增加，列数必须一致。
- `axis=1`：列方向增加，行数必须一致。

## 2. `vstack` 与 `hstack`

### 参数速览（本节）

适用 API（分项）：

1. `np.vstack(tup, dtype=None, casting='same_kind')`
2. `np.hstack(tup, dtype=None, casting='same_kind')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `tup`（`vstack`） | `[A, B]` | 沿行方向堆叠（等价于 `axis=0`） |
| `tup`（`hstack`） | `[A, C]` | 沿列方向堆叠（二维下等价于 `axis=1`） |
| 形状约束 | 非拼接维度兼容 | 非拼接轴长度必须一致 |
### 示例代码

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9]])
C = np.array([[10], [20]])

print(np.vstack([A, B]))
print(np.hstack([A, C]))
```

### 结果输出

```text
[[1 2 3]
 [4 5 6]
 [7 8 9]]
----------------
[[ 1  2  3 10]
 [ 4  5  6 20]]
```

## 3. `stack`：沿新轴堆叠

### 参数速览（本节）

适用 API：`np.stack(arrays, axis=0, out=None, dtype=None, casting='same_kind')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `arrays` | `[A, B]` | 输入数组形状必须完全一致 |
| `axis` | `0`、`2` | 指定新轴插入位置 |
| 输出形状 | `(2,2,2)` | 本例两次堆叠均得到三维数组 |
### 示例代码

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

s0 = np.stack([A, B], axis=0)
s2 = np.stack([A, B], axis=2)

print(s0.shape)
print(s0)
print(s2.shape)
print(s2)
```

### 结果输出

```text
(2, 2, 2)
----------------
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
----------------
(2, 2, 2)
----------------
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
```

## 4. `dstack`：深度堆叠

### 参数速览（本节）

适用 API：`np.dstack(tup)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `tup` | `[A, B]` | 沿第三轴（depth）堆叠 |
| 输入形状 | `A/B: (2,2)` | 通常输入为二维数组 |
| 输出形状 | `(2,2,2)` | 结果第三维长度等于输入数组个数 |
### 示例代码

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dstack([A, B]))
```

### 结果输出

```text
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
```

## 5. `split`：等分拆分

### 参数速览（本节）

适用 API：`np.split(ary, indices_or_sections, axis=0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `ary` | `arr(4x3)` | 待拆分数组 |
| `indices_or_sections` | `2`、`3` | 传整数表示等分块数（必须整除） |
| `axis` | `0`、`1` | `axis=0` 按行拆分，`axis=1` 按列拆分 |
### 示例代码

```python
import numpy as np

arr = np.arange(12).reshape(4, 3)

parts_row = np.split(arr, 2, axis=0)
parts_col = np.split(arr, 3, axis=1)

for p in parts_row:
    print(p)
for p in parts_col:
    print(p.flatten())
```

### 结果输出

```text
[[0 1 2]
 [3 4 5]]
----------------
[[ 6  7  8]
 [ 9 10 11]]
----------------
[0 3 6 9]
----------------
[ 1  4  7 10]
----------------
[ 2  5  8 11]
```

## 6. `vsplit` / `hsplit`

`vsplit` 与 `split(axis=0)` 等价，`hsplit` 与 `split(axis=1)` 等价。

## 7. `array_split`：不均匀拆分

### 参数速览（本节）

适用 API：`np.array_split(ary, indices_or_sections, axis=0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `ary` | `np.arange(10)` | 待拆分一维数组 |
| `indices_or_sections` | `3` | 允许不能整除，自动生成不等长分块 |
| 分块规则 | 前长后短 | 常见分配为前面的块多一个元素 |
### 示例代码

```python
import numpy as np

arr = np.arange(10)
parts = np.array_split(arr, 3)
for p in parts:
    print(p)
```

### 结果输出

```text
[0 1 2 3]
----------------
[4 5 6]
----------------
[7 8 9]
```

## 常见坑

1. `split` 必须整除，否则报错；不整除请用 `array_split`。
2. `stack` 要求输入数组形状一致。
3. `concatenate` 容易把 `axis` 写反，建议先打印 `shape`。

## 小结

- 拼接和拆分是数据批处理、窗口处理、特征组合的常用操作。
- 先想清楚“沿哪个轴变化”，再选 API。
