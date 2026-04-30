---
title: NumPy 变形
outline: deep
---

# NumPy 变形

## 本章目标

1. 掌握 `reshape` 的规则与 `-1` 自动推导维度
2. 区分 `flatten`（副本）与 `ravel`（视图）的语义差异
3. 掌握 `transpose` 对多维数组的轴重排
4. 掌握 `squeeze` / `expand_dims` / `np.newaxis` 对维度长度的增删
5. 理解 `np.resize` 与 `reshape` 的差别

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `arr.reshape(...)` | 方法 | 改变形状，元素总数不变 |
| `arr.flatten(...)` | 方法 | 展平为一维，**总是返回副本** |
| `arr.ravel(...)` | 方法 | 展平为一维，**尽量返回视图** |
| `arr.T` | 属性 | 转置视图（反转所有维度） |
| `np.transpose(...)` | 函数 | 转置，可指定轴顺序 |
| `np.squeeze(...)` | 函数 | 删除长度为 1 的维��� |
| `np.expand_dims(...)` | 函数 | 在指定轴插入长度为 1 的维度 |
| `np.newaxis` | 常量 | 等价于 `None`，在切片中插入新轴 |
| `np.resize(...)` | 函数 | 调整大小，可改变元素总数（不足时循环填充） |

## 1. 形状变换

### `ndarray.reshape`

#### 作用

改变数组形状而不改变元素总数。返回视图（内存连续时）或副本。一个维度可写 `-1` 让 NumPy 自动推导。

#### 重点方法

```python
arr.reshape(*shape, order='C')
# 或传元组:
arr.reshape(shape, order='C')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `shape` | `int` 或 `tuple[int, ...]` | 目标形状，可用 `-1` 自动推导一个维度；元素总数必须一致 | `(3, 4)`、`(2, -1)`、`(-1, 6)` |
| `order` | `str` | 读写顺序：`'C'` 行优先 / `'F'` 列优先 / `'A'` 任意 / `'K'` 保持内存顺序，默认为 `'C'` | `'F'` |

#### 示例代码

```python
import numpy as np

arr = np.arange(1, 13)
print(f"原数组: {arr}, shape={arr.shape}")
print(f"reshape(3, 4):\n{arr.reshape(3, 4)}")
print(f"reshape(4, 3):\n{arr.reshape(4, 3)}")
print(f"reshape(2, -1):\n{arr.reshape(2, -1)}")
print(f"reshape(-1, 6):\n{arr.reshape(-1, 6)}")
```

#### 输出

```text
原数组: [ 1  2  3  4  5  6  7  8  9 10 11 12], shape=(12,)
reshape(3, 4):
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
reshape(4, 3):
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
reshape(2, -1):
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
reshape(-1, 6):
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
```

#### 理解重点

- `-1` 最多出现一次，由 NumPy 根据元素总数与其他维度反推
- 变形前后元素总数必须相同，否则抛 `ValueError`
- `reshape` 通常返回视图，修改结果会影响原数组

## 2. 展平

### `ndarray.flatten`

#### 作用

将数组展平为一维，**总是返回副本**。修改返回结果不影响原数组。

#### 重点方法

```python
arr.flatten(order='C')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `order` | `str` | 读取顺序：`'C'` 行优先 / `'F'` 列优先 / `'A'` / `'K'`，默认为 `'C'` | `'F'` |

### `ndarray.ravel`

#### 作用

将数组展平为一维，**尽量返回视图**（若内存连续）。修改结果可能影响原数组。比 `flatten` 高效但须注意副作用。

#### 重点方法

```python
arr.ravel(order='C')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `order` | `str` | 读取顺序，同 `flatten`，默认为 `'C'` | `'F'` |

### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

flat = arr.flatten()
flat[0] = 999  # 不影响原数组

rav = arr.ravel()
rav[1] = 888   # 影响原数组（视图）

print(f"flatten 副本修改不影响原数组:\n{arr}")
```

### 输出

```text
flatten 副本修改不影响原数组:
[[  1 888   3]
 [  4   5   6]]
```

### 理解重点

- `flatten` = "一定是副本"，安全但有拷贝开销——适合传出去的数据
- `ravel` = "能视图就视图"，高效但可能联动修改——适合临时读取
- 不确定时用 `flatten` 或显式 `.copy()`，安全性优先

## 3. 转置与轴重排

### `ndarray.T`

#### 作用

返回数组转置视图，等价于反转所有轴。对二维即行列交换。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `ndarray` | 转置视图，修改会影响原数组 |

### `np.transpose`

#### 作用

比 `.T` 更灵活，可通过 `axes` 参数指定任意轴的排列顺序，适合多维张量的轴重排。

#### 重点方法

```python
np.transpose(a, axes=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `arr3d` |
| `axes` | `tuple[int, ...]` 或 `None` | 新的轴顺序，默认为 `None`（反转所有轴），元组长度需等于 `a.ndim` | `(1, 0, 2)` |

### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"arr.T:\n{arr.T}")

arr3d = np.arange(24).reshape(2, 3, 4)
print(f"arr_3d.shape: {arr3d.shape}")
print(f"arr_3d.T.shape: {arr3d.T.shape}")
print(f"transpose((1,0,2)).shape: "
      f"{np.transpose(arr3d, axes=(1, 0, 2)).shape}")
```

### 输出

```text
arr.T:
[[1 4]
 [2 5]
 [3 6]]
arr_3d.shape: (2, 3, 4)
arr_3d.T.shape: (4, 3, 2)
transpose((1,0,2)).shape: (3, 2, 4)
```

## 4. 维度增删

### `np.squeeze`

#### 作用

删除数组中长度为 1 的维度。常用于从 `(1, n)` 或 `(1, 1, n)` 恢复到 `(n,)`。

#### 重点方法

```python
np.squeeze(a, axis=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `(1, 1, 3)` |
| `axis` | `int` 或 `tuple[int, ...]` 或 `None` | 只删除指定轴（必须长度为 1），默认为 `None`（删除所有长度为 1 的轴） | `0` |

### `np.expand_dims`

#### 作用

在指定位置插入一个长度为 1 的维度。常用于给一维向量增加 batch 或 channel 维度。

#### 重点方法

```python
np.expand_dims(a, axis)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[1, 2, 3]` |
| `axis` | `int` | 插入新维度的位置，支持负索引 | `0`、`1`、`-1` |

### `np.newaxis`

#### 作用

一个常量（实际是 `None`），在切片中使用可在指定位置插入长度为 1 的维度，是 `expand_dims` 的语法糖。

### 示例代码

```python
import numpy as np

# squeeze
arr = np.array([[[1, 2, 3]]])
print(f"原形状: {arr.shape}")
print(f"squeeze 后: {np.squeeze(arr).shape}")

# expand_dims
v = np.array([1, 2, 3])
print(f"原 v.shape: {v.shape}")
print(f"expand_dims(axis=0).shape: "
      f"{np.expand_dims(v, axis=0).shape}")
print(f"expand_dims(axis=1).shape: "
      f"{np.expand_dims(v, axis=1).shape}")

# newaxis
print(f"v[np.newaxis, :].shape: {v[np.newaxis, :].shape}")
print(f"v[:, np.newaxis].shape: {v[:, np.newaxis].shape}")
```

### 输出

```text
原形状: (1, 1, 3)
squeeze 后: (3,)
原 v.shape: (3,)
expand_dims(axis=0).shape: (1, 3)
expand_dims(axis=1).shape: (3, 1)
v[np.newaxis, :].shape: (1, 3)
v[:, np.newaxis].shape: (3, 1)
```

### 理解重点

- `(n,)`（一维向量）与 `(1, n)`（行向量）与 `(n, 1)`（列向量）在广播中语义不同——它们不可互换
- 模型输入加 batch 维度：`x[np.newaxis, ...]` 是常用写法
- `squeeze` 删尺寸为 1 的轴比 `reshape` 更语义化

## 5. 改变元素总数

### `np.resize`

#### 作用

与 `reshape` 不同，`np.resize` **可以改变元素总数**。空间不足时循环重复原数组元素填充；空间多余时截断。

#### 重点方法

```python
np.resize(a, new_shape)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[1, 2, 3, 4]` |
| `new_shape` | `int` 或 `tuple[int, ...]` | 目标形状，可大于或小于原元素总数 | `(2, 4)`、`(3, 3)` |

#### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(f"resize((2, 4)):\n{np.resize(arr, (2, 4))}")
print(f"resize((3, 3)):\n{np.resize(arr, (3, 3))}")
```

#### 输出

```text
resize((2, 4)):
[[1 2 3 4]
 [1 2 3 4]]
resize((3, 3)):
[[1 2 3]
 [4 1 2]
 [3 4 1]]
```

#### 理解重点

- `np.resize(arr, shape)`（函数形式）循环填充；`arr.resize(shape)`（方法形式）原地修改且不循环填充——两者行为不同
- `reshape` 元素总数不变；`resize` 总数可变——功能不同，不可互换

## 常见坑

1. `reshape` 失败通常因为元素总数不匹配——先查 `arr.size` 再写目标形状
2. `ravel` 返回视图还是副本取决于内存布局，**不要假设**——确定要独立数据时用 `flatten` 或 `.copy()`
3. 列向量 `(n, 1)` 与行向量 `(1, n)` 在广播和矩阵乘法中语义完全不同——不要省 `np.newaxis`
4. `np.resize` 和 `arr.resize` 行为不同：函数形式循环填充，方法形式原地填 0 扩展
5. `transpose(axes=...)` 中 `axes` 长度必须等于数组维度

## 小结

- **总数不变**用 `reshape`；**总数可变**用 `resize`
- **只插入/删除长度为 1 的维度**优先 `np.newaxis` / `squeeze`，比 `reshape` 更语义化
- 牢记"副本 vs 视图"：`flatten` = 副本，`ravel` / `reshape` = 通常视图
