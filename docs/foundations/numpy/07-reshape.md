---
title: NumPy 变形
outline: deep
---

# NumPy 变形

## 本章目标

1. 掌握 `reshape` 的规则与 `-1` 自动推导维度。
2. 区分 `flatten` 与 `ravel`（副本 vs 视图）。
3. 掌握 `transpose` 对多维数组的轴重排。
4. 掌握 `squeeze` / `expand_dims` / `np.newaxis` 对维度长度的增删。
5. 理解 `np.resize` 与 `reshape` 的差别。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `arr.reshape(...)` | 方法 | 改变形状，元素总数不变 |
| `arr.flatten(...)` | 方法 | 展平为一维，**返回副本** |
| `arr.ravel(...)` | 方法 | 展平为一维，**尽量返回视图** |
| `arr.T` | 属性 | 转置视图（反转所有维度） |
| `np.transpose(...)` | 函数 | 转置，可指定轴顺序 |
| `np.squeeze(...)` | 函数 | 删除长度为 1 的维度 |
| `np.expand_dims(...)` | 函数 | 在指定轴插入长度为 1 的维度 |
| `np.newaxis` | 常量 | 等价于 `None`，在切片中插入新轴 |
| `np.resize(...)` | 函数 | 调整大小，可改变元素总数（不足时循环填充） |

## 形状变换

### `np.ndarray.reshape`

#### 作用

改变数组形状而**不改变元素总数**。返回视图（内存连续时）或副本。一个维度可写 `-1` 让 NumPy 自动推导。

#### 重点方法

```python
arr.reshape(*shape, order='C')
# 或传元组：
arr.reshape(shape, order='C')
```

#### 参数

| 参数名  | 本例取值                                | 说明                                                                   |
| ------- | --------------------------------------- | ---------------------------------------------------------------------- |
| `shape` | `(3, 4)`、`(4, 3)`、`(2, -1)`、`(-1, 6)` | 目标形状，可用 `-1` 自动推导一个维度；元素总数必须一致                 |
| `order` | `'C'`（默认）                           | 读写顺序：`'C'` 行优先、`'F'` 列优先、`'A'` 任意、`'K'` 保持内存顺序 |

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

- `-1` 最多出现**一次**；由 NumPy 根据其余维度推导。
- 变形前后元素总数必须相同，否则抛 `ValueError`。
- `reshape` 通常返回**视图**，修改结果会影响原数组。

## 展平

### `np.ndarray.flatten`

#### 作用

将数组展平为一维，**总是返回副本**，修改不会影响原数组。

#### 重点方法

```python
arr.flatten(order='C')
```

#### 参数

| 参数名  | 本例取值      | 说明                                                   |
| ------- | ------------- | ------------------------------------------------------ |
| `order` | `'C'`（默认） | 读取顺序：`'C'` 行优先、`'F'` 列优先、`'A'`、`'K'`      |

### `np.ndarray.ravel`

#### 作用

将数组展平为一维，**尽量返回视图**（若内存连续）。修改结果可能影响原数组。

#### 重点方法

```python
arr.ravel(order='C')
```

#### 参数

| 参数名  | 本例取值      | 说明                                                   |
| ------- | ------------- | ------------------------------------------------------ |
| `order` | `'C'`（默认） | 读取顺序，同 `flatten`                                 |

### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

flat = arr.flatten()
flat[0] = 999  # 不影响原数组

rav = arr.ravel()
rav[1] = 888   # 影响原数组（视图）

print(f"flatten 修改后，原数组:\n{arr}")
```

### 输出

```text
flatten 修改后，原数组:
[[  1 888   3]
 [  4   5   6]]
```

### 理解重点

- `flatten` = "一定是副本"，安全但有拷贝开销。
- `ravel` = "能视图就视图"，高效但可能影响原数组。
- 需要确定语义时优先 `flatten`；注重性能且不写结果时用 `ravel`。

## 转置与轴重排

### `np.ndarray.T`

#### 作用

返回数组转置视图，等价于反转所有轴。对二维就是常规的行列交换。

### `np.transpose`

#### 作用

比 `.T` 更灵活，可通过 `axes` 参数指定任意轴的排列顺序，适合多维张量。

#### 重点方法

```python
np.transpose(a, axes=None)
```

#### 参数

| 参数名 | 本例取值          | 说明                                                              |
| ------ | ----------------- | ----------------------------------------------------------------- |
| `a`    | 任意形状数组      | 输入数组                                                          |
| `axes` | `None`、`(1,0,2)` | 新的轴顺序；`None` 时反转所有轴；元组长度需等于 `a.ndim`          |

### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"arr.T:\n{arr.T}")

arr_3d = np.arange(24).reshape(2, 3, 4)
print(f"arr_3d.shape: {arr_3d.shape}")
print(f"arr_3d.T.shape: {arr_3d.T.shape}")
print(f"transpose((1,0,2)).shape: {np.transpose(arr_3d, axes=(1, 0, 2)).shape}")
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

## 维度增删

### `np.squeeze`

#### 作用

删除数组中**所有长度为 1 的维度**，或指定轴删除。常用于从单样本维度恢复到扁平形状。

#### 重点方法

```python
np.squeeze(a, axis=None)
```

#### 参数

| 参数名 | 本例取值       | 说明                                                         |
| ------ | -------------- | ------------------------------------------------------------ |
| `a`    | 形状 `(1,1,3)` | 输入数组                                                     |
| `axis` | `None`（默认） | `None` 删除所有长度为 1 的轴；指定 `int` 或元组只删选中的轴 |

### `np.expand_dims`

#### 作用

在指定位置**插入一个长度为 1 的维度**，常用于给一维向量增加 batch / channel 维。

#### 重点方法

```python
np.expand_dims(a, axis)
```

#### 参数

| 参数名 | 本例取值     | 说明                                                         |
| ------ | ------------ | ------------------------------------------------------------ |
| `a`    | `[1, 2, 3]`  | 输入数组                                                     |
| `axis` | `0`、`1`、`-1` | 插入新维度的位置（支持负索引）                               |

### `np.newaxis`

#### 作用

一个**常量**（实际是 `None`），在切片中使用可在指定位置插入长度为 1 的维度。是 `expand_dims` 的语法糖。

#### 语法

```python
arr[np.newaxis, :]   # 在开头插入新轴
arr[:, np.newaxis]   # 在末尾插入新轴
arr[np.newaxis, :, np.newaxis]  # 可同时插入多处
```

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
print(f"expand_dims(axis=0).shape: {np.expand_dims(v, axis=0).shape}")
print(f"expand_dims(axis=1).shape: {np.expand_dims(v, axis=1).shape}")

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

- `(n,)`（一维）、`(1, n)`（行向量）、`(n, 1)`（列向量）在广播中语义不同，要按需转换。
- 写模型输入批次维度时，`x[np.newaxis, ...]` 是常用写法。

## 改变元素总数

### `np.resize`

#### 作用

与 `reshape` 不同，`np.resize` **可以改变元素总数**。空间不足时会**循环重复**原数组元素填充；空间多余时截断。

#### 重点方法

```python
np.resize(a, new_shape)
```

#### 参数

| 参数名      | 本例取值          | 说明                                       |
| ----------- | ----------------- | ------------------------------------------ |
| `a`         | `[1, 2, 3, 4]`    | 输入数组                                   |
| `new_shape` | `(2, 4)`、`(3, 3)`| 目标形状，可大于或小于原元素总数           |

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

- `reshape` 元素总数不变；`resize` 可变（循环填充）。
- `arr.resize(shape)`（方法形式）是**原地**修改且不循环；与 `np.resize(arr, shape)`（函数形式）行为不同，易混淆。

## 常见坑

1. `reshape` 失败通常是因为元素总数不匹配。
2. `ravel` 返回视图还是副本取决于内存布局，**不要假设**；需要确定独立数据用 `flatten` 或 `.copy()`。
3. 列向量 `(n, 1)` 与行向量 `(1, n)` 在广播和矩阵乘法中语义完全不同，不要省 `np.newaxis`。
4. `np.resize` 和 `arr.resize` 行为不同：函数形式会循环填充，方法形式会原地填 0。
5. `transpose(axes=...)` 中 `axes` 长度必须等于数组维度，否则抛 `ValueError`。

## 小结

- 变形操作是"组织数据形状"的基础能力。
- **总数不变**用 `reshape`；**总数可变**用 `resize`。
- **只插入 / 删除长度 1 的维度**优先 `np.newaxis` / `squeeze`，比 `reshape` 更语义化。
- 牢记"副本 vs 视图"：`flatten` 是副本，`ravel` / `reshape` 通常是视图。
