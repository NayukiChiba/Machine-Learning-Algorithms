---
title: NumPy 创建数组
outline: deep
---

# NumPy 创建数组

> 对应脚本：`Basic/Numpy/02_creation.py`  
> 运行方式：`python Basic/Numpy/02_creation.py`（仓库根目录）

## 本章目标

1. 掌握 `np.array`、`zeros`、`ones`、`eye`、`full` 的创建方式。
2. 理解 `arange` 与 `linspace` 的使用差异。
3. 学会生成可复现的随机数组（`seed`）。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `np.array` | 从列表/嵌套列表创建数组 | `demo_from_list` |
| `np.zeros` / `np.ones` | 创建全 0 / 全 1 数组 | `demo_zeros_ones_eye` |
| `np.eye` | 创建单位矩阵或偏移对角矩阵 | `demo_zeros_ones_eye` |
| `np.full` | 用固定值填充数组 | `demo_zeros_ones_eye` |
| `np.arange` | 按步长生成序列（半开区间） | `demo_arange_linspace` |
| `np.linspace` | 生成等间距序列（按点数） | `demo_arange_linspace` |
| `np.random.rand/random/randint/randn/normal` | 生成不同分布随机数 | `demo_random` |
| `np.random.seed` | 固定随机种子，保证复现 | `demo_seed` |

## 1. 从列表创建数组：`np.array`

### 方法重点

- `dtype` 可以强制类型，避免后续隐式类型转换。
- 嵌套列表会自动推断为多维数组。
- `shape` 和 `ndim` 是后续所有变换的基础。

### 参数速览（本节）

适用 API：`np.array(object, dtype=None, copy=True, ndmin=0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `object` | `[1, 2, 3, 4, 5]`、`[[1, 2, 3], [4, 5, 6]]` | 输入序列或嵌套序列并推断维度 |
| `dtype` | `np.float64`（第三个示例） | 强制数组元素类型为浮点 |
| `copy` | `True`（默认） | 返回新数组副本 |
| `ndmin` | `0`（默认） | 不强制补维 |
### 示例代码

```python
import numpy as np

arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_float = np.array([1, 2, 3], dtype=np.float64)

print(arr_1d, arr_1d.shape, arr_1d.ndim)
print(arr_2d, arr_2d.shape, arr_2d.ndim)
print(arr_float, arr_float.dtype)
```

### 结果输出

```text
[1 2 3 4 5] (5,) 1
----------------
[[1 2 3]
 [4 5 6]] (2, 3) 2
----------------
[1. 2. 3.] float64
```

## 2. 特殊数组：`zeros` / `ones` / `eye` / `full`

### 方法重点

- `zeros(shape)`：常用于初始化缓存、mask、累加容器。
- `ones(shape)`：常用于基线向量、偏置初始化。
- `eye(N, M=None, k=0)`：`k=0` 主对角线，`k=1` 上偏一条对角线。
- `full(shape, fill_value)`：快速构造常量矩阵。

### 参数速览（本节）

1. `np.zeros(shape, dtype=float, order='C')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `shape` | `(3, 4)` | 指定输出数组形状 |
| `dtype` | `float`（默认） | 元素类型为浮点 |

2. `np.ones(shape, dtype=float, order='C')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `shape` | `(2, 3)` | 创建全 1 数组 |

3. `np.eye(N, M=None, k=0, dtype=float, order='C')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `N` / `M` | `3` / `None` | `M=None` 时生成 `N x N` 方阵 |
| `k` | `0`、`1` | `k=0` 主对角线，`k=1` 上偏一条对角线 |

4. `np.full(shape, fill_value, dtype=None, order='C')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `shape` / `fill_value` | `(2, 2)` / `7` | 用常量填充整个数组 |
### 示例代码

```python
import numpy as np

print(np.zeros((3, 4)))
print(np.ones((2, 3)))
print(np.eye(3))
print(np.eye(3, k=1))
print(np.full((2, 2), 7))
```

### 结果输出

```text
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
----------------
[[1. 1. 1.]
 [1. 1. 1.]]
----------------
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
----------------
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]]
----------------
[[7 7]
 [7 7]]
```

## 3. 序列数组：`arange` 与 `linspace`

### 方法重点

- `np.arange(start, stop, step)`：像 `range`，但返回数组；区间是 `[start, stop)`。
- `np.linspace(start, stop, num)`：按“点数”均匀切分，默认包含终点。
- 浮点步长建议优先 `linspace`，可读性和稳定性更好。

### 参数速览（本节）

1. `np.arange([start,] stop, [step], dtype=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `start` / `stop` / `step` | `0` / `10` / `2`；`10` / `0` / `-1` | 生成半开区间 `[start, stop)` 序列，支持负步长 |
| `dtype` | `None`（默认） | 类型由输入自动推断 |

2. `np.linspace(start, stop, num=50, endpoint=True, dtype=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `start` / `stop` | `0` / `1`；`0` / `2 * np.pi` | 指定插值区间端点 |
| `num` | `5`、`10` | 指定采样点个数 |
| `endpoint` | `True`（默认） | 结果包含终点 `stop` |
### 示例代码

```python
import numpy as np

print(np.arange(0, 10, 2))
print(np.arange(10, 0, -1))
print(np.linspace(0, 1, 5))
print(np.linspace(0, 2 * np.pi, 10))
```

### 结果输出

```text
[0 2 4 6 8]
----------------
[10  9  8  7  6  5  4  3  2  1]
----------------
[0.   0.25 0.5  0.75 1.  ]
----------------
[0.         0.6981317  1.3962634  2.0943951  2.7925268  3.4906585
 4.1887902  4.88692191 5.58505361 6.28318531]
```

## 4. 随机数组创建

### 方法重点

- `rand` / `random`：均匀分布 $[0, 1)$。
- `randint`：离散整数采样。
- `randn`：标准正态分布（均值 0，标准差 1）。
- `normal(loc, scale, size)`：可指定均值和标准差。

### 参数速览（本节）

1. `np.random.rand(d0, d1, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `d0, d1, ...` | `2, 3` | 以位置参数给出输出形状，生成 `[0, 1)` 均匀分布 |

2. `np.random.random(size=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `size` | `(2, 3)` | 与 `rand(2, 3)` 等价 |

3. `np.random.randint(low, high=None, size=None, dtype=int)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `low` / `high` | `0` / `10` | 整数采样区间为 `[0, 10)` |
| `size` | `(3, 3)` | 输出矩阵形状 |

4. `np.random.randn(d0, d1, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `d0, d1, ...` | `5` | 生成标准正态一维样本 |

5. `np.random.normal(loc=0.0, scale=1.0, size=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `loc` / `scale` / `size` | `10` / `2` / `5` | 生成均值 10、标准差 2 的正态样本 |
### 示例代码

```python
import numpy as np

np.random.seed(42)
print(np.random.rand(2, 3))
print(np.random.random(size=(2, 3)))
print(np.random.randint(0, 10, (3, 3)))

arr = np.random.randn(5)
print(arr)
print(arr.mean(), arr.std())

print(np.random.normal(loc=10, scale=2, size=5))
```

### 结果输出

```text
[[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]]
----------------
[[0.05808361 0.86617615 0.60111501]
 [0.70807258 0.02058449 0.96990985]]
----------------
[[5 1 4]
 [0 9 5]
 [8 0 9]]
----------------
[-0.25104397 -0.16386712 -1.47632969  1.48698096 -0.02445518]
----------------
-0.0857 0.9428
----------------
[10.71110263 10.83402222 11.66492371  9.41320171  9.94032286]
```

## 5. 随机种子与可复现性

### 方法重点

- 相同的 `seed` + 相同的调用顺序 => 相同结果。
- 机器学习实验必须固定随机种子，便于复现和对比。

### 参数速览（本节）

1. `np.random.seed(seed=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `seed` | `42` | 固定随机序列起点，保证可复现 |

2. `np.random.random(size)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `size` | `(2, 2)` | 生成 2x2 的均匀分布随机数组 |
### 示例代码

```python
import numpy as np

np.random.seed(42)
arr1 = np.random.random((2, 2))

np.random.seed(42)
arr2 = np.random.random((2, 2))

print(arr1)
print(arr2)
print(np.array_equal(arr1, arr2))
```

### 结果输出

```text
[[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
----------------
[[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
----------------
True
```

## 常见坑

1. `arange(0, 1, 0.1)` 可能出现浮点误差，优先用 `linspace`。
2. 只在脚本开头设置一次种子后，中间插入新的随机调用会改变后续序列。
3. 未指定 `dtype` 时，NumPy 会自动推断，可能与预期不一致。

## 小结

- 创建数组是后续索引、运算、线代和机器学习处理的入口。
- 本章最重要能力：看到问题时知道该用哪一种创建方式。