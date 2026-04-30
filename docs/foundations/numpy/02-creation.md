---
title: NumPy 创建数组
outline: deep
---

# NumPy 创建数组

## 本章目标

1. 掌握 `np.array`、`zeros`、`ones`、`eye`、`full` 创建指定值数组
2. 理解 `arange`（按步长）与 `linspace`（按点数）的使用差异
3. 学会生成可复现的随机数组（`seed`），掌握常用随机分布 API

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.array(...)` | 函数 | 从列表或嵌套列表创建 `ndarray` |
| `np.zeros(...)` | 函数 | 创建全 0 数组 |
| `np.ones(...)` | 函数 | 创建全 1 数组 |
| `np.eye(...)` | 函数 | 创建单位矩阵或对角线偏移矩阵 |
| `np.full(...)` | 函数 | 用固定值填充数组 |
| `np.arange(...)` | 函数 | 按步长生成等差序列（半开区间 `[start, stop)`） |
| `np.linspace(...)` | 函数 | 按点数生成等间距序列（默认包含终点） |
| `np.random.seed(...)` | 函数 | 固定随机种子，保证可复现 |
| `np.random.rand(...)` | 函数 | 生成 `[0, 1)` 均匀分布随机数 |
| `np.random.random(...)` | 函数 | 与 `rand` 等价，使用 `size` 关键字参数 |
| `np.random.randint(...)` | 函数 | 生成离散整数随机样本 |
| `np.random.randn(...)` | 函数 | 生成标准正态分布 $\mathcal{N}(0, 1)$ 随机样本 |
| `np.random.normal(...)` | 函数 | 生成指定均值与标准差的正态分布样本 |
| `np.array_equal(...)` | 函数 | 判断两个数组是否完全相等 |

## 1. 从列表创建数组

### `np.array`

#### 作用

从 Python 列表或嵌套列表创建 `ndarray`。嵌套列表自动推断为多维数组，可通过 `dtype` 指定元素类型。

#### 重点方法

```python
np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `object` | `array_like` | 输入数据，可为列表、元组、嵌套序列、其他数组 | `[1, 2, 3]`、`[[1, 2], [3, 4]]` |
| `dtype` | `dtype` 或 `None` | 元素数据类型，默认为 `None`（自动推断） | `np.float64` |
| `copy` | `bool` | `True` 总是复制数据，`False` 在可行时共享内存，默认为 `True` | `False` |
| `order` | `str` | 内存布局：`'K'` 保持原样 / `'C'` 行优先 / `'F'` 列优先 / `'A'` 任意，默认为 `'K'` | `'C'` |
| `subok` | `bool` | `True` 保留子类类型，`False` 强制基础 `ndarray`，默认为 `False` | `True` |
| `ndmin` | `int` | 返回数组的最小维度，不足时在前方补 1，默认为 `0` | `2` |
| `like` | `array_like` 或 `None` | 参考数组原型，用于兼容第三方数组库（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arrFloat = np.array([1, 2, 3], dtype=np.float64)

print(arr1d, arr1d.shape, arr1d.ndim)
print(arr2d)
print(arr2d.shape, arr2d.ndim)
print(arrFloat, arrFloat.dtype)
```

#### 输出

```text
[1 2 3 4 5] (5,) 1
[[1 2 3]
 [4 5 6]]
(2, 3) 2
[1. 2. 3.] float64
```

#### 理解重点

- 显式指定 `dtype` 可避免后续隐式类型转换——不确定类型时先定好
- `shape` 与 `ndim` 是后续所有变换的基础：拿到数组先看形状

## 2. 特殊数组

### `np.zeros`

#### 作用

创建指定形状的全 0 数组，常用于初始化缓存、mask、累加容器。

#### 重点方法

```python
np.zeros(shape, dtype=float, order='C', *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `shape` | `int` 或 `tuple[int, ...]` | 输出数组形状 | `(3, 4)` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `float`（即 `float64`） | `np.int32` |
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先，默认为 `'C'` | `'F'` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.zeros((3, 4)))
```

#### 输出

```text
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```

### `np.ones`

#### 作用

创建指定形状的全 1 数组，常用于基线向量、偏置初始化。

#### 重点方法

```python
np.ones(shape, dtype=None, order='C', *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `shape` | `int` 或 `tuple[int, ...]` | 输出数组形状 | `(2, 3)` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `None`（等价 `float64`） | `np.int32` |
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先，默认为 `'C'` | `'F'` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.ones((2, 3)))
```

#### 输出

```text
[[1. 1. 1.]
 [1. 1. 1.]]
```

### `np.eye`

#### 作用

创建单位矩阵或带对角线偏移的矩阵。$k=0$ 为主对角线，$k>0$ 向右上偏移，$k<0$ 向左下偏移。

#### 重点方法

```python
np.eye(N, M=None, k=0, dtype=float, order='C', *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `N` | `int` | 行数 | `3` |
| `M` | `int` 或 `None` | 列数，默认为 `None`（等于 `N`，生成方阵） | `4` |
| `k` | `int` | 对角线偏移：`0` 主对角线 / `>0` 向右上 / `<0` 向左下，默认为 `0` | `1` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `float` | `np.int32` |
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先，默认为 `'C'` | `'F'` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.eye(3))
print(np.eye(3, k=1))
```

#### 输出

```text
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]]
```

### `np.full`

#### 作用

用指定常量值填充整个数组，快速构造常量矩阵。

#### 重点方法

```python
np.full(shape, fill_value, dtype=None, order='C', *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `shape` | `int` 或 `tuple[int, ...]` | 输出数组形状 | `(2, 2)` |
| `fill_value` | `scalar` 或 `array_like` | 填充值，标量或可广播到 `shape` 的数组 | `7` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `None`（由 `fill_value` 推断） | `np.float32` |
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先，默认为 `'C'` | `'F'` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.full((2, 2), 7))
```

#### 输出

```text
[[7 7]
 [7 7]]
```

## 3. 序列数组

### `np.arange`

#### 作用

生成半开区间 $[start, stop)$ 的等差序列，类似 Python 的 `range`，但返回 `ndarray`。支持浮点步长与负步长。

#### 重点方法

```python
np.arange([start,] stop, [step,] dtype=None, *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `start` | `int` 或 `float` | 起始值（包含），省略时默认为 `0` | `0`、`10` |
| `stop` | `int` 或 `float` | 终止值（不包含） | `10`、`0` |
| `step` | `int` 或 `float` | 步长，省略时默认为 `1`，支持负数和浮点数 | `2`、`-1` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `None`（自动推断） | `np.float32` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.arange(0, 10, 2))
print(np.arange(10, 0, -1))
```

#### 输出

```text
[0 2 4 6 8]
[10  9  8  7  6  5  4  3  2  1]
```

### `np.linspace`

#### 作用

在指定区间 $[start, stop]$ 内按点数均匀切分，默认包含终点。浮点场景下比 `arange` 更稳定。

#### 重点方法

```python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `start` | `array_like` | 起始值，可为标量或数组 | `0` |
| `stop` | `array_like` | 终止值，可为标量或数组 | `2 * np.pi` |
| `num` | `int` | 采样点个数，默认为 `50` | `5`、`10` |
| `endpoint` | `bool` | 结果是否包含 `stop`，默认为 `True` | `False` |
| `retstep` | `bool` | `True` 时额外返回步长 `(array, step)`，默认为 `False` | `True` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `None`（自动推断） | `np.float32` |
| `axis` | `int` | `start` / `stop` 为数组时结果沿哪个轴排布，默认为 `0` | `1` |

#### 示例代码

```python
import numpy as np

print(np.linspace(0, 1, 5))
print(np.linspace(0, 2 * np.pi, 10))
```

#### 输出

```text
[0.   0.25 0.5  0.75 1.  ]
[0.         0.6981317  1.3962634  2.0943951  2.7925268  3.4906585
 4.1887902  4.88692191 5.58505361 6.28318531]
```

#### 理解重点

- `arange` 按**步长**切分，区间是 $[start, stop)$——适用于已知步长的场景
- `linspace` 按**点数**切分，默认包含终点——适用于已知采样点数的场景
- 浮点步长优先选 `linspace`，可读性和稳定性都更好

## 4. 随机数组

### `np.random.seed`

#### 作用

固定随机数生成器的起点，保证随机序列可复现。机器学习实验必须固定随机种子。

#### 重点方法

```python
np.random.seed(seed=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `seed` | `int` 或 `None` | 非负整数（$0$ 到 $2^{32}-1$）固定序列起点，`None` 使用系统随机源 | `42` |

### `np.random.rand`

#### 作用

生成 $[0, 1)$ 均匀分布随机数，以位置参数指定各维度长度。

#### 重点方法

```python
np.random.rand(d0, d1, ..., dn)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `d0, d1, ...` | `int` | 以位置参数逐维指定输出形状，省略所有参数返回单个标量 | `2, 3` |

### `np.random.random`

#### 作用

与 `rand` 等价，生成 $[0, 1)$ 均匀分布随机数。区别在于使用 `size` 关键字参数指定形状。

#### 重点方法

```python
np.random.random(size=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `size` | `int` 或 `tuple[int, ...]` 或 `None` | 输出形状，默认为 `None`（返回单个标量） | `(2, 3)` |

### `np.random.randint`

#### 作用

在 $[low, high)$ 区间内生成离散整数随机样本。

#### 重点方法

```python
np.random.randint(low, high=None, size=None, dtype=int)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `low` | `int` | 下界（包含）；若 `high=None`，则采样区间变为 $[0, low)$ | `0` |
| `high` | `int` 或 `None` | 上界（不包含） | `10` |
| `size` | `int` 或 `tuple[int, ...]` 或 `None` | 输出形状，默认为 `None`（返回单个标量） | `(3, 3)` |
| `dtype` | `dtype` | 整数元素类型，默认为 `int`（即 `int64`） | `np.int32` |

### `np.random.randn`

#### 作用

生成标准正态分布 $\mathcal{N}(0, 1)$ 的随机样本。概率密度函数：

$$
f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}
$$

#### 重点方法

```python
np.random.randn(d0, d1, ..., dn)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `d0, d1, ...` | `int` | 以位置参数逐维指定输出形状，省略所有参数返回单个标量 | `5` |

### `np.random.normal`

#### 作用

生成指定均值 $\mu$ 和标准差 $\sigma$ 的正态分布样本。概率密度函数：

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

#### 重点方法

```python
np.random.normal(loc=0.0, scale=1.0, size=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `loc` | `float` | 正态分布的均值 $\mu$（分布中心），默认为 `0.0` | `10` |
| `scale` | `float` | 正态分布的标准差 $\sigma$，必须非负，默认为 `1.0` | `2` |
| `size` | `int` 或 `tuple[int, ...]` 或 `None` | 输出形状，默认为 `None`（返回单个标量） | `5` |

### 综合示例

> 随机数 API 共享全局状态，独立运行时结果可能不同。下例在 `seed(42)` 后按顺序调用，保证输出可复现。

#### 示例代码

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

#### 输出

```text
[[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]]
[[0.05808361 0.86617615 0.60111501]
 [0.70807258 0.02058449 0.96990985]]
[[5 1 4]
 [0 9 5]
 [8 0 9]]
[-0.25104397 -0.16386712 -1.47632969  1.48698096 -0.02445518]
-0.0857 0.9428
[10.71110263 10.83402222 11.66492371  9.41320171  9.94032286]
```

#### 理解重点

- `rand` / `random` 用于均匀分布 $[0, 1)$
- `randint` 用于离散整数采样
- `randn` 是标准正态分布 $\mathcal{N}(0, 1)$；`normal` 可指定 $\mu$ 和 $\sigma$

## 5. 随机种子与可复现性

### `np.array_equal`

#### 作用

判断两个数组是否完全相等（形状相同且每个元素值相同）。常用于验证实验结果的可复现性。

#### 重点方法

```python
np.array_equal(a1, a2, equal_nan=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a1` | `array_like` | 待比较的第一个数组 | `arr1` |
| `a2` | `array_like` | 待比较的第二个数组 | `arr2` |
| `equal_nan` | `bool` | 是否将两个 `NaN` 视为相等（NumPy 1.19+），默认为 `False` | `True` |

### 综合示例

#### 示例代码

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

#### 输出

```text
[[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
[[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
True
```

#### 理解重点

- 相同的 `seed` + 相同的调用顺序 ⇒ 相同的随机结果
- 中间插入新的随机调用会消耗随机状态，改变后续所有序列

## 常见坑

1. `arange(0, 1, 0.1)` 可能出现浮点累积误差——优先用 `linspace`
2. 只在一处设置 `seed` 后，中间插入新随机调用会改变后续序列——设置 `seed` 要对应具体的随机操作之前
3. 未指定 `dtype` 时 NumPy 自动推断，可能与预期不符——整数输入默认 `int64`，浮点输入默认 `float64`

## 小结

- 创建数组是后续索引、运算、线代和机器学习的入口
- 看到问题先想用哪种创建方式：固定值（`zeros`/`ones`/`full`/`eye`）、序列（`arange`/`linspace`）、随机（`rand`/`randn`/`randint`）
- 养成"固定随机种子"的习惯——`seed(42)` 一行，省去无数调试时间
