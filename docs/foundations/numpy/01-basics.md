---
title: NumPy 基础与数组概念
outline: deep
---

# NumPy 基础与数组概念

## 本章目标

1. 理解 NumPy 的核心对象 `ndarray` 及其与 Python 列表的本质差异
2. 掌握 `np.array` 创建数组、`shape` 与 `ndim` 等基础属性
3. 理解广播机制的运算语义
4. 理解向量化带来的性能优势

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.__version__` | 属性 | 查看 NumPy 版本号 |
| `np.get_printoptions()` | 函数 | 读取当前数组打印配置 |
| `np.array(...)` | 函数 | 从 Python 序列创建 `ndarray` |
| `arr * scalar` / `arr + scalar` | 运算符 | 广播：逐元素算术运算（非列表拼接） |
| `arr.shape` | 属性 | 数组各维度长度组成的元组 |
| `arr.ndim` | 属性 | 数组的维度数量 |

## 1. 环境与打印配置

### `np.get_printoptions`

#### 作用

读取当前 NumPy 的数组打印配置。只负责查询，不修改配置。

#### 重点方法

```python
np.get_printoptions()
```

#### 返回内容

| 键名 | 类型 | 含义 |
|---|---|---|
| `precision` | `int` | 小数显示精度，默认为 `8` |
| `threshold` | `int` | 元素总数超过此值时省略中间内容，默认为 `1000` |
| `edgeitems` | `int` | 省略时首尾各保留的元素数，默认为 `3` |
| `linewidth` | `int` | 每行最大字符宽度，默认为 `75` |
| `suppress` | `bool` | 是否尽量抑制科学计数法，默认为 `False` |
| `nanstr` | `str` | NaN 的显示文本，默认为 `'nan'` |
| `infstr` | `str` | 无穷大的显示文本，默认为 `'inf'` |
| `sign` | `str` | 正负号显示策略，默认为 `'-'` |
| `floatmode` | `str` | 浮点数格式模式，默认为 `'maxprec'` |
| `formatter` | `dict` 或 `None` | 自定义格式化器，默认为 `None` |
| `legacy` | `bool` | 兼容旧版打印行为，默认为 `False` |

可使用 `np.set_printoptions(**kwargs)` 修改这些配置。

#### 示例代码

```python
import numpy as np

print(np.get_printoptions())
```

#### 输出

```python
{'edgeitems': 3,
 'threshold': 1000,
 'floatmode': 'maxprec',
 'precision': 8,
 'suppress': False,
 'linewidth': 75,
 'nanstr': 'nan',
 'infstr': 'inf',
 'sign': '-',
 'formatter': None,
 'legacy': False,
 'override_repr': None}
```

### `np.__version__`

#### 作用

查看当前安装的 NumPy 版本号。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `str` | NumPy 版本号字符串 |

#### 示例代码

```python
import numpy as np

print(np.__version__)
```

#### 输出

```text
2.1.3
```

## 2. ndarray 与 Python 列表的差异

### `np.array`

#### 作用

从 Python 列表或嵌套序列创建 `ndarray`，是 NumPy 最基础的数组构造入口。

#### 重点方法

```python
np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `object` | `array_like` | 输入数据，可为列表、元组、嵌套序列、其他数组 | `[1, 2, 3, 4, 5]` |
| `dtype` | `dtype` 或 `None` | 元素数据类型，默认为 `None`（自动推断） | `np.float64` |
| `copy` | `bool` | 是否复制数据，默认为 `True` | `False` |
| `order` | `str` | 内存布局：`'K'` 保持原样 / `'C'` 行优先 / `'F'` 列优先 / `'A'` 任意，默认为 `'K'` | `'C'` |
| `subok` | `bool` | `True` 时保留子类类型，默认为 `False`（强制基础 `ndarray`） | `True` |
| `ndmin` | `int` | 返回数组的最小维度，不足时在前方补 1，默认为 `0` | `2` |
| `like` | `array_like` 或 `None` | 参考数组原型，用于兼容第三方数组库（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

pyList = [1, 2, 3, 4, 5]
npArray = np.array([1, 2, 3, 4, 5])

print(f"Python列表: {pyList}")
print(f"NumPy数组: {npArray}")
```

#### 输出

```text
Python列表: [1, 2, 3, 4, 5]
NumPy数组: [1 2 3 4 5]
```

#### 理解重点

- 列表有逗号分隔，数组没有——这是最直观的视觉差异
- 显式指定 `dtype` 可避免后续隐式类型转换

### 广播机制

#### 作用

NumPy 的 `*` 和 `+` 对数组执行**逐元素**运算，并通过广播自动扩展标量或低维数组。这与 Python 列表的拼接/重复语义完全不同。

#### 乘法对比

```python
import numpy as np

pyList = [1, 2, 3, 4, 5]
npArray = np.array([1, 2, 3, 4, 5])

print(pyList * 2)    # 列表：重复拼接
print(npArray * 2)   # 数组：逐元素乘法
```

```text
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
[ 2  4  6  8 10]
```

#### 加法对比

```python
print(pyList + [6])     # 列表：拼接
print(npArray + 6)      # 数组：逐元素加法（标量广播）
```

```text
[1, 2, 3, 4, 5, 6]
[ 7  8  9 10 11]
```

#### 理解重点

- 列表 `* n` 是重复拼接，数组 `* n` 是逐元素乘法
- 列表 `+` 是拼接，数组 `+` 是逐元素加法
- 标量与数组运算时，标量自动广播到每个元素——这是向量化的基础

## 3. 向量化性能优势

#### 作用

NumPy 的向量化运算将循环下沉到 C 层执行，避免了 Python 解释器的逐元素开销。数据规模越大，优势越明显。

#### 示例代码

```python
import time
import numpy as np

size = 10_000_000

# Python 列表推导
pyList = list(range(size))
start = time.time()
result = [x * 2 for x in pyList]
pyTime = time.time() - start

# NumPy 向量化
npArray = np.arange(size)
start = time.time()
result = npArray * 2
npTime = time.time() - start

print(f"数据规模: {size:,}")
print(f"Python列表耗时: {pyTime:.4f}秒")
print(f"NumPy数组耗时: {npTime:.4f}秒")
print(f"NumPy快了约 {pyTime / npTime:.1f} 倍")
```

#### 输出

```text
数据规模: 10,000,000
Python列表耗时: 0.4064秒
NumPy数组耗时: 0.0191秒
NumPy快了约 21.3 倍
```

#### 理解重点

- 向量化把循环下沉到 C 层，解释器开销更低
- 性能差距随数据量增大而拉大——小数据（<1000）差异不明显
- 实际倍数受硬件、BLAS 实现、数据类型影响

## 4. ndarray 基础属性

### `arr.shape`

#### 作用

返回数组各维度长度组成的元组。一维向量的 `shape` 为 `(n,)`，注意尾部的逗号表示它是元组。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `tuple[int, ...]` | 各维度长度，如 `(2, 3)` 表示 2 行 3 列 |

### `arr.ndim`

#### 作用

返回数组的维度数量（轴的数量），等于 `len(arr.shape)`。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 维度数量，标量为 `0`，向量为 `1`，矩阵为 `2` |

### 常用属性速览

| 属性 | 类型 | 含义 |
|---|---|---|
| `shape` | `tuple[int, ...]` | 各维度长度 |
| `ndim` | `int` | 维度数量 |
| `size` | `int` | 元素总数，等于各维度长度的乘积 |
| `dtype` | `numpy.dtype` | 元素的数据类型 |
| `itemsize` | `int` | 每个元素占用的字节数 |
| `nbytes` | `int` | 数组总字节数，等于 `size × itemsize` |

### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3], dtype=np.float32)

print(arr.shape, type(arr.shape))
print(arr.ndim, type(arr.ndim))
print(arr.dtype, type(arr.dtype))
print(arr.itemsize, arr.nbytes)
```

### 输出

```text
(3,) tuple
1 int
float32 numpy.dtype
4 12
```

### 理解重点

- 一维向量的 `shape` 是 `(3, )`，注意尾部逗号——它是单元素元组，去掉逗号就成了整数 `3`
- 二维理解为"行 × 列"，三维及以上建议用"轴"来思考
- 大数组先用 `nbytes / 1024**2` 估算内存占用量

## 常见坑

1. 把列表的 `+` / `*` 语义误用到数组上——列表是拼接/重复，数组是逐元素运算
2. 忽略 `shape` 导致后续广播或矩阵运算维度不匹配
3. 性能对比时数据规模太小（<1000），倍数差异不明显，结论不可靠

## 小结

- NumPy 的核心不是"更简短的语法"，而是"统一的数组对象 + 向量化计算"
- 从本章开始建立 `shape`、`dtype`、广播的思维模型，后续所有章节都建立在这之上
- `np.array` 是数据的入口，`shape` 和 `ndim` 是理解数据形状的第一手信息
