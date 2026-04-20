---
title: NumPy 基础与数组概念
outline: deep
---

# NumPy 基础与数组概念

## 本章目标

1. 理解 NumPy 的核心对象 `ndarray`。
2. 明确 Python 列表与 NumPy 数组的运算语义差异。
3. 理解向量化带来的性能优势。
4. 掌握数组最基础属性：`shape`、`ndim`。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.__version__` | 属性 | 查看 NumPy 版本 |
| `np.get_printoptions()` | 函数 | 查看数组打印配置 |
| `np.array(...)` | 函数 | 创建 `ndarray` |
| `np.arange(...)` | 函数 | 快速创建等差序列 |
| `arr.shape` | 属性 | 数组形状 |
| `arr.ndim` | 属性 | 数组维度 |

## 环境与打印配置

### `np.get_printoptions()`

#### 作用

用来读取当前 NumPy· 的数组打印配置。它只负责查询，不会修改配置。

#### 返回值

-  类型：`dict[str, Any]`
- 含义：当前生效的打印配置字典

#### 返回内容

| key名称     | value类型      | 含义                                 |
| ----------- | -------------- | ------------------------------------ |
| `precision` | `int`          | 小数显示精度                         |
| `hreshold`  | `int`          | 元素超过这个数量时会省略中间内容     |
| `edgeitems` | `int`          | 省略时头尾各保留多少项               |
| `linewidth` | `int`          | 每行显示的最大字符宽度               |
| `suppress`  | `bool`         | 是否尽量抑制科学计数法               |
| `nanstr`    | `str`          | NaN 的显示文本                       |
| `infstr`    | `str`          | 无穷大的显示文本                     |
| `sign`      | `str`          | 正负号显示策略                       |
| `floatmode` | `str`          | 浮点数格式模式                       |
| `formatter` | `dict`或`None` | 自定义格式化器                       |
| `legacy`    | `bool`         | 兼容旧版本打印行为的开关（版本相关） |

可以使用`np.set_printoptions(**args)`来更改设置，在`**args`中填上上述的参数名字

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

查看numpy版本

#### 返回内容

* 类型：`str`
* 含义：`numpy`版本号

#### 示例代码

```python
import numpy as np
print(np.__version__)
```

#### 结果输出

```text
2.1.3
```

## `ndarray` 与 Python 列表的差异

### 创建`np.ndarray`数据

#### 重点方法

```python
np.array(object, dtype=None, copy=True, ndmin=0)
```

#### 参数

| 参数名   | 本例取值          | 说明                               |
| -------- | ----------------- | ---------------------------------- |
| `object` | `[1, 2, 3, 4, 5]` | 输入 Python 列表并创建一维数组     |
| `dtype`  | `None`            | 不显式指定，按输入自动推断整数类型 |
| `ndmin`  | `0`               | 不强制补维度，保持输入的自然维度   |

#### 示例代码

```python
import numpy as np

py_list = [1, 2, 3, 4, 5]
np_array = np.array([1, 2, 3, 4, 5])
```

#### 输出

```
Python列表: [1, 2, 3, 4, 5]
NumPy数组: [1 2 3 4 5]
```

### 广播机制

#### 乘法

* 方法：`np_array * 2`

* 示例：

```
print(py_list * 2)
print(np_array * 2)
```

* 结果：

```python
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
[ 2  4  6  8 10]
```



#### 加法

* 方法：`np_array + 6`
* 示例：

```python
print(py_list + [6])
print(np_array + 6)
```

* 结果：

```python
[1, 2, 3, 4, 5, 6]
[ 7  8  9 10 11]
```

#### 理解重点

- 列表 `* 2` 是重复拼接，数组 `* 2` 是数值乘法。
- 列表 `+` 是拼接，数组 `+` 是逐元素运算（支持广播）。

## 向量化性能优势

脚本比较了 100 万规模数据：

- Python 列表推导：`[x * 2 for x in py_list]`
- NumPy 向量化：`np_array * 2`

### 示例代码

```python
import time
import numpy as np
size = 10000000

# Python 列表运算
py_list = list(range(size))
start = time.time()
result_list = [x * 2 for x in py_list]
py_time = time.time() - start

# NumPy 数组运算
np_array = np.arange(size)
start = time.time()
result_np = np_array * 2
np_time = time.time() - start

print(f"数据规模: {size:,}")
print(f"Python列表耗时: {py_time:.4f}秒")
print(f"NumPy数组耗时: {np_time:.4f}秒")
print(f"NumPy快了约 {py_time / np_time:.1f} 倍")
```

### 结果

```text
数据规模: 10,000,000
Python列表耗时: 0.4064秒
NumPy数组耗时: 0.0191秒
NumPy快了约 21.3 倍
```

### 理解重点

- 向量化把循环下沉到 C 层，解释器开销更低。
- 实际倍数与硬件、BLAS 实现、数据规模有关。

## `ndarray` 类型数据的基本属性

下面是你要的“属性 + 含义 + 数据类型”表格，可直接放到文档里。

### ndarray 常用属性

| 属性       | 返回数据类型              | 含义                               |
| ---------- | ------------------------- | ---------------------------------- |
| `shape`    | `tuple[int, ...]`         | 数组各维度长度，如 (2, 3)          |
| `ndim`     | `int`                     | 维度数量                           |
| `size`     | `int`                     | 元素总数                           |
| `dtype`    | `numpy.dtype`             | 元素的数据类型描述对象             |
| `itemsize` | `int`                     | 每个元素占用字节数                 |
| `nbytes`   | `int`                     | 数组总字节数，等于 size × itemsize |
| `T`        | `numpy.ndarray`           | 转置视图（主要用于二维及以上）     |
| `strides`  | `tuple[int, ...]`         | 各轴移动一步对应的字节跨度         |
| `flags`    | `numpy.flagsobj`          | 内存布局和可写性等标志信息         |
| `real`     | `numpy.ndarray`           | 实部视图（复数数组常用）           |
| `imag`     | `numpy.ndarray`           | 虚部视图（实数数组通常为 0）       |
| `flat`     | `numpy.flatiter`          | 按一维顺序访问元素的迭代器         |
| `base`     | `numpy.ndarray` 或 `None` | 若为视图则指向原数组，否则为 None  |

### 常见元素数据类型（dtype）

| dtype 名称                         | 说明                | 每元素字节数（常见） |
| ---------------------------------- | ------------------- | -------------------- |
| `bool_`                            | 布尔类型 True/False | 1                    |
| `int8 / int16 / int32 / int64`     | 有符号整数          | 1 / 2 / 4 / 8        |
| `uint8 / uint16 / uint32 / uint64` | 无符号整数          | 1 / 2 / 4 / 8        |
| `float16 / float32 / float64`      | 浮点数              | 2 / 4 / 8            |
| `complex64 / complex128`           | 复数（实部+虚部）   | 8 / 16               |
| `str_ / unicode_`                  | 字符串类型          | 变长（与长度相关）   |
| `object_`                          | Python 对象类型     | 指针大小相关         |

### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3], dtype=np.float32)

print(arr.shape, type(arr.shape))
print(arr.ndim, type(arr.ndim))     
print(arr.dtype, type(arr.dtype))   
print(arr.itemsize, arr.nbytes)     
```

### 结果输出

```text
(3,) tuple
1 int
float32 numpy.dtype
4 12
```

### 理解重点

- 一维向量的 `shape` 是 `(n,)`，注意尾部逗号。
- 二维常理解为“行 × 列”。
- 三维以上建议用“轴”来思考，而不是“行列”。

## 常见坑

1. 把列表运算语义误用到数组上。
2. 忽略 `shape` 导致后续广播/矩阵运算报错。
3. 性能对比时数据规模太小，差异不明显。

## 小结

- NumPy 的核心不是“更简短的语法”，而是“统一的数组对象 + 向量化计算”。
- 从本章开始建立 `shape`、`dtype`、广播的思维方式，会让后续章节更顺。