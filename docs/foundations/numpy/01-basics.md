---
title: NumPy 基础与数组概念
outline: deep
---

# NumPy 基础与数组概念

> 对应脚本：`Basic/Numpy/01_basics.py`  
> 运行方式：`python Basic/Numpy/01_basics.py`

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

## 1. 环境与打印配置

### 参数速览（本节）

适用 API（分项）：

1. `np.get_printoptions()`
2. `np.__version__`

| 参数名                     | 本例取值                                    | 说明                                |
| ----------------------- | --------------------------------------- | --------------------------------- |
| 返回值（`get_printoptions`） | `dict`                                  | 返回当前打印配置字典                        |
| 返回字段                    | `precision` / `threshold` / `linewidth` | 分别表示浮点显示位数、摘要阈值、每行宽度              |
| 返回值（`__version__`）      | `str`                                   | 返回 NumPy 版本字符串，用于确认 API 行为是否与文档一致 |
### 示例代码

```python
import numpy as np

print(np.__version__)
opts = np.get_printoptions()
print(opts["precision"], opts["threshold"], opts["linewidth"])
```

### 结果输出（示例）

```text
2.1.3
----------------
8 1000 75
```

### 理解重点

- 版本号用于排查 API 差异。
- 打印参数会影响数组在终端显示的精度和换行。

## 2. `ndarray` 与 Python 列表的差异

### 参数速览（本节）

适用 API/表达式（分项）：

1. `np.array(object, dtype=None, copy=True, ndmin=0)`
2. `np_array * 2`
3. `np_array + 6`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `object` | `[1, 2, 3, 4, 5]` | 输入 Python 列表并创建一维数组 |
| `dtype` | `None` | 不显式指定，按输入自动推断整数类型 |
| `ndmin` | `0` | 不强制补维度，保持输入的自然维度 |
| 标量操作数 | `2` / `6` | 按逐元素规则计算，标量通过广播扩展到数组形状 |
### 示例代码

```python
import numpy as np

py_list = [1, 2, 3, 4, 5]
np_array = np.array([1, 2, 3, 4, 5])

print(py_list * 2)
print(np_array * 2)

print(py_list + [6])
print(np_array + 6)
```

### 结果输出

```text
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
----------------
[ 2  4  6  8 10]
----------------
[1, 2, 3, 4, 5, 6]
----------------
[ 7  8  9 10 11]
```

### 理解重点

- 列表 `* 2` 是重复拼接，数组 `* 2` 是数值乘法。
- 列表 `+` 是拼接，数组 `+` 是逐元素运算（支持广播）。

## 3. 向量化性能优势

脚本比较了 100 万规模数据：

- Python 列表推导：`[x * 2 for x in py_list]`
- NumPy 向量化：`np_array * 2`

### 示例输出（一次运行）

```text
数据规模: 1,000,000
----------------
Python列表耗时: 0.0368秒
----------------
NumPy数组耗时: 0.0032秒
----------------
NumPy快了约 11.3 倍
```

### 理解重点

- 向量化把循环下沉到 C 层，解释器开销更低。
- 实际倍数与硬件、BLAS 实现、数据规模有关。

## 4. `ndarray` 基本形状概念

### 参数速览（本节）

适用 API/属性（分项）：

1. `np.array(object, dtype=None, copy=True, ndmin=0)`
2. `arr.shape`
3. `arr.ndim`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `object` | 一维/二维/三维嵌套列表 | 输入结构直接决定初始形状 |
| `dtype` | `None` | 默认自动推断，本节样例均为整数 |
| `ndmin` | `0` | 不额外补维度 |
| 返回值（`arr.shape`） | `tuple` | 返回形状元组 |
| 返回值（`arr.ndim`） | `int` | 返回维度数量 |
### 示例代码

```python
import numpy as np

arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(arr_1d.shape, arr_1d.ndim)
print(arr_2d.shape, arr_2d.ndim)
print(arr_3d.shape, arr_3d.ndim)
```

### 结果输出

```text
(5,) 1
----------------
(2, 3) 2
----------------
(2, 2, 2) 3
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