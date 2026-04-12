---
title: NumPy 运算与统计
outline: deep
---

# NumPy 运算与统计

> 对应脚本：`Basic/Numpy/05_operations.py`  
> 运行方式：`python Basic/Numpy/05_operations.py`

## 本章目标

1. 掌握 NumPy 的逐元素算术与比较运算。
2. 掌握统计函数及 `axis` 的语义。
3. 掌握常见数学函数与逻辑函数的用法。

## 重点方法速览

| 分类 | 方法 |
|---|---|
| 算术 | `+` `-` `*` `/` `**` `//` `%` |
| 比较 | `==` `!=` `>` `<`、`np.array_equal`、`np.any`、`np.all` |
| 统计 | `sum` `mean` `std` `var` `min` `max` `argmin` `argmax` `cumsum` `cumprod` |
| 数学 | `np.sin` `np.cos` `np.exp` `np.log` `np.log10` `np.floor` `np.ceil` `np.round` `np.abs` |
| 逻辑 | `np.logical_and` `np.logical_or` `np.logical_not` `np.logical_xor` |

## 1. 算术运算（元素级）

### 参数速览（本节）

适用运算/API（分项）：

1. `+` `-` `*` `/` `**` `//` `%`
2. `np.add(a, b, out=None, where=True, dtype=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 左右操作数 | `a`、`b`、`2` | 默认执行逐元素计算 |
| `out` | `None`（默认） | 可把结果写入指定数组 |
| `where` | `True`（默认） | 仅在条件为真位置执行运算 |
| `dtype` | `None`（默认） | 控制计算类型（部分 ufunc 支持） |
### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
print(a // 2)
print(a % 2)
```

### 结果输出

```text
[ 6  8 10 12]
----------------
[-4 -4 -4 -4]
----------------
[ 5 12 21 32]
----------------
[0.2        0.33333333 0.42857143 0.5       ]
----------------
[ 1  4  9 16]
----------------
[0 1 1 2]
----------------
[1 0 1 0]
```

### 理解重点

- NumPy 默认是逐元素运算，而不是矩阵乘法。
- 整数做 `/` 会得到浮点结果。

## 2. 比较运算与聚合判断

### 参数速览（本节）

适用 API/表达式（分项）：

1. `a == b`
2. `a > b`
3. `np.array_equal(a1, a2, equal_nan=False)`
4. `np.any(a, axis=None, out=None, keepdims=False, where=True)`
5. `np.all(a, axis=None, out=None, keepdims=False, where=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 比较操作符 | `==`、`>` | 返回同形状布尔数组 |
| `a1` / `a2` | `a` / `b` | 判断两个数组是否完全相等 |
| `equal_nan` | `False`（默认） | 控制是否将 `NaN` 视为相等 |
| `axis` / `where`（`any`） | `None` / `True` | 是否存在至少一个真值，可按轴聚合 |
| `axis` / `where`（`all`） | `None` / `True` | 是否全部为真，可按轴聚合 |
### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([4, 3, 2, 1])

print(a == b)
print(a > b)
print(np.array_equal(a, b))
print(np.any(a == b))
print(np.all(a != b))
```

### 结果输出

```text
[False False False False]
----------------
[False False  True  True]
----------------
False
----------------
False
----------------
True
```

## 3. 统计运算

脚本固定随机种子：`np.random.seed(42)`。

### 参数速览（本节）

适用 API（分项）：

1. `sum(a, axis=None, dtype=None, keepdims=False, where=True)`
2. `mean(a, axis=None, dtype=None, keepdims=False, where=True)`
3. `std(a, axis=None, dtype=None, ddof=0, keepdims=False, where=True)`
4. `var(a, axis=None, dtype=None, ddof=0, keepdims=False, where=True)`
5. `min/max/argmin/argmax(a, axis=None, ...)`
6. `cumsum/cumprod(a, axis=None, dtype=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `axis` / `dtype`（`sum`） | `None` / `None` | 对全部元素求和 |
| `axis` / `dtype`（`mean`） | `None` / `None` | 计算整体均值 |
| `ddof`（`std`） | `0` | 计算总体标准差 |
| `ddof`（`var`） | `0` | 计算总体方差 |
| `axis`（`min/max/argmin/argmax`） | `None` | 获取全局最值及其索引 |
| `axis`（`cumsum/cumprod`） | `None` | 沿扁平序列做累计和/累计积 |
### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.randint(1, 100, size=10)

print(arr)
print(arr.sum(), arr.mean(), arr.std(), arr.var())
print(arr.min(), arr.max(), arr.argmin(), arr.argmax())
print(arr.cumsum())
print(arr[:5].cumprod())
```

### 结果输出

```text
[52 93 15 72 61 21 83 87 75 75]
----------------
634 63.4 25.37 643.64
----------------
15 93 2 1
----------------
[ 52 145 160 232 293 314 397 484 559 634]
----------------
[       52      4836     72540   5222880 318595680]
```

## 4. `axis` 参数详解

以数组：

```text
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

为例：

- `sum()`：全部元素求和。
- `sum(axis=0)`：按列聚合。
- `sum(axis=1)`：按行聚合。

### 示例输出

```text
sum() = 45
----------------
sum(axis=0) = [12 15 18]
----------------
mean(axis=0) = [4. 5. 6.]
----------------
sum(axis=1) = [ 6 15 24]
----------------
mean(axis=1) = [2. 5. 8.]
```

## 5. 数学函数

### 参数速览（本节）

适用 API（分项）：

1. `np.sin(x, out=None, where=True)`
2. `np.cos(x, out=None, where=True)`
3. `np.exp(x, out=None, where=True)`
4. `np.log(x, out=None, where=True)`
5. `np.log10(x, out=None, where=True)`
6. `np.round(a, decimals=0, out=None)`
7. `np.floor(a)`
8. `np.ceil(a)`
9. `np.abs(a)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x`（`sin/cos`） | `[0, π/6, π/4, π/3, π/2]` | 输入按弧度解释 |
| `x`（`exp`） | `[1, 2, 3, 4, 5]` | 计算指数函数 |
| `x`（`log/log10`） | `[1, 2, 3, 4, 5]` | 要求 `x > 0` |
| `decimals`（`round`） | `0`（默认） | 控制四舍五入保留位数 |
| `a`（`floor/ceil/abs`） | `arr3` | 分别做向下取整、向上取整、绝对值 |

### 三角函数与指数对数

```python
import numpy as np

arr = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(np.sin(arr).round(3))
print(np.cos(arr).round(3))

arr2 = np.array([1, 2, 3, 4, 5])
print(np.exp(arr2).round(3))
print(np.log(arr2).round(3))
print(np.log10(arr2).round(3))
```

### 结果输出

```text
[0.    0.5   0.707 0.866 1.   ]
----------------
[1.    0.866 0.707 0.5   0.   ]
----------------
[  2.718   7.389  20.086  54.598 148.413]
----------------
[0.    0.693 1.099 1.386 1.609]
----------------
[0.    0.301 0.477 0.602 0.699]
```

### 取整函数输出

```text
arr3 = [ 1.2  2.5  3.7 -1.2 -2.5]
----------------
floor: [ 1.  2.  3. -2. -3.]
----------------
ceil:  [ 2.  3.  4. -1. -2.]
----------------
round: [ 1.  2.  4. -1. -2.]
----------------
abs:   [1.2 2.5 3.7 1.2 2.5]
```

## 6. 逻辑运算

### 参数速览（本节）

适用 API（分项）：

1. `np.logical_and(x1, x2, out=None, where=True)`
2. `np.logical_or(x1, x2, out=None, where=True)`
3. `np.logical_not(x, out=None, where=True)`
4. `np.logical_xor(x1, x2, out=None, where=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x1` / `x2`（`logical_and`） | `a` / `b` | 两数组按位逻辑与 |
| `x1` / `x2`（`logical_or`） | `a` / `b` | 两数组按位逻辑或 |
| `x`（`logical_not`） | `a` | 单数组按位逻辑非 |
| `x1` / `x2`（`logical_xor`） | `a` / `b` | 两数组按位异或 |
| 输入类型 | 布尔数组 | 非布尔输入会先转换为布尔值 |
### 示例代码

```python
import numpy as np

a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

print(np.logical_and(a, b))
print(np.logical_or(a, b))
print(np.logical_not(a))
print(np.logical_xor(a, b))
```

### 结果输出

```text
[ True False False False]
----------------
[ True  True  True False]
----------------
[False False  True  True]
----------------
[False  True  True False]
```

## 常见坑

1. `*` 是逐元素乘法，不是矩阵乘法；矩阵乘法请用 `@`。
2. `axis` 方向理解反了会导致结果维度不对。
3. `np.round` 在 `.5` 附近遵循银行家舍入策略，可能和直觉不同。

## 小结

- 运算与统计是 NumPy 在数据分析中的核心价值。
- 熟练使用 `axis`，就能高效处理任意维度数据。
