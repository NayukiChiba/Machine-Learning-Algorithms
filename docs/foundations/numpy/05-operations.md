---
title: NumPy 运算与统计
outline: deep
---

# NumPy 运算与统计

## 本章目标

1. 掌握 NumPy 的逐元素算术运算与比较运算
2. 掌握常用统计函数：求和、均值、方差、极值、累积
3. 理解 `axis` 参数在多维聚合中的语义
4. 熟练使用三角、指对、取整等常见数学函数
5. 掌握布尔数组的逻辑运算

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `+` `-` `*` `/` `**` `//` `%` | 运算符 | 逐元素算术运算 |
| `==` `!=` `>` `<` `>=` `<=` | 运算符 | 逐元素比较，返回布尔数组 |
| `np.array_equal(...)` | 函数 | 判断两个数组是否完全相等 |
| `np.any(...)` / `np.all(...)` | 函数 | 至少一个为真 / 全部为真 |
| `arr.sum(...)` / `arr.mean(...)` | 方法 | 求和 / 均值，支持按轴聚合 |
| `arr.std(...)` / `arr.var(...)` | 方法 | 标准差 / 方差 |
| `arr.min(...)` / `arr.max(...)` | 方法 | 最小值 / 最大值 |
| `arr.argmin(...)` / `arr.argmax(...)` | 方法 | 极值所在索引 |
| `arr.cumsum(...)` / `arr.cumprod(...)` | 方法 | 累积和 / 累积积 |
| `np.sin(...)` / `np.cos(...)` | 函数 | 三角函数（弧度） |
| `np.exp(...)` / `np.log(...)` | 函数 | 指数 / 自然对数 |
| `np.floor(...)` / `np.ceil(...)` / `np.round(...)` | 函数 | 向下 / 向上取整 / 四舍五入 |
| `np.abs(...)` | 函数 | 绝对值 |
| `np.logical_and/or/not/xor(...)` | 函数 | 布尔数组的逻辑运算 |

## 1. 算术运算

#### 作用

NumPy 的算术运算符全部为逐元素操作（element-wise）。标量与数组运算时标量自动广播到每个位置。`*` 是逐元素乘而非矩阵乘法。

### 运算符一览

| 运算符 | 含义 | 示例 |
|---|---|---|
| `+` | 逐元素加 | `a + b` |
| `-` | 逐元素减 | `a - b` |
| `*` | 逐元素乘（非矩阵乘法） | `a * b` |
| `/` | 逐元素除，返回浮点 | `a / b` |
| `**` | 逐元素幂 | `a ** 2` |
| `//` | 逐元素整除（地板除） | `a // 2` |
| `%` | 逐元素取模 | `a % 2` |

### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")
print(f"a // 2 = {a // 2}")
print(f"a % 2 = {a % 2}")
```

### 输出

```text
a + b = [ 6  8 10 12]
a - b = [-4 -4 -4 -4]
a * b = [ 5 12 21 32]
a / b = [0.2        0.33333333 0.42857143 0.5       ]
a ** 2 = [ 1  4  9 16]
a // 2 = [0 1 1 2]
a % 2 = [1 0 1 0]
```

### 理解重点

- `*` 是逐元素乘，**不是**矩阵乘法——矩阵乘法用 `@` 或 `np.dot`
- 整数数组做 `/` 返回浮点数组；做 `//` 保持整数

## 2. 比较与聚合

### 比较运算符

| 运算符 | 含义 | 返回 |
|---|---|---|
| `==` | 逐元素等于 | 布尔数组 |
| `!=` | 逐元素不等 | 布尔数组 |
| `>` | 逐元素大于 | 布尔数组 |
| `<` | 逐元素小于 | 布尔数组 |
| `>=` | 逐元素大于等于 | 布尔数组 |
| `<=` | 逐元素小于等于 | 布尔数组 |

### `np.array_equal`

#### 作用

判断两个数组是否完全相等（形状一致且每个元素值相等）。这是整体判断，不同于逐元素的 `a == b`。

#### 重点方法

```python
np.array_equal(a1, a2, equal_nan=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a1` | `array_like` | 第一个数组 | `a` |
| `a2` | `array_like` | 第二个数组 | `b` |
| `equal_nan` | `bool` | 是否将两个 `NaN` 视为相等（NumPy 1.19+），默认为 `False` | `True` |

### `np.any` / `np.all`

#### 作用

- `np.any`：至少一个元素为真（逻辑或）
- `np.all`：所有元素均为真（逻辑与）

两者均支持 `axis` 按轴聚合。

#### 重点方法

```python
np.any(a, axis=None, out=None, keepdims=False, *, where=True)
np.all(a, axis=None, out=None, keepdims=False, *, where=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组，非布尔值会按 `bool(x)` 转换 | `a == b` |
| `axis` | `int` 或 `None` | 聚合轴，默认为 `None`（对全部元素） | `0` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |
| `keepdims` | `bool` | 是否保留被聚合的轴（长度为 1），默认为 `False` | `True` |
| `where` | `array_like` | 只对 `True` 位置参与聚合，默认为 `True` | —— |

### 综合示例

#### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([4, 3, 2, 1])

print(f"a == b: {a == b}")
print(f"a > b: {a > b}")
print(f"np.array_equal(a, b): {np.array_equal(a, b)}")
print(f"np.any(a == b): {np.any(a == b)}")
print(f"np.all(a != b): {np.all(a != b)}")
```

#### 输出

```text
a == b: [False False False False]
a > b: [False False  True  True]
np.array_equal(a, b): False
np.any(a == b): False
np.all(a != b): True
```

## 3. 统计函数

以下 API 是 `ndarray` 的方法，同时存在同名的 `np.xxx` 函数，两者等价。

### `ndarray.sum`

#### 作用

对数组元素求和，支持按 `axis` 聚合。全局求和公式：

$$
\text{sum} = \sum_{i} a_i
$$

#### 重点方法

```python
arr.sum(axis=None, dtype=None, out=None, keepdims=False, initial=<no value>, where=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `axis` | `int` 或 `tuple[int, ...]` 或 `None` | 聚合轴，默认为 `None`（全部元素） | `0`、`1` |
| `dtype` | `dtype` 或 `None` | 累加中间类型，整数数组可指定 `np.int64` 避免溢出 | `np.int64` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |
| `keepdims` | `bool` | 保留被聚合的轴（长度为 1），便于后续广播，默认为 `False` | `True` |
| `initial` | `scalar` | 聚合初始值 | `0` |
| `where` | `array_like` | 只对 `True` 位置参与聚合，默认为 `True` | —— |

### `ndarray.mean`

#### 作用

计算算术平均值，签名与 `sum` 高度一致。公式：

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

#### 重点方法

```python
arr.mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)
```

参数含义同 `sum`，不再重复。

### `ndarray.std` / `ndarray.var`

#### 作用

计算标准差 / 方差。`ddof` 控制自由度修正：

- $ddof=0$（默认）：总体方差 $\sigma^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$
- $ddof=1$：样本方差 $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$

#### 重点方法

```python
arr.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)
arr.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `axis` | `int` 或 `tuple` 或 `None` | 聚合轴 | `0`、`1` |
| `ddof` | `int` | 自由度修正，默认为 `0`（总体）；`1` 为样本（无偏） | `1` |
| `keepdims` | `bool` | 是否保留聚合轴，默认为 `False` | `True` |

### `ndarray.min` / `ndarray.max`

#### 作用

返回数组的最小值 / 最大值，支持按轴聚合。

#### 重点方法

```python
arr.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
arr.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
```

### `ndarray.argmin` / `ndarray.argmax`

#### 作用

返回最小值 / 最大值所在的**索引**。多维时 `axis=None` 返回扁平后的整数索引。

#### 重点方法

```python
arr.argmin(axis=None, out=None, keepdims=False)
arr.argmax(axis=None, out=None, keepdims=False)
```

### `ndarray.cumsum` / `ndarray.cumprod`

#### 作用

沿指定轴计算累积和 / 累积积。累积和公式：

$$
\text{cumsum}_k = \sum_{i=1}^{k} x_i
$$

#### 重点方法

```python
arr.cumsum(axis=None, dtype=None, out=None)
arr.cumprod(axis=None, dtype=None, out=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `axis` | `int` 或 `None` | `None` 先展平再累积，默认为 `None` | `0` |
| `dtype` | `dtype` 或 `None` | 中间类型，整数乘积易溢出可升高精度 | `np.int64` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |

### 综合示例

#### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.randint(1, 100, size=10)

print(f"数组: {arr}")
print(f"sum={arr.sum()}, mean={arr.mean():.2f}")
print(f"std={arr.std():.2f}, var={arr.var():.2f}")
print(f"min={arr.min()}, max={arr.max()}")
print(f"argmin={arr.argmin()}, argmax={arr.argmax()}")
print(f"cumsum: {arr.cumsum()}")
print(f"前 5 个 cumprod: {arr[:5].cumprod()}")
```

#### 输出

```text
数组: [52 93 15 72 61 21 83 87 75 75]
sum=634, mean=63.40
std=25.37, var=643.64
min=15, max=93
argmin=2, argmax=1
cumsum: [ 52 145 160 232 293 314 397 484 559 634]
前 5 个 cumprod: [       52      4836     72540   5222880 318595680]
```

## 4. 沿轴聚合（axis）

#### 作用

对多维数组聚合时，`axis` 指定沿哪个轴方向进行。核心直觉（二维）：

- `axis=0`：沿**行方向**走 → 对每一列聚合，结果形状 = 列数
- `axis=1`：沿**列方向**走 → 对每一行聚合，结果形状 = 行数
- `axis=None`（默认）：对所有元素聚合为标量

**口诀**："聚合轴就是被吃掉的轴"——`axis=0` 聚合后行维度消失，剩下列。

#### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原数组:\n{arr}")
print(f"全体 sum(): {arr.sum()}")
print(f"按列 sum(axis=0): {arr.sum(axis=0)}")
print(f"按行 sum(axis=1): {arr.sum(axis=1)}")
print(f"按列 mean(axis=0): {arr.mean(axis=0)}")
print(f"按行 mean(axis=1): {arr.mean(axis=1)}")
```

#### 输出

```text
原数组:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
全体 sum(): 45
按列 sum(axis=0): [12 15 18]
按行 sum(axis=1): [ 6 15 24]
按列 mean(axis=0): [4. 5. 6.]
按行 mean(axis=1): [2. 5. 8.]
```

#### 理解重点

- 三维以上建议始终用 `keepdims=True` 调试，结果形状比不带 `keepdims` 直观
- `axis=0` 聚合结果取第一行看：`[12, 15, 18]` = `[1+4+7, 2+5+8, 3+6+9]`

## 5. 数学函数

### `np.sin` / `np.cos`

#### 作用

计算三角函数，输入按**弧度**计算（非角度）。转换公式：$\text{rad} = \theta° \times \frac{\pi}{180}$。

#### 重点方法

```python
np.sin(x, /, out=None, *, where=True, dtype=None)
np.cos(x, /, out=None, *, where=True, dtype=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 输入弧度值 | `[0, np.pi/2, np.pi]` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |
| `where` | `array_like` | 只对 `True` 位置执行运算 | —— |
| `dtype` | `dtype` 或 `None` | 指定输出类型 | —— |

### `np.exp`

#### 作用

计算自然指数 $e^x$，其中 $e \approx 2.71828$。

#### 重点方法

```python
np.exp(x, /, out=None, *, where=True, dtype=None)
```

### `np.log` / `np.log10`

#### 作用

`np.log` 计算自然对数 $\ln x$，`np.log10` 计算常用对数 $\log_{10} x$。输入必须 $>0$，否则返回 `-inf` 或 `nan`。

#### 重点方法

```python
np.log(x, /, out=None, *, where=True, dtype=None)
np.log10(x, /, out=None, *, where=True, dtype=None)
```

### `np.floor` / `np.ceil`

#### 作用

向下取整 / 向上取整。注意"向下"对负数而言是更负方向：$floor(-1.2) = -2$。

#### 重点方法

```python
np.floor(x, /, out=None, *, where=True, dtype=None)
np.ceil(x, /, out=None, *, where=True, dtype=None)
```

### `np.round`

#### 作用

四舍五入到指定小数位。NumPy 采用**银行家舍入**（`.5` 向偶数舍入），结果可能与"严格四舍五入"不同。

#### 重点方法

```python
np.round(a, decimals=0, out=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[1.5, 2.5, 3.5]` |
| `decimals` | `int` | 保留小数位数，可为负数（舍入到 10/100 位），默认为 `0` | `2` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |

### `np.abs`

#### 作用

计算绝对值。复数输入时返回模长：$|a + bi| = \sqrt{a^2 + b^2}$。

#### 重点方法

```python
np.abs(x, /, out=None, *, where=True, dtype=None)
```

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"sin: {np.sin(arr).round(3)}")
print(f"cos: {np.cos(arr).round(3)}")

arr2 = np.array([1, 2, 3, 4, 5])
print(f"exp: {np.exp(arr2).round(3)}")
print(f"log: {np.log(arr2).round(3)}")
print(f"log10: {np.log10(arr2).round(3)}")

arr3 = np.array([1.2, 2.5, 3.7, -1.2, -2.5])
print(f"floor: {np.floor(arr3)}")
print(f"ceil:  {np.ceil(arr3)}")
print(f"round: {np.round(arr3)}")
print(f"abs:   {np.abs(arr3)}")
```

#### 输出

```text
sin: [0.    0.5   0.707 0.866 1.   ]
cos: [1.    0.866 0.707 0.5   0.   ]
exp: [  2.718   7.389  20.086  54.598 148.413]
log: [0.    0.693 1.099 1.386 1.609]
log10: [0.    0.301 0.477 0.602 0.699]
floor: [ 1.  2.  3. -2. -3.]
ceil:  [ 2.  3.  4. -1. -2.]
round: [ 1.  2.  4. -1. -2.]
abs:   [1.2 2.5 3.7 1.2 2.5]
```

## 6. 逻辑函数

专门用于布尔数组的逻辑运算。输入非布尔时先按 `bool(x)` 转换。

### `np.logical_and`

#### 作用

按位逻辑与，等价于 `a & b`。

#### 重点方法

```python
np.logical_and(x1, x2, /, out=None, *, where=True, dtype=None)
```

### `np.logical_or`

#### 作用

按位逻辑或，等价于 `a | b`。

#### 重点方法

```python
np.logical_or(x1, x2, /, out=None, *, where=True, dtype=None)
```

### `np.logical_not`

#### 作用

按位逻辑非，等价于 `~a`。

#### 重点方法

```python
np.logical_not(x, /, out=None, *, where=True, dtype=None)
```

### `np.logical_xor`

#### 作用

按位异或：相同为 `False`，不同为 `True`。

#### 重点方法

```python
np.logical_xor(x1, x2, /, out=None, *, where=True, dtype=None)
```

### 综合示例

#### 示例代码

```python
import numpy as np

a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

print(f"logical_and: {np.logical_and(a, b)}")
print(f"logical_or:  {np.logical_or(a, b)}")
print(f"logical_not(a): {np.logical_not(a)}")
print(f"logical_xor: {np.logical_xor(a, b)}")
```

#### 输出

```text
logical_and: [ True False False False]
logical_or:  [ True  True  True False]
logical_not(a): [False False  True  True]
logical_xor: [False  True  True False]
```

## 常见坑

1. `*` 是逐元素乘，**不是**矩阵乘法——矩阵乘法用 `@` 或 `np.dot`
2. `axis` 方向容易理解反：`axis=0` 是对列聚合（行被吃掉），不是"对第 0 行操作"
3. `np.round` 遵循**银行家舍入**（`.5` 向偶数舍入），`2.5` → `2`，`3.5` → `4`
4. `ddof=0`（默认）是**有偏**方差 / 标准差；推断统计应用 `ddof=1` 做无偏估计
5. `np.log(0)` 返回 `-inf` 并发出 `RuntimeWarning`；负数返回 `nan`
6. 大整数数组做 `cumprod` 极易溢出，应先 `.astype(np.int64)` 或转浮点

## 小结

- 运算与统计是 NumPy 在数据分析中的核心价值：无循环处理任意维度数据
- 熟练使用 `axis` 是 NumPy 进阶的分水岭
- 统计函数的 `ddof`、`keepdims` 两个参数对结果影响大，每次调用时明确意图
- ufunc 的通用参数（`out`、`where`、`dtype`）是内存和性能优化的入口
