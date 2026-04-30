---
title: NumPy 实用函数
outline: deep
---

# NumPy 实用函数

## 本章目标

1. 掌握排序 `sort` 与索引排序 `argsort` 的区别
2. 掌握去重 `unique` 及其附加输出（索引、计数、逆映射）
3. 掌握集合运算：`intersect1d` / `union1d` / `setdiff1d` / `setxor1d` / `isin`
4. 掌握搜索：`argmax` / `argmin` / `where` / `nonzero`
5. 掌握裁剪 `clip` 与取整函数
6. 理解引用 / 视图 / 副本三种语义的区别

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.sort(...)` | 函数 | 返回排序后的新数组 |
| `np.argsort(...)` | 函数 | 返回排序后的索引 |
| `np.unique(...)` | 函数 | 返回唯一值，可选索引/计数/逆映射 |
| `np.intersect1d(...)` | 函数 | 两数组交集 |
| `np.union1d(...)` | 函数 | 两数组并集 |
| `np.setdiff1d(...)` | 函数 | 差集 $a \setminus b$ |
| `np.setxor1d(...)` | 函数 | 对称差集 |
| `np.isin(...)` | 函数 | 元素级成员检测（替代已弃用的 `in1d`） |
| `np.argmax(...)` / `np.argmin(...)` | 函数 | 最大 / 最小值索引 |
| `np.where(...)` | 函数 | 条件索引或三元替换 |
| `np.nonzero(...)` | 函数 | 非零元素索引 |
| `np.clip(...)` | 函数 | 截断到 $[a_{min}, a_{max}]$ 区间 |
| `np.floor/ceil/round/trunc(...)` | 函数 | 向下/向上/四舍五入/截断取整 |
| `arr.view(...)` | 方法 | 返回共享数据的视图 |
| `arr.copy(...)` | 方法 | 返回独立数据的副本 |

## 1. 排序

### `np.sort`

#### 作用

返回排序后的新数组，不修改原数组。支持多种排序算法和多维沿轴排序。

#### 重点方法

```python
np.sort(a, axis=-1, kind=None, order=None, stable=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[3, 1, 4, 1, 5]` |
| `axis` | `int` 或 `None` | 沿哪个轴排序，默认为 `-1`（最后一轴）；`None` 先展平 | `0` |
| `kind` | `str` 或 `None` | 算法：`'quicksort'` / `'mergesort'` / `'heapsort'` / `'stable'`，默认为 `'quicksort'` | `'stable'` |
| `stable` | `bool` 或 `None` | `True` 保持相等元素原序（等价 `kind='stable'`） | `True` |

### `np.argsort`

#### 作用

返回将数组排序所需的索引。`arr[np.argsort(arr)]` 等价于 `np.sort(arr)`。常用于跟随排序多个数组。

#### 重点方法

```python
np.argsort(a, axis=-1, kind=None, order=None, stable=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[3, 1, 4, 1, 5]` |
| `axis` | `int` 或 `None` | 沿哪个轴排序，默认为 `-1` | `0` |
| `kind` | `str` 或 `None` | 排序算法 | `'stable'` |
| `stable` | `bool` 或 `None` | 是否稳定排序 | `True` |

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"sort: {np.sort(arr)}")
print(f"argsort: {np.argsort(arr)}")
print(f"arr[argsort]: {arr[np.argsort(arr)]}")

arr2d = np.array([[3, 1, 2], [6, 4, 5]])
print(f"按行排序 axis=1:\n{np.sort(arr2d, axis=1)}")
print(f"按列排序 axis=0:\n{np.sort(arr2d, axis=0)}")
```

#### 输出

```text
sort: [1 1 2 3 4 5 6 9]
argsort: [1 3 6 0 2 4 7 5]
arr[argsort]: [1 1 2 3 4 5 6 9]
按行排序 axis=1:
[[1 2 3]
 [4 5 6]]
按列排序 axis=0:
[[3 1 2]
 [6 4 5]]
```

#### 理解重点

- `argsort` 返回"排序后位置对应的原索引"——可用于按 `scores` 排对应的 `names`
- 降序排序用 `np.sort(arr)[::-1]` 或 `arr[np.argsort(arr)[::-1]]`

## 2. 去重与计数

### `np.unique`

#### 作用

返回数组中的唯一值（默认已升序排列）。可选返回首次出现位置、逆映射、出现次数。

#### 重点方法

```python
np.unique(ar, return_index=False, return_inverse=False,
          return_counts=False, axis=None, *, equal_nan=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `ar` | `array_like` | 输入数组 | `[1, 2, 2, 3, 3, 3]` |
| `return_index` | `bool` | `True` 返回每个唯一值首次出现的索引，默认为 `False` | `True` |
| `return_inverse` | `bool` | `True` 返回把原数组映射回唯一值的逆索引，默认为 `False` | `True` |
| `return_counts` | `bool` | `True` 返回每个唯一值的出现次数，默认为 `False` | `True` |
| `axis` | `int` 或 `None` | `None` 先展平；指定轴时按行/列去重，默认为 `None` | `0` |
| `equal_nan` | `bool` | 是否将多个 `NaN` 视为同一值，默认为 `True` | —— |

#### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

print(f"唯一值: {np.unique(arr)}")
print(f"含首次索引: {np.unique(arr, return_index=True)}")
print(f"含计数: {np.unique(arr, return_counts=True)}")
```

#### 输出

```text
唯一值: [1 2 3 4]
含首次索引: (array([1, 2, 3, 4]), array([0, 1, 3, 6]))
含计数: (array([1, 2, 3, 4]), array([1, 2, 3, 4]))
```

#### 理解重点

- `return_counts=True` 是一行代码做频数统计的最快方式
- 值-计数对：`dict(zip(*np.unique(arr, return_counts=True)))`

## 3. 集合运算

所有集合运算都先展平再去重，结果一维且升序排列。

### 接口速览

| API | 数学符号 | 含义 |
|---|---|---|
| `np.intersect1d(a, b)` | $A \cap B$ | 交集 |
| `np.union1d(a, b)` | $A \cup B$ | 并集 |
| `np.setdiff1d(a, b)` | $A \setminus B$ | 差集（顺序敏感） |
| `np.setxor1d(a, b)` | $A \triangle B$ | 对称差集（只在一边出现） |
| `np.isin(a, b)` | $a_i \in B$ | 成员检测，返回布尔数组 |

### `np.isin`

#### 作用

对数组 `element` 的每个元素判断是否出现在 `test_elements` 中，返回同形状布尔数组。替代已弃用的 `np.in1d`。

#### 重点方法

```python
np.isin(element, test_elements, assume_unique=False, invert=False, *, kind=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `element` | `array_like` | 被检测的数组 | `a` |
| `test_elements` | `array_like` | 参考集合 | `[2, 4]` |
| `assume_unique` | `bool` | 假设两边已去重以加速，默认为 `False` | `True` |
| `invert` | `bool` | `True` 时返回"不在集合中"的掩码，默认为 `False` | `True` |
| `kind` | `str` 或 `None` | 算法提示：`'sort'` 或 `'table'`（NumPy 1.23+） | —— |

### 综合示例

#### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])

print(f"交集: {np.intersect1d(a, b)}")
print(f"并集: {np.union1d(a, b)}")
print(f"差集 a-b: {np.setdiff1d(a, b)}")
print(f"差集 b-a: {np.setdiff1d(b, a)}")
print(f"对称差: {np.setxor1d(a, b)}")
print(f"a 是否在 [2, 4]: {np.isin(a, [2, 4])}")
```

#### 输出

```text
交集: [3 4 5]
并集: [1 2 3 4 5 6 7]
差集 a-b: [1 2]
差集 b-a: [6 7]
对称差: [1 2 6 7]
a 是否在 [2, 4]: [False  True False  True False]
```

## 4. 搜索

### `np.argmax` / `np.argmin`

#### 作用

返回最大值 / 最小值的索引。多维时 `axis=None` 返回扁平索引。

#### 重点方法

```python
np.argmax(a, axis=None, out=None, keepdims=False)
np.argmin(a, axis=None, out=None, keepdims=False)
```

### `np.nonzero`

#### 作用

返回非零元素的索引元组，每个轴对应一个索引数组。等价于 `np.where(arr != 0)`。

#### 重点方法

```python
np.nonzero(a)
```

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([1, 5, 2, 8, 3, 9, 4, 7])
print(f"argmax: {np.argmax(arr)}, argmin: {np.argmin(arr)}")

idx = np.where(arr > 5)
print(f"大于 5 的索引: {idx[0]}, 值: {arr[idx]}")

arr2 = np.array([0, 1, 0, 2, 0, 3])
print(f"非零索引: {np.nonzero(arr2)[0]}")
```

#### 输出

```text
argmax: 5, argmin: 0
大于 5 的索引: [3 5 7], 值: [8 9 7]
非零索引: [1 3 5]
```

## 5. 裁剪与取整

### `np.clip`

#### 作用

将数组元素截断到 $[a_{min}, a_{max}]$ 区间：小于 $a_{min}$ 取 $a_{min}$，大于 $a_{max}$ 取 $a_{max}$。

#### 重点方法

```python
np.clip(a, a_min, a_max, out=None, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[1, 5, 10, 15, 20]` |
| `a_min` | `scalar` 或 `None` | 下界，`None` 表示不设下界 | `5` |
| `a_max` | `scalar` 或 `None` | 上界，`None` 表示不设上界 | `15` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |

### 取整函数对比

| 函数 | 对 `-1.2` 结果 | 对 `2.5` 结果 | 规则 |
|---|---|---|---|
| `np.floor` | `-2.0` | `2.0` | 向下（更负方向） |
| `np.ceil` | `-1.0` | `3.0` | 向上（更正方向） |
| `np.round` | `-1.0` | `2.0` | 银行家舍入（`.5` 向偶数） |
| `np.trunc` | `-1.0` | `2.0` | 截断（向零方向） |

#### 示例代码

```python
import numpy as np

arr = np.array([1, 5, 10, 15, 20])
print(f"clip(5, 15): {np.clip(arr, 5, 15)}")

x = np.array([1.2, 2.5, 3.7, -1.2, -2.5, -3.7])
print(f"floor: {np.floor(x)}")
print(f"ceil:  {np.ceil(x)}")
print(f"round: {np.round(x)}")
print(f"trunc: {np.trunc(x)}")
```

#### 输出

```text
clip(5, 15): [ 5  5 10 15 15]
floor: [ 1.  2.  3. -2. -3. -4.]
ceil:  [ 2.  3.  4. -1. -2. -3.]
round: [ 1.  2.  4. -1. -2. -4.]
trunc: [ 1.  2.  3. -1. -2. -3.]
```

## 6. 引用、视图、副本

#### 三种语义对比

| 操作 | 共享数据 | 修改影响原数组 | 场景 |
|---|---|---|---|
| `ref = arr` | 是（同一对象） | 是 | Python 赋值 |
| `arr.view()` | 是 | 是 | 重读 dtype / 不同视角 |
| `arr.copy()` | 否 | 否 | 真正独立副本 |

### `arr.view`

#### 作用

返回共享底层数据的新数组对象。可重解读为不同 `dtype`（只要字节数匹配）。修改视图影响原数组。

#### 重点方法

```python
arr.view(dtype=None, type=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `dtype` | `dtype` 或 `None` | 新的数据类型，默认为 `None`（保持原类型） | `np.int32` |
| `type` | `type` 或 `None` | 新的数组子类 | —— |

### `arr.copy`

#### 作用

返回完全独立的数据副本，修改不影响原数组。

#### 重点方法

```python
arr.copy(order='C')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先 / `'A'` 任意 / `'K'` 保持原样，默认为 `'C'` | `'F'` |

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# 引用
ref = arr
ref[0] = 100
print(f"改引用后，原数组: {arr}")

arr[0] = 1  # 复位

# 视图
v = arr.view()
v[1] = 200
print(f"改视图后，原数组: {arr}")

arr[1] = 2  # 复位

# 副本
c = arr.copy()
c[2] = 300
print(f"改副本后，原数组不变: {arr}")
```

#### 输出

```text
改引用后，原数组: [100   2   3   4   5]
改视图后，原数组: [  1 200   3   4   5]
改副本后，原数组不变: [1 2 3 4 5]
```

#### 理解重点

- 切片通常返回视图、花式索引/布尔索引返回副本——容易混淆
- 不确定时用 `arr.base` 判断：`None` 表示独立数据，非 `None` 表示视图
- 需要传出去且不希望被改的数据，显式 `.copy()` 最安全

## 常见坑

1. `argsort` 返回的是索引不是排序值——别直接当值用
2. `np.unique` 默认返回排序后的唯一值——非原序
3. `np.in1d` 已弃用——新代码用 `np.isin`
4. `np.round` 是银行家舍入（`.5` 向偶数）——需严格四舍五入时用 `np.floor(x + 0.5)`
5. 视图 vs 副本：误把视图当副本会出现"神秘联动修改"——大型项目优先显式 `.copy()`
6. `arr.view(np.int32)` 时 `float64` 的 8 字节被重读为两个 `int32`，形状也会变——这不是类型转换

## 小结

- 本章集合了日常数据处理中最常用的"工具抽屉"
- 集合运算、`unique`、`clip` 是数据清洗/EDA 的高频工具
- 搞清"引用 vs 视图 vs 副本"能避开 NumPy 最常见的一类 bug
