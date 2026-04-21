---
title: NumPy 实用函数
outline: deep
---

# NumPy 实用函数

## 本章目标

1. 掌握排序 `sort` 与索引排序 `argsort` 的区别。
2. 掌握去重 `unique` 及其附加输出（索引、计数、逆映射）。
3. 掌握集合运算 `intersect1d` / `union1d` / `setdiff1d` / `setxor1d` / `isin`。
4. 掌握搜索 `argmax` / `argmin` / `where` / `nonzero`。
5. 掌握裁剪 `clip` 与取整 `floor` / `ceil` / `round` / `trunc`。
6. 理解 `引用` / `view` / `copy` 三种语义的区别。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.sort(...)` | 函数 | 返回排序后的**新数组** |
| `np.argsort(...)` | 函数 | 返回排序后的**索引** |
| `np.unique(...)` | 函数 | 返回去重、可选索引 / 计数 / 逆映射 |
| `np.intersect1d(...)` | 函数 | 两数组**交集** |
| `np.union1d(...)` | 函数 | 两数组**并集** |
| `np.setdiff1d(...)` | 函数 | **差集** `a − b` |
| `np.setxor1d(...)` | 函数 | **对称差集** |
| `np.isin(...)` | 函数 | 元素级成员检测（替代已弃用的 `in1d`） |
| `np.argmax(...)` / `np.argmin(...)` | 函数 | 最大 / 最小值索引 |
| `np.where(...)` | 函数 | 条件索引或替换 |
| `np.nonzero(...)` | 函数 | 非零元素索引 |
| `np.clip(...)` | 函数 | 将元素限制到 `[a_min, a_max]` 区间 |
| `np.floor/ceil/round/trunc(...)` | 函数 | 向下 / 向上 / 四舍五入 / 截断取整 |
| `arr.view(...)` | 方法 | 返回共享数据的视图 |
| `arr.copy(...)` | 方法 | 返回独立数据的副本 |

## 排序

### `np.sort`

#### 作用

返回排序后的**新数组**（不修改原数组）。支持多种算法和多维沿轴排序。

#### 重点方法

```python
np.sort(a, axis=-1, kind=None, order=None, stable=None)
```

#### 参数

| 参数名   | 本例取值                | 说明                                                                   |
| -------- | ----------------------- | ---------------------------------------------------------------------- |
| `a`      | `[3, 1, 4, 1, 5, 9, 2, 6]` | 输入数组                                                             |
| `axis`   | `-1`（默认）            | 沿哪个轴排序；`None` 先展平                                            |
| `kind`   | `None`（默认）          | 算法：`'quicksort'` / `'mergesort'` / `'heapsort'` / `'stable'`        |
| `order`  | `None`（默认）          | 对结构化数组指定排序字段顺序                                           |
| `stable` | `None`（默认）          | `True` 等价 `kind='stable'`；保持相等元素原序                          |

### `np.argsort`

#### 作用

返回将数组排序所需要的**索引**。`arr[np.argsort(arr)]` 等价于 `np.sort(arr)`。常用于**跟随排序**多个数组。

#### 重点方法

```python
np.argsort(a, axis=-1, kind=None, order=None, stable=None)
```

#### 参数

| 参数名   | 本例取值        | 说明                                             |
| -------- | --------------- | ------------------------------------------------ |
| `a`      | `arr`           | 输入数组                                         |
| `axis`   | `-1`（默认）    | 沿哪个轴排序                                     |
| `kind`   | `None`（默认）  | 排序算法                                         |
| `stable` | `None`（默认）  | 是否稳定排序                                     |

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"sort: {np.sort(arr)}")
print(f"argsort: {np.argsort(arr)}")
print(f"arr[argsort]: {arr[np.argsort(arr)]}")

arr_2d = np.array([[3, 1, 2], [6, 4, 5]])
print(f"按行排序 axis=1:\n{np.sort(arr_2d, axis=1)}")
print(f"按列排序 axis=0:\n{np.sort(arr_2d, axis=0)}")
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

- `argsort` 返回"排序后位置对应的原索引"；可用于跟随排序多个数组（如按 `scores` 排序对应的 `names`）。
- 降序排序用 `np.sort(arr)[::-1]` 或 `arr[np.argsort(arr)[::-1]]`。

## 去重与计数

### `np.unique`

#### 作用

返回数组中的**唯一值**（默认已排序）。可选返回首次出现位置、逆映射、计数等。

#### 重点方法

```python
np.unique(ar, return_index=False, return_inverse=False,
          return_counts=False, axis=None, equal_nan=True)
```

#### 参数

| 参数名           | 本例取值         | 说明                                                            |
| ---------------- | ---------------- | --------------------------------------------------------------- |
| `ar`             | `[1,2,2,3,3,3,4,4,4,4]` | 输入数组                                                |
| `return_index`   | `False`（默认）  | `True` 返回每个唯一值**首次出现**的索引                         |
| `return_inverse` | `False`（默认）  | `True` 返回把原数组映射回唯一值的**逆索引**                     |
| `return_counts`  | `False`（默认）  | `True` 返回每个唯一值的**出现次数**                             |
| `axis`           | `None`（默认）   | `None` 先展平；指定轴时在该轴上"去重行 / 列"                    |
| `equal_nan`      | `True`（默认）   | 是否将多个 `NaN` 视为同一个值                                   |

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

- `return_counts=True` 是一行代码做"值频数统计"的最快方式。
- 统计值-计数对：`dict(zip(*np.unique(arr, return_counts=True)))`。

## 集合运算

所有集合运算都**先展平再去重**，结果一维且按升序排列。

### `np.intersect1d`

#### 作用

返回两个数组的**交集**。

#### 重点方法

```python
np.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)
```

#### 参数

| 参数名           | 本例取值         | 说明                                              |
| ---------------- | ---------------- | ------------------------------------------------- |
| `ar1` / `ar2`    | `a`、`b`         | 两个输入数组                                      |
| `assume_unique`  | `False`（默认）  | `True` 时跳过内部去重，加速（须自行保证无重复）   |
| `return_indices` | `False`（默认）  | `True` 额外返回交集元素在两个数组中的索引         |

### `np.union1d`

#### 作用

返回两个数组的**并集**。

#### 重点方法

```python
np.union1d(ar1, ar2)
```

### `np.setdiff1d`

#### 作用

返回**差集** $ar1 \setminus ar2$。顺序敏感，$a - b$ 与 $b - a$ 结果不同。

#### 重点方法

```python
np.setdiff1d(ar1, ar2, assume_unique=False)
```

### `np.setxor1d`

#### 作用

返回两个数组的**对称差集**（只在其中一个出现的元素）。

#### 重点方法

```python
np.setxor1d(ar1, ar2, assume_unique=False)
```

### `np.isin`

#### 作用

对数组 `ar1` 的每个元素，判断是否出现在 `ar2` 中，返回与 `ar1` 同形状的布尔数组。**替代已弃用的 `np.in1d`**。

#### 重点方法

```python
np.isin(element, test_elements, assume_unique=False, invert=False, kind=None)
```

#### 参数

| 参数名           | 本例取值          | 说明                                                       |
| ---------------- | ----------------- | ---------------------------------------------------------- |
| `element`        | `a`               | 被检测的数组                                               |
| `test_elements`  | `[2, 4]`          | 参考集合                                                   |
| `assume_unique`  | `False`（默认）   | 假设两边已去重以加速                                       |
| `invert`         | `False`（默认）   | `True` 时返回"不在集合中"的布尔掩码                        |
| `kind`           | `None`（默认）    | 算法提示：`'sort'` 或 `'table'`（NumPy 1.23+）             |

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

## 搜索

### `np.argmax` / `np.argmin`

#### 作用

返回数组中最大 / 最小值的**索引**。多维时 `axis=None` 返回扁平索引。

#### 重点方法

```python
np.argmax(a, axis=None, out=None, keepdims=False)
np.argmin(a, axis=None, out=None, keepdims=False)
```

#### 参数

| 参数名     | 本例取值        | 说明                                 |
| ---------- | --------------- | ------------------------------------ |
| `a`        | `arr`           | 输入数组                             |
| `axis`     | `None`（默认）  | 指定轴；`None` 返回扁平后的索引       |
| `keepdims` | `False`（默认） | 是否保留聚合轴                       |

### `np.nonzero`

#### 作用

返回数组中**非零元素**的索引（一个元组，每个元素对应一个轴的索引数组）。等价于 `np.where(arr != 0)`。

#### 重点方法

```python
np.nonzero(a)
```

#### 参数

| 参数名 | 本例取值          | 说明         |
| ------ | ----------------- | ------------ |
| `a`    | `[0,1,0,2,0,3]`   | 输入数组     |

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([1, 5, 2, 8, 3, 9, 4, 7])
print(f"argmax: {np.argmax(arr)}, argmin: {np.argmin(arr)}")

idx = np.where(arr > 5)
print(f"大于 5 的索引: {idx[0]}")
print(f"大于 5 的值: {arr[idx]}")

arr2 = np.array([0, 1, 0, 2, 0, 3])
print(f"非零索引: {np.nonzero(arr2)[0]}")
```

#### 输出

```text
argmax: 5, argmin: 0
大于 5 的索引: [3 5 7]
大于 5 的值: [8 9 7]
非零索引: [1 3 5]
```

## 裁剪与取整

### `np.clip`

#### 作用

将数组元素**截断到 `[a_min, a_max]` 区间**。小于下界取下界，大于上界取上界。

#### 重点方法

```python
np.clip(a, a_min, a_max, out=None, **kwargs)
```

#### 参数

| 参数名   | 本例取值 | 说明                                                     |
| -------- | -------- | -------------------------------------------------------- |
| `a`      | `arr`    | 输入数组                                                 |
| `a_min`  | `5`      | 下界；`None` 表示不设下界                                |
| `a_max`  | `15`     | 上界；`None` 表示不设上界                                |
| `out`    | `None`（默认） | 目标数组                                           |

### `np.floor` / `np.ceil` / `np.round` / `np.trunc`

#### 作用

- `np.floor`: 向下取整（更负方向）
- `np.ceil`: 向上取整（更正方向）
- `np.round`: 四舍五入（**银行家舍入**，`.5` 向偶数舍入）
- `np.trunc`: 截断取整（向零方向）

#### 重点方法

```python
np.floor(x, /, out=None, *, where=True, dtype=None)
np.ceil(x, /, out=None, *, where=True, dtype=None)
np.round(a, decimals=0, out=None)
np.trunc(x, /, out=None, *, where=True, dtype=None)
```

#### 参数差异

| 函数        | 对 `-1.2` 结果 | 对 `2.5` 结果 |
| ----------- | -------------- | ------------- |
| `np.floor`  | `-2.0`         | `2.0`         |
| `np.ceil`   | `-1.0`         | `3.0`         |
| `np.round`  | `-1.0`         | `2.0`（偶数） |
| `np.trunc`  | `-1.0`         | `2.0`         |

### 综合示例

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

## 引用、视图、拷贝

### 三种语义对比

| 操作              | 是否共享数据 | 修改是否影响原数组 | 场景                         |
| ----------------- | ------------ | ------------------ | ---------------------------- |
| `arr_ref = arr`   | 是（同一对象）| 是                 | Python 赋值语义              |
| `arr.view()`      | 是           | 是                 | 改变 dtype / 查看不同视图    |
| `arr.copy()`      | 否           | 否                 | 真正独立的副本               |

### `arr.view`

#### 作用

返回共享底层数据的**新数组对象**。可重解读为不同 `dtype`（只要字节数匹配）。修改视图会影响原数组。

#### 重点方法

```python
arr.view(dtype=None, type=None)
```

#### 参数

| 参数名  | 本例取值       | 说明                                       |
| ------- | -------------- | ------------------------------------------ |
| `dtype` | `None`（默认） | 新的数据类型；`None` 则保持原类型          |
| `type`  | `None`（默认） | 新的数组子类                               |

### `arr.copy`

#### 作用

返回**完全独立**的数据副本，修改不影响原数组。

#### 重点方法

```python
arr.copy(order='C')
```

#### 参数

| 参数名  | 本例取值      | 说明                                                             |
| ------- | ------------- | ---------------------------------------------------------------- |
| `order` | `'C'`（默认） | 内存布局：`'C'` 行优先、`'F'` 列优先、`'A'` 任意、`'K'` 保持原状 |

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# 引用
arr_ref = arr
arr_ref[0] = 100
print(f"改引用后，原数组: {arr}")

arr[0] = 1  # 复位

# 视图
arr_view = arr.view()
arr_view[1] = 200
print(f"改视图后，原数组: {arr}")

arr[1] = 2  # 复位

# 副本
arr_copy = arr.copy()
arr_copy[2] = 300
print(f"改副本后，原数组不变: {arr}")
```

#### 输出

```text
改引用后，原数组: [100   2   3   4   5]
改视图后，原数组: [  1 200   3   4   5]
改副本后，原数组不变: [1 2 3 4 5]
```

#### 理解重点

- **切片** 通常返回视图、**花式索引** / **布尔索引** 返回副本——容易混淆。
- 不确定时用 `arr.base` 判断：`None` 表示独立数据，非 `None` 表示视图。
- 需要传出去且不希望被改的数组，显式 `.copy()` 最安全。

## 常见坑

1. `argsort` 返回的是**索引**，不是"排序后的值本身"，不要直接当值用。
2. `np.unique` 默认返回的是**排序后的**唯一值；如需保持原序需手写。
3. `np.in1d` 已弃用，新代码用 `np.isin`。
4. `np.round` 是**银行家舍入**（`.5` 向偶数）；需要严格四舍五入可 `np.floor(x + 0.5)`（正数）。
5. 视图与副本的差异：误把视图当副本会出现"神秘联动修改"，大型项目优先显式 `.copy()`。
6. `arr.view(np.int32)` 时，原 `float64` 的 8 字节会被重解读为两个 `int32`，形状也会变；不是类型转换。

## 小结

- 本章集合了日常数据处理中最常用的"工具抽屉"，熟练后能显著减少手写循环。
- 集合运算、`unique`、`clip` 是数据清洗 / EDA 的高频工具。
- 搞清"引用 vs 视图 vs 副本"能避开 NumPy 最常见的一类 bug。
