---
title: Pandas 基础与核心数据结构
outline: deep
---

# Pandas 基础与核心数据结构

> 对应脚本：`Basic/Pandas/01_basics.py`  
> 运行方式：`python Basic/Pandas/01_basics.py`（仓库根目录）

## 本章目标

1. 理解 Pandas 的两大核心数据结构：`Series` 和 `DataFrame`。
2. 掌握从列表、字典创建 `Series` 和 `DataFrame` 的方法。
3. 学会使用 `head`、`tail`、`info`、`describe` 等基本查看方法。
4. 掌握 `shape`、`ndim`、`size`、`dtypes` 等常用属性。

## 重点方法与概念速览

| 名称 | 类型 | 作用 | 本章位置 |
|---|---|---|---|
| `pd.Series(...)` | 构造器 | 创建一维带标签数组 | `demo_series` |
| `pd.DataFrame(...)` | 构造器 | 创建二维表格数据 | `demo_dataframe` |
| `df.head(n)` / `df.tail(n)` | 方法 | 查看前/后 n 行 | `demo_basic_view` |
| `df.info()` | 方法 | 显示列类型与非空计数 | `demo_basic_view` |
| `df.describe()` | 方法 | 数值列统计摘要 | `demo_basic_view` |
| `df.shape` / `df.ndim` / `df.size` | 属性 | 形状、维度、元素总数 | `demo_attributes` |
| `df.dtypes` | 属性 | 各列数据类型 | `demo_attributes` |

## 1. Series 数据结构

### 方法重点

- `Series` 是带索引的一维数组，可以看作一列 Excel 数据。
- 可以从列表、字典创建，字典的键自动成为索引。
- 支持自定义索引，通过标签访问元素。

### 参数速览（本节）

适用 API：`pd.Series(data=None, index=None, dtype=None, name=None, copy=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data` | `[1, 2, 3, 4, 5]`、`[10, 20, 30]`、`dict` | 输入数据，支持列表、字典、标量等 |
| `index` | `None`、`["a", "b", "c"]` | 自定义索引标签，默认为 `0, 1, 2, ...` |
| `dtype` | `None`（默认） | 数据类型，默认自动推断 |
| `name` | `None`（默认） | Series 的名称 |

### 示例代码

```python
import pandas as pd

# 从列表创建
s1 = pd.Series([1, 2, 3, 4, 5])
print(s1)
print(f"索引: {s1.index.tolist()}")
print(f"值: {s1.values}")

# 指定索引
s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s2)
print(f"访问 s2['b']: {s2['b']}")

# 从字典创建
data = {"apple": 100, "banana": 200, "orange": 150}
s3 = pd.Series(data)
print(s3)
```

### 结果输出

```text
0    1
1    2
2    3
3    4
4    5
dtype: int64
索引: [0, 1, 2, 3, 4]
值: [1 2 3 4 5]
----------------
a    10
b    20
c    30
dtype: int64
访问 s2['b']: 20
----------------
apple     100
banana    200
orange    150
dtype: int64
```

### 理解重点

- `Series` 的核心是"值 + 索引"，索引赋予了数据标签语义。
- 从字典创建时，字典键自动成为索引，比列表创建更直观。
- `s.values` 返回底层 NumPy 数组，`s.index` 返回索引对象。

## 2. DataFrame 数据结构

### 方法重点

- `DataFrame` 是由多列 `Series` 组成的二维表格。
- 最常用的创建方式是从字典创建，字典键为列名，值为列数据。
- `shape`、`columns`、`index`、`dtypes` 是最基础的属性。

### 参数速览（本节）

适用 API：`pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data` | `{"Name": [...], "Age": [...], "City": [...]}` | 输入数据，支持字典、数组、列表等 |
| `index` | `None`（默认） | 行索引，默认为 `0, 1, 2, ...` |
| `columns` | `None`（默认） | 列名，字典创建时自动推断 |
| `dtype` | `None`（默认） | 统一的数据类型，通常不设置 |

### 示例代码

```python
import pandas as pd

# 从字典创建
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["Beijing", "Shanghai", "Guangzhou"],
}
df = pd.DataFrame(data)
print(df)

# 查看基本信息
print(f"形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"索引: {df.index.tolist()}")
print(f"数据类型:\n{df.dtypes}")
```

### 结果输出

```text
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
2  Charlie   35  Guangzhou
----------------
形状: (3, 3)
列名: ['Name', 'Age', 'City']
索引: [0, 1, 2]
数据类型:
Name    object
Age      int64
City    object
dtype: object
```

### 理解重点

- `DataFrame` 可以理解为"Excel 工作表"或"SQL 数据表"。
- `shape` 返回 `(行数, 列数)` 元组，是检查数据规模的第一步。
- 字符串列的 `dtype` 是 `object`，数值列自动推断为 `int64` 或 `float64`。

## 3. 基本数据查看方法

### 方法重点

- `head(n)` / `tail(n)` 是快速预览数据的标准操作。
- `info()` 一次性展示列数、非空数、类型和内存占用。
- `describe()` 自动统计数值列的均值、标准差、分位数等。

### 参数速览（本节）

1. `df.head(n=5)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n` | `3` | 返回前 n 行，默认 5 |

2. `df.tail(n=5)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n` | `3` | 返回后 n 行，默认 5 |

3. `df.info(verbose=None, memory_usage=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `verbose` | `None`（默认） | 是否显示完整列信息 |
| `memory_usage` | `None`（默认） | 是否显示内存使用量 |

4. `df.describe(percentiles=None, include=None, exclude=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `percentiles` | `None`（默认） | 默认输出 25%、50%、75% 分位数 |
| `include` | `None`（默认） | 默认只统计数值列 |

### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "A": np.random.randn(10),
    "B": np.random.randint(0, 100, 10),
    "C": ["cat", "dog", "bird", "cat", "dog",
          "bird", "cat", "dog", "bird", "cat"],
})

print(df.head(3))
print(df.tail(3))
df.info()
print(df.describe())
```

### 结果输出

```text
head(3):
          A   B     C
0  0.496714  63   cat
1 -0.138264  59   dog
2  0.647689  20  bird
----------------
tail(3):
          A   B     C
7  0.767435  88   dog
8 -0.469474  48  bird
9  0.542560  90   cat
----------------
info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   A       10 non-null     float64
 1   B       10 non-null     int32
 2   C       10 non-null     object
dtypes: float64(1), int32(1), object(1)
memory usage: 332.0+ bytes
----------------
describe():
               A          B
count  10.000000  10.000000
mean    0.448061  55.300000
std     0.723008  25.289655
min    -0.469474  20.000000
25%    -0.210169  36.000000
50%     0.519637  58.000000
75%     0.737498  72.000000
max     1.579213  90.000000
```

### 理解重点

- `head`/`tail` 是拿到数据后的第一步操作，快速了解数据长什么样。
- `info()` 最关键的信息是非空计数——能立即发现缺失值。
- `describe()` 默认只统计数值列，传 `include='all'` 可同时统计字符串列。

## 4. DataFrame 常用属性

### 方法重点

- `shape`、`ndim`、`size` 描述数据的几何特征。
- `columns` 和 `index` 分别返回列名和行索引对象。
- `values` 返回底层 NumPy 二维数组。
- `dtypes` 是调试类型相关问题的必备属性。

### 参数速览（本节）

适用属性（分项）：

1. `df.shape` — 返回 `(行数, 列数)` 元组
2. `df.ndim` — DataFrame 始终为 `2`
3. `df.size` — 总元素数 = 行数 × 列数
4. `df.columns` — 列名 Index 对象
5. `df.index` — 行索引 Index 对象
6. `df.values` — 底层 NumPy 数组
7. `df.dtypes` — 各列数据类型 Series

| 属性名 | 返回类型 | 说明 |
|---|---|---|
| `shape` | `tuple` | `(行数, 列数)` |
| `ndim` | `int` | DataFrame 为 2，Series 为 1 |
| `size` | `int` | 元素总数 |
| `columns` | `Index` | 列名集合 |
| `index` | `RangeIndex` | 行索引 |
| `values` | `np.ndarray` | 底层二维数组 |
| `dtypes` | `Series` | 各列的 dtype |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4.0, 5.0, 6.0],
    "C": ["x", "y", "z"],
})

print(f"shape: {df.shape}")
print(f"ndim: {df.ndim}")
print(f"size: {df.size}")
print(f"columns: {df.columns.tolist()}")
print(f"index: {df.index.tolist()}")
print(f"values:\n{df.values}")
print(f"dtypes:\n{df.dtypes}")
```

### 结果输出

```text
shape: (3, 3)
ndim: 2
size: 9
columns: ['A', 'B', 'C']
index: [0, 1, 2]
values:
[[1 4.0 'x']
 [2 5.0 'y']
 [3 6.0 'z']]
dtypes:
A      int64
B    float64
C     object
dtype: object
```

### 理解重点

- `shape` 是数据探索的第一步，用于确认行列规模。
- `values` 会将混合类型统一为 `object`，纯数值列可放心使用。
- `dtypes` 中 `object` 通常表示字符串或混合类型。

## 常见坑

1. 混淆 `Series` 和 `DataFrame`：单列选取 `df["col"]` 返回 `Series`，双括号 `df[["col"]]` 返回 `DataFrame`。
2. `describe()` 默认忽略非数值列，需要 `include='all'` 才能看到字符串列统计。
3. `values` 属性在混合类型 DataFrame 中返回 `object` 数组，可能影响后续 NumPy 运算。

## 小结

- Pandas 的核心是 `Series`（一维带标签）和 `DataFrame`（二维表格），理解它们是后续所有操作的基础。
- `head` / `info` / `describe` 三步走是数据探索的标准流程。
- 关注 `shape` 和 `dtypes`，它们决定了后续操作是否正确。
