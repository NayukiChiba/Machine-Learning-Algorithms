---
title: Pandas 基础与核心数据结构
outline: deep
---

# Pandas 基础与核心数据结构

## 本章目标

1. 理解 Pandas 的两大核心数据结构：`Series` 和 `DataFrame`
2. 掌握从列表、字典、NumPy 数组创建 `Series` 和 `DataFrame`
3. 学会用 `head`、`tail`、`info`、`describe` 快速探查数据
4. 掌握 `shape`、`dtypes`、`columns`、`index` 等常用属性

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.Series(...)` | 构造器 | 创建一维带标签数组 |
| `pd.DataFrame(...)` | 构造器 | 创建二维表格数据 |
| `df.head(...)` | 方法 | 查看前 n 行 |
| `df.tail(...)` | 方法 | 查看后 n 行 |
| `df.info(...)` | 方法 | 列类型、非空计数、内存占用 |
| `df.describe(...)` | 方法 | 数值列统计摘要 |
| `df.shape` | 属性 | `(行数, 列数)` 元组 |
| `df.dtypes` | 属性 | 各列数据类型 |
| `df.columns` | 属性 | 列名 `Index` 对象 |
| `df.index` | 属性 | 行索引对象 |
| `df.values` | 属性 | 底层 NumPy 数组（推荐用 `df.to_numpy()`） |

## 1. Series 数据结构

### `pd.Series`

#### 作用

创建一维**带索引标签**的数组，可看作一列 Excel 数据。索引赋予数据标签语义——可通过标签而非位置访问元素。支持从列表、字典、标量、NumPy 数组创建。

#### 重点方法

```python
pd.Series(data=None, index=None, dtype=None, name=None, copy=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `array_like`、`dict`、标量 | 输入数据，可为列表、元组、字典、标量、`ndarray` | `[1, 2, 3]`、`{"a": 1}`、`5` |
| `index` | `array_like` 或 `None` | 自定义索引标签；`None` 时默认为 `RangeIndex` | `["a", "b", "c"]` |
| `dtype` | `dtype` 或 `None` | 元素数据类型，默认为 `None`（自动推断） | `np.int64`、`"float32"` |
| `name` | `str` 或 `None` | Series 的名称，显示在输出底部，默认为 `None` | `"prices"` |
| `copy` | `bool` 或 `None` | 是否复制输入数据，默认为 `None`（按源类型决定） | `True` |

#### 示例代码

```python
import pandas as pd
import numpy as np

# 从列表创建
s1 = pd.Series([1, 2, 3, 4, 5])
print(f"s1:\n{s1}")
print(f"索引: {s1.index.tolist()}")
print(f"值: {s1.values}")

# 指定索引
s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(f"\ns2:\n{s2}")
print(f"访问 s2['b']: {s2['b']}")

# 从字典创建（键 → 索引，值 → 数据）
s3 = pd.Series({"apple": 100, "banana": 200, "orange": 150})
print(f"\ns3:\n{s3}")
```

#### 输出

```text
s1:
0    1
1    2
2    3
3    4
4    5
dtype: int64
索引: [0, 1, 2, 3, 4]
值: [1 2 3 4 5]

s2:
a    10
b    20
c    30
dtype: int64
访问 s2['b']: 20

s3:
apple     100
banana    200
orange    150
dtype: int64
```

#### 理解重点

- `Series` = 值 + 索引——索引是数据的一部分，不只是行号
- 字典创建最直观：键自动成为索引，值成为数据
- `s.values` 返回底层 NumPy 数组；`s.index` 返回 `Index` 对象——两者是 Series 与 NumPy 互操作的桥梁

## 2. DataFrame 数据结构

### `pd.DataFrame`

#### 作用

创建二维带标签的表格，可类比 Excel 工作表或 SQL 表。每列可以是不同 `dtype`——这是与 NumPy 二维数组最本质的区别。

#### 重点方法

```python
pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `dict`、`ndarray`、`list[Series]` 等 | 输入数据；字典的键自动成为列名 | `{"Name": [...], "Age": [...]}` |
| `index` | `array_like` 或 `None` | 行索引，默认为 `None`（`RangeIndex`） | `["r1", "r2", "r3"]` |
| `columns` | `array_like` 或 `None` | 列名；字典创建时自动取键名，默认为 `None` | `["col1", "col2"]` |
| `dtype` | `dtype` 或 `None` | 统一设置所有列的数据类型，默认为 `None`（按列推断） | `np.float64` |
| `copy` | `bool` 或 `None` | 是否复制输入数据，默认为 `None` | `True` |

#### 示例代码

```python
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["Beijing", "Shanghai", "Guangzhou"],
}
df = pd.DataFrame(data)

print(df)
print(f"\n形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"索引: {df.index.tolist()}")
print(f"数据类型:\n{df.dtypes}")
```

#### 输出

```text
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
2  Charlie   35  Guangzhou

形状: (3, 3)
列名: ['Name', 'Age', 'City']
索引: [0, 1, 2]
数据类型:
Name    object
Age      int64
City    object
dtype: object
```

#### 理解重点

- 从字典创建：**键 → 列名**，值列表 → 列数据，所有值长度必须一致
- 字符串列默认 `dtype=object`（底层存 Python 对象指针），内存效率不如 `category` 类型
- `shape` 返回 `(行数, 列数)`——拿到数据后先查 shape 是最基本的健全检查

## 3. 数据查看

### `head` / `tail`

#### 作用

- `head(n)`：查看前 `n` 行，默认 5。拿到新数据后的第一步操作
- `tail(n)`：查看后 `n` 行，默认 5。检查数据尾部是否完整

#### 重点方法

```python
df.head(n=5)
df.tail(n=5)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n` | `int` | 返回的行数，默认为 `5`；负数返回除末尾 `|n|` 行外的全部 | `3`、`10` |

### `DataFrame.info`

#### 作用

打印 DataFrame 的摘要信息：列名、非空计数、dtype、内存占用。快速发现缺失值和类型问题的首选工具。**直接 print 到 stdout，不返回值**。

#### 重点方法

```python
df.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `verbose` | `bool` 或 `None` | 是否显示完整列信息，默认为 `None`（列少时自动开启） | `False` |
| `buf` | `file-like` 或 `None` | 输出缓冲区，默认为 `None`（`sys.stdout`） | `open("log.txt", "w")` |
| `max_cols` | `int` 或 `None` | 显示列数上限，默认为 `None`（自动适配终端宽度） | `10` |
| `memory_usage` | `bool`、`str` 或 `None` | 是否显示内存占用；`'deep'` 精确计算 object 列，默认为 `None` | `"deep"` |
| `show_counts` | `bool` 或 `None` | 是否显示非空计数，默认为 `None`（行数 < 阈值时自动开启） | `True` |

### `DataFrame.describe`

#### 作用

生成数值列的统计摘要：计数、均值、标准差、最小值、四分位数、最大值。EDA（探索性数据分析）的标准第一步。默认只统计数值列。

#### 重点方法

```python
df.describe(percentiles=None, include=None, exclude=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `percentiles` | `list[float]` 或 `None` | 要计算的分位数，默认为 `None`（即 `[.25, .5, .75]`） | `[.1, .5, .9]` |
| `include` | `str`、`list[str]` 或 `None` | 要统计的 dtype；`'all'` 包含数值和字符串列，默认为 `None`（只数值） | `"all"`、`[np.number]` |
| `exclude` | `str`、`list[str]` 或 `None` | 要排除的 dtype，默认为 `None` | `[np.object]` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "A": np.random.randn(10),
    "B": np.random.randint(0, 100, 10),
    "C": ["cat", "dog", "bird", "cat", "dog",
          "bird", "cat", "dog", "bird", "cat"],
    "D": [1.2, np.nan, 3.4, 4.5, np.nan, 6.7, 7.8, 8.9, np.nan, 10.0],
})

print(f"head(3):\n{df.head(3)}")
print(f"\ntail(3):\n{df.tail(3)}")
print("\ninfo():")
df.info()
print(f"\ndescribe():\n{df.describe()}")
print(f"\ndescribe(include='all'):\n{df.describe(include='all')}")
```

#### 输出

```text
head(3):
          A   B     C    D
0  0.496714  63   cat  1.2
1 -0.138264  59   dog  NaN
2  0.647689  20  bird  3.4

tail(3):
          A   B     C    D
7  0.767435  88   dog  8.9
8 -0.469474  48  bird  NaN
9  0.542560  90   cat  10.0

info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   A       10 non-null     float64
 1   B       10 non-null     int32
 2   C       10 non-null     object
 3   D       7 non-null      float64
dtypes: float64(2), int32(1), object(1)
memory usage: 412.0+ bytes

describe():
               A          B         D
count  10.000000  10.000000   7.00000
mean    0.448061  55.300000   6.07143
std     0.723008  25.289655   3.18927
min    -0.469474  20.000000   1.20000
25%    -0.210169  36.000000   3.40000
50%     0.519637  58.000000   6.70000
75%     0.737498  72.000000   8.05000
max     1.579213  90.000000  10.00000

describe(include='all'):
          A         B     C         D
count   10.000  10.0000    10    7.000
unique    NaN      NaN      3      NaN
top       NaN      NaN    cat      NaN
freq      NaN      NaN      4      NaN
mean      0.448   55.300   NaN    6.071
std       0.723   25.290   NaN    3.189
min      -0.469   20.000   NaN    1.200
25%      -0.210   36.000   NaN    3.400
50%       0.519   58.000   NaN    6.700
75%       0.738   72.000   NaN    8.050
max       1.579   90.000   NaN   10.000
```

#### 理解重点

- **数据探索三板斧**：`head` → `info` → `describe`——拿到数据的标准流程
- `info` 最关键的信息是 **Non-Null Count**——立即暴露哪些列有缺失值
- `describe` 默认只看数值列；`include='all'` 同时显示字符串列的 unique/top/freq
- `info()` 不返回字符串——不要写 `s = df.info()`，需要捕获时用 `buf` 参数

## 4. 常用属性

### 属性速览

| 属性 | 返回类型 | 含义 |
|---|---|---|
| `df.shape` | `tuple[int, int]` | `(行数, 列数)` |
| `df.ndim` | `int` | 维度数；DataFrame 恒为 `2` |
| `df.size` | `int` | 元素总数 = 行数 × 列数 |
| `df.columns` | `Index` | 列名 `Index` 对象 |
| `df.index` | `Index` | 行索引对象 |
| `df.dtypes` | `Series` | 各列数据类型 |
| `df.values` | `numpy.ndarray` | 底层 NumPy 二维数组（推荐用 `df.to_numpy()` 替代） |
| `df.T` | `DataFrame` | 转置（行列交换） |
| `df.empty` | `bool` | 是否为空 DataFrame |
| `df.axes` | `list[Index]` | `[行索引, 列索引]` |

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
print(f"dtypes:\n{df.dtypes}")
print(f"values:\n{df.values}")
print(f"empty: {df.empty}")
```

### 输出

```text
shape: (3, 3)
ndim: 2
size: 9
columns: ['A', 'B', 'C']
index: [0, 1, 2]
dtypes:
A      int64
B    float64
C     object
dtype: object
values:
[[1 4.0 'x']
 [2 5.0 'y']
 [3 6.0 'z']]
empty: False
```

### 理解重点

- `shape` 和 `dtypes` 是所有数据探索任务的起点
- `values` 对混合类型 DataFrame 返回 `object` 数组，数值计算会变慢——纯数值列才用它
- 现代代码推荐用 `df.to_numpy()` 替代 `df.values`（语义更明确，参数更可控）
- `empty` 在读取文件后做健全检查很有用：`if df.empty: raise ValueError(...)`

## 常见坑

1. 单列选取 `df["col"]` 返回 `Series`，`df[["col"]]` 返回单列 `DataFrame`——形状和方法集都不同，容易在后续操作中踩坑
2. `describe()` 默认忽略非数值列；看字符串列分布用 `include='all'` 或 `include=[object]`
3. `values` 对混合类型 DataFrame 返回 `object` 数组，参与 NumPy 运算会变慢甚至失败——纯数值列才用它
4. `df.info()` 不返回字符串而是直接 print——不要写 `s = df.info()` 试图赋值
5. 字典创建 DataFrame 时所有值列表长度必须一致，否则抛 `ValueError`
6. `RangeIndex` 不是普通列表——需要转 list 时用 `df.index.tolist()`

## 小结

- Pandas 的两大核心是 **`Series`**（一维带标签）和 **`DataFrame`**（二维表格）
- 拿到数据的标准流程：`head` → `info` → `describe` → `shape` / `dtypes`
- `Series` 和 `DataFrame` 都建立在 NumPy 之上，但多了**标签索引**和**丰富的数据处理方法**
- 时刻关注 `shape` 和 `dtypes`——它们决定了后续操作的正确性
