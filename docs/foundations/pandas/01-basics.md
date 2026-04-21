---
title: Pandas 基础与核心数据结构
outline: deep
---

# Pandas 基础与核心数据结构

## 本章目标

1. 理解 Pandas 的两大核心数据结构：`Series` 和 `DataFrame`。
2. 掌握从列表、字典创建 `Series` 和 `DataFrame` 的方法。
3. 学会使用 `head`、`tail`、`info`、`describe` 等基本查看方法。
4. 掌握 `shape`、`ndim`、`size`、`dtypes` 等常用属性。

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
| `df.ndim` | 属性 | 维度数量（`DataFrame` 恒为 2） |
| `df.size` | 属性 | 元素总数 = 行 × 列 |
| `df.columns` | 属性 | 列名 `Index` 对象 |
| `df.index` | 属性 | 行索引 `Index` 对象 |
| `df.values` | 属性 | 底层 NumPy 数组 |
| `df.dtypes` | 属性 | 各列数据类型 `Series` |

## Series 数据结构

### `pd.Series`

#### 作用

创建一维**带索引标签**的数组。可看作一列 Excel 数据，索引赋予了数据标签语义。可从列表、字典、标量、NumPy 数组等多种来源创建。

#### 重点方法

```python
pd.Series(data=None, index=None, dtype=None, name=None, copy=None, fastpath=False)
```

#### 参数

| 参数名   | 本例取值                            | 说明                                                                     |
| -------- | ----------------------------------- | ------------------------------------------------------------------------ |
| `data`   | `[1, 2, 3, 4, 5]`、`dict`           | 输入数据，可为列表、元组、字典、标量、`ndarray` 等                       |
| `index`  | `None`（默认）、`["a", "b", "c"]`   | 自定义索引标签；`None` 时默认为 `RangeIndex(0, ..., n)`；字典键自动用作索引 |
| `dtype`  | `None`（默认）                      | 元素数据类型；`None` 时自动推断                                          |
| `name`   | `None`（默认）、`"prices"`          | Series 的名称，会显示在输出底部                                          |
| `copy`   | `None`（默认）                      | 是否复制输入数据；`None` 时按源类型决定                                  |

#### 示例代码

```python
import pandas as pd

# 从列表创建
s1 = pd.Series([1, 2, 3, 4, 5])
print(f"s1:\n{s1}")
print(f"索引: {s1.index.tolist()}")
print(f"值: {s1.values}")

# 指定索引
s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(f"\ns2:\n{s2}")
print(f"访问 s2['b']: {s2['b']}")

# 从字典创建（键自动成为索引）
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

- `Series` = 值 + 索引。索引可以是任意可哈希对象（整数、字符串、日期）。
- 字典创建最直观：键 → 索引，值 → 数据。
- `s.values` 返回底层 NumPy 数组；`s.index` 返回 `Index` 对象。

## DataFrame 数据结构

### `pd.DataFrame`

#### 作用

创建二维带标签的表格，可类比 Excel 工作表或 SQL 表。每列可以是不同 dtype。

#### 重点方法

```python
pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
```

#### 参数

| 参数名    | 本例取值                                        | 说明                                                                   |
| --------- | ----------------------------------------------- | ---------------------------------------------------------------------- |
| `data`    | `{"Name": [...], "Age": [...], "City": [...]}` | 输入数据，可为字典、二维数组、Series 列表、另一个 DataFrame 等         |
| `index`   | `None`（默认）                                  | 行索引；`None` 时默认为 `RangeIndex`                                   |
| `columns` | `None`（默认）                                  | 列名；字典创建时会自动用字典键；二维数组创建时必须显式指定才有列名     |
| `dtype`   | `None`（默认）                                  | 为所有列统一设置 dtype；`None` 时按列自动推断                          |
| `copy`    | `None`（默认）                                  | 是否复制输入数据                                                       |

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

- 从字典创建：**键 → 列名**，值 → 列数据。
- 字符串列默认 dtype 是 `object`（底层是 Python 对象指针）。
- `shape` 返回 `(行数, 列数)`，是做数据探索第一步的检查点。

## 数据查看

### `DataFrame.head`

#### 作用

查看 DataFrame 的前 `n` 行。拿到新数据后的**第一步**操作。

#### 重点方法

```python
df.head(n=5)
```

#### 参数

| 参数名 | 本例取值     | 说明                                           |
| ------ | ------------ | ---------------------------------------------- |
| `n`    | `3`、`5`（默认） | 返回的行数；为负数时返回"除末尾 \|n\| 行外"   |

### `DataFrame.tail`

#### 作用

查看 DataFrame 的后 `n` 行。

#### 重点方法

```python
df.tail(n=5)
```

#### 参数

| 参数名 | 本例取值     | 说明                                           |
| ------ | ------------ | ---------------------------------------------- |
| `n`    | `3`、`5`（默认） | 返回的行数；为负数时返回"除开头 \|n\| 行外"   |

### `DataFrame.info`

#### 作用

打印 DataFrame 的摘要信息，包括列类型、非空计数、内存占用。快速发现缺失值的首选工具。

#### 重点方法

```python
df.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None)
```

#### 参数

| 参数名         | 本例取值        | 说明                                                                   |
| -------------- | --------------- | ---------------------------------------------------------------------- |
| `verbose`      | `None`（默认）  | 是否显示完整列信息；列数很多时可设 `False` 简化输出                    |
| `buf`          | `None`（默认）  | 输出缓冲区；`None` 写到 `sys.stdout`                                   |
| `max_cols`     | `None`（默认）  | 显示列数上限                                                           |
| `memory_usage` | `None`（默认）  | 是否显示内存占用；`'deep'` 精确计算 object 列占用                      |
| `show_counts`  | `None`（默认）  | 是否显示非空计数（仅当行数 < 1,690,785 时默认开启）                    |

### `DataFrame.describe`

#### 作用

生成数值列的统计摘要：计数、均值、标准差、最小值、四分位数、最大值。是 EDA（探索性数据分析）的标准第一步。

#### 重点方法

```python
df.describe(percentiles=None, include=None, exclude=None)
```

#### 参数

| 参数名        | 本例取值                    | 说明                                                                   |
| ------------- | --------------------------- | ---------------------------------------------------------------------- |
| `percentiles` | `None`（默认 `[.25, .5, .75]`）| 要计算的分位数列表，值应在 `[0, 1]`                                  |
| `include`     | `None`（默认数值）、`'all'`、`[np.number, 'object']` | 要统计的 dtype；`'all'` 同时包含数值和字符串 |
| `exclude`     | `None`（默认）              | 要排除的 dtype                                                         |

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
})

print(f"head(3):\n{df.head(3)}")
print(f"\ntail(3):\n{df.tail(3)}")
print("\ninfo():")
df.info()
print(f"\ndescribe():\n{df.describe()}")
```

#### 输出

```text
head(3):
          A   B     C
0  0.496714  63   cat
1 -0.138264  59   dog
2  0.647689  20  bird

tail(3):
          A   B     C
7  0.767435  88   dog
8 -0.469474  48  bird
9  0.542560  90   cat

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

#### 理解重点

- **数据探索三板斧**：`head` → `info` → `describe`。
- `info` 最关键的是"Non-Null Count"——立即发现哪些列有缺失值。
- `describe` 默认只看数值列；要看字符串列用 `df.describe(include='all')`。

## DataFrame 常用属性

### 属性速览

| 属性       | 返回类型                 | 含义                               |
| ---------- | ------------------------ | ---------------------------------- |
| `shape`    | `tuple[int, int]`        | `(行数, 列数)`                     |
| `ndim`     | `int`                    | 维度数；DataFrame 恒为 2           |
| `size`     | `int`                    | 元素总数 = 行数 × 列数             |
| `columns`  | `Index`                  | 列名 `Index` 对象                  |
| `index`    | `RangeIndex` / `Index`   | 行索引对象                         |
| `values`   | `numpy.ndarray`          | 底层 NumPy 二维数组                |
| `dtypes`   | `Series`                 | 各列数据类型                       |
| `T`        | `DataFrame`              | 转置（行列交换）                   |
| `empty`    | `bool`                   | 是否为空 DataFrame                 |
| `axes`     | `list[Index, Index]`     | `[index, columns]`                 |

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

### 输出

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

- `shape` 与 `dtypes` 是所有数据探索任务的起点。
- `values` 将混合类型统一为 `object` 数组，可能丢失数值效率；**纯数值列**才建议用 `values`。
- 现代代码中，`df.to_numpy()` 比 `df.values` 更推荐（明确语义、更可控）。

## 常见坑

1. **单列选取的返回类型**：`df["col"]` 返回 `Series`，`df[["col"]]` 返回**单列 `DataFrame`**，形状和方法集都不同。
2. `describe()` 默认忽略非数值列；要看字符串列分布用 `include='all'` 或 `include=[object]`。
3. `values` 属性对混合类型 DataFrame 返回 `object` 数组，参与 NumPy 运算会变慢甚至失败。
4. `df.info()` 不返回字符串，而是直接 print；不要写 `s = df.info()`。
5. 字典创建 DataFrame 时，所有值的长度必须一致，否则抛 `ValueError`。
6. `RangeIndex` 看起来像整数列表，但它是一个特殊对象；要强制转 list 用 `df.index.tolist()`。

## 小结

- Pandas 的核心是 **`Series`（一维带标签）** 和 **`DataFrame`（二维表格）**。
- 拿到数据后的标准流程：`head` → `info` → `describe` → `shape` / `dtypes`。
- `Series` 和 `DataFrame` 都建立在 NumPy 之上，但**多了标签**和**更丰富的方法**。
- 时刻关注 `shape` 和 `dtypes`——它们决定了后续操作的正确性。
