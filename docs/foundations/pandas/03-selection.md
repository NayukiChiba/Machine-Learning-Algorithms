---
title: Pandas 数据选择与过滤
outline: deep
---

# Pandas 数据选择与过滤

> 对应脚本：`Basic/Pandas/03_selection.py`  
> 运行方式：`python Basic/Pandas/03_selection.py`（仓库根目录）

## 本章目标

1. 掌握单列和多列选择的语法差异。
2. 理解 `loc`（标签索引）和 `iloc`（位置索引）的区别。
3. 掌握布尔条件过滤和多条件组合。
4. 学会使用 `query` 方法进行字符串表达式查询。

## 重点方法速览

| 方法/语法 | 作用 | 本章位置 |
|---|---|---|
| `df["col"]` | 选取单列，返回 Series | `demo_column_select` |
| `df[["c1", "c2"]]` | 选取多列，返回 DataFrame | `demo_column_select` |
| `df[start:stop]` | 行切片 | `demo_row_select` |
| `df.loc[row, col]` | 按标签选取 | `demo_loc_iloc` |
| `df.iloc[row, col]` | 按位置选取 | `demo_loc_iloc` |
| `df[condition]` | 布尔条件过滤 | `demo_filter` |
| `df.query(expr)` | 字符串表达式查询 | `demo_query` |

## 示例数据

本章所有 demo 使用同一个示例 DataFrame：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 35, 28, 32],
    "City": ["Beijing", "Shanghai", "Beijing", "Guangzhou", "Shanghai"],
    "Salary": [8000, 12000, 15000, 9000, 11000],
    "Score": np.random.randint(60, 100, 5),
})
```

## 1. 列选择

### 方法重点

- 单列选取返回 `Series`，双括号多列选取返回 `DataFrame`。
- 单列选取也可以用属性访问 `df.Name`，但当列名与方法重名或含空格时不可用。

### 参数速览（本节）

适用语法（分项）：

1. `df["col"]` — 单列选取
2. `df[["col1", "col2"]]` — 多列选取

| 语法 | 返回类型 | 说明 |
|---|---|---|
| `df["Name"]` | `Series` | 单列选取 |
| `df[["Name", "Age"]]` | `DataFrame` | 多列选取，注意双层方括号 |

### 示例代码

```python
# 单列选择
print(df["Name"])
print(type(df["Name"]))

# 多列选择
print(df[["Name", "Age"]])
```

### 结果输出

```text
0      Alice
1        Bob
2    Charlie
3      David
4        Eve
Name: Name, dtype: object
<class 'pandas.core.series.Series'>
----------------
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35
3    David   28
4      Eve   32
```

### 理解重点

- `df["Name"]` 和 `df[["Name"]]` 结果不同：前者是 `Series`，后者是单列 `DataFrame`。
- 多列选取传入列名列表，常用于特征子集提取。

## 2. 行选择

### 方法重点

- 行切片语法与 Python 列表一致，不包含结束位置。
- `head(n)` / `tail(n)` 是行选择的快捷方式。

### 参数速览（本节）

适用语法（分项）：

1. `df[start:stop]` — 行切片
2. `df.head(n=5)` — 前 n 行

| 语法 | 本例取值 | 说明 |
|---|---|---|
| `df[1:3]` | 第 1、2 行 | 位置切片，不包含 stop |
| `df.head(2)` | 前 2 行 | 等价于 `df[:2]` |

### 示例代码

```python
# 切片选择
print(df[1:3])

# head
print(df.head(2))
```

### 结果输出

```text
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
----------------
    Name  Age      City  Salary  Score
0  Alice   25   Beijing    8000     91
1    Bob   30  Shanghai   12000     87
```

## 3. `loc` 和 `iloc` 索引器

### 方法重点

- `loc` 基于**标签**索引，切片包含终点。
- `iloc` 基于**整数位置**索引，切片不包含终点（与 Python 一致）。
- 两者都支持行列同时选取：`df.loc[rows, cols]`。

### 参数速览（本节）

适用 API（分项）：

1. `df.loc[row_label, col_label]`
2. `df.iloc[row_pos, col_pos]`

| 参数 | 本例取值 | 说明 |
|---|---|---|
| `loc["b", "Name"]` | `"Bob"` | 标签索引取单个元素 |
| `loc["a":"c", ["Name", "Age"]]` | 3 行 2 列子表 | 标签切片**包含**终点 `"c"` |
| `iloc[1, 0]` | `"Bob"` | 位置索引取单个元素 |
| `iloc[0:3, 0:2]` | 3 行 2 列子表 | 位置切片**不包含**终点 |

### 示例代码

```python
df.index = ["a", "b", "c", "d", "e"]

# loc - 标签索引
print(df.loc["b", "Name"])
print(df.loc["a":"c", ["Name", "Age"]])

# iloc - 位置索引
print(df.iloc[1, 0])
print(df.iloc[0:3, 0:2])
```

### 结果输出

```text
Bob
----------------
      Name  Age
a    Alice   25
b      Bob   30
c  Charlie   35
----------------
Bob
----------------
      Name  Age
a    Alice   25
b      Bob   30
c  Charlie   35
```

### 理解重点

- `loc` 切片**两端都包含**（闭区间），`iloc` 切片**不包含右端**（半开区间）。
- 当索引是默认的 `0, 1, 2, ...` 时，`loc[0:2]` 返回 3 行，`iloc[0:2]` 返回 2 行。
- 推荐使用 `loc` / `iloc` 而非裸切片 `df[1:3]`，语义更明确。

## 4. 条件过滤

### 方法重点

- 布尔索引是 Pandas 最强大的过滤方式。
- 多条件组合必须用 `&`（与）/ `|`（或），不能用 `and` / `or`。
- 每个条件需要加括号，避免运算优先级问题。
- `isin()` 适合等值匹配多个候选值。

### 参数速览（本节）

适用语法/API（分项）：

1. `df[df["col"] > value]` — 单条件过滤
2. `df[(cond1) & (cond2)]` — 多条件与
3. `df[(cond1) | (cond2)]` — 多条件或
4. `df[df["col"].isin(list)]` — 等值多选

| 语法 | 说明 |
|---|---|
| `df[df["Age"] > 28]` | 年龄大于 28 |
| `(df["Age"] > 25) & (df["Salary"] > 10000)` | 年龄大于 25 且薪资大于 10000 |
| `(df["City"] == "Beijing") \| (df["City"] == "Shanghai")` | 城市为北京或上海 |
| `df["City"].isin(["Beijing", "Shanghai"])` | 等价的简写 |

### 示例代码

```python
# 单条件过滤
print(df[df["Age"] > 28])

# 多条件 AND
print(df[(df["Age"] > 25) & (df["Salary"] > 10000)])

# 多条件 OR
print(df[(df["City"] == "Beijing") | (df["City"] == "Shanghai")])

# isin
print(df[df["City"].isin(["Beijing", "Shanghai"])])
```

### 结果输出

```text
Age > 28:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
----------------
Age > 25 且 Salary > 10000:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
----------------
City 是 Beijing 或 Shanghai:
      Name  Age      City  Salary  Score
0    Alice   25   Beijing    8000     91
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
```

### 理解重点

- `isin()` 比多个 `==` 用 `|` 连接更简洁，尤其候选值多时。
- 布尔索引返回**副本**，修改结果不影响原 DataFrame。
- 过滤后索引保持原值不变，可用 `reset_index(drop=True)` 重置。

## 5. `query` 方法

### 方法重点

- `query` 用字符串表达式实现过滤，语法更接近 SQL。
- 可通过 `@variable` 引用 Python 外部变量。
- 对于复杂多条件过滤，`query` 比布尔索引更易读。

### 参数速览（本节）

适用 API：`df.query(expr, inplace=False, **kwargs)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `expr` | `"Age > 28 and Salary > 10000"` | 查询表达式字符串 |
| `inplace` | `False`（默认） | 是否原地修改 |
| `@variable` | `@min_age` | 引用外部 Python 变量 |

### 示例代码

```python
# query 表达式
print(df.query("Age > 28 and Salary > 10000"))

# 引用外部变量
min_age = 30
print(df.query("Age >= @min_age"))
```

### 结果输出

```text
Age > 28 and Salary > 10000:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
----------------
Age >= @min_age:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
```

### 理解重点

- `query` 内部使用 `and` / `or` / `not`（不是 `&` / `|` / `~`）。
- `@` 前缀引用外部变量，避免将变量值拼接到字符串中。
- 对于简单条件布尔索引更直接，复杂条件 `query` 可读性更好。

## 常见坑

1. `loc` 切片包含终点，`iloc` 不包含——两者行为不一致，容易混淆。
2. 多条件过滤忘记加括号：`df[df["A"] > 1 & df["B"] < 5]` 会因运算优先级报错。
3. 布尔条件中使用 `and` / `or` 而非 `&` / `|`，会触发 `ValueError`。
4. `query` 中列名含空格或特殊字符时需要用反引号包裹：`` query("`col name` > 5") ``。

## 小结

- 列选择是特征工程的基础，行过滤是数据清洗的核心。
- `loc`（标签）和 `iloc`（位置）是最精确的索引方式，优先使用。
- 布尔索引和 `query` 是条件过滤的两大工具，根据复杂度选择。
