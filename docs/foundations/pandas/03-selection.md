---
title: Pandas 数据选择与过滤
outline: deep
---

# Pandas 数据选择与过滤

## 本章目标

1. 掌握单列和多列选择的语法差异。
2. 理解 `loc`（标签索引）和 `iloc`（位置索引）的区别。
3. 掌握布尔条件过滤和多条件组合。
4. 学会使用 `query` 方法进行字符串表达式查询。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df["col"]` | 语法 | 选取单列，返回 `Series` |
| `df[["c1", "c2"]]` | 语法 | 选取多列，返回 `DataFrame` |
| `df[start:stop]` | 语法 | 行切片（按位置） |
| `df.loc[...]` | 索引器 | 按**标签**选取行列 |
| `df.iloc[...]` | 索引器 | 按**位置**选取行列 |
| `df.at[...]` | 索引器 | 按标签取**单个元素**（比 `loc` 快） |
| `df.iat[...]` | 索引器 | 按位置取**单个元素**（比 `iloc` 快） |
| `df[bool_mask]` | 语法 | 布尔索引过滤 |
| `Series.isin(...)` | 方法 | 元素级成员检测 |
| `df.query(...)` | 方法 | 字符串表达式查询 |

## 示例数据

本章所有示例使用同一个 DataFrame：

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

## 列选择

### 单列与多列

| 语法                      | 返回类型     | 说明                                  |
| ------------------------- | ------------ | ------------------------------------- |
| `df["Name"]`              | `Series`     | 单列                                  |
| `df[["Name"]]`            | `DataFrame`  | 单列 DataFrame（注意双层方括号）      |
| `df[["Name", "Age"]]`     | `DataFrame`  | 多列子集                              |
| `df.Name`                 | `Series`     | 属性访问（列名含空格/关键字时不可用） |

### 示例代码

```python
# 单列（Series）
print(f"type(df['Name']) = {type(df['Name']).__name__}")
print(df["Name"])

# 多列（DataFrame）
print(f"\ndf[['Name', 'Age']]:")
print(df[["Name", "Age"]])
```

### 输出

```text
type(df['Name']) = Series
0      Alice
1        Bob
2    Charlie
3      David
4        Eve
Name: Name, dtype: object

df[['Name', 'Age']]:
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35
3    David   28
4      Eve   32
```

### 理解重点

- **单括号 vs 双括号**：`df["x"]` 是 `Series`，`df[["x"]]` 是单列 `DataFrame`。
- 多列选取常用于提取特征子集：`X = df[["feat1", "feat2", "feat3"]]`。

## 行选择（位置切片）

### 作用

DataFrame 支持类 Python 列表的整数切片 `df[start:stop]`，按**位置**返回子集。结束位置**不包含**。

### 示例代码

```python
print(f"df[1:3]:\n{df[1:3]}")
print(f"\ndf.head(2):\n{df.head(2)}")
```

### 输出

```text
df[1:3]:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72

df.head(2):
    Name  Age      City  Salary  Score
0  Alice   25   Beijing    8000     91
1    Bob   30  Shanghai   12000     87
```

### 理解重点

- 裸切片 `df[start:stop]` 按位置，遵循 Python 半开区间规则。
- **生产代码推荐显式使用 `df.iloc[1:3]`**，语义更明确。

## 按标签与位置精确索引

### `DataFrame.loc`

#### 作用

按**标签**选取行列。支持单标签、标签列表、标签切片、布尔数组。标签切片**包含终点**。

#### 重点方法

```python
df.loc[row_indexer, col_indexer]
```

#### 参数

| 参数         | 支持类型                                                  |
| ------------ | --------------------------------------------------------- |
| `row_indexer`| 单标签、标签列表、标签切片（闭区间）、布尔数组、callable  |
| `col_indexer`| 同上                                                      |

### `DataFrame.iloc`

#### 作用

按**整数位置**选取行列。规则与 Python 切片一致（**不包含终点**）。

#### 重点方法

```python
df.iloc[row_pos, col_pos]
```

#### 参数

| 参数       | 支持类型                                                   |
| ---------- | ---------------------------------------------------------- |
| `row_pos`  | 整数、整数列表、整数切片（半开）、布尔数组、callable        |
| `col_pos`  | 同上                                                       |

### `DataFrame.at` / `DataFrame.iat`

#### 作用

- `at`：按标签取**单个元素**，比 `loc` 更快。
- `iat`：按位置取**单个元素**，比 `iloc` 更快。

仅支持单个行 + 单个列标签/位置，不支持切片或列表。

### 综合示例

#### 示例代码

```python
df.index = ["a", "b", "c", "d", "e"]

# loc: 标签索引
print(f"loc['b', 'Name']: {df.loc['b', 'Name']}")
print(f"\nloc['a':'c', ['Name', 'Age']]:\n{df.loc['a':'c', ['Name', 'Age']]}")

# iloc: 位置索引
print(f"\niloc[1, 0]: {df.iloc[1, 0]}")
print(f"\niloc[0:3, 0:2]:\n{df.iloc[0:3, 0:2]}")

# at / iat: 单元素
print(f"\nat['b', 'Age']: {df.at['b', 'Age']}")
print(f"iat[1, 1]: {df.iat[1, 1]}")
```

#### 输出

```text
loc['b', 'Name']: Bob

loc['a':'c', ['Name', 'Age']]:
      Name  Age
a    Alice   25
b      Bob   30
c  Charlie   35

iloc[1, 0]: Bob

iloc[0:3, 0:2]:
      Name  Age
a    Alice   25
b      Bob   30
c  Charlie   35

at['b', 'Age']: 30
iat[1, 1]: 30
```

#### 理解重点

- **关键区别**：`loc` 切片**闭区间**（包含终点）；`iloc` 切片**半开**（不含终点）。
- 当索引是默认 `0, 1, 2, ...` 时特别容易混淆：`loc[0:2]` 返回 3 行（包含 2），`iloc[0:2]` 返回 2 行。
- 取单个元素优先 `at` / `iat`（比 `loc` / `iloc` 快 3~5 倍）。

## 条件过滤（布尔索引）

### 基本语法

```python
df[df["col"] > value]                         # 单条件
df[(cond1) & (cond2)]                         # 与
df[(cond1) | (cond2)]                         # 或
df[~cond]                                     # 非
df[df["col"].isin([v1, v2])]                  # 多值等于
df[df["col"].between(low, high)]              # 区间
df[df["col"].str.contains("pattern")]         # 字符串包含
```

### `Series.isin`

#### 作用

逐元素判断是否在给定值集合中，返回布尔 Series。替代多个 `==` 用 `|` 连接。

#### 重点方法

```python
Series.isin(values)
```

#### 参数

| 参数名   | 本例取值                        | 说明                                  |
| -------- | ------------------------------- | ------------------------------------- |
| `values` | `["Beijing", "Shanghai"]`       | 值集合（列表、Series、字典、数组）    |

### 综合示例

#### 示例代码

```python
# 单条件
print(f"Age > 28:\n{df[df['Age'] > 28]}")

# 多条件 AND
print(f"\nAge > 25 且 Salary > 10000:\n{df[(df['Age'] > 25) & (df['Salary'] > 10000)]}")

# 多条件 OR
print(f"\nCity 是 Beijing 或 Shanghai:\n{df[(df['City'] == 'Beijing') | (df['City'] == 'Shanghai')]}")

# isin 等价写法
print(f"\nisin:\n{df[df['City'].isin(['Beijing', 'Shanghai'])]}")

# between
print(f"\nAge 在 [28, 32]:\n{df[df['Age'].between(28, 32)]}")
```

#### 输出

```text
Age > 28:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Age > 25 且 Salary > 10000:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

City 是 Beijing 或 Shanghai:
      Name  Age      City  Salary  Score
0    Alice   25   Beijing    8000     91
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
```

### 理解重点

- 多条件组合**必须**用 `&` / `|` / `~`（位运算），**不能**用 `and` / `or` / `not`。
- 每个条件**必须**加括号，否则因运算符优先级出错：`df["A"] > 1 & df["B"] < 5` ✗。
- `isin` 对多个等值匹配最简洁；`between` 对区间匹配最简洁。
- 布尔索引返回**副本**（修改不影响原 DataFrame）。过滤后行索引保持不变，可 `reset_index(drop=True)` 重置。

## 字符串表达式查询

### `DataFrame.query`

#### 作用

用字符串表达式过滤行，语法接近 SQL。可通过 `@var` 引用外部 Python 变量。

#### 重点方法

```python
df.query(expr, *, inplace=False, **kwargs)
```

#### 参数

| 参数名    | 本例取值                            | 说明                                                                 |
| --------- | ----------------------------------- | -------------------------------------------------------------------- |
| `expr`    | `"Age > 28 and Salary > 10000"`     | 查询表达式（字符串）                                                 |
| `inplace` | `False`（默认）                     | `True` 时原地过滤（不推荐）                                          |
| `**kwargs`| `local_dict={'x': 5}` 等            | 底层 `pd.eval` 的可选参数                                            |

#### `query` 支持的语法

| 语法                        | 示例                                            |
| --------------------------- | ----------------------------------------------- |
| 比较                        | `"Age > 28"`                                    |
| 逻辑                        | `"Age > 28 and Salary > 10000"`                 |
| 或 / 非                     | `"City == 'Beijing' or City == 'Shanghai'"`     |
| `in` / `not in`             | `"City in ['Beijing', 'Shanghai']"`             |
| 引用外部变量                | `"Age >= @min_age"`                             |
| 列名含空格或关键字          | `` "`col name` > 5" ``（反引号包裹）            |

#### 示例代码

```python
# 基本表达式
print(df.query("Age > 28 and Salary > 10000"))

# 引用外部变量
min_age = 30
print(df.query("Age >= @min_age"))

# in 语法
print(df.query("City in ['Beijing', 'Shanghai']"))
```

#### 输出

```text
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
      Name  Age      City  Salary  Score
0    Alice   25   Beijing    8000     91
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68
```

### 理解重点

- `query` 内部使用 `and` / `or` / `not`（不是 `&` / `|` / `~`）——与布尔索引相反！
- `@var` 安全引用外部变量，避免把值拼接到字符串里（既丑又危险）。
- 列名带空格 / 点 / 保留字时用**反引号**包裹：`` `col name` ``。
- 复杂查询 `query` 更易读；简单查询布尔索引更直接。

## 链式索引的陷阱

### 问题

`df[df["A"] > 0]["B"] = 1` 这样的**链式索引赋值**会触发 `SettingWithCopyWarning`，且修改可能不生效。

### 正确写法

```python
# 用 loc 一步到位
df.loc[df["A"] > 0, "B"] = 1

# 或先赋给变量
mask = df["A"] > 0
df.loc[mask, "B"] = 1
```

### 理解重点

- 链式索引 `df[x][y]` 先切片再取列，产生的是临时对象；在其上赋值 Pandas 无法保证回写到原 DataFrame。
- **规则**：赋值场景用 `df.loc[行, 列] = 值`，不要用链式索引。

## 常见坑

1. `loc` 切片**包含**终点，`iloc` 切片**不包含**——这是最易混淆的点。
2. 多条件过滤忘加括号：`df["A"] > 1 & df["B"] < 5` 会报错，正确写法 `(df["A"] > 1) & (df["B"] < 5)`。
3. 在布尔索引中误用 `and` / `or`（应该用 `&` / `|`）。
4. 在 `query` 中误用 `&` / `|`（应该用 `and` / `or`）。
5. 链式索引赋值 `df[mask]["col"] = x` 会触发 `SettingWithCopyWarning`，应改用 `df.loc[mask, "col"] = x`。
6. `df.col` 属性访问在列名含空格、点、Python 关键字时失效，一律用 `df["col"]` 更安全。

## 小结

- **列选择**：`df["col"]`（单列 Series）/ `df[["c1","c2"]]`（多列 DataFrame）。
- **行选择**：`df.iloc[...]`（位置）/ `df.loc[...]`（标签）。
- **条件过滤**：布尔索引（`&` / `|` / `~`）与 `query`（`and` / `or` / `not`）二选一。
- 赋值场景永远用 `df.loc[行, 列] = 值`，避免链式索引的副作用。
