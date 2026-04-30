---
title: Pandas 数据选择与过滤
outline: deep
---

# Pandas 数据选择与过滤

## 本章目标

1. 掌握单列和多列选择的语法差异及返回类型区别
2. 理解 `loc`（标签索引）和 `iloc`（位置索引）的切片规则差异
3. 掌握布尔条件过滤与多条件组合的正确写法
4. 学会使用 `isin` 和 `query` 进行高效数据筛选

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df["col"]` | 语法 | 选取单列，返回 `Series` |
| `df[["c1", "c2"]]` | 语法 | 选取多列，返回 `DataFrame` |
| `df[start:stop]` | 语法 | 行切片（按位置，半开区间） |
| `df.loc[...]` | 索引器 | 按**标签**选取行列（切片闭区间） |
| `df.iloc[...]` | 索引器 | 按**位置**选取行列（切片半开） |
| `df.at[...]` / `df.iat[...]` | 索引器 | 按标签/位置取**单个元素**（比 loc/iloc 快） |
| `Series.isin(...)` | 方法 | 元素级成员检测 |
| `Series.between(...)` | 方法 | 区间内检测 |
| `df.query(...)` | 方法 | 字符串表达式查询 |

## 1. 列选择

### 语法速览

| 语法 | 返回类型 | 说明 |
|---|---|---|
| `df["Name"]` | `Series` | 单列，最常用写法 |
| `df[["Name"]]` | `DataFrame` | 单列 DataFrame（注意双层方括号） |
| `df[["Name", "Age"]]` | `DataFrame` | 多列子集 |
| `df.Name` | `Series` | 属性访问（列名含空格/关键字时不可用） |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["Beijing", "Shanghai", "Guangzhou"],
})

# 单列返回 Series
print(f"type(df['Name']) = {type(df['Name']).__name__}")
print(df["Name"])

# 多列返回 DataFrame
print(f"\ndf[['Name', 'Age']]:")
print(df[["Name", "Age"]])
```

### 输出

```text
type(df['Name']) = Series
0      Alice
1        Bob
2    Charlie
Name: Name, dtype: object

df[['Name', 'Age']]:
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35
```

### 理解重点

- **单括号 vs 双括号**：`df["x"]` 返回 `Series`，`df[["x"]]` 返回单列 `DataFrame`——二者可用的方法不同
- 多列选取常用于提取特征子集：`X = df[["feat1", "feat2", "feat3"]]`
- 属性访问 `df.Name` 仅在列名是合法 Python 标识符时可用；列名含空格/点号/关键字时一律用 `df["col"]`

## 2. 行切片

### `df[start:stop]`

#### 作用

DataFrame 支持类 Python 列表的整数切片，按**位置**行切片（半开区间）。

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 35, 28, 32],
})

print(f"df[1:3]:\n{df[1:3]}")
print(f"\ndf[:2]:\n{df[:2]}")
```

#### 输出

```text
df[1:3]:
      Name  Age
1      Bob   30
2  Charlie   35

df[:2]:
    Name  Age
0  Alice   25
1    Bob   30
```

#### 理解重点

- 裸切片 `df[start:stop]` 按位置，遵循 Python 半开区间（不含 stop）
- 生产代码推荐显式使用 `df.iloc[start:stop]`——语义更明确

## 3. 按标签索引

### `DataFrame.loc`

#### 作用

按**标签**选取行列。支持单标签、标签列表、标签切片、布尔数组。标签切片**包含终点**（闭区间），与 `iloc` 不同。

#### 重点方法

```python
df.loc[row_indexer, col_indexer]
```

#### 参数

| 参数 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `row_indexer` | 单标签、标签列表、标签切片、布尔数组、callable | 行选择器；标签切片为**闭区间** | `"a"`、`["a","c"]`、`"a":"c"`、`df.A > 0` |
| `col_indexer` | 同上 | 列选择器 | `"Name"`、`["Name","Age"]`、`:`（全部列） |

### 示例代码

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
}, index=["a", "b", "c", "d", "e"])

# 单元素
print(f"loc['b', 'Name']: {df.loc['b', 'Name']}")

# 标签切片（闭区间！包含 'c'）
print(f"\nloc['a':'c', ['Name', 'Age']]:\n{df.loc['a':'c', ['Name', 'Age']]}")

# 布尔数组 + 列选择
print(f"\nloc[Age > 28, ['Name', 'Salary']]:\n{df.loc[df['Age'] > 28, ['Name', 'Salary']]}")

# 全部行指定列
print(f"\nloc[:, 'City']:\n{df.loc[:, 'City']}")
```

### 输出

```text
loc['b', 'Name']: Bob

loc['a':'c', ['Name', 'Age']]:
      Name  Age
a    Alice   25
b      Bob   30
c  Charlie   35

loc[Age > 28, ['Name', 'Salary']]:
      Name  Salary
b      Bob   12000
c  Charlie   15000
e      Eve   11000

loc[:, 'City']:
a      Beijing
b     Shanghai
c      Beijing
d    Guangzhou
e     Shanghai
Name: City, dtype: object
```

### 理解重点

- **最关键的区别**：`loc` 切片是**闭区间**（包含终点），`iloc` 切片是**半开**（不含终点）
- 当索引是默认 `0, 1, 2, ...` 时特别容易混淆：`df.loc[0:2]` 返回 3 行（含 2），`df.iloc[0:2]` 返回 2 行
- `loc` 可以同时按标签选行 + 按名称选列——一步完成行列筛选
- 赋值场景一律用 `loc`：`df.loc[mask, "col"] = value`——避免链式索引的 `SettingWithCopyWarning`

## 4. 按位置索引

### `DataFrame.iloc`

#### 作用

按**整数位置**选取行列。规则与 Python 列表索引一致：整数从 0 开始，切片**不包含终点**（半开区间）。

#### 重点方法

```python
df.iloc[row_pos, col_pos]
```

#### 参数

| 参数 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `row_pos` | 整数、整数列表、整数切片、布尔数组、callable | 行位置选择器 | `0`、`[0, 2]`、`0:3`、`[True, False, ...]` |
| `col_pos` | 同上 | 列位置选择器 | `1`、`[0, 2]`、`:`（全部列） |

### 示例代码

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

# 单元素
print(f"iloc[1, 0]: {df.iloc[1, 0]}")

# 位置切片（半开！不含索引 3）
print(f"\niloc[0:3, 0:2]:\n{df.iloc[0:3, 0:2]}")

# 整数列表
print(f"\niloc[[0, 2, 4], [0, 3]]:\n{df.iloc[[0, 2, 4], [0, 3]]}")

# 全部行指定列
print(f"\niloc[:, -2:]:\n{df.iloc[:, -2:]}")
```

### 输出

```text
iloc[1, 0]: Bob

iloc[0:3, 0:2]:
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35

iloc[[0, 2, 4], [0, 3]]:
      Name  Salary
0    Alice    8000
2  Charlie   15000
4      Eve   11000

iloc[:, -2:]:
   Salary  Score
0    8000     91
1   12000     87
2   15000     72
3    9000     88
4   11000     68
```

### 理解重点

- `iloc` 始终按位置——与索引标签无关，即使索引是自定义字符串也按 0, 1, 2, ... 定位
- 负索引 `-1` 表示最后一行/列，`-2` 表示倒数第二——与 Python list 一致
- 取多个不连续行用列表 `iloc[[0, 2, 5]]`，连续行用切片 `iloc[0:5]`

## 5. 单元素快速访问

### `at` / `iat`

#### 作用

- `at[row_label, col_label]`：按标签取**单个元素**，比 `loc` 快 3~5 倍
- `iat[row_pos, col_pos]`：按位置取**单个元素**，比 `iloc` 快 3~5 倍

仅支持单个行 + 单个列（不支持切片或列表）。适合循环中逐元素读写或精确修改某一格。

### 速览

| 索引器 | 索引方式 | 对标 | 适用场景 |
|---|---|---|---|
| `df.at[label, col]` | 标签 | `loc` | 已知行列标签 |
| `df.iat[pos, col]` | 位置 | `iloc` | 已知行列位置 |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6],
}, index=["x", "y", "z"])

print(f"at['y', 'B']: {df.at['y', 'B']}")
print(f"iat[1, 1]: {df.iat[1, 1]}")

# 快速修改某个值
df.at["z", "A"] = 99
print(f"\n修改后:\n{df}")
```

### 输出

```text
at['y', 'B']: 5
iat[1, 1]: 5

修改后:
    A  B
x   1  4
y   2  5
z  99  6
```

### 理解重点

- `at` / `iat` 只能取**单个元素**——传入切片或列表会报错
- 遍历 DataFrame 逐元素修改时优先用 `at` / `iat`，比 `loc` / `iloc` 快
- 日常读写非性能瓶颈时用 `loc` / `iloc` 更灵活

## 6. 布尔索引

### 基本语法

| 语法 | 含义 | 说明 |
|---|---|---|
| `df[df["col"] > value]` | 单条件 | 返回符合条件的行 |
| `df[(cond1) & (cond2)]` | 与 | **必须用 `&`**，不能用 `and` |
| `df[(cond1) \| (cond2)]` | 或 | **必须用 `\|`**，不能用 `or` |
| `df[~cond]` | 非 | **必须用 `~`**，不能用 `not` |
| `df[df["col"].isin([v1, v2])]` | 多值匹配 | 替代多个 `==` 用 `\|` 连接 |
| `df[df["col"].between(low, high)]` | 区间 | 闭区间 `[low, high]` |
| `df[df["col"].str.contains("pat")]` | 字符串包含 | 详见 ch04 `.str` 访问器 |

### `Series.isin`

#### 作用

逐元素判断是否在给定值集合中，返回布尔 `Series`。替代多个 `==` 用 `|` 连接。

#### 重点方法

```python
Series.isin(values)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `values` | `list`、`set`、`Series`、`dict` | 值集合 | `["Beijing", "Shanghai"]` |

### `Series.between`

#### 作用

逐元素判断是否在区间内。默认闭区间 `[left, right]`（两端包含）。

#### 重点方法

```python
Series.between(left, right, inclusive="both")
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `left` | 标量 | 区间下界（包含） | `28` |
| `right` | 标量 | 区间上界（包含） | `32` |
| `inclusive` | `str` | 端点包含策略：`"both"` / `"left"` / `"right"` / `"neither"`，默认为 `"both"` | `"left"` |

### 综合示例

#### 示例代码

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

# 单条件
print(f"Age > 28:\n{df[df['Age'] > 28]}")

# 多条件 AND（必须用 &，每个条件必须加括号！）
print(f"\nAge > 25 & Salary > 10000:\n{df[(df['Age'] > 25) & (df['Salary'] > 10000)]}")

# isin —— 替代多个 OR
print(f"\nisin(['Beijing', 'Shanghai']):\n{df[df['City'].isin(['Beijing', 'Shanghai'])]}")

# between —— 区间筛选
print(f"\nAge.between(28, 32):\n{df[df['Age'].between(28, 32)]}")

# 取反
print(f"\n非 Beijing:\n{df[~df['City'].isin(['Beijing'])]}")
```

#### 输出

```text
Age > 28:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Age > 25 & Salary > 10000:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

isin(['Beijing', 'Shanghai']):
      Name  Age      City  Salary  Score
0    Alice   25   Beijing    8000     91
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Age.between(28, 32):
    Name  Age      City  Salary  Score
1    Bob   30  Shanghai   12000     87
3  David   28  Guangzhou    9000     88
4    Eve   32  Shanghai   11000     68

非 Beijing:
    Name  Age      City  Salary  Score
1    Bob   30  Shanghai   12000     87
3  David   28  Guangzhou    9000     88
4    Eve   32  Shanghai   11000     68
```

### 理解重点

- 多条件组合**必须**用 `&` / `|` / `~`（位运算符），**不能**用 `and` / `or` / `not`
- 每个条件**必须加括号**：`(df.A > 1) & (df.B < 5)`——否则因运算符优先级报错
- 布尔索引返回**副本**，修改不影响原 DataFrame；过滤后行索引保持不变，用 `reset_index(drop=True)` 重置
- `isin` 替代多个 `==` 的 OR 连接——性能更好、代码更短
- `between` 默认闭区间（两端包含），用 `inclusive` 参数调整

## 7. 表达式查询

### `DataFrame.query`

#### 作用

用字符串表达式过滤行，语法接近 SQL。可通过 `@` 引用外部 Python 变量。内部使用 `and` / `or` / `not`（非 `&` / `|` / `~`），与布尔索引相反。

#### 重点方法

```python
df.query(expr, *, inplace=False, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `expr` | `str` | 查询表达式字符串 | `"Age > 28 and Salary > 10000"` |
| `inplace` | `bool` | 是否原地过滤，默认为 `False` | `True` |

#### `query` 支持的语法

| 语法 | 示例 |
|---|---|
| 比较运算 | `"Age > 28"` |
| 逻辑与 / 或 / 非 | `"Age > 28 and Salary > 10000"` |
| `in` / `not in` | `"City in ['Beijing', 'Shanghai']"` |
| 引用外部变量 | `"Age >= @minAge"` |
| 列名含空格/关键字 | `` "`col name` > 5" ``（反引号包裹） |

#### 示例代码

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

# 基本表达式
print(df.query("Age > 28 and Salary > 10000"))

# 引用外部变量（用 @）
minAge = 30
print(f"\nAge >= {minAge}:\n{df.query('Age >= @minAge')}")

# in 语法
print(f"\nCities:\n{df.query('City in [\"Beijing\", \"Shanghai\"]')}")

# not in
print(f"\n非 Beijing/Shanghai:\n{df.query('City not in [\"Beijing\", \"Shanghai\"]')}")
```

#### 输出

```text
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Age >= 30:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Cities:
      Name  Age      City  Salary  Score
0    Alice   25   Beijing    8000     91
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

非 Beijing/Shanghai:
    Name  Age       City  Salary  Score
3  David   28  Guangzhou    9000     88
```

### 理解重点

- `query` 内部用 `and` / `or` / `not`——与布尔索引（`&` / `|` / `~`）**相反**
- `@var` 安全引用外部变量，避免字符串拼接（既丑又有注入风险）
- 列名含空格/点号/保留字时用**反引号**包裹：`` `col name` ``
- 复杂多条件查询 `query` 更易读；简单单条件布尔索引更直接

## 链式索引的陷阱

### 问题

`df[df["A"] > 0]["B"] = 1` 这样的**链式索引赋值**会触发 `SettingWithCopyWarning`，且修改可能不生效。原因是 `df[mask]` 返回临时副本，在其上赋值无法回写到原 DataFrame。

### 正确写法

```python
# 用 loc 一步到位
df.loc[df["A"] > 0, "B"] = 1

# 或先计算掩码再赋值
mask = df["A"] > 0
df.loc[mask, "B"] = 1
```

### 理解重点

- 链式索引 `df[x][y]` 分两步：先切片得临时对象，再取列——赋值无法保证回写
- 规则：**赋值场景用 `df.loc[行, 列] = 值`**，读取场景链式索引可接受

## 常见坑

1. `loc` 切片**包含**终点，`iloc` 切片**不包含**——当索引是 `0, 1, 2, ...` 时容易误判
2. 多条件过滤忘加括号：`df["A"] > 1 & df["B"] < 5` 会报错——正确写法 `(df["A"] > 1) & (df["B"] < 5)`
3. 布尔索引误用 `and` / `or`——应该用 `&` / `|`
4. 在 `query` 中误用 `&` / `|`——应该用 `and` / `or`
5. 链式索引赋值 `df[mask]["col"] = x` 会触发 `SettingWithCopyWarning`——改用 `df.loc[mask, "col"] = x`
6. `df.col` 属性访问在列名含空格/点号/Python 关键字时失效——一律用 `df["col"]` 更安全
7. 布尔索引返回副本——修改后若需保留结果应显式赋值给变量

## 小结

- 列选择：`df["col"]`（Series）/ `df[["c1", "c2"]]`（DataFrame）——双括号是关键
- 行选择：`df.iloc[...]`（位置，半开）/ `df.loc[...]`（标签，闭区间）——切片规则不同
- 条件过滤：布尔索引用 `&` / `|` / `~`；`query` 用 `and` / `or` / `not`——不要混用
- 赋值场景永远用 `df.loc[行, 列] = 值`，避免链式索引的副作用
