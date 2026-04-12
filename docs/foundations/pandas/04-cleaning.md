---
title: Pandas 数据清洗与处理
outline: deep
---

# Pandas 数据清洗与处理

> 对应脚本：`Basic/Pandas/04_cleaning.py`  
> 运行方式：`python Basic/Pandas/04_cleaning.py`（仓库根目录）

## 本章目标

1. 掌握缺失值的检测、删除与填充策略。
2. 学会检测和删除重复值。
3. 理解数据类型转换方法（`astype`、`to_datetime`）。
4. 熟悉字符串向量化操作（`str` 访问器）。
5. 掌握值替换的基本用法。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `df.isnull()` / `df.isna()` | 检测缺失值 | `demo_missing_values` |
| `df.dropna()` | 删除含缺失值的行/列 | `demo_missing_values` |
| `df.fillna(value)` | 填充缺失值 | `demo_missing_values` |
| `df.duplicated()` | 检测重复行 | `demo_duplicates` |
| `df.drop_duplicates()` | 删除重复行 | `demo_duplicates` |
| `s.astype(dtype)` | 类型转换 | `demo_type_conversion` |
| `pd.to_datetime(...)` | 转换为日期类型 | `demo_type_conversion` |
| `s.str.*` | 字符串向量化操作 | `demo_string_ops` |
| `s.replace(...)` | 值替换 | `demo_replace` |

## 1. 缺失值处理

### 方法重点

- `isnull()` 和 `isna()` 完全等价，返回布尔 DataFrame。
- `dropna()` 默认删除**任何列**含缺失值的行。
- `fillna()` 支持常量填充、前向填充（`ffill`）、后向填充（`bfill`）。

### 参数速览（本节）

1. `df.isnull()` — 无参数，返回与原数据同形状的布尔 DataFrame

2. `df.dropna(axis=0, how='any', subset=None, inplace=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `axis` | `0`（默认） | `0` 删除行，`1` 删除列 |
| `how` | `'any'`（默认） | `'any'` 任一缺失即删，`'all'` 全部缺失才删 |
| `subset` | `None`（默认） | 仅检查指定列的缺失值 |

3. `df.fillna(value=None, method=None, axis=None, inplace=False, limit=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `value` | `0` | 用固定值填充 |
| `method` | `'ffill'` | 前向填充（用前一个有效值） |
| `limit` | `None`（默认） | 连续填充的最大数量 |

### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1, 2, np.nan, 4, 5],
    "B": [np.nan, 2, 3, np.nan, 5],
    "C": ["x", "y", None, "z", "w"],
})

# 检测缺失值
print(df.isnull())
print(df.isnull().sum())

# 删除含缺失值的行
print(df.dropna())

# 用 0 填充
print(df.fillna(0))

# 前向填充
print(df.fillna(method="ffill"))
```

### 结果输出

```text
isnull():
       A      B      C
0  False   True  False
1  False  False  False
2   True  False   True
3  False   True  False
4  False  False  False
----------------
缺失值统计:
A    1
B    2
C    1
dtype: int64
----------------
dropna():
     A    B  C
1  2.0  2.0  y
4  5.0  5.0  w
----------------
fillna(0):
     A    B  C
0  1.0  0.0  x
1  2.0  2.0  y
2  0.0  3.0  0
3  4.0  0.0  z
4  5.0  5.0  w
----------------
fillna(method='ffill'):
     A    B  C
0  1.0  NaN  x
1  2.0  2.0  y
2  2.0  3.0  y
3  4.0  3.0  z
4  5.0  5.0  w
```

### 理解重点

- `isnull().sum()` 是快速统计缺失值的标准做法。
- `dropna()` 可能丢失大量数据，优先考虑 `fillna()`。
- 前向填充（`ffill`）适合时间序列数据，后向填充（`bfill`）适合回溯场景。

## 2. 重复值处理

### 方法重点

- `duplicated()` 返回布尔 Series，标记重复行（首次出现不算重复）。
- `drop_duplicates()` 默认保留第一次出现的行。
- `subset` 参数可指定仅基于某些列判断重复。

### 参数速览（本节）

1. `df.duplicated(subset=None, keep='first')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `subset` | `None`（默认） | 基于所有列判断重复 |
| `keep` | `'first'`（默认） | 首次出现标记为 False |

2. `df.drop_duplicates(subset=None, keep='first', inplace=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `subset` | `None`、`["A"]` | 基于指定列去重 |
| `keep` | `'first'`（默认） | 保留首次出现的行 |

### 示例代码

```python
df = pd.DataFrame({
    "A": [1, 1, 2, 2, 3],
    "B": ["a", "a", "b", "c", "c"],
})

print(df.duplicated())
print(df.drop_duplicates())
print(df.drop_duplicates(subset=["A"]))
```

### 结果输出

```text
duplicated():
0    False
1     True
2    False
3    False
4    False
dtype: bool
----------------
drop_duplicates():
   A  B
0  1  a
2  2  b
3  2  c
4  3  c
----------------
drop_duplicates(subset=['A']):
   A  B
0  1  a
2  2  b
4  3  c
```

### 理解重点

- `duplicated()` 默认第一次出现的行不算"重复"，`keep='last'` 则保留最后一次。
- `subset` 指定列时，只要这些列相同就算重复，其他列的值可以不同。
- `keep=False` 标记所有重复行（包括首次出现），用于发现数据质量问题。

## 3. 数据类型转换

### 方法重点

- `astype()` 是最通用的类型转换方法。
- `pd.to_datetime()` 专门处理日期字符串到时间类型的转换。
- 类型转换是数据清洗的重要步骤，影响后续计算和内存。

### 参数速览（本节）

1. `s.astype(dtype, copy=True, errors='raise')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `dtype` | `int`、`float` | 目标数据类型 |
| `errors` | `'raise'`（默认） | 转换失败时报错 |

2. `pd.to_datetime(arg, format=None, errors='raise', ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `arg` | `df["C"]` | 待转换的日期列 |
| `format` | `None`（默认） | 自动推断格式 |

### 示例代码

```python
df = pd.DataFrame({
    "A": ["1", "2", "3", "4", "5"],
    "B": [1.1, 2.2, 3.3, 4.4, 5.5],
    "C": ["2023-01-01", "2023-01-02", "2023-01-03",
          "2023-01-04", "2023-01-05"],
})
print(df.dtypes)

# astype 转换
df["A"] = df["A"].astype(int)
df["B"] = df["B"].astype(int)
print(df.dtypes)

# 日期转换
df["C"] = pd.to_datetime(df["C"])
print(df["C"].dtype)
```

### 结果输出

```text
转换前:
A    object
B    float64
C    object
dtype: object
----------------
astype 转换后:
A     int32
B     int32
C    object
dtype: object
----------------
日期转换后:
C 列类型: datetime64[ns]
```

### 理解重点

- 字符串列的 `dtype` 为 `object`，需要 `astype(int)` 才能做数值运算。
- `pd.to_datetime()` 比 `astype("datetime64")` 更灵活，支持多种日期格式。
- `errors='coerce'` 可将无法转换的值设为 `NaT`（Not a Time），避免整列报错。

## 4. 字符串操作

### 方法重点

- Pandas 通过 `str` 访问器提供向量化字符串操作。
- 所有 Python 字符串方法几乎都有对应的 `str` 版本。
- `str` 方法自动跳过 `NaN` 值，不会报错。

### 参数速览（本节）

适用 API（分项，均通过 `s.str` 访问）：

| 方法 | 说明 |
|---|---|
| `s.str.strip()` | 去除首尾空白 |
| `s.str.lower()` | 转小写 |
| `s.str.upper()` | 转大写 |
| `s.str.contains(pat)` | 是否包含子串（返回布尔） |
| `s.str.split(pat)` | 按分隔符拆分 |

### 示例代码

```python
df = pd.DataFrame({
    "Name": ["  Alice  ", "BOB", "charlie", "David Lee"],
    "Email": ["alice@example.com", "bob@test.org",
              "charlie@example.com", "david@test.org"],
})

print(df["Name"].str.strip())
print(df["Name"].str.lower())
print(df["Name"].str.upper())
print(df["Email"].str.contains("example"))
print(df["Email"].str.split("@"))
```

### 结果输出

```text
str.strip():
0        Alice
1          BOB
2      charlie
3    David Lee
Name: Name, dtype: object
----------------
str.lower():
0      alice
1        bob
2    charlie
3    david lee
Name: Name, dtype: object
----------------
str.upper():
0        ALICE
1          BOB
2      CHARLIE
3    DAVID LEE
Name: Name, dtype: object
----------------
str.contains('example'):
0     True
1    False
2     True
3    False
Name: Email, dtype: bool
----------------
str.split('@'):
0      [alice, example.com]
1           [bob, test.org]
2    [charlie, example.com]
3        [david, test.org]
Name: Email, dtype: object
```

### 理解重点

- `str` 访问器是向量化操作，比循环调用字符串方法快得多。
- `str.contains()` 支持正则表达式，`regex=False` 可关闭。
- `str.split(..., expand=True)` 可将拆分结果直接展开为多列。

## 5. 值替换

### 方法重点

- `replace()` 支持单值替换和字典批量替换。
- 可以同时作用于整个 DataFrame 或单个 Series。
- 替换不会修改原数据，返回新对象（除非 `inplace=True`）。

### 参数速览（本节）

适用 API：`s.replace(to_replace, value=None, inplace=False, regex=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `to_replace` | `1`、`{"yes": 1, "no": 0, "maybe": -1}` | 被替换的值或字典映射 |
| `value` | `100` | 单值替换时的目标值 |
| `regex` | `False`（默认） | 是否支持正则表达式 |

### 示例代码

```python
df = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": ["yes", "no", "yes", "no", "maybe"],
})

# 单值替换
print(df["A"].replace(1, 100))

# 字典替换
print(df["B"].replace({"yes": 1, "no": 0, "maybe": -1}))
```

### 结果输出

```text
replace(1, 100):
0    100
1      2
2      3
3      4
4      5
Name: A, dtype: int64
----------------
replace({'yes': 1, 'no': 0, 'maybe': -1}):
0    1
1    0
2    1
3    0
4   -1
Name: B, dtype: int64
```

### 理解重点

- 字典替换是将分类标签映射为数值的常用方法。
- `replace` 和 `map` 功能类似，但 `replace` 只替换匹配项，`map` 会将不匹配项设为 `NaN`。
- `regex=True` 时可以用正则表达式进行模式替换。

## 常见坑

1. `fillna(method='ffill')` 在首行就是 NaN 时无法填充。
2. `astype(int)` 遇到 NaN 会报错，需先处理缺失值或使用 `Int64`（可空整数类型）。
3. `drop_duplicates` 默认基于所有列判断，大数据集下可能很慢，建议指定 `subset`。
4. `str` 方法只能用于 `object` 类型的 Series，数值列调用会报错。

## 小结

- 数据清洗是分析流水线中最耗时但最关键的步骤。
- 缺失值处理三板斧：检测（`isnull`）→ 删除（`dropna`）或填充（`fillna`）。
- `str` 访问器和 `replace` 是文本列清洗的核心工具。
