---
title: Pandas 数据清洗与处理
outline: deep
---

# Pandas 数据清洗与处理

## 本章目标

1. 掌握缺失值的检测、删除与填充策略。
2. 学会检测和删除重复值。
3. 掌握数据类型转换（`astype` / `pd.to_datetime` / `pd.to_numeric`）。
4. 熟悉字符串向量化操作（`str` 访问器）。
5. 掌握值替换的常见用法。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.isnull(...)` / `df.isna(...)` | 方法 | 检测缺失值（两者完全等价） |
| `df.notnull(...)` / `df.notna(...)` | 方法 | 检测非缺失值 |
| `df.dropna(...)` | 方法 | 删除含缺失值的行 / 列 |
| `df.fillna(...)` | 方法 | 填充缺失值 |
| `df.duplicated(...)` | 方法 | 标记重复行 |
| `df.drop_duplicates(...)` | 方法 | 删除重复行 |
| `s.astype(...)` | 方法 | 类型转换 |
| `pd.to_datetime(...)` | 函数 | 转换为日期时间类型 |
| `pd.to_numeric(...)` | 函数 | 转换为数值类型（支持错误处理） |
| `s.str.xxx(...)` | 访问器 | 字符串向量化操作 |
| `s.replace(...)` | 方法 | 值替换 |
| `s.map(...)` | 方法 | 按字典 / 函数逐元素映射 |

## 缺失值处理

### 缺失值表示

Pandas 用 `NaN`（`float`）、`None`（`object`）、`NaT`（日期时间）等表示缺失值，统称为 NA。

### `DataFrame.isnull` / `isna`

#### 作用

返回与原数据同形状的布尔 DataFrame，`True` 表示该位置为缺失。两方法**完全等价**。

#### 重点方法

```python
df.isnull()
df.isna()
```

### `DataFrame.dropna`

#### 作用

删除含有缺失值的行或列。

#### 重点方法

```python
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False, ignore_index=False)
```

#### 参数

| 参数名         | 本例取值         | 说明                                                                   |
| -------------- | ---------------- | ---------------------------------------------------------------------- |
| `axis`         | `0`（默认）、`1`  | `0` 删除行，`1` 删除列                                                 |
| `how`          | `'any'`（默认）、`'all'` | `'any'`：任一位置缺失即删；`'all'`：全部缺失才删              |
| `thresh`       | `None`（默认）、整数 | 保留至少有 `thresh` 个非缺失值的行 / 列                            |
| `subset`       | `None`（默认）、列名列表 | 只检查指定列的缺失情况                                         |
| `inplace`      | `False`（默认）   | 是否原地修改                                                           |
| `ignore_index` | `False`（默认）   | 是否重置索引为 `RangeIndex`                                            |

### `DataFrame.fillna`

#### 作用

填充缺失值。支持常量、前向 / 后向填充、按列字典填充等。

#### 重点方法

```python
df.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)
```

#### 参数

| 参数名     | 本例取值                          | 说明                                                          |
| ---------- | --------------------------------- | ------------------------------------------------------------- |
| `value`    | `0`、`{'A': 0, 'B': mean}`        | 填充值；可为标量、字典（按列）、Series、DataFrame             |
| `method`   | `None`、`'ffill'`、`'bfill'`      | 前向 / 后向填充（不能与 `value` 同用）                        |
| `axis`     | `None`（默认）、`0`、`1`          | 填充方向                                                      |
| `inplace`  | `False`（默认）                   | 是否原地修改                                                  |
| `limit`    | `None`、整数                      | 连续填充的最大次数                                            |

注：新版本（Pandas 2.0+）推荐 `df.ffill()` / `df.bfill()` 替代 `method` 参数。

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1, 2, np.nan, 4, 5],
    "B": [np.nan, 2, 3, np.nan, 5],
    "C": ["x", "y", None, "z", "w"],
})

print(f"原数据:\n{df}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\ndropna():\n{df.dropna()}")
print(f"\nfillna(0):\n{df.fillna(0)}")
print(f"\nffill:\n{df.ffill()}")
```

#### 输出

```text
原数据:
     A    B     C
0  1.0  NaN     x
1  2.0  2.0     y
2  NaN  3.0  None
3  4.0  NaN     z
4  5.0  5.0     w

缺失值统计:
A    1
B    2
C    1
dtype: int64

dropna():
     A    B  C
1  2.0  2.0  y
4  5.0  5.0  w

fillna(0):
     A    B  C
0  1.0  0.0  x
1  2.0  2.0  y
2  0.0  3.0  0
3  4.0  0.0  z
4  5.0  5.0  w

ffill:
     A    B  C
0  1.0  NaN  x
1  2.0  2.0  y
2  2.0  3.0  y
3  4.0  3.0  z
4  5.0  5.0  w
```

### 理解重点

- `isnull()` / `isna()` 完全等价，选用习惯即可。
- `dropna` 默认删除**任何列缺失**的行；`how='all'` 才是"全部缺失"才删。
- 填充策略常用组合：数值列用均值 / 中位数，类别列用众数或 `'Unknown'`，时序列用 `ffill`。

## 重复值处理

### `DataFrame.duplicated`

#### 作用

标记重复行。返回布尔 Series：**第一次出现标 `False`，后续标 `True`**（可调整）。

#### 重点方法

```python
df.duplicated(subset=None, keep='first')
```

#### 参数

| 参数名   | 本例取值                                 | 说明                                                                  |
| -------- | ---------------------------------------- | --------------------------------------------------------------------- |
| `subset` | `None`（默认）、列名列表                 | 只检查指定列的重复                                                    |
| `keep`   | `'first'`（默认）、`'last'`、`False`     | 标记策略：保留首次为 `False`、保留末次为 `False`、所有重复都为 `True` |

### `DataFrame.drop_duplicates`

#### 作用

删除重复行。

#### 重点方法

```python
df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
```

#### 参数

| 参数名         | 本例取值                           | 说明                                               |
| -------------- | ---------------------------------- | -------------------------------------------------- |
| `subset`       | `None`（默认）、`['A']`            | 只根据指定列判断重复                               |
| `keep`         | `'first'`（默认）、`'last'`、`False` | 保留策略；`False` 表示**所有重复都删除**         |
| `inplace`      | `False`（默认）                    | 是否原地修改                                       |
| `ignore_index` | `False`（默认）                    | 是否重置索引                                       |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({"A": [1, 1, 2, 2, 3], "B": ["a", "a", "b", "c", "c"]})

print(f"原数据:\n{df}")
print(f"\nduplicated():\n{df.duplicated()}")
print(f"\ndrop_duplicates():\n{df.drop_duplicates()}")
print(f"\ndrop_duplicates(subset=['A']):\n{df.drop_duplicates(subset=['A'])}")
```

### 输出

```text
原数据:
   A  B
0  1  a
1  1  a
2  2  b
3  2  c
4  3  c

duplicated():
0    False
1     True
2    False
3    False
4    False
dtype: bool

drop_duplicates():
   A  B
0  1  a
2  2  b
3  2  c
4  3  c

drop_duplicates(subset=['A']):
   A  B
0  1  a
2  2  b
4  3  c
```

### 理解重点

- `duplicated()` 默认 `keep='first'`：第一次出现不标记（`False`）。
- 想找**所有重复**的行：`keep=False`。
- 业务去重常按**关键列**：`drop_duplicates(subset=['id', 'date'])`。

## 数据类型转换

### `Series.astype`

#### 作用

将 Series 转换为指定 dtype。不能处理转换失败（遇到无法转换的值会抛错）。

#### 重点方法

```python
s.astype(dtype, copy=True, errors='raise')
```

#### 参数

| 参数名   | 本例取值                  | 说明                                                                 |
| -------- | ------------------------- | -------------------------------------------------------------------- |
| `dtype`  | `int`、`np.float32`、`'category'`、`{'A': int, 'B': str}` | 目标 dtype                      |
| `copy`   | `True`（默认）            | 是否返回副本                                                         |
| `errors` | `'raise'`（默认）、`'ignore'` | `'raise'` 无法转换时抛错；`'ignore'` 静默保留原值                |

### `pd.to_datetime`

#### 作用

将字符串或数字转换为 `datetime64` 类型。支持多种格式、错误处理、时区等。

#### 重点方法

```python
pd.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False,
               utc=False, format=None, exact=True, unit=None,
               infer_datetime_format=False, origin='unix', cache=True)
```

#### 参数

| 参数名                | 本例取值                              | 说明                                                        |
| --------------------- | ------------------------------------- | ----------------------------------------------------------- |
| `arg`                 | `"2023-01-01"`、Series、列表          | 输入                                                        |
| `errors`              | `'raise'`（默认）、`'coerce'`、`'ignore'` | `'coerce'` 将无法解析设为 `NaT`                       |
| `format`              | `None`、`'%Y-%m-%d'`                   | 日期格式字符串，给出后速度更快                              |
| `dayfirst`            | `False`（默认）                        | 是否按"日/月/年"解析                                        |
| `utc`                 | `False`（默认）                        | 是否转为 UTC 时区                                           |
| `unit`                | `None`、`'s'`、`'ms'`、`'ns'`         | 输入为数字时代表的时间单位                                  |

### `pd.to_numeric`

#### 作用

将字符串 / 混合类型转换为数值，支持错误处理。

#### 重点方法

```python
pd.to_numeric(arg, errors='raise', downcast=None)
```

#### 参数

| 参数名     | 本例取值                                  | 说明                                                          |
| ---------- | ----------------------------------------- | ------------------------------------------------------------- |
| `arg`      | Series、列表                              | 输入                                                          |
| `errors`   | `'raise'`（默认）、`'coerce'`、`'ignore'` | `'coerce'` 将无法转换的设为 `NaN`                             |
| `downcast` | `None`（默认）、`'integer'`、`'float'`、`'signed'`、`'unsigned'` | 尝试降精度以节省内存       |

### 综合示例

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "A": ["1", "2", "3", "4", "5"],
    "B": [1.1, 2.2, 3.3, 4.4, 5.5],
    "C": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
})

df["A"] = df["A"].astype(int)
df["B"] = df["B"].astype(int)
df["C"] = pd.to_datetime(df["C"])

print(df)
print(f"\ndtypes:\n{df.dtypes}")
```

#### 输出

```text
   A  B          C
0  1  1 2023-01-01
1  2  2 2023-01-02
2  3  3 2023-01-03
3  4  4 2023-01-04
4  5  5 2023-01-05

dtypes:
A             int64
B             int64
C    datetime64[ns]
dtype: object
```

### 理解重点

- `astype(int)` 对浮点**截断**（`1.9 → 1`）；需要四舍五入先 `round()`。
- `pd.to_datetime` 遇到脏数据用 `errors='coerce'` 生成 `NaT`，再用 `dropna` 清除。
- `category` 类型可显著节省内存：`df['city'] = df['city'].astype('category')`。

## 字符串向量化操作

### `Series.str` 访问器

Series 通过 `.str` 访问器可以调用**向量化字符串方法**，自动跳过 `NaN`。

### 常用方法一览

| 方法                          | 作用                                      |
| ----------------------------- | ----------------------------------------- |
| `s.str.lower()` / `upper()`   | 大小写转换                                |
| `s.str.strip(...)`            | 去除两端空白                              |
| `s.str.lstrip()` / `rstrip()` | 去除左 / 右空白                           |
| `s.str.len()`                 | 每个字符串长度                            |
| `s.str.contains(pat, regex=True)` | 是否包含模式，返回布尔 Series         |
| `s.str.startswith(p)` / `endswith(p)` | 前缀 / 后缀匹配                   |
| `s.str.replace(pat, repl, regex=True)` | 替换                             |
| `s.str.split(sep, expand=False)` | 按分隔符切分                          |
| `s.str.extract(pat)`          | 正则提取（捕获组）                        |
| `s.str.cat(sep='')`           | 连接所有字符串                            |
| `s.str.pad(width, side, fillchar)` | 补齐宽度                            |
| `s.str.zfill(width)`          | 左侧补零                                  |
| `s.str[i]` / `s.str[i:j]`     | 按位置取字符                              |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["  Alice  ", "BOB", "charlie", "David Lee"],
    "Email": ["alice@example.com", "bob@test.org",
              "charlie@example.com", "david@test.org"],
})

print(f"strip + lower:\n{df['Name'].str.strip().str.lower()}")
print(f"\n包含 'example':\n{df['Email'].str.contains('example')}")
print(f"\nsplit('@'):\n{df['Email'].str.split('@')}")
print(f"\n提取用户名 / 域名:\n{df['Email'].str.split('@', expand=True)}")
```

### 输出

```text
strip + lower:
0        alice
1          bob
2      charlie
3    david lee
Name: Name, dtype: object

包含 'example':
0     True
1    False
2     True
3    False
Name: Email, dtype: bool

split('@'):
0    [alice, example.com]
1         [bob, test.org]
2    [charlie, example.com]
3       [david, test.org]
Name: Email, dtype: object

提取用户名 / 域名:
         0            1
0    alice  example.com
1      bob     test.org
2  charlie  example.com
3    david     test.org
```

### 理解重点

- `str` 访问器自动跳过 `NaN`（返回 `NaN`），比写 `for` 循环安全得多。
- `str.split(sep, expand=True)` 直接展开成多列 DataFrame——常用于"名字中分出姓 / 名"。
- 字符串方法可链式调用：`s.str.strip().str.lower().str.replace(' ', '_')`。

## 值替换

### `Series.replace`

#### 作用

将指定值替换为新值。支持单值、列表、字典、正则等多种模式。

#### 重点方法

```python
s.replace(to_replace=None, value=None, inplace=False, limit=None,
          regex=False, method=None)
```

#### 参数

| 参数名       | 本例取值                                      | 说明                                                             |
| ------------ | --------------------------------------------- | ---------------------------------------------------------------- |
| `to_replace` | `1`、`[1, 2]`、`{'yes': 1, 'no': 0}`          | 被替换值；可为标量、列表、字典（键值对映射）、正则               |
| `value`      | `100`、`None`                                 | 替换为的值；用字典形式时可省略                                   |
| `inplace`    | `False`（默认）                               | 是否原地修改                                                     |
| `regex`      | `False`（默认）、`True`                       | 是否将 `to_replace` 作为正则模式                                 |
| `method`     | `None`（默认）、`'pad'`、`'ffill'`、`'bfill'` | `value=None` 且非字典时的填充策略（已不推荐）                    |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                   "B": ["yes", "no", "yes", "no", "maybe"]})

print(f"replace(1, 100):\n{df['A'].replace(1, 100)}")
print(f"\n字典替换:\n{df['B'].replace({'yes': 1, 'no': 0, 'maybe': -1})}")

# 多个值换成同一个
print(f"\n多值→同值:\n{df['A'].replace([1, 2, 3], 0)}")

# 正则
df2 = pd.DataFrame({"s": ["abc123", "def456", "ghi789"]})
print(f"\n正则替换:\n{df2['s'].replace(r'\\d+', 'NUM', regex=True)}")
```

### 输出

```text
replace(1, 100):
0    100
1      2
2      3
3      4
4      5
Name: A, dtype: int64

字典替换:
0    1
1    0
2    1
3    0
4   -1
Name: B, dtype: int64

多值→同值:
0    0
1    0
2    0
3    4
4    5
Name: A, dtype: int64

正则替换:
0    abcNUM
1    defNUM
2    ghiNUM
Name: s, dtype: object
```

### 理解重点

- **字典形式**是最直观的映射写法，尤其适合"类别 → 整数编码"。
- `regex=True` 时所有字符串模式按正则解析，注意转义。
- `replace` 适合已知旧值替换；若需**任意映射**（含函数）用 `s.map(...)` 更合适。

## 常见坑

1. `isnull` 只识别 `NaN` / `None` / `NaT`，**不识别**空字符串 `""`；需要先 `s.replace('', np.nan)`。
2. `fillna(method='ffill')` 在新版本已弃用，改用 `df.ffill()`。
3. `dropna(inplace=True)` 后 `df` 索引**不重置**；需要 `reset_index(drop=True)` 或加 `ignore_index=True`。
4. `duplicated(keep=False)` 会把**所有**重复行都标为 `True`（包括第一个），和默认行为不同。
5. `astype(int)` 遇到 `NaN` 会抛错；先 `fillna` 或用可空整数类型 `Int64`（大写）。
6. `str.contains(pat)` 遇到 `NaN` 默认返回 `NaN`；布尔索引前要先 `fillna(False)` 或用 `na=False` 参数。
7. 链式字符串调用 `s.str.strip().lower()` 是**错的**，每步都要 `.str`：`s.str.strip().str.lower()`。

## 小结

- **清洗四步曲**：检测缺失 → 处理缺失 → 去重 → 类型转换。
- 缺失值处理思路：能删就删（`dropna`），不能删就填（`fillna` + 合适策略）。
- `str` 访问器是 Pandas 最强的字符串处理工具，比 apply 快得多。
- 值替换 `replace`（键值对）与 `map`（函数映射）搭配使用，覆盖绝大多数映射需求。
