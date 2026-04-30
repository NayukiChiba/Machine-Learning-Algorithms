---
title: Pandas 数据清洗与处理
outline: deep
---

# Pandas 数据清洗与处理

## 本章目标

1. 掌握缺失值的检测、删除与填充三种策略
2. 学会检测和删除重复行
3. 掌握 `astype`、`pd.to_numeric`、`pd.to_datetime` 三种类型转换
4. 熟悉 `.str` 访问器的字符串向量化操作
5. 掌握 `replace` 和 `map` 的值映射/替换

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.isnull()` / `df.isna()` | 方法 | 检测缺失值（两者等价） |
| `df.notnull()` / `df.notna()` | 方法 | 检测非缺失值 |
| `df.dropna(...)` | 方法 | 删除含缺失值的行/列 |
| `df.fillna(...)` | 方法 | 填充缺失值 |
| `df.duplicated(...)` | 方法 | 检测重复行 |
| `df.drop_duplicates(...)` | 方法 | 删除重复行 |
| `df.astype(...)` | 方法 | 转换列的数据类型 |
| `pd.to_numeric(...)` | 函数 | 安全转换为数值（错误可设为 NaN） |
| `pd.to_datetime(...)` | 函数 | 解析日期时间字符串 |
| `Series.str.xxx(...)` | 访问器 | 字符串向量化操作 |
| `Series.replace(...)` | 方法 | 值替换 |
| `Series.map(...)` | 方法 | 字典映射/函数应用 |

## 1. 缺失值检测

### `isnull` / `notnull`

#### 作用

- `isnull()`（别名 `isna()`）：逐元素判断是否为缺失值（`NaN` / `None` / `NaT`），返回同形状布尔 DataFrame
- `notnull()`（别名 `notna()`）：与 `isnull` 相反——逐元素判断是否为非缺失值

#### 重点方法

```python
df.isnull()
df.notnull()
df.isnull().sum()     # 每列缺失计数（常用组合）
df.isnull().mean()    # 每列缺失比例
```

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1, 2, np.nan, 4, 5],
    "B": [np.nan, 2.0, 3.0, np.nan, 5.0],
    "C": ["x", None, "y", "z", None],
})

print(f"原始数据:\n{df}")
print(f"\nisnull():\n{df.isnull()}")
print(f"\n每列缺失计数:\n{df.isnull().sum()}")
print(f"\n每列缺失比例:\n{df.isnull().mean().round(2)}")
print(f"\nnotnull():\n{df.notnull()}")
```

#### 输出

```text
原始数据:
     A    B     C
0  1.0  NaN     x
1  2.0  2.0  None
2  NaN  3.0     y
3  4.0  NaN     z
4  5.0  5.0  None

isnull():
       A      B      C
0  False   True  False
1  False  False   True
2   True  False  False
3  False   True  False
4  False  False   True

每列缺失计数:
A    1
B    2
C    2
dtype: int64

每列缺失比例:
A    0.2
B    0.4
C    0.4
dtype: float64

notnull():
       A      B      C
0   True  False   True
1   True   True  False
2  False   True   True
3   True  False   True
4   True   True  False
```

#### 理解重点

- `isnull().sum()` 是发现缺失值的最快方式——一行代码显示每列缺失计数
- `isnull().mean()` 直接得到缺失比例——适合设定缺失率阈值做列筛选
- `None` 和 `np.nan` 在 Pandas 中等价——`isnull()` 都能检测
- 布尔 DataFrame 可直接用于条件过滤：`df[df["A"].notnull()]`

## 2. 删除缺失值

### `DataFrame.dropna`

#### 作用

删除包含缺失值的行或列。通过 `axis`、`how`、`thresh`、`subset` 精细控制删除条件。

#### 重点方法

```python
df.dropna(*, axis=0, how='any', thresh=None, subset=None, inplace=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `axis` | `int` | `0` 删行、`1` 删列，默认为 `0` | `1` |
| `how` | `str` | `'any'` 有缺失即删、`'all'` 全缺失才删，默认为 `'any'` | `"all"` |
| `thresh` | `int` 或 `None` | 至少要有 N 个非缺失值才保留，默认为 `None` | `2` |
| `subset` | `list[str]` 或 `None` | 只在指定列上检测缺失，默认为 `None`（全列） | `["A", "B"]` |
| `inplace` | `bool` | 是否原地修改，默认为 `False` | `True` |

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, np.nan, 35, np.nan, 32],
    "Salary": [8000.0, 12000.0, np.nan, np.nan, 11000.0],
})

print(f"原始数据:\n{df}")
print(f"\ndropna() 默认删行:\n{df.dropna()}")
print(f"\ndropna(axis=1) 删列:\n{df.dropna(axis=1)}")
print(f"\ndropna(subset=['Age']) 只看 Age 列:\n{df.dropna(subset=['Age'])}")
print(f"\ndropna(thresh=2) 至少 2 个非缺失才保留:\n{df.dropna(thresh=2)}")
```

#### 输出

```text
原始数据:
      Name   Age   Salary
0    Alice  25.0   8000.0
1      Bob   NaN  12000.0
2  Charlie  35.0      NaN
3    David   NaN      NaN
4      Eve  32.0  11000.0

dropna() 默认删行:
    Name   Age   Salary
0  Alice  25.0   8000.0
4    Eve  32.0  11000.0

dropna(axis=1) 删列:
      Name
0    Alice
1      Bob
2  Charlie
3    David
4      Eve

dropna(subset=['Age']) 只看 Age 列:
      Name   Age   Salary
0    Alice  25.0   8000.0
2  Charlie  35.0      NaN
4      Eve  32.0  11000.0

dropna(thresh=2) 至少 2 个非缺失才保留:
      Name   Age   Salary
0    Alice  25.0   8000.0
1      Bob   NaN  12000.0
2  Charlie  35.0      NaN
4      Eve  32.0  11000.0
```

#### 理解重点

- `how='all'` 只在整行/列全 NaN 时删除——比 `how='any'` 保守
- `subset` 只在指定列判断缺失——其余列有 NaN 不影响保留
- `thresh` 比 `how` 更精细："每行至少要有 N 个有效值"

## 3. 填充缺失值

### `DataFrame.fillna`

#### 作用

用指定值或策略填充缺失值。支持常数值、前/后向填充、均值/中位数/众数填充、字典按列填充。

#### 重点方法

```python
df.fillna(value=None, *, method=None, axis=None, inplace=False,
          limit=None, downcast=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `value` | 标量、`dict`、`Series` | 填充值；`dict` 格式为 `{列名: 值}` | `0`、`{"A": 0, "B": 1}` |
| `method` | `str` 或 `None` | 填充策略：`'ffill'` 前向填充 / `'bfill'` 后向填充，与 `value` 互斥，默认为 `None` | `"ffill"` |
| `axis` | `int` | 填充方向轴，默认为 `None`（即 `0`） | `1` |
| `inplace` | `bool` | 是否原地修改，默认为 `False` | `True` |
| `limit` | `int` 或 `None` | 前/后向填充时最多连续填充几个 | `1` |

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1, np.nan, np.nan, 4, 5],
    "B": [np.nan, 2, 3, np.nan, 5],
    "C": ["x", None, "y", None, "z"],
})

print(f"原始数据:\n{df}")
print(f"\nfillna(0):\n{df.fillna(0)}")
print(f"\nffill 前向填充:\n{df.fillna(method='ffill')}")
print(f"\n按列填充 dict:\n{df.fillna({'A': -1, 'B': 99, 'C': 'missing'}))")
print(f"\n均值填充 A 列:\n{df.fillna({'A': df['A'].mean()})}")
```

#### 输出

```text
原始数据:
     A    B     C
0  1.0  NaN     x
1  NaN  2.0  None
2  NaN  3.0     y
3  4.0  NaN  None
4  5.0  5.0     z

fillna(0):
     A    B  C
0  1.0  0.0  x
1  0.0  2.0  0
2  0.0  3.0  y
3  4.0  0.0  0
4  5.0  5.0  z

ffill 前向填充:
     A    B  C
0  1.0  NaN  x
1  1.0  2.0  x
2  1.0  3.0  y
3  4.0  3.0  y
4  5.0  5.0  z

按列填充 dict:
     A     B        C
0  1.0  99.0        x
1 -1.0   2.0  missing
2 -1.0   3.0        y
3  4.0  99.0  missing
4  5.0   5.0        z

均值填充 A 列:
     A    B     C
0  1.0  NaN     x
1  3.33 2.0  None
2  3.33 3.0     y
3  4.0  NaN  None
4  5.0  5.0     z
```

#### 理解重点

- 常量填充最快但可能引入偏差——均值/中位数填充更保守
- `method='ffill'` 适合时间序列（假设值不会突变）
- `dict` 形式的 `value` 最灵活——每列可以有不同的填充策略
- `method` 和 `value` 互斥——不能同时使用

## 4. 重复值处理

### `duplicated` / `drop_duplicates`

#### 作用

- `duplicated()`：检测重复行，返回布尔 Series（首次出现标记为 `False`）
- `drop_duplicates()`：删除重复行，保留首次出现（或末次）

#### 重点方法

```python
df.duplicated(subset=None, keep='first')
df.drop_duplicates(subset=None, *, keep='first', inplace=False, ignore_index=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `subset` | `list[str]` 或 `None` | 只在指定列上判断重复，默认为 `None`（全列） | `["Name"]`、`["A", "B"]` |
| `keep` | `str` | 保留哪一行：`'first'` / `'last'` / `False`（全删），默认为 `'first'` | `"last"`、`False` |
| `inplace` | `bool` | 是否原地修改（仅 `drop_duplicates`），默认为 `False` | `True` |
| `ignore_index` | `bool` | 是否重置索引（仅 1.0+），默认为 `False` | `True` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
    "Age": [25, 30, 25, 30, 35],
    "City": ["Beijing", "Shanghai", "Beijing", "Shanghai", "Guangzhou"],
})

print(f"原始数据:\n{df}")
print(f"\nduplicated():\n{df.duplicated()}")
print(f"\ndrop_duplicates():\n{df.drop_duplicates()}")
print(f"\ndrop_duplicates(subset=['Name']):\n{df.drop_duplicates(subset=['Name'])}")
print(f"\ndrop_duplicates(keep='last'):\n{df.drop_duplicates(keep='last')}")
```

#### 输出

```text
原始数据:
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
2    Alice   25    Beijing
3      Bob   30   Shanghai
4  Charlie   35  Guangzhou

duplicated():
0    False
1    False
2     True
3     True
4    False
dtype: bool

drop_duplicates():
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
4  Charlie   35  Guangzhou

drop_duplicates(subset=['Name']):
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
4  Charlie   35  Guangzhou

drop_duplicates(keep='last'):
      Name  Age       City
2    Alice   25    Beijing
3      Bob   30   Shanghai
4  Charlie   35  Guangzhou
```

#### 理解重点

- `duplicated()` 可用于在删除前先确认重复情况——`df[df.duplicated()]` 查看重复行
- `keep=False` 删除**所有**重复行（包括首次出现）——适合完全去重
- `subset` 只在指定列判断——不同名字但有相同年龄的人不会被误删

## 5. 类型转换

### `DataFrame.astype`

#### 作用

将列转换为指定的数据类型。支持 NumPy dtype、Python 类型、Pandas 可空类型（`Int64` 等）。

#### 重点方法

```python
df.astype(dtype, copy=None, errors='raise')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `dtype` | `dtype`、`str`、`dict` | 目标类型；`dict` 格式为 `{列名: 类型}` | `int`、`"float32"`、`{"A": int, "B": float}` |
| `copy` | `bool` 或 `None` | 是否返回副本 | `True` |
| `errors` | `str` | 转换失败处理：`'raise'` / `'ignore'`，默认为 `'raise'` | `"ignore"` |

### `pd.to_numeric`

#### 作用

将 Series 安全转换为数值类型。转换失败的值可通过 `errors` 设为 NaN（`'coerce'`），比 `astype` 更安全。

#### 重点方法

```python
pd.to_numeric(arg, errors='raise', downcast=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arg` | `Series`、`list` | 输入数据 | `df["col"]` |
| `errors` | `str` | 错误处理：`'raise'` / `'coerce'`（转 NaN）/ `'ignore'`，默认为 `'raise'` | `"coerce"` |
| `downcast` | `str` 或 `None` | 整数/浮点降级：`'integer'` / `'float'`，默认为 `None` | `"integer"` |

### `pd.to_datetime`

#### 作用

将 Series 或标量解析为 `datetime64` 类型。支持多种日期格式自动推断，也可用 `format` 参数加速。

#### 重点方法

```python
pd.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False,
               utc=False, format=None, unit=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arg` | `Series`、`list`、`str` | 输入日期数据 | `df["date"]` |
| `errors` | `str` | 错误处理：`'raise'` / `'coerce'` / `'ignore'`，默认为 `'raise'` | `"coerce"` |
| `format` | `str` 或 `None` | 日期格式字符串，指定后解析更快 | `"%Y-%m-%d"` |
| `dayfirst` | `bool` | `True` 时优先解析为日/月顺序，默认为 `False` | `True` |
| `utc` | `bool` | 是否转为 UTC 时间，默认为 `False` | `True` |
| `unit` | `str` | 整数输入的时间单位：`'s'` / `'ms'` / `'ns'` | `"s"` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "ID": ["1", "2", "x", "4", "5"],
    "Price": ["10.5", "20.3", "bad", "40.1", "50.0"],
    "Date": ["2024-01-01", "2024-02-15", "invalid", "2024-04-20", "2024-05-30"],
})

print(f"原始类型:\n{df.dtypes}\n")

# astype：已知安全时直接用
df["ID_clean"] = pd.to_numeric(df["ID"], errors="coerce")
print(f"to_numeric (errors='coerce'):\n{df['ID_clean']}")

# to_numeric
df["Price_clean"] = pd.to_numeric(df["Price"], errors="coerce")
print(f"\nto_numeric (errors='coerce'):\n{df['Price_clean']}")

# to_datetime
df["Date_clean"] = pd.to_datetime(df["Date"], errors="coerce")
print(f"\nto_datetime (errors='coerce'):\n{df['Date_clean']}")
```

#### 输出

```text
原始类型:
ID       object
Price    object
Date     object
dtype: object

to_numeric (errors='coerce'):
0    1.0
1    2.0
2    NaN
3    4.0
4    5.0
Name: ID_clean, dtype: float64

to_numeric (errors='coerce'):
0    10.5
1    20.3
2     NaN
3    40.1
4    50.0
Name: Price_clean, dtype: float64

to_datetime (errors='coerce'):
0   2024-01-01
1   2024-02-15
2          NaT
3   2024-04-20
4   2024-05-30
Name: Date_clean, dtype: datetime64[ns]
```

#### 理解重点

- `astype` 直接转换——不合法值会抛异常；已知数据干净时用
- `to_numeric(errors='coerce')` 是清洗脏数据的安全网——非法值变 NaN，不中断流程
- `to_datetime(format=...)` 指定格式字符串可以禁掉自动推断——**大幅加速**大批量解析
- `pd.to_datetime` 在 [ch07 时间序列](07-timeseries.md) 中有更完整的日期时间处理

## 6. 字符串操作（`.str` 访问器）

### `.str` 访问器

#### 作用

通过 `Series.str.xxx()` 对字符串列执行向量化操作（一次操作整列，无需循环）。覆盖大小写转换、切片、包含检测、正则提取、拼接等 30+ 方法。

### 常用方法速览

| 方法 | 作用 | 示例 |
|---|---|---|
| `str.lower()` / `str.upper()` | 大小写转换 | `s.str.upper()` |
| `str.strip()` | 去除首尾空白 | `s.str.strip()` |
| `str.contains(pat)` | 正则匹配（返回布尔） | `s.str.contains("pat")` |
| `str.startswith(pat)` | 开头匹配 | `s.str.startswith("A")` |
| `str.replace(pat, repl)` | 替换子串/正则 | `s.str.replace("-", "_")` |
| `str.split(pat)` | 按分隔符拆分（返回列表列） | `s.str.split(",")` |
| `str.extract(pat)` | 正则提取捕获组 | `s.str.extract(r"(\d+)")` |
| `str.len()` | 字符串长度 | `s.str.len()` |
| `str.cat(sep=...)` | 拼接 Series 元素 | `s.str.cat(sep=", ")` |
| `str.slice(start, stop)` | 按位置切片 | `s.str.slice(0, 3)` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["  Alice  ", "BOB", "charlie", "DAVID", "eve  "],
    "Email": ["alice@co.com", "bob@co.com", "charlie@org.cn",
              "david@co.com", "eve@org.cn"],
    "Phone": ["010-1234", "021-5678", "010-9012", "0755-3456", "021-7890"],
})

# 大小写 + 去空格
df["NameClean"] = df["Name"].str.strip().str.title()
print(f"Name 清洗:\n{df[['Name', 'NameClean']]}")

# 正则包含
print(f"\n邮箱含 .com:\n{df[df['Email'].str.contains('.com')]}")

# 正则提取
df["Domain"] = df["Email"].str.extract(r"@(.+)")
print(f"\n域名提取:\n{df[['Email', 'Domain']]}")

# 拆分
df["AreaCode"] = df["Phone"].str.split("-").str[0]
print(f"\n区号拆分:\n{df[['Phone', 'AreaCode']]}")
```

#### 输出

```text
Name 清洗:
       Name NameClean
0    Alice     Alice
1       BOB       Bob
2  charlie   Charlie
3     DAVID     David
4     eve        Eve

邮箱含 .com:
    Name        Email      Phone
0  Alice   alice@co.com  010-1234
1    BOB     bob@co.com  021-5678
3  DAVID  david@co.com  0755-3456

域名提取:
       Email   Domain
0   alice@co.com   co.com
1     bob@co.com   co.com
2  charlie@org.cn  org.cn
3  david@co.com   co.com
4    eve@org.cn   org.cn

区号拆分:
       Phone AreaCode
0  010-1234      010
1  021-5678      021
2  010-9012      010
3  0755-3456     0755
4  021-7890      021
```

#### 理解重点

- `.str` 访问器**只能用于字符串列**——对数值列调用会抛异常
- `.str` 可链式调用：`df["col"].str.strip().str.lower().str.replace("_", "-")`
- `str.contains` 默认使用正则——搜索 `.` 等元字符需转义：`str.contains(r"\.com")`
- `str.extract` 用括号捕获组（`(...)`）提取子模式——多组返回多列

## 7. 值替换与映射

### `Series.replace`

#### 作用

将 Series 中的特定值替换为新值。支持单值替换、多值替换（列表）、字典映射。

#### 重点方法

```python
Series.replace(to_replace, value=None, *, inplace=False, regex=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `to_replace` | 标量、`list`、`dict`、正则 | 被替换值；`dict` 时为 `{旧值: 新值}` | `-1`、`[1, 2]`、`{"A": "a"}` |
| `value` | 标量、`list` 或 `None` | 替换值；`to_replace` 是 `dict` 时须为 `None` | `0`、`[10, 20]` |
| `regex` | `bool` | 是否将 `to_replace` 视为正则表达式，默认为 `False` | `True` |

### `Series.map`

#### 作用

将 Series 的每个值按字典或函数映射为新值。未映射的值变为 NaN（除非函数处理）。

#### 重点方法

```python
Series.map(arg, na_action=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arg` | `dict`、`Series`、函数 | 映射规则 | `{"cat": "猫", "dog": "狗"}`、`lambda x: x*2` |
| `na_action` | `str` 或 `None` | `'ignore'` 时跳过 NaN（不传入函数），默认为 `None` | `"ignore"` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Code": ["A", "B", "C", "A", "D"],
    "Score": [85, 92, 78, 90, -1],
})

# replace：将 -1（缺考标记）替换为 NaN
df["ScoreClean"] = df["Score"].replace(-1, np.nan)
print(f"replace(-1, NaN):\n{df[['Score', 'ScoreClean']]}")

# map：将编码映射为全称
codeMap = {"A": "Excellent", "B": "Good", "C": "Fair", "D": "Poor"}
df["Grade"] = df["Code"].map(codeMap)
print(f"\nmap 编码映射:\n{df[['Code', 'Grade']]}")

# replace dict 形式
df["ScoreCat"] = df["Score"].replace({
    85: "优秀", 92: "优秀", 78: "一般", 90: "优秀", -1: "缺考"
})
print(f"\nreplace dict 形式:\n{df[['Score', 'ScoreCat']]}")
```

#### 输出

```text
replace(-1, NaN):
   Score  ScoreClean
0     85        85.0
1     92        92.0
2     78        78.0
3     90        90.0
4     -1         NaN

map 编码映射:
  Code      Grade
0    A  Excellent
1    B       Good
2    C       Fair
3    A  Excellent
4    D       Poor

replace dict 形式:
   Score ScoreCat
0     85       优秀
1     92       优秀
2     78       一般
3     90       优秀
4     -1       缺考
```

#### 理解重点

- `replace` 和 `map` 的默认行为不同：`map` 对未映射值返回 NaN；`replace` 不匹配则保持原值
- `map(dict)` 适合编码→全称映射；`replace(dict)` 适合脏数据清洗
- `replace` 的 `regex=True` 可以做正则替换——比 `.str.replace` 更灵活
- `map` 可以传函数：`df["col"].map(lambda x: x * 2)`——等价于 `apply`

## 常见坑

1. `NaN` 不等于任何值（包括 NaN 自身）——用 `isnull()` 检测，不要用 `== np.nan`
2. `dropna()` 默认 `how='any'`——某行只要有一个 NaN 就被删除，容易误删大量数据
3. `fillna(method='ffill')` 和 `fillna(value=0)` 互斥——不能同时指定
4. `astype(int)` 遇到 NaN 会抛错——因为 int 不支持 NaN；先 `fillna` 或使用可空 `Int64` 类型
5. `.str` 访问器遇到 NaN 返回 NaN 而非报错——但 NaN 不参与后续字符串匹配
6. `map` 未匹配到的值变成 NaN——与 `replace` 行为不同（`replace` 不匹配保持原值）
7. `to_datetime` 不指定 `format` 时自动推断——大数据集应显式指定格式以加速 10~100 倍

## 小结

- 缺失值处理的标准流程：`isnull().sum()` 评估 → `fillna` 或 `dropna` 处理
- 类型转换的安全选择：`pd.to_numeric(errors='coerce')` / `pd.to_datetime(errors='coerce')`
- `.str` 访问器是 Pandas 字符串操作的统一入口——无需写循环
- `replace` 和 `map` 功能相似但默认行为不同：`map` 未匹配变 NaN，`replace` 保持原值
