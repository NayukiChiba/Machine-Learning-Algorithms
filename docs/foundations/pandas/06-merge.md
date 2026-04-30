---
title: Pandas 数据合并与连接
outline: deep
---

# Pandas 数据合并与连接

## 本章目标

1. 掌握 `pd.concat` 的行拼接与列拼接
2. 掌握 `pd.merge` 的四种连接模式（inner / left / right / outer）
3. 学会用 `left_on` / `right_on` 合并不同列名的键
4. 掌握 `df.join` 按索引的快捷连接
5. 学会用 `indicator` 参数诊断合并质量

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.concat(...)` | 函数 | 按行或列拼接多个 DataFrame |
| `pd.merge(...)` | 函数 | 数据库风格连接（类似 SQL JOIN） |
| `df.join(...)` | 方法 | 按索引连接（`merge` 的索引版快捷方式） |
| `df.combine_first(...)` | 方法 | 用另一个 DataFrame 填充缺失值 |

## 1. 行列拼接

### `pd.concat`

#### 作用

沿指定轴将多个 DataFrame 或 Series 拼接在一起。`axis=0` 纵向堆叠（加行），`axis=1` 横向拼接（加列）。

#### 重点方法

```python
pd.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None,
          verify_integrity=False, sort=False, copy=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `objs` | `list[DataFrame]`、`dict` | 待拼接对象序列；`dict` 时键用作多层索引 | `[df1, df2]` |
| `axis` | `int` | `0` 按行拼接（上下堆叠）、`1` 按列拼接（左右拼接），默认为 `0` | `1` |
| `join` | `str` | 非拼接轴上索引对齐方式：`'outer'` 并集 / `'inner'` 交集，默认为 `'outer'` | `"inner"` |
| `ignore_index` | `bool` | `True` 时重置索引，默认为 `False` | `True` |
| `keys` | `list` | 为每组数据添加标签（形成多层索引的顶层），默认为 `None` | `["df1", "df2"]` |
| `verify_integrity` | `bool` | `True` 时检查结果索引是否有重复，默认为 `False` | `True` |
| `sort` | `bool` | `axis=1` 时是否对非拼接轴排序，默认为 `False` | `True` |

#### 示例代码

```python
import pandas as pd

df1 = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Age": [25, 30],
    "City": ["Beijing", "Shanghai"],
})

df2 = pd.DataFrame({
    "Name": ["Charlie", "David"],
    "Age": [35, 28],
    "City": ["Guangzhou", "Shenzhen"],
})

# 按行拼接
rows = pd.concat([df1, df2], axis=0, ignore_index=True)
print(f"axis=0 按行拼接:\n{rows}")

# 按列拼接
left = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
right = pd.DataFrame({"City": ["Beijing", "Shanghai"], "Score": [85, 92]})
cols = pd.concat([left, right], axis=1)
print(f"\naxis=1 按列拼接:\n{cols}")
```

#### 输出

```text
axis=0 按行拼接:
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
2  Charlie   35  Guangzhou
3    David   28   Shenzhen

axis=1 按列拼接:
    Name  Age      City  Score
0  Alice   25   Beijing     85
1    Bob   30  Shanghai     92
```

#### 理解重点

- `ignore_index=True` 重置为连续整数索引——避免拼接后出现重复索引
- `keys` 参数可标记每组数据来源：`pd.concat([df1, df2], keys=["A", "B"])` 创建 MultiIndex
- `join='inner'` 只保留所有 DataFrame 共有的列（按列拼接时）或行（按行拼接时）

## 2. 数据库风格合并

### `pd.merge`

#### 作用

类似 SQL JOIN，按指定列（键）将两个 DataFrame 的行对齐合并。支持 inner / left / right / outer 四种连接模式。

#### 重点方法

```python
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
         indicator=False, validate=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `left` | `DataFrame` | 左侧 DataFrame | `df1` |
| `right` | `DataFrame` | 右侧 DataFrame | `df2` |
| `how` | `str` | 连接方式：`'inner'` / `'left'` / `'right'` / `'outer'`，默认为 `'inner'` | `"left"` |
| `on` | `str`、`list[str]` | 两侧同名的连接键列名；`None` 时自动用交集列名 | `"ID"`、`["A", "B"]` |
| `left_on` | `str`、`list[str]` | 左侧连接键（两侧列名不同时使用） | `"left_id"` |
| `right_on` | `str`、`list[str]` | 右侧连接键（与 `left_on` 配合） | `"right_id"` |
| `left_index` | `bool` | `True` 时用左侧索引作为连接键，默认为 `False` | `True` |
| `right_index` | `bool` | `True` 时用右侧索引作为连接键，默认为 `False` | `True` |
| `suffixes` | `tuple[str, str]` | 同名列（非键）的后缀，默认为 `('_x', '_y')` | `('_L', '_R')` |
| `indicator` | `bool`、`str` | `True` 时新增 `_merge` 列标记每行的来源，默认为 `False` | `True` |
| `validate` | `str` 或 `None` | 验证连接键的唯一性：`'one_to_one'` / `'one_to_many'` / `'many_to_one'` / `'many_to_many'`，默认为 `None` | `"one_to_one"` |

### 四种 `how` 模式

| `how` | 行为 | SQL 类比 |
|---|---|---|
| `"inner"` | 只保留两侧都匹配的行 | `INNER JOIN` |
| `"left"` | 保留左侧所有行，右侧无匹配的填充 NaN | `LEFT JOIN` |
| `"right"` | 保留右侧所有行，左侧无匹配的填充 NaN | `RIGHT JOIN` |
| `"outer"` | 保留两侧所有行，无匹配的填充 NaN | `FULL OUTER JOIN` |

### 综合示例

#### 示例代码

```python
import pandas as pd

employees = pd.DataFrame({
    "ID": [1, 2, 3, 4],
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Dept": ["Sales", "IT", "IT", "HR"],
})

salaries = pd.DataFrame({
    "ID": [1, 2, 3, 5],
    "Salary": [8000, 12000, 15000, 9000],
})

print("员工表:")
print(employees)
print(f"\n薪资表:")
print(salaries)

# inner：只保留两边都有的 ID
print(f"\ninner merge (交集):")
print(pd.merge(employees, salaries, on="ID", how="inner"))

# left：保留所有员工，缺薪资的填 NaN
print(f"\nleft merge (保留所有员工):")
print(pd.merge(employees, salaries, on="ID", how="left"))

# outer：保留所有 ID
print(f"\nouter merge (全外连接):")
print(pd.merge(employees, salaries, on="ID", how="outer"))

# 带 indicator 的 outer
print(f"\nouter + indicator:")
print(pd.merge(employees, salaries, on="ID", how="outer", indicator=True))
```

#### 输出

```text
员工表:
   ID     Name  Dept
0   1    Alice Sales
1   2      Bob    IT
2   3  Charlie    IT
3   4    David    HR

薪资表:
   ID  Salary
0   1    8000
1   2   12000
2   3   15000
3   5    9000

inner merge (交集):
   ID     Name  Dept  Salary
0   1    Alice Sales    8000
1   2      Bob    IT   12000
2   3  Charlie    IT   15000

left merge (保留所有员工):
   ID     Name  Dept   Salary
0   1    Alice Sales   8000.0
1   2      Bob    IT  12000.0
2   3  Charlie    IT  15000.0
3   4    David    HR      NaN

outer merge (全外连接):
   ID     Name  Dept   Salary
0   1    Alice Sales   8000.0
1   2      Bob    IT  12000.0
2   3  Charlie    IT  15000.0
3   4    David    HR      NaN
4   5      NaN   NaN   9000.0

outer + indicator:
   ID     Name  Dept   Salary      _merge
0   1    Alice Sales   8000.0        both
1   2      Bob    IT  12000.0        both
2   3  Charlie    IT  15000.0        both
3   4    David    HR      NaN   left_only
4   5      NaN   NaN   9000.0  right_only
```

#### 理解重点

- `how='left'` 是最常用的合并模式——以主表为基准，补充辅助信息
- `left_on` / `right_on` 在两侧键列名不同时使用——列名无需统一
- `indicator=True` 生成 `_merge` 列标记每行来源——用于合并质量诊断
- `validate` 在预期一对一/一对多时加验证——防止意外的多对多产生重复行
- 多键合并：`on=["A", "B"]` 同时按两列匹配

## 3. 按索引连接

### `DataFrame.join`

#### 作用

按索引连接两个 DataFrame，是 `pd.merge(left_index=True, right_index=True)` 的便捷封装。可同时按索引+列连接。

#### 重点方法

```python
df.join(other, on=None, how='left', lsuffix='', rsuffix='',
        sort=False, validate=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `other` | `DataFrame`、`Series`、`list` | 要连接的 DataFrame | `df2` |
| `on` | `str`、`list[str]` 或 `None` | 主表用列键（从表仍用索引），默认为 `None` | `"ID"` |
| `how` | `str` | 连接方式，同 `merge`，默认为 `'left'` | `"inner"`、`"outer"` |
| `lsuffix` | `str` | 左侧同名列的后缀，默认为 `''` | `"_L"` |
| `rsuffix` | `str` | 右侧同名列的后缀，默认为 `''` | `"_R"` |
| `validate` | `str` 或 `None` | 同 `merge` 的键验证，默认为 `None` | `"one_to_one"` |

#### 示例代码

```python
import pandas as pd

df1 = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
}, index=["a", "b", "c"])

df2 = pd.DataFrame({
    "City": ["Beijing", "Shanghai", "Guangzhou"],
    "Score": [85, 92, 78],
}, index=["a", "b", "d"])

# join 按索引（默认 left）
print(f"df1.join(df2):\n{df1.join(df2)}")

# inner join
print(f"\ndf1.join(df2, how='inner'):\n{df1.join(df2, how='inner')}")
```

#### 输出

```text
df1.join(df2):
      Name  Age       City  Score
a    Alice   25    Beijing   85.0
b      Bob   30   Shanghai   92.0
c  Charlie   35        NaN    NaN

df1.join(df2, how='inner'):
    Name  Age      City  Score
a  Alice   25   Beijing     85
b    Bob   30  Shanghai     92
```

#### 理解重点

- `join` 的核心场景：主表用普通列，从表已经把键设为了索引
- `on` + `join` 组合可比 `merge` 更简洁：`df.set_index("key").join(other.set_index("key"))`

## 4. 缺失值补齐

### `DataFrame.combine_first`

#### 作用

用另一个 DataFrame 的值填充当前 DataFrame 的缺失值（NaN）。按索引和列名对齐，只在当前值为 NaN 时填充。

#### 重点方法

```python
df.combine_first(other)
```

#### 示例代码

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame({
    "A": [1, np.nan, 3],
    "B": [np.nan, 5, np.nan],
}, index=[0, 1, 2])

df2 = pd.DataFrame({
    "A": [10, 20, 30],
    "B": [40, 50, 60],
}, index=[0, 1, 2])

print(f"df1 (有缺失):\n{df1}")
print(f"\ndf2 (备用值):\n{df2}")
print(f"\ncombine_first:\n{df1.combine_first(df2)}")
```

#### 输出

```text
df1 (有缺失):
     A    B
0  1.0  NaN
1  NaN  5.0
2  3.0  NaN

df2 (备用值):
    A   B
0  10  40
1  20  50
2  30  60

combine_first:
     A    B
0  1.0  40.0
1  20.0  5.0
2  3.0  60.0
```

#### 理解重点

- `combine_first` 只在单元格为 NaN 时才填充——已有值不会被覆盖
- 按索引和列名同时对齐——两侧行列标签不必完全相同（多余的行列会保留）

## 连接方法选择指南

| 场景 | 推荐方法 |
|---|---|
| 多个相同结构数据上下堆叠 | `pd.concat(axis=0)` |
| 左右拼接（按位置对齐） | `pd.concat(axis=1)` |
| 按列值匹配合并 | `pd.merge(on=...)` |
| 按索引匹配合并 | `df.join(...)` |
| 用备用数据填充 NaN | `df.combine_first(...)` |

## 常见坑

1. `pd.merge` 不指定 `on` 时自动用所有共同列名作为键——可能无意中将多列合并导致结果行数爆炸
2. 多对多合并会产生笛卡尔积——用 `validate` 参数提前验证键的唯一性
3. `concat` 默认保留原始索引——行拼接后可能出现重复索引，用 `ignore_index=True` 重置
4. `join` 默认 `how='left'`，而 `merge` 默认 `how='inner'`——行为不一致，容易误用
5. 合并后同名列（非键）被加后缀 `_x` / `_y`——用 `suffixes` 自定义有意义的后缀
6. `combine_first` 要求索引和列名对齐——数据来源结构不同时需先调整

## 小结

- 纯堆叠用 `concat`；按键匹配用 `merge`；按索引匹配用 `join`
- `merge` 的四种 `how` 对应 SQL 四种 JOIN——`'left'` 最常用，以主表为基准
- `indicator` 和 `validate` 是合并质量诊断的两个关键参数
- 合并后务必检查行数——意外的笛卡尔积是最常见的合并 bug
