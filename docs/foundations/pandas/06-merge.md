---
title: Pandas 数据合并与连接
outline: deep
---

# Pandas 数据合并与连接

> 对应脚本：`Basic/Pandas/06_merge.py`  
> 运行方式：`python Basic/Pandas/06_merge.py`（仓库根目录）

## 本章目标

1. 掌握 `pd.concat` 的纵向和横向合并。
2. 理解 `pd.merge` 的四种连接方式（inner/left/right/outer）。
3. 学会处理不同列名的合并（`left_on` / `right_on`）。
4. 理解 `df.join` 基于索引的连接方式。
5. 学会使用 `indicator` 参数追踪数据来源。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `pd.concat(objs, axis)` | 纵向/横向拼接 | `demo_concat` |
| `pd.merge(left, right, on, how)` | 数据库风格连接 | `demo_merge` |
| `pd.merge(..., left_on, right_on)` | 不同列名合并 | `demo_merge_on_different_keys` |
| `df.join(other, how)` | 基于索引连接 | `demo_join` |
| `pd.merge(..., indicator=True)` | 标记数据来源 | `demo_merge_indicator` |

## 1. Concat 合并

### 方法重点

- `concat` 是最基础的拼接操作，按轴方向堆叠 DataFrame。
- `axis=0` 纵向拼接（堆叠行），`axis=1` 横向拼接（堆叠列）。
- 列名不完全一致时，默认外连接（缺失位置填 NaN）。

### 参数速览（本节）

适用 API：`pd.concat(objs, axis=0, join='outer', ignore_index=False, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `objs` | `[df1, df2]` | 待合并的 DataFrame 列表 |
| `axis` | `0`、`1` | `0` 纵向，`1` 横向 |
| `join` | `'outer'`（默认） | `'outer'` 保留所有列，`'inner'` 只保留公共列 |
| `ignore_index` | `False`、`True` | 是否重置索引 |

### 示例代码

```python
import pandas as pd

df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

# 纵向合并
print(pd.concat([df1, df2], axis=0, ignore_index=True))

# 横向合并
print(pd.concat([df1, df2], axis=1))

# 不同列合并
df3 = pd.DataFrame({"A": [1, 2], "C": [5, 6]})
print(pd.concat([df1, df3], ignore_index=True))
```

### 结果输出

```text
纵向合并:
   A  B
0  1  3
1  2  4
2  5  7
3  6  8
----------------
横向合并:
   A  B  A  B
0  1  3  5  7
1  2  4  6  8
----------------
不同列合并 (外连接):
   A    B    C
0  1  3.0  NaN
1  2  4.0  NaN
2  1  NaN  5.0
3  2  NaN  6.0
```

### 理解重点

- 纵向合并时 `ignore_index=True` 很重要，否则索引会重复。
- 不同列合并默认外连接，缺失列填 `NaN`。
- `concat` 不做匹配连接，只是简单堆叠。

## 2. Merge 连接

### 方法重点

- `merge` 类似 SQL 的 JOIN 操作，基于公共键匹配行。
- 四种连接方式：`inner`（交集）、`left`、`right`、`outer`（并集）。
- 默认 `how='inner'`，只保留两表都有的键。

### 参数速览（本节）

适用 API：`pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, suffixes=('_x', '_y'), ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `left` / `right` | `employees` / `departments` | 左表和右表 |
| `on` | `"dept_id"` | 连接键（两表同名列） |
| `how` | `"inner"`、`"left"`、`"right"`、`"outer"` | 连接方式 |
| `suffixes` | `('_x', '_y')`（默认） | 重名列的后缀 |

### 示例代码

```python
employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "David"],
    "dept_id": [10, 20, 10, 30],
})

departments = pd.DataFrame({
    "dept_id": [10, 20, 40],
    "dept_name": ["Sales", "IT", "HR"],
})

# 内连接（默认）
print(pd.merge(employees, departments, on="dept_id"))

# 左连接
print(pd.merge(employees, departments, on="dept_id", how="left"))

# 右连接
print(pd.merge(employees, departments, on="dept_id", how="right"))

# 外连接
print(pd.merge(employees, departments, on="dept_id", how="outer"))
```

### 结果输出

```text
内连接:
   emp_id     name  dept_id dept_name
0       1    Alice       10     Sales
1       3  Charlie       10     Sales
2       2      Bob       20        IT
----------------
左连接:
   emp_id     name  dept_id dept_name
0       1    Alice       10     Sales
1       2      Bob       20        IT
2       3  Charlie       10     Sales
3       4    David       30       NaN
----------------
右连接:
   emp_id     name  dept_id dept_name
0     1.0    Alice       10     Sales
1     3.0  Charlie       10     Sales
2     2.0      Bob       20        IT
3     NaN      NaN       40        HR
----------------
外连接:
   emp_id     name  dept_id dept_name
0     1.0    Alice       10     Sales
1     3.0  Charlie       10     Sales
2     2.0      Bob       20        IT
3     4.0    David       30       NaN
4     NaN      NaN       40        HR
```

### 理解重点

- **内连接**只保留两表公共键的行，丢失不匹配数据。
- **左连接**保留左表所有行，右表无匹配的填 `NaN`。
- **外连接**保留两表所有行，是最完整但可能有最多 `NaN` 的结果。
- 类比 SQL：`inner` = `INNER JOIN`，`left` = `LEFT JOIN`，`outer` = `FULL OUTER JOIN`。

## 3. 不同列名合并

### 方法重点

- 当两表的连接键列名不同时，使用 `left_on` / `right_on`。
- 合并后两个键列都会保留，通常需要手动删除多余的一列。

### 参数速览（本节）

适用 API：`pd.merge(left, right, left_on=..., right_on=..., how=...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `left_on` | `"id"` | 左表的连接键列名 |
| `right_on` | `"key"` | 右表的连接键列名 |
| `how` | `"outer"` | 外连接 |

### 示例代码

```python
df1 = pd.DataFrame({"id": [1, 2, 3], "value1": ["a", "b", "c"]})
df2 = pd.DataFrame({"key": [1, 2, 4], "value2": ["x", "y", "z"]})

result = pd.merge(df1, df2, left_on="id", right_on="key", how="outer")
print(result)
```

### 结果输出

```text
    id value1  key value2
0  1.0      a  1.0      x
1  2.0      b  2.0      y
2  3.0      c  NaN    NaN
3  NaN    NaN  4.0      z
```

### 理解重点

- `left_on` 和 `right_on` 必须成对使用。
- 合并后 `id` 和 `key` 都存在，可用 `drop()` 删除多余列。
- 也可以用 `left_index=True` / `right_index=True` 基于索引合并。

## 4. Join 操作

### 方法重点

- `join` 是 DataFrame 的实例方法，默认基于**索引**连接。
- 默认是左连接（`how='left'`）。
- 语义上等价于 `merge` 加 `left_index=True, right_index=True`。

### 参数速览（本节）

适用 API：`df1.join(other, on=None, how='left', lsuffix='', rsuffix='', ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `other` | `df2` | 待连接的 DataFrame |
| `how` | `'left'`（默认）、`'outer'` | 连接方式 |
| `lsuffix` / `rsuffix` | `''`（默认） | 重名列后缀 |

### 示例代码

```python
df1 = pd.DataFrame({"A": [1, 2, 3]}, index=["a", "b", "c"])
df2 = pd.DataFrame({"B": [4, 5, 6]}, index=["a", "b", "d"])

# 左连接
print(df1.join(df2))

# 外连接
print(df1.join(df2, how="outer"))
```

### 结果输出

```text
左连接:
   A    B
a  1  4.0
b  2  5.0
c  3  NaN
----------------
外连接:
     A    B
a  1.0  4.0
b  2.0  5.0
c  3.0  NaN
d  NaN  6.0
```

### 理解重点

- `join` 基于索引，`merge` 基于列——这是两者的核心区别。
- 当两表有同名列时，必须指定 `lsuffix` / `rsuffix`，否则报错。
- 多数情况下 `merge` 更灵活，`join` 在索引对齐场景下更简洁。

## 5. 合并指示器

### 方法重点

- `indicator=True` 在结果中添加 `_merge` 列，标记每行的数据来源。
- `suffixes` 参数处理两表同名列的命名冲突。
- 指示器常用于数据质量检查：发现只存在于一侧的记录。

### 参数速览（本节）

适用 API：`pd.merge(..., indicator=True, suffixes=('_left', '_right'))`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `indicator` | `True` | 添加 `_merge` 列 |
| `suffixes` | `("_left", "_right")` | 同名列的后缀 |

`_merge` 列取值：

| 值 | 含义 |
|---|---|
| `left_only` | 只存在于左表 |
| `right_only` | 只存在于右表 |
| `both` | 两表都存在 |

### 示例代码

```python
df1 = pd.DataFrame({"key": [1, 2, 3], "value": ["a", "b", "c"]})
df2 = pd.DataFrame({"key": [2, 3, 4], "value": ["x", "y", "z"]})

result = pd.merge(
    df1, df2, on="key", how="outer",
    suffixes=("_left", "_right"), indicator=True,
)
print(result)
```

### 结果输出

```text
   key value_left value_right      _merge
0    1          a         NaN   left_only
1    2          b           x        both
2    3          c           y        both
3    4        NaN           z  right_only
```

### 理解重点

- `indicator` 是调试合并结果的利器，能快速定位不匹配的数据。
- `_merge` 列的值是 Categorical 类型，可以直接过滤：`result[result["_merge"] == "left_only"]`。
- `suffixes` 只在两表有同名非连接列时生效。

## 常见坑

1. `concat` 纵向合并后索引重复，导致后续 `loc` 索引结果不唯一。
2. `merge` 时连接键有重复值，会产生笛卡尔积，结果行数可能远超预期。
3. 忘记指定 `how` 参数，默认内连接可能丢失数据。
4. `join` 遇到同名列不指定后缀会直接报错。

## 小结

- `concat` 用于简单堆叠，`merge` 用于匹配连接——两者功能互补。
- 理解四种连接方式（inner/left/right/outer）是关键，类比 SQL JOIN 有助于记忆。
- `indicator` 参数是数据合并质量检查的好帮手。
