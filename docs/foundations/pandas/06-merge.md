---
title: Pandas 数据合并与连接
outline: deep
---

# Pandas 数据合并与连接

## 本章目标

1. 掌握 `pd.concat` 的行 / 列拼接。
2. 掌握数据库风格连接 `pd.merge` 的四种 `how` 模式（inner / left / right / outer）。
3. 学会用 `left_on` / `right_on` 合并不同列名的键。
4. 掌握 `DataFrame.join` 按索引的快捷连接。
5. 熟悉 `indicator` 参数用于诊断合并质量。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.concat(...)` | 函数 | 按行或列拼接多个 DataFrame |
| `pd.merge(...)` | 函数 | 数据库风格连接（按列键） |
| `df.merge(...)` | 方法 | 同 `pd.merge`，以 `self` 为左表 |
| `df.join(...)` | 方法 | 按**索引**连接（快捷写法） |
| `df.combine_first(...)` | 方法 | 用另一 DataFrame 的值补齐 `NaN` |
| `indicator` | 参数 | 合并结果中标记来源（`left_only` / `right_only` / `both`） |

## 按行 / 列拼接

### `pd.concat`

#### 作用

沿**现有轴**将多个 DataFrame / Series 拼接起来。与 NumPy 的 `concatenate` 类似，但会保留索引对齐。

#### 重点方法

```python
pd.concat(objs, axis=0, join='outer', ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          sort=False, copy=True)
```

#### 参数

| 参数名             | 本例取值                     | 说明                                                                   |
| ------------------ | ---------------------------- | ---------------------------------------------------------------------- |
| `objs`             | `[df1, df2]`                 | 待拼接的序列（列表或字典）                                             |
| `axis`             | `0`（默认）、`1`             | `0` 纵向（叠行），`1` 横向（拼列）                                     |
| `join`             | `'outer'`（默认）、`'inner'` | 非拼接轴的合并方式；`'outer'` 保留所有列，`'inner'` 只保留共有列       |
| `ignore_index`     | `False`（默认）、`True`      | 是否重置索引为 `RangeIndex`                                            |
| `keys`             | `None`（默认）、`['A', 'B']` | 给每个对象加一个顶层标签，生成 MultiIndex                              |
| `verify_integrity` | `False`（默认）              | 是否验证拼接后的索引没有重复                                           |
| `sort`             | `False`（默认）              | 当 `join='outer'` 且列不同时，是否对列排序                             |

#### 示例代码

```python
import pandas as pd

df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

print(f"纵向:\n{pd.concat([df1, df2], axis=0, ignore_index=True)}")
print(f"\n横向:\n{pd.concat([df1, df2], axis=1)}")

df3 = pd.DataFrame({"A": [1, 2], "C": [5, 6]})
print(f"\n不同列 outer:\n{pd.concat([df1, df3], ignore_index=True)}")
```

#### 输出

```text
纵向:
   A  B
0  1  3
1  2  4
2  5  7
3  6  8

横向:
   A  B  A  B
0  1  3  5  7
1  2  4  6  8

不同列 outer:
   A    B    C
0  1  3.0  NaN
1  2  4.0  NaN
2  1  NaN  5.0
3  2  NaN  6.0
```

#### 理解重点

- 纵向拼接用 `ignore_index=True` 重置行索引，否则会保留原索引（可能重复）。
- 横向拼接要求**索引对齐**；不对齐的行会出现 `NaN`（相当于 outer join）。
- 列不同时，缺失列会填 `NaN`；想只保留共有列用 `join='inner'`。

## 数据库风格连接

### `pd.merge`

#### 作用

按**列键**做数据库风格连接。支持 inner / left / right / outer 四种模式。

#### 重点方法

```python
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=False,
         suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
```

#### 参数

| 参数名           | 本例取值                                         | 说明                                                   |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------ |
| `left` / `right` | 两个 DataFrame                                   | 左表和右表                                             |
| `how`            | `'inner'`（默认）、`'left'`、`'right'`、`'outer'`、`'cross'` | 连接类型                                    |
| `on`             | `'dept_id'`                                      | 两表**共有**的连接键列名                               |
| `left_on`        | `'id'`                                           | 左表连接键                                             |
| `right_on`       | `'key'`                                          | 右表连接键                                             |
| `left_index`     | `False`（默认）                                  | 是否用左表索引作为连接键                               |
| `right_index`    | `False`（默认）                                  | 是否用右表索引作为连接键                               |
| `suffixes`       | `('_x', '_y')`（默认）                           | 同名非连接列的后缀                                     |
| `indicator`      | `False`（默认）、`True`、`'_merge'`              | 是否添加 `_merge` 列标记来源                           |
| `validate`       | `None`、`'1:1'`、`'1:m'`、`'m:1'`、`'m:m'`        | 验证连接键的基数关系（不符则抛错）                     |

### 四种 `how` 模式

| `how`      | 含义                            | 结果行数                          |
| ---------- | ------------------------------- | --------------------------------- |
| `'inner'`  | 只保留两表**共有**键            | 两表交集                          |
| `'left'`   | 保留**左表**所有行，右表匹配    | 左表 + 右表匹配（不匹配填 `NaN`） |
| `'right'`  | 保留**右表**所有行，左表匹配    | 右表 + 左表匹配                   |
| `'outer'`  | 保留**两表**所有行              | 并集                              |
| `'cross'`  | 笛卡儿积（Pandas 1.2+）          | `len(left) × len(right)`          |

### 综合示例

#### 示例代码

```python
import pandas as pd

employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "David"],
    "dept_id": [10, 20, 10, 30],
})

departments = pd.DataFrame({
    "dept_id": [10, 20, 40],
    "dept_name": ["Sales", "IT", "HR"],
})

print(f"inner:\n{pd.merge(employees, departments, on='dept_id')}")
print(f"\nleft:\n{pd.merge(employees, departments, on='dept_id', how='left')}")
print(f"\nright:\n{pd.merge(employees, departments, on='dept_id', how='right')}")
print(f"\nouter:\n{pd.merge(employees, departments, on='dept_id', how='outer')}")
```

#### 输出

```text
inner:
   emp_id     name  dept_id dept_name
0       1    Alice       10     Sales
1       3  Charlie       10     Sales
2       2      Bob       20        IT

left:
   emp_id     name  dept_id dept_name
0       1    Alice       10     Sales
1       2      Bob       20        IT
2       3  Charlie       10     Sales
3       4    David       30       NaN

right:
   emp_id     name  dept_id dept_name
0     1.0    Alice       10     Sales
1     3.0  Charlie       10     Sales
2     2.0      Bob       20        IT
3     NaN      NaN       40        HR

outer:
   emp_id     name  dept_id dept_name
0     1.0    Alice       10     Sales
1     3.0  Charlie       10     Sales
2     2.0      Bob       20        IT
3     4.0    David       30       NaN
4     NaN      NaN       40        HR
```

### 不同列名的键：`left_on` / `right_on`

#### 示例代码

```python
df1 = pd.DataFrame({"id": [1, 2, 3], "value1": ["a", "b", "c"]})
df2 = pd.DataFrame({"key": [1, 2, 4], "value2": ["x", "y", "z"]})

result = pd.merge(df1, df2, left_on="id", right_on="key", how="outer")
print(result)
```

#### 输出

```text
    id value1  key value2
0  1.0      a  1.0      x
1  2.0      b  2.0      y
2  3.0      c  NaN    NaN
3  NaN    NaN  4.0      z
```

### 合并诊断：`indicator`

#### 示例代码

```python
df1 = pd.DataFrame({"key": [1, 2, 3], "value": ["a", "b", "c"]})
df2 = pd.DataFrame({"key": [2, 3, 4], "value": ["x", "y", "z"]})

result = pd.merge(
    df1, df2, on="key", how="outer",
    suffixes=("_left", "_right"),
    indicator=True,
)
print(result)
```

#### 输出

```text
   key value_left value_right      _merge
0    1          a         NaN   left_only
1    2          b           x        both
2    3          c           y        both
3    4        NaN           z  right_only
```

### 理解重点

- `indicator=True` 添加 `_merge` 列：`left_only` / `right_only` / `both`，用于快速定位缺失匹配。
- `suffixes=('_left', '_right')` 区分两表同名列；默认 `('_x', '_y')` 不够直观。
- `validate='1:1'` 等参数会在不满足时抛错，用于保证数据质量。

## 按索引连接

### `DataFrame.join`

#### 作用

按**索引**（或左表的列 + 右表的索引）连接两个 DataFrame。是 `merge(..., left_index=True, right_index=True)` 的快捷写法。

#### 重点方法

```python
df.join(other, on=None, how='left', lsuffix='', rsuffix='',
        sort=False, validate=None)
```

#### 参数

| 参数名    | 本例取值                                          | 说明                                                                |
| --------- | ------------------------------------------------- | ------------------------------------------------------------------- |
| `other`   | 另一个 DataFrame / Series / DataFrame 列表        | 待连接对象                                                          |
| `on`      | `None`（默认）、左表列名                          | 用左表的列匹配 `other` 的索引                                       |
| `how`     | `'left'`（默认）、`'right'`、`'outer'`、`'inner'` | 连接类型；**默认是左连接**（与 `merge` 默认 `'inner'` 不同）        |
| `lsuffix` | `''`（默认）                                      | 同名列的左表后缀                                                    |
| `rsuffix` | `''`（默认）                                      | 同名列的右表后缀                                                    |
| `sort`    | `False`（默认）                                   | 是否按连接键排序                                                    |

### 示例代码

```python
df1 = pd.DataFrame({"A": [1, 2, 3]}, index=["a", "b", "c"])
df2 = pd.DataFrame({"B": [4, 5, 6]}, index=["a", "b", "d"])

print(f"join（默认 left）:\n{df1.join(df2)}")
print(f"\njoin(how='outer'):\n{df1.join(df2, how='outer')}")
```

### 输出

```text
join（默认 left）:
   A    B
a  1  4.0
b  2  5.0
c  3  NaN

join(how='outer'):
     A    B
a  1.0  4.0
b  2.0  5.0
c  3.0  NaN
d  NaN  6.0
```

### 理解重点

- `join` 默认 `how='left'`，与 `merge` 默认 `'inner'` 不同，容易踩坑。
- 多表同时按索引连接：`df.join([df2, df3, df4])` 比多次 `merge` 更简洁。

## `concat` vs `merge` vs `join`

| 方法       | 连接依据                | 默认方式  | 典型场景                             |
| ---------- | ----------------------- | --------- | ------------------------------------ |
| `concat`   | **轴方向**（索引 / 列） | `outer`   | 拼接多批相同结构的数据               |
| `merge`    | **列键**（或索引）      | `inner`   | 按业务键关联两个表（SQL join）       |
| `join`     | **索引**                | `left`    | 按索引快速合并（`merge` 的快捷写法） |

## 常见坑

1. `concat` 默认 `ignore_index=False`，结果行索引会有重复；数据拼接时**一律加 `ignore_index=True`**。
2. `merge` 的默认 `how='inner'` 会**丢弃不匹配的行**，要小心数据量意外减少；关键链路用 `how='left'` + `indicator=True` 诊断。
3. 合并后同名列自动加 `_x` / `_y` 后缀，可读性差——用 `suffixes=('_left', '_right')` 明确区分。
4. 多对多（`m:m`）合并会产生**笛卡儿积**，数据量急剧膨胀；用 `validate='1:m'` 等显式验证。
5. `join` 和 `merge` 的默认 `how` 不同（`'left'` vs `'inner'`），写代码时显式指定更安全。
6. `left_on` / `right_on` 合并后会**保留两个键列**，可能需要 `drop`。

## 小结

- **`concat`** 拼接结构相同的数据（训练集 / 验证集 / 测试集合并）。
- **`merge`** 是 SQL 风格连接，业务场景最常用；必考参数 `how` / `on` / `suffixes` / `indicator`。
- **`join`** 索引合并的快捷写法，多表同时合并尤其方便。
- 合并前后**务必检查行数**，用 `indicator=True` 可视化地看清匹配情况。
