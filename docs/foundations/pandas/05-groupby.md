---
title: Pandas 数据分组与聚合
outline: deep
---

# Pandas 数据分组与聚合

> 对应脚本：`Basic/Pandas/05_groupby.py`  
> 运行方式：`python Basic/Pandas/05_groupby.py`（仓库根目录）

## 本章目标

1. 理解 GroupBy 的 "分组 → 应用 → 合并" 三步模型。
2. 掌握常用聚合函数（`sum`、`mean`、`count`、`agg`）。
3. 学会对不同列使用不同聚合函数。
4. 理解 `transform` 和 `apply` 的区别与使用场景。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `df.groupby(by)` | 按列分组 | `demo_groupby` |
| `grouped.sum()` / `.mean()` / `.count()` | 单一聚合 | `demo_agg` |
| `grouped.agg(...)` | 多函数/多列聚合 | `demo_agg`、`demo_multi_column_agg` |
| `grouped.transform(func)` | 广播回原形状 | `demo_transform` |
| `grouped.apply(func)` | 自定义分组操作 | `demo_apply` |

## 示例数据

本章所有 demo 使用同一个示例 DataFrame：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Bonus": [1000, 1200, 1500, 1400, 800, 900],
    "Years": [3, 5, 4, 6, 2, 3],
})
```

## 1. GroupBy 基本操作

### 方法重点

- `groupby()` 返回分组对象（惰性计算），不会立即执行。
- `ngroups` 属性查看分组数量。
- 遍历分组对象会得到 `(组名, 子DataFrame)` 元组。

### 参数速览（本节）

适用 API：`df.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, dropna=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `by` | `"Department"` | 按此列分组 |
| `as_index` | `True`（默认） | 分组键作为结果索引 |
| `sort` | `True`（默认） | 按分组键排序 |
| `dropna` | `True`（默认） | 分组键为 NaN 的行被丢弃 |

### 示例代码

```python
grouped = df.groupby("Department")
print(type(grouped))
print(f"分组数量: {grouped.ngroups}")

for name, group in grouped:
    print(f"\n--- {name} ---")
    print(group)
```

### 结果输出

```text
<class 'pandas.core.groupby.generic.DataFrameGroupBy'>
分组数量: 3
----------------
--- HR ---
  Department Employee  Salary  Bonus  Years
4         HR      Eve    7000    800      2
5         HR    Frank    7500    900      3

--- IT ---
  Department Employee  Salary  Bonus  Years
2         IT  Charlie   12000   1500      4
3         IT    David   11000   1400      6

--- Sales ---
  Department Employee  Salary  Bonus  Years
0      Sales    Alice    8000   1000      3
1      Sales      Bob    9000   1200      5
```

### 理解重点

- `groupby` 是惰性操作，真正的计算发生在调用聚合函数时。
- 分组结果默认按分组键排序，传 `sort=False` 可以保持原始顺序。
- 可以按多列分组：`df.groupby(["col1", "col2"])`。

## 2. 聚合函数

### 方法重点

- 常用内置聚合：`sum`、`mean`、`count`、`max`、`min`、`std`。
- `agg()` 可以同时应用多个聚合函数。

### 参数速览（本节）

1. `grouped["col"].sum()` / `.mean()` / `.count()` — 单一聚合

2. `grouped["col"].agg(func_list)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `func_list` | `["sum", "mean", "max"]` | 同时应用多个聚合函数 |

### 示例代码

```python
grouped = df.groupby("Department")

print(grouped["Salary"].sum())
print(grouped["Salary"].mean())
print(grouped["Employee"].count())
print(grouped["Salary"].agg(["sum", "mean", "max"]))
```

### 结果输出

```text
sum():
Department
HR        14500
IT        23000
Sales     17000
Name: Salary, dtype: int64
----------------
mean():
Department
HR         7250.0
IT        11500.0
Sales      8500.0
Name: Salary, dtype: float64
----------------
count():
Department
HR       2
IT       2
Sales    2
Name: Employee, dtype: int64
----------------
agg(['sum', 'mean', 'max']):
              sum     mean    max
Department
HR          14500   7250.0   7500
IT          23000  11500.0  12000
Sales       17000   8500.0   9000
```

### 理解重点

- `count()` 统计非 NaN 值数量，`size()` 统计包含 NaN 的总行数。
- `agg()` 返回的列名为聚合函数名，可通过 `columns` 重命名。

## 3. 多列聚合

### 方法重点

- 可以对不同列指定不同的聚合函数。
- 字典形式 `agg({"col1": "sum", "col2": ["mean", "max"]})` 最灵活。
- 命名聚合支持自定义结果列名。

### 参数速览（本节）

1. `grouped.agg(dict)` — 字典形式

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `dict` | `{"Salary": ["mean", "sum"], "Bonus": "sum", "Years": "mean"}` | 不同列不同聚合 |

2. `grouped["col"].agg(**kwargs)` — 命名聚合

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `total` | `("sum")` | 自定义列名 total |
| `average` | `("mean")` | 自定义列名 average |
| `range` | `lambda x: x.max() - x.min()` | 自定义聚合函数 |

### 示例代码

```python
grouped = df.groupby("Department")

# 不同列不同聚合函数
result = grouped.agg({
    "Salary": ["mean", "sum"],
    "Bonus": "sum",
    "Years": "mean",
})
print(result)

# 命名聚合
result = grouped["Salary"].agg(
    total=("sum"),
    average=("mean"),
    range=lambda x: x.max() - x.min(),
)
print(result)
```

### 结果输出

```text
不同列不同聚合:
            Salary         Bonus Years
              mean    sum   sum  mean
Department
HR          7250.0  14500  1700   2.5
IT         11500.0  23000  2900   5.0
Sales       8500.0  17000  2200   4.0
----------------
命名聚合:
            total  average  range
Department
HR          14500   7250.0    500
IT          23000  11500.0   1000
Sales       17000   8500.0   1000
```

### 理解重点

- 字典聚合产生多级列索引（MultiIndex columns），可通过 `columns.droplevel()` 简化。
- `lambda` 自定义聚合时，输入是该组的 Series。
- 命名聚合语法简洁且结果列名可控，是推荐的方式。

## 4. Transform 方法

### 方法重点

- `transform` 的核心特点：返回与原 DataFrame **等长**的结果。
- 常用于"在原数据上添加分组统计列"的场景。
- 典型用例：分组均值、分组标准化（z-score）。

### 参数速览（本节）

适用 API：`grouped["col"].transform(func)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `func` | `"mean"`、`lambda x: (x - x.mean()) / x.std()` | 聚合函数或自定义函数 |

### 示例代码

```python
# 添加部门平均工资列
df["Dept_Mean_Salary"] = df.groupby("Department")["Salary"].transform("mean")
print(df)

# 分组标准化 (z-score)
df["Salary_Zscore"] = df.groupby("Department")["Salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(df[["Department", "Employee", "Salary", "Salary_Zscore"]])
```

### 结果输出

```text
添加部门平均工资列:
  Department Employee  Salary  Bonus  Years  Dept_Mean_Salary
0      Sales    Alice    8000   1000      3            8500.0
1      Sales      Bob    9000   1200      5            8500.0
2         IT  Charlie   12000   1500      4           11500.0
3         IT    David   11000   1400      6           11500.0
4         HR      Eve    7000    800      2            7250.0
5         HR    Frank    7500    900      3            7250.0
----------------
标准化分数:
  Department Employee  Salary  Salary_Zscore
0      Sales    Alice    8000      -0.707107
1      Sales      Bob    9000       0.707107
2         IT  Charlie   12000       0.707107
3         IT    David   11000      -0.707107
4         HR      Eve    7000      -0.707107
5         HR    Frank    7500       0.707107
```

### 理解重点

- `transform` vs `agg`：`agg` 将每组缩减为一行，`transform` 保持原始行数。
- `transform` 的结果可直接赋值为新列，常用于特征工程。
- 组内只有一个值时，`std()` 返回 `NaN`，z-score 会全变成 `NaN`。

## 5. Apply 方法

### 方法重点

- `apply` 是最灵活的分组操作，可以返回任意形状的结果。
- 比 `agg` 和 `transform` 更通用，但速度更慢。
- 自定义函数的输入是每个组的 DataFrame。

### 参数速览（本节）

适用 API：`grouped.apply(func, include_groups=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `func` | `top_employee`、`summary` | 自定义函数 |
| `include_groups` | `False` | 不将分组键列传入函数 |

### 示例代码

```python
grouped = df.groupby("Department")

# 每组薪资最高的员工
def top_employee(group):
    return group.nlargest(1, "Salary")

print(grouped.apply(top_employee, include_groups=False))

# 自定义汇总
def summary(group):
    return pd.Series({
        "count": len(group),
        "total_salary": group["Salary"].sum(),
        "avg_years": group["Years"].mean(),
    })

print(grouped.apply(summary, include_groups=False))
```

### 结果输出

```text
每个部门薪资最高的员工:
              Employee  Salary  Bonus  Years
Department
HR         5    Frank    7500    900      3
IT         2  Charlie   12000   1500      4
Sales      1      Bob    9000   1200      5
----------------
自定义汇总:
            count  total_salary  avg_years
Department
HR            2.0       14500.0        2.5
IT            2.0       23000.0        5.0
Sales         2.0       17000.0        4.0
```

### 理解重点

- `apply` 的灵活性来自于：函数可以返回标量、Series 或 DataFrame。
- 能用 `agg`/`transform` 解决的场景不要用 `apply`，后者效率更低。
- `include_groups=False` 避免分组键被当作数据传入函数。

## 常见坑

1. `groupby` 后直接看 `grouped` 不会显示数据——它是惰性对象。
2. `agg` 字典形式产生 MultiIndex 列，可能导致后续列访问困难。
3. `transform` 的函数必须返回与输入等长的结果，否则报错。
4. `apply` 可能触发双重调用（第一次用于推断返回类型），自定义函数不要有副作用。

## 小结

- GroupBy 的核心模型是 "Split → Apply → Combine"。
- `agg` 用于聚合缩减，`transform` 用于广播回原形状，`apply` 用于自定义逻辑。
- 优先使用 `agg` / `transform`，它们更快且语义更明确。
