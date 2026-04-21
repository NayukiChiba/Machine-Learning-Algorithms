---
title: Pandas 分组与聚合
outline: deep
---

# Pandas 分组与聚合

## 本章目标

1. 理解 `groupby` 的"split-apply-combine"思维模型。
2. 掌握单列聚合与多列不同聚合的写法。
3. 区分 `agg`、`transform`、`apply` 三者的语义与返回形状。
4. 学会用 `agg` 做命名聚合生成规整结果。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.groupby(...)` | 方法 | 按列 / 条件分组，返回 `GroupBy` 对象 |
| `gb.agg(...)` | 方法 | 对分组做聚合，返回**每组一行**的结果 |
| `gb.transform(...)` | 方法 | 对分组做变换，返回**与原数据同长度**的 Series/DataFrame |
| `gb.apply(...)` | 方法 | 对每组应用自定义函数，返回任意形状 |
| `gb.filter(...)` | 方法 | 按组级别条件过滤（保留整个组） |
| `gb.size()` | 方法 | 每组行数（含 NaN） |
| `gb.count()` | 方法 | 每组非空计数（按列） |
| `gb.ngroups` | 属性 | 分组数量 |
| `gb.groups` | 属性 | 分组字典 `{key: [indices]}` |

## 示例数据

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

## 分组的"SAC"模型

**Split-Apply-Combine** 是 `groupby` 的核心思想：

1. **Split**：按指定列 / 函数把数据切成多个组
2. **Apply**：对每组独立应用聚合 / 变换 / 过滤函数
3. **Combine**：把每组结果合并成最终输出

## 分组

### `DataFrame.groupby`

#### 作用

按一个或多个列（或函数、Series、数组）将 DataFrame 分组，返回 `GroupBy` 对象。分组本身**不执行计算**，只有后续调用 `agg` / `transform` / `apply` 等才会触发。

#### 重点方法

```python
df.groupby(by=None, axis=0, level=None, as_index=True, sort=True,
           group_keys=True, observed=False, dropna=True)
```

#### 参数

| 参数名       | 本例取值                                     | 说明                                                                   |
| ------------ | -------------------------------------------- | ---------------------------------------------------------------------- |
| `by`         | `"Department"`、`["A", "B"]`、函数、Series    | 分组依据                                                               |
| `axis`       | `0`（默认）                                  | 分组轴；几乎总是 `0`                                                   |
| `level`      | `None`                                       | 多级索引时按哪一级分组                                                 |
| `as_index`   | `True`（默认）                               | `True` 时分组键作为结果索引；`False` 时保留为列                        |
| `sort`       | `True`（默认）                               | 是否对分组键排序；`False` 可加速                                       |
| `group_keys` | `True`（默认）                               | `apply` 时是否把分组键加入结果层级索引                                 |
| `observed`   | `False`（默认）                              | 对 `category` 列是否只保留出现过的分类                                 |
| `dropna`     | `True`（默认）                               | 是否丢弃分组键含 `NaN` 的行                                            |

### 遍历分组

`GroupBy` 对象可迭代，每次得到 `(group_name, group_df)` 元组。

### 示例代码

```python
grouped = df.groupby("Department")
print(f"分组数: {grouped.ngroups}")
print(f"分组键: {list(grouped.groups.keys())}")

for name, group in grouped:
    print(f"\n--- {name} ---")
    print(group)
```

### 输出

```text
分组数: 3
分组键: ['HR', 'IT', 'Sales']

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

## 聚合

### `GroupBy.agg`

#### 作用

对每组执行**聚合函数**（返回一个标量），结果是"每组一行"。支持：
- 单个函数：`agg('mean')`、`agg(np.sum)`、`agg(lambda x: ...)`
- 函数列表：`agg(['sum', 'mean', 'max'])`
- 按列不同聚合：`agg({'Salary': 'mean', 'Bonus': 'sum'})`
- 命名聚合：`agg(total=('Salary', 'sum'), avg=('Salary', 'mean'))`

#### 重点方法

```python
gb.agg(func=None, *args, **kwargs)
# 同义方法
gb.aggregate(func=None, *args, **kwargs)
```

#### 参数

| 参数名     | 本例取值                                         | 说明                                                       |
| ---------- | ------------------------------------------------ | ---------------------------------------------------------- |
| `func`     | `'mean'`、`['sum', 'mean']`、字典、命名聚合       | 聚合函数规则                                               |
| `*args`    | 传给 `func` 的额外位置参数                       | 例如 `agg(np.quantile, q=0.9)`                             |
| `**kwargs` | 命名聚合：`total=('Salary', 'sum')`              | 新列名 = `(原列名, 聚合函数)`                              |

### 常用聚合函数

| 函数                 | 作用                   |
| -------------------- | ---------------------- |
| `'sum'`              | 求和                   |
| `'mean'`             | 均值                   |
| `'median'`           | 中位数                 |
| `'min'` / `'max'`    | 极值                   |
| `'std'` / `'var'`    | 标准差 / 方差          |
| `'count'`            | 非空计数               |
| `'size'`             | 行数（含 NaN）         |
| `'first'` / `'last'` | 组内首 / 末值          |
| `'nunique'`          | 唯一值个数             |

### 综合示例

#### 示例代码

```python
grouped = df.groupby("Department")

# 单列单函数
print(f"Salary.sum():\n{grouped['Salary'].sum()}")

# 单列多函数
print(f"\nSalary.agg(['sum', 'mean', 'max']):")
print(grouped["Salary"].agg(["sum", "mean", "max"]))

# 多列不同函数
print(f"\n多列不同聚合:")
print(grouped.agg({
    "Salary": ["mean", "sum"],
    "Bonus": "sum",
    "Years": "mean",
}))

# 命名聚合（推荐）
print(f"\n命名聚合:")
print(grouped.agg(
    total=("Salary", "sum"),
    average=("Salary", "mean"),
    bonus_sum=("Bonus", "sum"),
))
```

#### 输出

```text
Salary.sum():
Department
HR       14500
IT       23000
Sales    17000
Name: Salary, dtype: int64

Salary.agg(['sum', 'mean', 'max']):
              sum    mean    max
Department
HR          14500  7250.0   7500
IT          23000 11500.0  12000
Sales       17000  8500.0   9000

多列不同聚合:
              Salary          Bonus Years
                mean    sum     sum  mean
Department
HR            7250.0  14500    1700   2.5
IT           11500.0  23000    2900   5.0
Sales         8500.0  17000    2200   4.0

命名聚合:
            total  average  bonus_sum
Department
HR          14500   7250.0       1700
IT          23000  11500.0       2900
Sales       17000   8500.0       2200
```

#### 理解重点

- **命名聚合** `agg(新名=('列', '函数'))` 是 Pandas 0.25+ 的推荐写法，结果列名干净可控。
- 字典方式 `agg({'col': 'fn'})` 在多函数时产生 MultiIndex 列，后续处理麻烦。
- `as_index=False` 可让分组键作为普通列而不是索引。

## 变换

### `GroupBy.transform`

#### 作用

对每组应用函数，返回**与原数据同长度**的结果（每行都有值）。适合**组内标准化**、**组内累计**、**组内填充**等场景。

#### 重点方法

```python
gb.transform(func, *args, engine=None, engine_kwargs=None, **kwargs)
```

#### 参数

| 参数名  | 本例取值                            | 说明                                                              |
| ------- | ----------------------------------- | ----------------------------------------------------------------- |
| `func`  | `'mean'`、`'rank'`、lambda          | 变换函数；必须返回与输入同长度的 Series                           |
| `*args` | 传给 `func` 的额外参数              | —                                                                 |

### 示例代码

```python
# 添加"部门平均工资"列
df["Dept_Mean_Salary"] = df.groupby("Department")["Salary"].transform("mean")

# 组内 Z-score 标准化
df["Salary_Zscore"] = df.groupby("Department")["Salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)

print(df[["Department", "Employee", "Salary", "Dept_Mean_Salary", "Salary_Zscore"]])
```

### 输出

```text
  Department Employee  Salary  Dept_Mean_Salary  Salary_Zscore
0      Sales    Alice    8000            8500.0      -0.707107
1      Sales      Bob    9000            8500.0       0.707107
2         IT  Charlie   12000           11500.0       0.707107
3         IT    David   11000           11500.0      -0.707107
4         HR      Eve    7000            7250.0      -0.707107
5         HR    Frank    7500            7250.0       0.707107
```

### 理解重点

- `agg` 返回每组一行；`transform` 返回每行一个值（**长度不变**）。
- 典型场景：给原数据增加"组内均值 / 排名 / 标准化值"列。
- 组内 Z-score 标准化是特征工程中非常常见的操作。

## 自由度最高的 `apply`

### `GroupBy.apply`

#### 作用

对每组应用函数，**返回任意形状**（Series / DataFrame / 标量）。Pandas 会自动根据返回类型 combine 结果。

#### 重点方法

```python
gb.apply(func, *args, include_groups=True, **kwargs)
```

#### 参数

| 参数名            | 本例取值                        | 说明                                                              |
| ----------------- | ------------------------------- | ----------------------------------------------------------------- |
| `func`            | 自定义函数                      | 接收一个 group (DataFrame)，返回任意对象                          |
| `include_groups`  | `True`（默认）、`False`          | 分组键列是否传入函数；Pandas 2.2+ 推荐显式设 `False`              |

### 示例代码

```python
grouped = df.groupby("Department")

# 每个部门工资最高的员工
def top_employee(group):
    return group.nlargest(1, "Salary")

print(grouped.apply(top_employee, include_groups=False))

# 自定义每组多字段汇总
def summary(group):
    return pd.Series({
        "count": len(group),
        "total_salary": group["Salary"].sum(),
        "avg_years": group["Years"].mean(),
    })

print(grouped.apply(summary, include_groups=False))
```

### 输出

```text
              Employee  Salary  Bonus  Years
Department
HR         5     Frank    7500    900      3
IT         2   Charlie   12000   1500      4
Sales      1       Bob    9000   1200      5

            count  total_salary  avg_years
Department
HR            2.0       14500.0        2.5
IT            2.0       23000.0        5.0
Sales         2.0       17000.0        4.0
```

### 理解重点

- `apply` 是最灵活的工具，但**最慢**——能用 `agg` / `transform` 就不要用 `apply`。
- 函数返回 `Series` 会自动展开成列；返回 `DataFrame` 会按组堆叠；返回标量会得到单值 Series。
- 第一组可能被 Pandas 调用**两次**用于类型推断，函数应该是**纯函数**、无副作用。

## 过滤

### `GroupBy.filter`

#### 作用

按**组级别**的条件过滤，保留（或丢弃）整个组。与按行过滤不同，`filter` 的返回值是**原 DataFrame 的子集**（不是每组一行）。

#### 重点方法

```python
gb.filter(func, dropna=True, *args, **kwargs)
```

#### 参数

| 参数名   | 本例取值       | 说明                                             |
| -------- | -------------- | ------------------------------------------------ |
| `func`   | 返回布尔值的函数 | 接收 group，返回 `True` 保留 / `False` 丢弃    |
| `dropna` | `True`（默认）| 是否丢弃条件函数返回 `NaN` 的组                  |

### 示例代码

```python
# 保留平均工资 > 8000 的部门
result = df.groupby("Department").filter(lambda g: g["Salary"].mean() > 8000)
print(result[["Department", "Employee", "Salary"]])
```

### 输出

```text
  Department Employee  Salary
0      Sales    Alice    8000
1      Sales      Bob    9000
2         IT  Charlie   12000
3         IT    David   11000
```

### 理解重点

- `filter` 在 group 级别做判断；`df[df.Salary > 8000]` 在行级别做判断，语义不同。
- 典型场景：只分析"样本数足够多"的组（如 `g['id'].count() >= 10`）。

## 三大方法对比

| 方法        | 函数返回       | 结果形状                 | 用途                              |
| ----------- | -------------- | ------------------------ | --------------------------------- |
| `agg`       | 标量           | 每组一行                 | 聚合统计                          |
| `transform` | 同长度 Series  | 与原数据同长度            | 组内标准化 / 排名 / 累计 / 填充   |
| `apply`     | 任意           | 任意（自动推断）          | 自由度最高，但最慢                |
| `filter`    | 布尔           | 原数据子集（整组保留）    | 按组级条件过滤                    |

## 常见坑

1. `groupby` 返回的是**懒对象**，不触发计算；打印 `GroupBy` 对象不会看到数据。
2. 字典形式 `agg({'col': 'fn'})` 在多函数时会产生 MultiIndex 列，**优先使用命名聚合**。
3. `apply` 的第一组可能被调用两次（Pandas 内部推断返回类型），函数应该是**纯函数**，无副作用。
4. 多列分组 `groupby(['A', 'B'])` 的结果索引是 MultiIndex，后续处理要 `reset_index()`。
5. `transform` 的函数必须返回**与输入同长度**的值；返回标量会被自动广播。
6. `groupby(..., dropna=True)`（默认）会**丢弃**分组键为 `NaN` 的行；需要保留应显式设 `dropna=False`。
7. 用 `as_index=False` 或 `reset_index()` 将分组键变回普通列，便于后续连接 / 存储。

## 小结

- 分组聚合是数据分析的核心工具，思维模型是 **Split-Apply-Combine**。
- 写代码的优先级：`agg` > `transform` > `apply`——选功能够用的最快的那个。
- 聚合时推荐**命名聚合**语法，结果列名清晰可控。
- `transform` 配合 `groupby` 是特征工程（组内统计特征）的利器。
