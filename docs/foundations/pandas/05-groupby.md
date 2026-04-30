---
title: Pandas 分组与聚合
outline: deep
---

# Pandas 分组与聚合

## 本章目标

1. 理解 `groupby` 的 Split-Apply-Combine 思维模型
2. 掌握单列聚合、多列不同聚合、命名聚合三种写法
3. 区分 `agg`、`transform`、`apply` 三者的返回形状与适用场景
4. 学会用 `transform` 做组内标准化，用 `filter` 做组级过滤

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.groupby(...)` | 方法 | 按列分组，返回 `GroupBy` 对象 |
| `gb.agg(...)` | 方法 | 聚合运算，返回**每组一行**的结果 |
| `gb.transform(...)` | 方法 | 组内变换，返回**与原数据同长度**的结果 |
| `gb.apply(...)` | 方法 | 自定义组级运算，返回任意形状 |
| `gb.filter(...)` | 方法 | 按组级别条件过滤，保留整个组 |
| `gb.size()` | 方法 | 每组行数（含 NaN） |
| `gb.count()` | 方法 | 每组非空计数（按列） |
| `gb.ngroups` | 属性 | 分组总数 |
| `gb.groups` | 属性 | 分组字典 `{key: [indices]}` |

## 1. 分组（Split-Apply-Combine）

Split-Apply-Combine 是 `groupby` 的核心思想：

1. **Split**：按指定列/函数把数据切成多个组
2. **Apply**：对每组独立应用聚合/变换/过滤函数
3. **Combine**：把各组结果合并成最终输出

### `DataFrame.groupby`

#### 作用

按一个或多个列将 DataFrame 分组，返回 `GroupBy` 对象。分组本身**不执行计算**——只有后续调用 `agg` / `transform` / `apply` 等方法才会触发。

#### 重点方法

```python
df.groupby(by=None, axis=0, level=None, as_index=True, sort=True,
           group_keys=True, observed=False, dropna=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `by` | `str`、`list[str]`、`Series`、函数 | 分组依据；多个列用列表 | `"Department"`、`["A", "B"]` |
| `axis` | `int` | 分组轴，默认为 `0`（按行分组） | `1` |
| `level` | `int`、`str` 或 `None` | 多级索引时按哪一级分组，默认为 `None` | `0` |
| `as_index` | `bool` | 分组键是否作为结果索引，默认为 `True`；`False` 时保留为普通列 | `False` |
| `sort` | `bool` | 是否对分组键排序，默认为 `True`；`False` 可加速 | `False` |
| `group_keys` | `bool` | `apply` 时是否把分组键加入结果索引，默认为 `True` | `True` |
| `observed` | `bool` | 对 `category` 列是否只保留出现过的分类，默认为 `False` | `True` |
| `dropna` | `bool` | 是否丢弃分组键含 NaN 的行，默认为 `True` | `False` |

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Bonus": [1000, 1200, 1500, 1400, 800, 900],
    "Years": [3, 5, 4, 6, 2, 3],
})

grouped = df.groupby("Department")
print(f"分组数: {grouped.ngroups}")
print(f"分组键: {list(grouped.groups.keys())}")

# 遍历分组
for name, group in grouped:
    print(f"\n--- {name} ---")
    print(group)
```

#### 输出

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

#### 理解重点

- `groupby` 返回的是**懒对象**——直接打印看不到数据，只有计算结果才触发运算
- 遍历 `GroupBy` 对象可得到 `(分组名, 子DataFrame)` 元组
- `as_index=False` 或 `reset_index()` 可将分组键变回普通列，便于后续连接/存储

## 2. 聚合

### `GroupBy.agg`

#### 作用

对每组执行**聚合函数**（返回标量），结果每组一行。支持多种聚合模式：

- 单个函数：`agg("mean")`、`agg(np.sum)`
- 函数列表：`agg(["sum", "mean", "max"])`
- 按列不同聚合：`agg({"Salary": "mean", "Bonus": "sum"})`
- 命名聚合（推荐）：`agg(total=("Salary", "sum"), avg=("Salary", "mean"))`

#### 重点方法

```python
gb.agg(func=None, *args, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `str`、`list`、`dict`、函数 | 聚合函数，字符串形式最常用 | `"mean"`、`["sum", "std"]` |
| `*args` | 传给 `func` 的额外参数 | 如 `agg(np.quantile, q=0.9)` | — |
| `**kwargs` | 命名聚合 | `新列名 = ("原列名", "聚合函数")` | `total=("Salary", "sum")` |

#### 常用内置聚合函数

| 字符串 | 含义 | 字符串 | 含义 |
|---|---|---|---|
| `"sum"` | 求和 | `"mean"` | 均值 |
| `"median"` | 中位数 | `"min"` / `"max"` | 极值 |
| `"std"` / `"var"` | 标准差/方差 | `"count"` | 非空计数 |
| `"size"` | 行数（含 NaN） | `"first"` / `"last"` | 组内首/末值 |
| `"nunique"` | 唯一值个数 | `"sem"` | 均值的标准误差 |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Bonus": [1000, 1200, 1500, 1400, 800, 900],
    "Years": [3, 5, 4, 6, 2, 3],
})

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
    bonusSum=("Bonus", "sum"),
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
            total  average  bonusSum
Department
HR          14500   7250.0      1700
IT          23000  11500.0      2900
Sales       17000   8500.0      2200
```

#### 理解重点

- **命名聚合** `agg(新名=("列", "函数"))` 是 Pandas 0.25+ 的推荐写法——结果列名干净可控
- 字典方式 `agg({"col": "fn"})` 在多函数时产生 MultiIndex 列，后续处理麻烦
- 聚合结果默认以分组键为索引；用 `as_index=False` 或 `.reset_index()` 让其变回列

## 3. 组内变换

### `GroupBy.transform`

#### 作用

对每组应用函数，返回**与原数据同长度**的结果（每行都有值）。适合组内标准化、组内排名、组内均值填充。

#### 重点方法

```python
gb.transform(func, *args, engine=None, engine_kwargs=None, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `str`、函数 | 变换函数；必须返回与输入同长度的值 | `"mean"`、`"rank"`、lambda |
| `*args` | 传给 `func` 的额外参数 | — | — |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Years": [3, 5, 4, 6, 2, 3],
})

# 组内均值（给原数据添加"部门平均工资"列）
df["DeptMeanSalary"] = df.groupby("Department")["Salary"].transform("mean")

# 组内 Z-score 标准化
df["SalaryZscore"] = df.groupby("Department")["Salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)

print(df[["Department", "Employee", "Salary", "DeptMeanSalary", "SalaryZscore"]])
```

#### 输出

```text
  Department Employee  Salary  DeptMeanSalary       SalaryZscore
0      Sales    Alice    8000          8500.0  -0.707107
1      Sales      Bob    9000          8500.0   0.707107
2         IT  Charlie   12000         11500.0   0.707107
3         IT    David   11000         11500.0  -0.707107
4         HR      Eve    7000          7250.0  -0.707107
5         HR    Frank    7500          7250.0   0.707107
```

#### 数学公式

组内 Z-score 标准化：

$$
z_i = \frac{x_i - \mu_g}{\sigma_g}
$$

其中 $\mu_g$ 是组 $g$ 的均值，$\sigma_g$ 是组 $g$ 的标准差。

#### 理解重点

- `agg` 返回每组一行；`transform` 返回每行一个值——**长度不变**是核心区别
- 典型场景：给原数据添加"组内均值/排名/累计值"列，作为特征工程的一部分
- 返回值会自动对齐原数据的索引，可直接赋为新列

## 4. 自定义应用

### `GroupBy.apply`

#### 作用

对每组应用自定义函数，**返回任意形状**（Series / DataFrame / 标量），Pandas 自动根据返回类型合并结果。是 `groupby` 中最灵活、但也最慢的方法。

#### 重点方法

```python
gb.apply(func, *args, include_groups=True, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | 函数 | 接收 group (DataFrame)，返回任意对象 | `lambda g: g.nlargest(1, "Salary")` |
| `include_groups` | `bool` | 分组键列是否传入函数；Pandas 2.2+ 推荐设 `False`，默认为 `True` | `False` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Bonus": [1000, 1200, 1500, 1400, 800, 900],
    "Years": [3, 5, 4, 6, 2, 3],
})

grouped = df.groupby("Department")

# 每组工资最高的员工
def topEmployee(group):
    return group.nlargest(1, "Salary")

print("每组最高工资:\n")
print(grouped.apply(topEmployee, include_groups=False))

# 自定义每组多字段汇总
def summary(group):
    return pd.Series({
        "count": len(group),
        "totalSalary": group["Salary"].sum(),
        "avgYears": group["Years"].mean(),
    })

print(f"\n自定义汇总:\n{grouped.apply(summary, include_groups=False)}")
```

#### 输出

```text
每组最高工资:

              Employee  Salary  Bonus  Years
Department
HR         5     Frank    7500    900      3
IT         2   Charlie   12000   1500      4
Sales      1       Bob    9000   1200      5

自定义汇总:
            count  totalSalary  avgYears
Department
HR            2.0       14500.0       2.5
IT            2.0       23000.0       5.0
Sales         2.0       17000.0       4.0
```

#### 理解重点

- `apply` 最灵活但也**最慢**——能用 `agg` / `transform` 就不要用 `apply`
- 第一组可能被 Pandas 调用**两次**用于类型推断——函数必须是**纯函数**、无副作用
- 返回 `Series` 会自动展开成列；返回 `DataFrame` 按组堆叠；返回标量得到单值

## 5. 组级过滤

### `GroupBy.filter`

#### 作用

按组级别的条件过滤——保留（或丢弃）**整个组**，而不是过滤单行。结果是与原数据形状相同的子集。

#### 重点方法

```python
gb.filter(func, dropna=True, *args, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | 函数（返回 `bool`） | 接收 group，返回 `True` 保留 / `False` 丢弃 | `lambda g: g.Salary.mean() > 8000` |
| `dropna` | `bool` | 是否丢弃函数返回 NaN 的组，默认为 `True` | `False` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Years": [3, 5, 4, 6, 2, 3],
})

# 保留平均工资 > 8000 的部门
result = df.groupby("Department").filter(lambda g: g["Salary"].mean() > 8000)
print(result[["Department", "Employee", "Salary"]])
```

#### 输出

```text
  Department Employee  Salary
0      Sales    Alice    8000
1      Sales      Bob    9000
2         IT  Charlie   12000
3         IT    David   11000
```

#### 理解重点

- `filter` 在组级别做判断，保留**整组所有行**；`df[df.Salary > 8000]` 在行级别做判断——语义不同
- 典型场景：只分析"样本量足够多"的组（`len(g) >= 10`）；剔除"数据不完整"的组

## 6. 附加方法与属性

### 速览

| 方法/属性 | 类型 | 返回 | 说明 |
|---|---|---|---|
| `gb.size()` | 方法 | `Series` | 每组行数（含 NaN），比 `count` 快 |
| `gb.count()` | 方法 | `DataFrame` | 按列的非空计数 |
| `gb.ngroups` | 属性 | `int` | 分组总数 |
| `gb.groups` | 属性 | `dict` | `{分组键: [行索引列表]}` |
| `gb.indices` | 属性 | `dict` | 同 `groups`，低级别接口 |
| `gb.first()` / `gb.last()` | 方法 | `DataFrame` | 每组的首/末行 |

## 7. 三大方法对比

| 方法 | 函数返回 | 结果形状 | 适用场景 |
|---|---|---|---|
| `agg` | 标量 | 每组一行 | 汇总统计（均值、总和、计数） |
| `transform` | 同长度 Series | 与原数据同长度 | 组内标准化、排名、均值填充、累计值 |
| `apply` | 任意 | 任意（自动推断） | 自由度最高——当 agg/transform 不够用时 |
| `filter` | 布尔 | 原数据子集 | 按组级条件保留/丢弃整组 |

## 常见坑

1. `groupby` 返回**懒对象**，不触发计算——直接打印看不到数据，必须调用 `agg` 等方法才会执行
2. 字典形式 `agg({"col": "fn"})` 在多函数时产生 MultiIndex 列——**优先使用命名聚合** `agg(新名=("列", "函数"))`
3. `apply` 的第一组可能被调用两次（Pandas 推断返回类型）——函数应该是**纯函数**，无副作用
4. 多列分组 `groupby(["A", "B"])` 结果索引是 MultiIndex——后续处理用 `reset_index()`
5. `transform` 的函数必须返回**与输入同长度**的值——返回标量会被自动广播，可能产生意外结果
6. `groupby(..., dropna=True)`（默认）会**丢弃**分组键为 NaN 的行——需要保留时显式设 `dropna=False`

## 小结

- 分组聚合的思维模型是 **Split-Apply-Combine**——先分、再算、后合
- 写代码的优先级：`agg` > `transform` > `apply`——选功能够用中最快的那个
- 聚合时推荐**命名聚合**语法——结果列名清晰可控
- `transform` 配合 `groupby` 是特征工程（组内统计特征）的利器
