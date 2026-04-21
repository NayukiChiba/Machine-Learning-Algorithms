---
title: Pandas 高级操作与性能优化
outline: deep
---

# Pandas 高级操作与性能优化

## 本章目标

1. 掌握透视表 `pivot_table` 与交叉表 `crosstab`。
2. 理解多级索引 `MultiIndex` 的创建与选择。
3. 体会向量化 vs 循环的性能差距。
4. 掌握内存优化技巧（`astype` 降精度、`category` 类型）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.pivot_table(...)` | 函数 | 透视表（可聚合） |
| `df.pivot(...)` | 方法 | 简单透视（不聚合，仅重塑） |
| `pd.crosstab(...)` | 函数 | 交叉频数表 |
| `pd.MultiIndex.from_arrays(...)` | 类方法 | 从数组创建多级索引 |
| `pd.MultiIndex.from_tuples(...)` | 类方法 | 从元组创建多级索引 |
| `df.stack(...)` / `df.unstack(...)` | 方法 | 长宽表相互转换 |
| `df.melt(...)` | 方法 | 宽表 → 长表 |
| `df.memory_usage(...)` | 方法 | 查看内存占用 |
| `astype('category')` | 方法 | 转换为分类类型以节省内存 |

## 透视表

### `pd.pivot_table`

#### 作用

从长表生成透视表，指定行、列、值与聚合函数。类似 Excel 数据透视表。

#### 重点方法

```python
pd.pivot_table(data, values=None, index=None, columns=None,
               aggfunc='mean', fill_value=None, margins=False,
               dropna=True, margins_name='All', observed=False, sort=True)
```

#### 参数

| 参数名        | 本例取值                    | 说明                                                                |
| ------------- | --------------------------- | ------------------------------------------------------------------- |
| `data`        | DataFrame                   | 源数据                                                              |
| `values`      | `"Sales"`                   | 被聚合的数值列；`None` 时聚合所有数值列                             |
| `index`       | `"Date"`、`["A", "B"]`      | 作为行索引的列                                                      |
| `columns`     | `"Region"`                  | 作为列索引的列                                                      |
| `aggfunc`     | `'mean'`（默认）、`'sum'`、`['mean', 'sum']`、字典 | 聚合函数                                       |
| `fill_value`  | `None`（默认）、`0`         | 空值填充                                                            |
| `margins`     | `False`（默认）、`True`     | 是否添加行 / 列小计（显示"All"）                                    |
| `margins_name`| `'All'`（默认）              | 小计行 / 列的名字                                                   |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Date": ["2023-01", "2023-01", "2023-02", "2023-02"],
    "Region": ["North", "South", "North", "South"],
    "Sales": [100, 150, 120, 180],
    "Quantity": [10, 15, 12, 18],
})

pivot = pd.pivot_table(df, values="Sales", index="Date",
                       columns="Region", aggfunc="sum")
print(f"透视表:\n{pivot}")

# 带小计
pivot_m = pd.pivot_table(df, values="Sales", index="Date",
                          columns="Region", aggfunc="sum", margins=True)
print(f"\n带小计:\n{pivot_m}")
```

### 输出

```text
透视表:
Region   North  South
Date
2023-01    100    150
2023-02    120    180

带小计:
Region   North  South  All
Date
2023-01    100    150  250
2023-02    120    180  300
All        220    330  550
```

### 理解重点

- `pivot_table` 自动聚合（默认 `mean`）；`df.pivot` 只重塑形状，**不聚合**。
- 当 `index` + `columns` 组合有重复时，必须用 `pivot_table`（会自动聚合）。
- `margins=True` 添加总计行 / 列，便于总览。

## 交叉频数表

### `pd.crosstab`

#### 作用

统计两个或多个分类变量的**频数**（计数）。是 `pivot_table(aggfunc='count')` 的快捷封装。

#### 重点方法

```python
pd.crosstab(index, columns, values=None, rownames=None, colnames=None,
            aggfunc=None, margins=False, margins_name='All',
            dropna=True, normalize=False)
```

#### 参数

| 参数名      | 本例取值                       | 说明                                                             |
| ----------- | ------------------------------ | ---------------------------------------------------------------- |
| `index`     | `df["Gender"]`                 | 作为行的 Series                                                  |
| `columns`   | `df["City"]`                   | 作为列的 Series                                                  |
| `values`    | `None`（默认）、Series          | 指定时需搭配 `aggfunc`，用于聚合而非计数                         |
| `aggfunc`   | `None`（默认 count）           | `values` 提供时的聚合函数                                        |
| `margins`   | `False`（默认）                | 是否添加总计                                                     |
| `normalize` | `False`（默认）、`True`、`'index'`、`'columns'` | 归一化为频率：整体 / 按行 / 按列                |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Gender": ["M", "F", "M", "F", "M", "F"],
    "City": ["Beijing", "Shanghai", "Beijing", "Beijing", "Shanghai", "Shanghai"],
})

print(f"交叉频数:\n{pd.crosstab(df['Gender'], df['City'])}")
print(f"\n按行归一化:\n{pd.crosstab(df['Gender'], df['City'], normalize='index')}")
```

### 输出

```text
交叉频数:
City    Beijing  Shanghai
Gender
F             1         2
M             2         1

按行归一化:
City     Beijing  Shanghai
Gender
F       0.333333  0.666667
M       0.666667  0.333333
```

## 多级索引

### `pd.MultiIndex`

#### 作用

表示二维及以上的**层级索引**。常见于 `groupby(['A', 'B'])` 结果、`pivot_table` 带多列索引、金融时间序列等。

#### 构造方式

```python
# 从数组
pd.MultiIndex.from_arrays([['A','A','B','B'], [1, 2, 1, 2]], names=['L', 'N'])

# 从元组
pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2)])

# 从笛卡儿积
pd.MultiIndex.from_product([['A', 'B'], [1, 2]])

# 从 DataFrame
pd.MultiIndex.from_frame(df[['col1', 'col2']])
```

### 索引选择

```python
df.loc['A']              # 取第一层 'A' 的所有行
df.loc[('A', 1)]         # 取具体 ('A', 1)
df.loc[['A', 'B']]       # 取第一层多个值
df.xs('A', level='Letter')  # 按层级名跨层索引
```

### 示例代码

```python
import pandas as pd

index = pd.MultiIndex.from_arrays(
    [["A", "A", "B", "B"], [1, 2, 1, 2]],
    names=["Letter", "Number"],
)
df = pd.DataFrame({"Value": [10, 20, 30, 40]}, index=index)

print(f"多级索引 DataFrame:\n{df}")
print(f"\ndf.loc['A']:\n{df.loc['A']}")
print(f"\ndf.loc[('A', 1)]: {df.loc[('A', 1)].iloc[0]}")
```

### 输出

```text
多级索引 DataFrame:
               Value
Letter Number
A      1          10
       2          20
B      1          30
       2          40

df.loc['A']:
        Value
Number
1          10
2          20

df.loc[('A', 1)]: 10
```

### 理解重点

- 多层索引用 **元组** 表示具体位置：`df.loc[('A', 1)]`。
- `df.reset_index()` 把多级索引变回普通列；`df.set_index(['a', 'b'])` 反之。
- `df.unstack(level=-1)` 把内层索引变成列（宽表）；`df.stack()` 反之（长表）。

## 长宽表转换

### `df.stack` / `df.unstack`

- `stack`: 宽 → 长，把**列**移到**内层行索引**
- `unstack`: 长 → 宽，把**行索引某层**移到**列**

### `df.melt`

将宽表转为长表（更常用的"长表化"方式）。

#### 重点方法

```python
df.melt(id_vars=None, value_vars=None, var_name=None,
        value_name='value', col_level=None, ignore_index=True)
```

| 参数名        | 本例取值                 | 说明                                                  |
| ------------- | ------------------------ | ----------------------------------------------------- |
| `id_vars`     | `['id']`                 | 保留为"主键"的列                                      |
| `value_vars`  | `['A', 'B']`             | 需要"融化"的列；省略则取所有非 `id_vars` 列           |
| `var_name`    | `None`、`'variable'`      | 变量名列的列名                                        |
| `value_name`  | `'value'`（默认）         | 值列的列名                                            |

## 性能优化

### 向量化 vs 循环

Pandas 建立在 NumPy 之上，**向量化操作比 Python 循环快数百到数千倍**。

#### 示例代码

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
df = pd.DataFrame({"A": np.random.randn(n), "B": np.random.randn(n)})

# 循环：慢
start = time.time()
result = []
for i in range(len(df)):
    result.append(df["A"].iloc[i] + df["B"].iloc[i])
loop_time = time.time() - start

# 向量化：快
start = time.time()
result = df["A"] + df["B"]
vec_time = time.time() - start

print(f"循环: {loop_time:.4f}s")
print(f"向量化: {vec_time:.4f}s")
print(f"提速: {loop_time / vec_time:.1f}×")
```

#### 输出（示例）

```text
循环: 25.3412s
向量化: 0.0041s
提速: ≈6000×
```

### 性能最佳实践

| 反模式 | 改进写法 |
|---|---|
| `for i in range(len(df)): df.iloc[i]` | 向量化 `df["A"] + df["B"]` |
| `df.apply(fn, axis=1)` | 向量化 / `numpy` 操作（`apply` 慢 10~50×） |
| 反复在循环中 `append` 到 DataFrame | 先收集到 `list`，最后 `pd.concat` 一次 |
| 反复读整个 DataFrame | `chunksize` 分块读 |
| `object` dtype 大数组 | 用 `category` 或明确数值 dtype |

## 内存优化

### `DataFrame.memory_usage`

#### 作用

查看每列占用的字节数。`deep=True` 会精确计算 `object` 列（字符串）的真实占用。

### 优化策略

1. **整数降精度**：`int64 → int32 → int16 → int8`（根据取值范围）
2. **浮点降精度**：`float64 → float32`（精度够用时）
3. **字符串用 `category`**：重复类别很多时节省成倍内存
4. **稀疏数据**：`SparseDtype`

### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "int_col": np.random.randint(0, 100, 10000),
    "float_col": np.random.randn(10000),
    "str_col": np.random.choice(["A", "B", "C"], 10000),
})

print(f"优化前:\n{df.memory_usage(deep=True)}")

df["int_col"] = df["int_col"].astype("int8")
df["float_col"] = df["float_col"].astype("float32")
df["str_col"] = df["str_col"].astype("category")

print(f"\n优化后:\n{df.memory_usage(deep=True)}")
```

### 输出（示例）

```text
优化前:
Index         132
int_col     80000
float_col   80000
str_col    570000
dtype: int64

优化后:
Index         132
int_col     10000
float_col   40000
str_col     10368
dtype: int64
```

### 理解重点

- `int8` 范围 `[-128, 127]`；`uint8` 范围 `[0, 255]`——根据实际取值选类型。
- `float32` 约 7 位有效数字；训练数据、特征矩阵几乎都够用。
- `category` 类型对重复字符串节省**显著**：上例从 570 KB 降到 10 KB。
- 用 `pd.to_numeric(..., downcast='integer')` 自动选最小整数类型。

## 常见坑

1. `pivot_table` 和 `pivot` **不同**：前者自动聚合可处理重复，后者只重塑。
2. `crosstab` 默认计数，不是求和；要求和需显式 `values=... , aggfunc='sum'`。
3. 多级索引选取 `df.loc[('A', 1)]` 的圆括号不能省，`df.loc['A', 1]` 会被解释为行列两个索引。
4. `apply(axis=1)` 看似方便实则慢，能用向量化 / `np.where` 不要用 `apply`。
5. `astype('category')` 后再做 `groupby` 需要 `observed=True`，否则会保留所有分类组（空组填 NaN）。
6. `iterrows` 在大数据上极慢，**永远不要**用它改 DataFrame；需要行级操作用 `apply` 或改写成向量化。

## 小结

- `pivot_table` / `crosstab` 是长表 → 宽表的两大工具；前者可聚合，后者专做频数。
- 多级索引是 `groupby` / `pivot_table` 的自然产物，掌握 `reset_index` / `unstack` 足够应对大多数场景。
- 向量化 > `apply` > 循环——这是 Pandas 性能的铁律。
- 大数据表常识：int 降精度、float32、category 三板斧，可节省大半内存。
