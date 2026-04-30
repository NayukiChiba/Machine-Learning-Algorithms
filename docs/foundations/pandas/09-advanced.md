---
title: Pandas 高级操作与性能优化
outline: deep
---

# Pandas 高级操作与性能优化

## 本章目标

1. 掌握 `pivot_table` 透视表与 `crosstab` 交叉表
2. 理解 `MultiIndex` 多级索引的创建与选择
3. 掌握 `stack` / `unstack` / `melt` 的长宽表转换
4. 学会用 `astype('category')` 和降精度做内存优化

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.pivot_table(...)` | 函数 | 透视表（可聚合，支持 MultiIndex） |
| `df.pivot(...)` | 方法 | 简单透视（不聚合，仅重塑） |
| `pd.crosstab(...)` | 函数 | 频数/比例交叉表 |
| `pd.MultiIndex.from_arrays(...)` | 构造器 | 从数组列表创建多级索引 |
| `pd.MultiIndex.from_tuples(...)` | 构造器 | 从元组列表创建多级索引 |
| `df.stack(...)` | 方法 | 列 → 行（宽变长） |
| `df.unstack(...)` | 方法 | 行 → 列（长变宽） |
| `df.melt(...)` | 方法 | 宽表 → 长表（多列熔化为键值对） |
| `df.memory_usage(...)` | 方法 | 每列内存占用 |
| `df.astype('category')` | 方法 | 转换为分类类型（节省内存） |

## 1. 透视表与简单透视

### `pd.pivot_table`

#### 作用

创建 Excel 风格的透视表。按行列分组后对值做聚合（求和、均值、计数等）。与 `groupby` 的区别：结果呈现为二维交叉表，行列都是分组维度。

#### 重点方法

```python
pd.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean',
               fill_value=None, margins=False, dropna=True,
               margins_name='All', observed=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `DataFrame` | 输入数据 | `df` |
| `values` | `str`、`list[str]` | 要聚合的值列 | `"Sales"`、`["Sales", "Profit"]` |
| `index` | `str`、`list[str]` | 行分组键（结果的行索引） | `"Region"`、`["Region", "Dept"]` |
| `columns` | `str`、`list[str]` | 列分组键（结果的列索引） | `"Year"` |
| `aggfunc` | `str`、`list[str]`、`dict`、函数 | 聚合函数，默认为 `'mean'` | `"sum"`、`["sum", "mean"]` |
| `fill_value` | 标量 或 `None` | 用此值替换 NaN，默认为 `None` | `0` |
| `margins` | `bool` | `True` 时添加"总计"行/列，默认为 `False` | `True` |
| `dropna` | `bool` | 是否丢弃全 NaN 的列，默认为 `True` | `False` |

### `DataFrame.pivot`

#### 作用

纯粹的形状重塑——不做聚合，要求行列对唯一。若行列组合有重复会报错。

#### 重点方法

```python
df.pivot(*, columns, index=None, values=None)
```

### 综合示例

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Region": ["East", "East", "East", "West", "West", "West"],
    "Year": [2022, 2023, 2022, 2022, 2023, 2023],
    "Product": ["A", "A", "B", "A", "B", "B"],
    "Sales": [100, 150, 200, 300, 250, 180],
})

print("原始数据:")
print(df)

# pivot_table：按 Region × Year 聚合 Sales
pt = pd.pivot_table(df, values="Sales", index="Region",
                    columns="Year", aggfunc="sum", fill_value=0)
print(f"\n透视表 (Region × Year, sum):\n{pt}")

# 带 margins（总计行列）
ptm = pd.pivot_table(df, values="Sales", index="Region",
                     columns="Year", aggfunc="sum", margins=True)
print(f"\n透视表 + margins:\n{ptm}")

# pivot：在不聚合时使用（行列对唯一）
piv = df.pivot(index="Region", columns="Year", values="Sales")
print(f"\npivot (无聚合):\n{piv}")
```

#### 输出

```text
原始数据:
  Region  Year Product  Sales
0   East  2022       A    100
1   East  2023       A    150
2   East  2022       B    200
3   West  2022       A    300
4   West  2023       B    250
5   West  2023       B    180

透视表 (Region × Year, sum):
Year    2022  2023
Region
East     300   150
West     300   430

透视表 + margins:
Year    2022  2023   All
Region
East     300   150   450
West     300   430   730
All      600   580  1180

pivot (无聚合):
Year    2022  2023
Region
East     100   150
West     300   215
```

#### 理解重点

- `pivot_table` 是分组聚合的二维呈现——类似于 Excel 的数据透视表
- `pivot` 只改变形状不做聚合——要求行列组合唯一，否则抛 `ValueError`
- `margins=True` 自动追加"总计"行列——适合快速生成汇总报告
- 多值列用列表 `values=["Sales", "Profit"]`——生成 MultiIndex 列

## 2. 交叉频数表

### `pd.crosstab`

#### 作用

计算两个（或更多）分类变量的频数/比例交叉表。底层可视为 `pivot_table` 在计数场景的快捷方式。

#### 重点方法

```python
pd.crosstab(index, columns, values=None, aggfunc=None, rownames=None,
            colnames=None, margins=False, margins_name='All',
            normalize=False, dropna=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `index` | `array_like`、`Series` | 行分类变量 | `df["Region"]` |
| `columns` | `array_like`、`Series` | 列分类变量 | `df["Product"]` |
| `values` | `array_like` 或 `None` | 要聚合的值列；`None` 时计算频数 | `df["Sales"]` |
| `aggfunc` | `str`、函数 或 `None` | 聚合函数；`values` 非空时必须指定 | `"sum"` |
| `normalize` | `bool`、`str` | `True` 归一化为比例；也可指定 `'index'` / `'columns'` / `'all'` | `True`、`"index"` |
| `margins` | `bool` | `True` 时添加总计行列，默认为 `False` | `True` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Region": ["East", "East", "West", "West", "East", "West"],
    "Product": ["A", "B", "A", "A", "B", "B"],
    "Sales": [100, 200, 300, 150, 180, 220],
})

# 频数
ct = pd.crosstab(df["Region"], df["Product"])
print(f"频数交叉表:\n{ct}")

# 比例（按行归一化）
ctn = pd.crosstab(df["Region"], df["Product"], normalize="index")
print(f"\n比例交叉表（行归一化）:\n{ctn.round(2)}")

# 按值聚合
cts = pd.crosstab(df["Region"], df["Product"], values=df["Sales"],
                  aggfunc="sum")
print(f"\nSales 汇总交叉表:\n{cts}")
```

#### 输出

```text
频数交叉表:
Product  A  B
Region
East     1  2
West     2  1

比例交叉表（行归一化）:
Product     A     B
Region
East     0.33  0.67
West     0.67  0.33

Sales 汇总交叉表:
Product    A    B
Region
East     100  380
West     450  220
```

#### 理解重点

- `crosstab` 默认计算频数——类似 R 的 `table()` 函数
- `normalize='index'` / `'columns'` / `'all'` 分别按行/列/所有归一化
- `values` + `aggfunc` 组合使 `crosstab` 可以替代 `pivot_table` 的简单场景

## 3. 多级索引

### `pd.MultiIndex`

#### 作用

MultiIndex（分层索引）允许在单个轴上有多层标签。DataFrame 的行和列都可以是多级索引——透视表的结果天然就是 MultiIndex。

#### 创建方式

```python
# 从数组列表
pd.MultiIndex.from_arrays([["A", "A", "B", "B"], [1, 2, 1, 2]],
                          names=["level1", "level2"])

# 从元组列表
pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1), ("B", 2)],
                          names=["level1", "level2"])

# 从笛卡尔积
pd.MultiIndex.from_product([["A", "B"], [1, 2]],
                           names=["level1", "level2"])

# 通过 groupby/set_index 创建
df.set_index(["col1", "col2"])
```

### MultiIndex 选择

| 语法 | 说明 |
|---|---|
| `df.loc["A"]` | 选择第一级为 "A" 的所有行 |
| `df.loc[("A", 1)]` | 选择精确的多级标签（用元组） |
| `df.loc[("A", slice(None))]` | 第一级为 "A"，第二级全部（等价于 `df.loc["A"]`） |
| `df.xs("A", level="level1")` | 按级别名称精确选择（跨切面） |

#### 示例代码

```python
import pandas as pd
import numpy as np

# 创建 MultiIndex 行
idx = pd.MultiIndex.from_product(
    [["East", "West"], ["Q1", "Q2", "Q3", "Q4"]],
    names=["Region", "Quarter"]
)
df = pd.DataFrame({
    "Sales": np.random.randint(100, 500, 8),
    "Profit": np.random.randint(10, 100, 8),
}, index=idx)

print("MultiIndex DataFrame:")
print(df)

# 按第一级选择
print(f"\nEast 全部:\n{df.loc['East']}")

# 精确多级选择
print(f"\nloc[('West', 'Q3')]:\n{df.loc[('West', 'Q3')]}")

# 按级别名称跨切面
print(f"\nxs('Q2', level='Quarter'):\n{df.xs('Q2', level='Quarter')}")
```

#### 输出

```text
MultiIndex DataFrame:
                     Sales  Profit
Region Quarter
East   Q1              101      67
       Q2              300      57
       Q3              433      50
       Q4              464      89
West   Q1              185      95
       Q2              268      24
       Q3              482      24
       Q4              362      80

East 全部:
         Sales  Profit
Quarter
Q1         101      67
Q2         300      57
Q3         433      50
Q4         464      89

loc[('West', 'Q3')]:
Sales     482
Profit     24
Name: (West, Q3), dtype: int32

xs('Q2', level='Quarter'):
        Sales  Profit
Region
East      300      57
West      268      24
```

#### 理解重点

- MultiIndex 常见于 `groupby`（多键分组）、`pivot_table`（多行列）、`set_index`（多列设为索引）的结果
- 元组选择 `df.loc[("A", 1)]` 用于精确定位多级标签
- `xs` 方法按级别名称跨切面选择——比 `loc` 更语义化
- 去 MultiIndex 用 `reset_index()` 或 `df.columns = ["_".join(c) for c in df.columns]`

## 4. 长宽表转换

### `stack` / `unstack` / `melt`

#### 作用

- `stack()`：列 → 行（宽变长）——把列标签"压"进行索引的最后一级
- `unstack()`：行 → 列（长变宽）——把行索引最后一级"展开"为列标签
- `melt()`：宽表 → 长表——将多列"熔化"为两列：变量名列和值列

#### 重点方法

```python
df.stack(level=-1, dropna=True)
df.unstack(level=-1, fill_value=None)
df.melt(id_vars=None, value_vars=None, var_name=None, value_name='value')
```

#### `melt` 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `id_vars` | `list[str]` | 保持不变的标识列 | `["Name", "Date"]` |
| `value_vars` | `list[str]` 或 `None` | 要熔化的值列；`None` 时熔化除 `id_vars` 外的所有列 | `["Q1", "Q2", "Q3", "Q4"]` |
| `var_name` | `str` | 熔化后变量名列的列名，默认为 `'variable'` | `"Quarter"` |
| `value_name` | `str` | 熔化后值列的列名，默认为 `'value'` | `"Sales"` |

#### 示例代码

```python
import pandas as pd

# 宽表
dfWide = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Q1": [100, 200, 150],
    "Q2": [120, 210, 160],
    "Q3": [130, 220, 170],
    "Q4": [140, 230, 180],
})

print(f"宽表:\n{dfWide}")

# melt：宽 → 长
dfLong = dfWide.melt(id_vars=["Name"], var_name="Quarter", value_name="Sales")
print(f"\nmelt 长表:\n{dfLong}")

# 用 pivot_table 长 → 宽（逆操作）
dfBack = pd.pivot_table(dfLong, values="Sales", index="Name",
                        columns="Quarter")
print(f"\npivot_table 回到宽表:\n{dfBack}")

# stack 示例
dfIdx = dfWide.set_index("Name")
print(f"\nstack 前:\n{dfIdx}")
print(f"\nstack 后:\n{dfIdx.stack()}")
```

#### 输出

```text
宽表:
      Name   Q1   Q2   Q3   Q4
0    Alice  100  120  130  140
1      Bob  200  210  220  230
2  Charlie  150  160  170  180

melt 长表:
       Name Quarter  Sales
0     Alice      Q1    100
1       Bob      Q1    200
2   Charlie      Q1    150
3     Alice      Q2    120
4       Bob      Q2    210
5   Charlie      Q2    160
6     Alice      Q3    130
7       Bob      Q3    220
8   Charlie      Q3    170
9     Alice      Q4    140
10      Bob      Q4    230
11  Charlie      Q4    180

pivot_table 回到宽表:
Quarter    Q1   Q2   Q3   Q4
Name
Alice     100  120  130  140
Bob       200  210  220  230
Charlie   150  160  170  180

stack 前:
           Q1   Q2   Q3   Q4
Name
Alice     100  120  130  140
Bob       200  210  220  230
Charlie   150  160  170  180

stack 后:
Name     Quarter
Alice    Q1         100
         Q2         120
         Q3         130
         Q4         140
Bob      Q1         200
         Q2         210
         Q3         220
         Q4         230
Charlie  Q1         150
         Q2         160
         Q3         170
         Q4         180
dtype: int64
```

#### 理解重点

- `melt` 比 `stack` 更直观——明确指定 `id_vars`（保持的列）和 `value_vars`（熔化的列）
- `melt` 的逆操作是 `pivot_table`——前者变长、后者变宽
- `stack` 适合列名本身就是"分类变量"的宽表——将列变成行索引的一层
- 长表是"整洁数据"（tidy data）的标准形态——seaborn、plotly 等可视化库的首选格式

## 5. 内存优化

### `df.memory_usage` / `astype('category')`

#### 作用

- `memory_usage()`：查看 DataFrame 每列的内存占用（字节）
- `astype('category')`：将低基数列（重复值多）转为分类类型——大幅节省内存
- 降精度为数值列节省内存：`float64 → float32`、`int64 → int32`

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "ID": range(10000),
    "City": np.random.choice(["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"], 10000),
    "Value": np.random.randn(10000),
})

print(f"原始内存:\n{df.memory_usage()}")
print(f"总内存: {df.memory_usage().sum() / 1024:.1f} KB")

# City 列转为 category
df["City"] = df["City"].astype("category")
print(f"\ncategory 后内存:\n{df.memory_usage()}")
print(f"总内存: {df.memory_usage().sum() / 1024:.1f} KB")

# 查看 category 节省比例
cityBefore = 10000 * 8  # object 列：每元素一个指针 ~8 字节
cityAfter = df["City"].memory_usage()
print(f"\nCity 列：转前 ~{cityBefore} bytes → 转后 {cityAfter} bytes")
print(f"节省: {(1 - cityAfter / cityBefore) * 100:.1f}%")
```

#### 输出

```text
原始内存:
Index      132
ID       80000
City     80000
Value    80000
dtype: int64
总内存: 234.5 KB

category 后内存:
Index      132
ID       80000
City     10548
Value    80000
dtype: int64
总内存: 166.7 KB

City 列：转前 ~80000 bytes → 转后 10548 bytes
节省: 86.8%
```

#### 理解重点

- `category` 类型的本质：存储整数编码 + 查找表——重复值越多，节省越多
- 典型场景：性别、国家、城市、产品类型等**低基数**（≤ 50 类）分类变量
- 数值降精度也有效：`df["col"].astype("float32")` 比 `float64` 省一半内存
- `df.info(memory_usage="deep")` 查看含 object 列深层内存的实际占用

## 常见坑

1. `pivot` 要求行列对唯一——重复会报错；需要聚合时用 `pivot_table`
2. `pivot_table` 默认 `aggfunc='mean'`——计算求和要用 `aggfunc='sum'`
3. `stack()` 要求所有列同 dtype——否则会降级为 `object`
4. `melt()` 不指定 `value_vars` 会熔化所有非 `id_vars` 列——注意检查是否预期
5. `category` 列的修改受限：不能直接赋新类别——需用 `.cat.add_categories()` 先添加
6. `memory_usage()` 默认不计算 object 列指向的字符串内存——用 `deep=True` 获得真实值

## 小结

- 透视表 `pivot_table` 是分组聚合的二维呈现——比多层 `groupby` 更直观
- `crosstab` 是频数统计的快捷方式——一行代码生成分类交叉表
- 长表是可视化库的首选格式——`melt` 将宽表转为长表
- 内存优化两步走：重复字符串列 → `category`；数值列 → 降精度（`float32` / `int32`）
