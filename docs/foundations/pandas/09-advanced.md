---
title: Pandas 高级操作与性能优化
outline: deep
---

# Pandas 高级操作与性能优化

> 对应脚本：`Basic/Pandas/09_advanced.py`  
> 运行方式：`python Basic/Pandas/09_advanced.py`（仓库根目录）

## 本章目标

1. 掌握透视表（`pivot_table`）和交叉表（`crosstab`）的使用。
2. 理解多级索引（MultiIndex）的创建与访问。
3. 认识向量化操作的性能优势。
4. 学会基本的内存优化技巧。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `pd.pivot_table(...)` | 创建透视表 | `demo_pivot_table` |
| `pd.crosstab(...)` | 创建交叉表 | `demo_crosstab` |
| `pd.MultiIndex.from_arrays(...)` | 创建多级索引 | `demo_multi_index` |
| 向量化运算 `df["A"] + df["B"]` | 避免 Python 循环 | `demo_vectorization` |
| `s.astype(...)` / `.astype("category")` | 内存优化 | `demo_memory_optimization` |

## 1. 透视表

### 方法重点

- 透视表类似 Excel 中的"数据透视表"功能。
- 核心是三要素：行索引（`index`）、列索引（`columns`）、聚合值（`values`）。
- `aggfunc` 指定聚合方式，默认为 `'mean'`。

### 参数速览（本节）

适用 API：`pd.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data` | `df` | 源 DataFrame |
| `values` | `"Sales"` | 需要聚合的值列 |
| `index` | `"Date"` | 行索引（透视后的行） |
| `columns` | `"Region"` | 列索引（透视后的列） |
| `aggfunc` | `"sum"` | 聚合函数 |
| `fill_value` | `None`（默认） | 缺失值填充 |
| `margins` | `False`（默认） | 是否添加汇总行/列 |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Date": ["2023-01", "2023-01", "2023-02", "2023-02"],
    "Region": ["North", "South", "North", "South"],
    "Sales": [100, 150, 120, 180],
    "Quantity": [10, 15, 12, 18],
})

pivot = pd.pivot_table(
    df, values="Sales", index="Date",
    columns="Region", aggfunc="sum",
)
print(pivot)
```

### 结果输出

```text
Region   North  South
Date
2023-01    100    150
2023-02    120    180
```

### 理解重点

- 透视表将"长格式"数据转为"宽格式"，便于对比分析。
- `aggfunc` 支持字符串（`'sum'`、`'mean'`）和函数（`np.sum`、自定义）。
- `margins=True` 会在边缘添加汇总行和汇总列。

## 2. 交叉表

### 方法重点

- `crosstab` 是 `pivot_table` 的便捷版本，专门用于频率统计。
- 默认统计出现次数（计数），无需指定 `values` 和 `aggfunc`。
- 传入 `normalize=True` 可以得到比例而非计数。

### 参数速览（本节）

适用 API：`pd.crosstab(index, columns, values=None, aggfunc=None, normalize=False, margins=False, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `index` | `df["Gender"]` | 行维度 |
| `columns` | `df["City"]` | 列维度 |
| `normalize` | `False`（默认） | `True` 返回比例 |
| `margins` | `False`（默认） | 是否添加汇总 |

### 示例代码

```python
df = pd.DataFrame({
    "Gender": ["M", "F", "M", "F", "M", "F"],
    "City": ["Beijing", "Shanghai", "Beijing",
             "Beijing", "Shanghai", "Shanghai"],
})

ct = pd.crosstab(df["Gender"], df["City"])
print(ct)
```

### 结果输出

```text
City    Beijing  Shanghai
Gender
F             1         2
M             2         1
```

### 理解重点

- 交叉表本质是对两个分类变量做频率统计。
- 等价于 `df.groupby(["Gender", "City"]).size().unstack(fill_value=0)`。
- 在特征分析中，交叉表常用于观察类别变量之间的关联。

## 3. 多级索引

### 方法重点

- MultiIndex 允许在行或列上使用多层索引，增加数据维度。
- 创建方式：`from_arrays`、`from_tuples`、`from_product`。
- 多级索引的访问使用元组：`df.loc[("A", 1)]`。

### 参数速览（本节）

适用 API：`pd.MultiIndex.from_arrays(arrays, names=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `arrays` | `[["A", "A", "B", "B"], [1, 2, 1, 2]]` | 每个数组对应一层索引 |
| `names` | `["Letter", "Number"]` | 各层的名称 |

### 示例代码

```python
arrays = [["A", "A", "B", "B"], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=["Letter", "Number"])
df = pd.DataFrame({"Value": [10, 20, 30, 40]}, index=index)

print(df)
print(df.loc["A"])
print(df.loc[("A", 1)])
```

### 结果输出

```text
多级索引 DataFrame:
               Value
Letter Number
A      1          10
       2          20
B      1          30
       2          40
----------------
df.loc['A']:
        Value
Number
1          10
2          20
----------------
df.loc[('A', 1)]:
Value    10
Name: (A, 1), dtype: int64
```

### 理解重点

- `df.loc["A"]` 选取第一层为 `"A"` 的所有行，返回降级的 DataFrame。
- `df.loc[("A", 1)]` 精确定位到两层都匹配的行。
- `reset_index()` 可以将多级索引还原为普通列。
- `xs` 方法可以在任意层级上选取数据：`df.xs(1, level="Number")`。

## 4. 向量化操作

### 方法重点

- Pandas 的核心性能优势在于向量化操作——避免 Python 级别的循环。
- 向量化操作底层调用 NumPy/C 实现，速度比逐行循环快几个数量级。
- 在可能的情况下，始终优先使用向量化操作。

### 参数速览（本节）

| 操作方式 | 示例 | 速度 |
|---|---|---|
| Python 循环 | `for i in range(len(df)): result.append(df["A"].iloc[i] + df["B"].iloc[i])` | 慢 |
| 向量化运算 | `df["A"] + df["B"]` | 快 |

### 示例代码

```python
import time
import numpy as np

n = 100000
df = pd.DataFrame({
    "A": np.random.randn(n),
    "B": np.random.randn(n),
})

# 循环方式
start = time.time()
result1 = []
for i in range(len(df)):
    result1.append(df["A"].iloc[i] + df["B"].iloc[i])
loop_time = time.time() - start

# 向量化方式
start = time.time()
result2 = df["A"] + df["B"]
vec_time = time.time() - start

print(f"循环耗时: {loop_time:.4f}秒")
print(f"向量化耗时: {vec_time:.4f}秒")
print(f"向量化快了约 {loop_time / vec_time:.1f} 倍")
```

### 结果输出（示例）

```text
循环耗时: 3.2541秒
向量化耗时: 0.0008秒
向量化快了约 4068.0 倍
```

### 理解重点

- 向量化运算将循环下沉到 C 层，消除了 Python 解释器的逐行开销。
- `df.apply()` 虽然比裸循环好，但仍然是行级别 Python 调用，不是真正的向量化。
- 当无法向量化时，考虑使用 `itertuples()`（比 `iterrows()` 快）。

## 5. 内存优化

### 方法重点

- `memory_usage(deep=True)` 精确查看各列内存占用。
- 降低整数/浮点精度：`int64` → `int8`，`float64` → `float32`。
- 低基数字符串列转 `category` 类型，可大幅节省内存。

### 参数速览（本节）

| 优化手段 | 操作 | 效果 |
|---|---|---|
| 降低整数精度 | `df["col"].astype("int8")` | 8 字节 → 1 字节 |
| 降低浮点精度 | `df["col"].astype("float32")` | 8 字节 → 4 字节 |
| 分类类型 | `df["col"].astype("category")` | 大幅减少重复字符串内存 |

### 示例代码

```python
import numpy as np

df = pd.DataFrame({
    "int_col": np.random.randint(0, 100, 10000),
    "float_col": np.random.randn(10000),
    "str_col": np.random.choice(["A", "B", "C"], 10000),
})

print("优化前内存使用:")
print(df.memory_usage(deep=True))

# 优化
df["int_col"] = df["int_col"].astype("int8")
df["float_col"] = df["float_col"].astype("float32")
df["str_col"] = df["str_col"].astype("category")

print("\n优化后内存使用:")
print(df.memory_usage(deep=True))
```

### 结果输出（示例）

```text
优化前内存使用:
Index          128
int_col      80000
float_col    80000
str_col     640000
dtype: int64

优化后内存使用:
Index        128
int_col    10000
float_col  40000
str_col    10342
dtype: int64
```

### 理解重点

- 字符串列是内存杀手——每个元素独立存储 Python 字符串对象。
- `category` 类型将字符串映射为整数编码，只存储一份字符串表。
- `int8` 范围是 `-128` 到 `127`，`int16` 范围是 `-32768` 到 `32767`——选择前确认数据范围。
- `float32` 精度约 7 位有效数字，大部分分析场景足够。

## 常见坑

1. 透视表的 `aggfunc` 默认是 `mean`，忘记改成 `sum` 会导致结果不符预期。
2. 多级索引的切片语法容易出错，建议用 `xs()` 或 `reset_index()` 后再操作。
3. 向量化运算中混入 Python 循环会显著拖慢速度。
4. `astype("int8")` 时数据超出范围会静默溢出，产生错误结果。

## 小结

- 透视表和交叉表是数据汇总分析的利器，理解"行-列-值"三要素是关键。
- 多级索引为复杂数据提供了层次化的组织方式，但会增加操作复杂度。
- **向量化优先**是 Pandas 性能优化的第一原则。
- 内存优化的核心思路：降低数值精度 + 字符串转 `category`。
