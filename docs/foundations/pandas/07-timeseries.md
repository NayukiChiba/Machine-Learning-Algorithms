---
title: Pandas 时间序列处理
outline: deep
---

# Pandas 时间序列处理

> 对应脚本：`Basic/Pandas/07_timeseries.py`  
> 运行方式：`python Basic/Pandas/07_timeseries.py`（仓库根目录）

## 本章目标

1. 掌握时间戳的创建与日期范围生成。
2. 理解 DatetimeIndex 的属性访问与索引能力。
3. 学会使用部分字符串进行时间序列切片。
4. 掌握 `resample`（重采样）和 `rolling`（滚动窗口）操作。
5. 理解时间偏移 `shift` 与变化率 `pct_change` 的用法。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `pd.to_datetime(...)` | 解析日期字符串 | `demo_datetime_create` |
| `pd.date_range(...)` | 生成日期范围 | `demo_datetime_create` |
| `ts.index.year` / `.month` / `.day` | 日期属性访问 | `demo_time_index` |
| `ts["2023-01"]` | 部分字符串索引 | `demo_time_slice` |
| `df.resample(rule)` | 时间重采样 | `demo_resample` |
| `ts.rolling(window)` | 滚动窗口 | `demo_rolling` |
| `ts.shift(periods)` | 时间偏移 | `demo_shift` |
| `ts.pct_change()` | 百分比变化 | `demo_shift` |

## 1. 时间序列创建

### 方法重点

- `pd.to_datetime()` 可以解析多种格式的日期字符串。
- `pd.date_range()` 按频率生成连续日期序列。
- 常用频率代码：`'D'`（天）、`'H'`（小时）、`'W'`（周）、`'ME'`（月末）、`'B'`（工作日）。

### 参数速览（本节）

1. `pd.to_datetime(arg, format=None, errors='raise')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `arg` | `["2023-01-01", "2023-01-02", "2023-01-03"]` | 日期字符串列表 |
| `format` | `None`（默认） | 自动推断格式 |

2. `pd.date_range(start=None, end=None, periods=None, freq='D')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `start` | `"2023-01-01"` | 起始日期 |
| `periods` | `5`、`3` | 生成的日期点数 |
| `freq` | `"D"`、`"H"`、`"W"`、`"ME"`、`"B"` | 频率代码 |

### 示例代码

```python
import pandas as pd

# 解析日期字符串
dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
print(dates)

# date_range 创建日期范围
dr = pd.date_range("2023-01-01", periods=5, freq="D")
print(dr)

# 不同频率
print("freq='H':", pd.date_range("2023-01-01", periods=3, freq="H").tolist())
print("freq='W':", pd.date_range("2023-01-01", periods=3, freq="W").tolist())
print("freq='ME':", pd.date_range("2023-01-01", periods=3, freq="ME").tolist())
print("freq='B':", pd.date_range("2023-01-01", periods=5, freq="B").tolist())
```

### 结果输出

```text
DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'],
              dtype='datetime64[ns]', freq=None)
----------------
DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03',
               '2023-01-04', '2023-01-05'],
              dtype='datetime64[ns]', freq='D')
----------------
freq='H': [..., Timestamp('2023-01-01 02:00:00')]
freq='W': [..., Timestamp('2023-01-15')]
freq='ME': [..., Timestamp('2023-03-31')]
freq='B': [..., Timestamp('2023-01-06')]
```

### 理解重点

- `date_range` 的三个参数 `start`/`end`/`periods`，指定任意两个即可确定序列。
- `freq='ME'` 生成的是**月末**日期（Month End），`'MS'` 是月初（Month Start）。
- `freq='B'` 跳过周末，适合金融数据。

## 2. 时间序列索引

### 方法重点

- 以 `DatetimeIndex` 为索引的 Series/DataFrame 支持丰富的日期属性访问。
- 通过 `.index.year`、`.index.month` 等属性可以直接提取时间分量。

### 参数速览（本节）

适用属性（通过 `ts.index` 访问）：

| 属性 | 返回值 | 说明 |
|---|---|---|
| `.year` | 年份数组 | 如 `[2023, 2023, ...]` |
| `.month` | 月份数组 | 1-12 |
| `.day` | 日期数组 | 1-31 |
| `.dayofweek` | 星期几数组 | 0=周一，6=周日 |

### 示例代码

```python
import numpy as np

dates = pd.date_range("2023-01-01", periods=10, freq="D")
ts = pd.Series(np.random.randn(10), index=dates)
print(ts)

print(f"year: {ts.index.year.tolist()}")
print(f"month: {ts.index.month.tolist()}")
print(f"day: {ts.index.day.tolist()}")
print(f"dayofweek: {ts.index.dayofweek.tolist()}")
```

### 结果输出

```text
2023-01-01    0.496714
2023-01-02   -0.138264
...
2023-01-10    0.542560
Freq: D, dtype: float64
----------------
year: [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
month: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
day: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dayofweek: [6, 0, 1, 2, 3, 4, 5, 6, 0, 1]
```

### 理解重点

- `dayofweek` 以周一为 0，与 Python 标准库 `datetime` 一致。
- 这些属性常用于特征工程：提取"是否工作日"、"月份"等特征。
- 对 DataFrame 列使用时，需要 `df["date"].dt.year` 语法（`dt` 访问器）。

## 3. 时间序列切片

### 方法重点

- DatetimeIndex 支持**部分字符串索引**，如 `ts["2023-01"]` 选取整月数据。
- 切片语法 `ts["2023-01-15":"2023-01-20"]` 两端**都包含**。
- 这种便捷语法只在索引为 DatetimeIndex 时可用。

### 参数速览（本节）

适用语法（分项）：

| 语法 | 说明 |
|---|---|
| `ts["2023-01"]` | 选取 2023 年 1 月的所有数据 |
| `ts["2023-01-15":"2023-01-20"]` | 日期范围切片（两端包含） |
| `ts["2023"]` | 选取 2023 年全部数据 |

### 示例代码

```python
dates = pd.date_range("2023-01-01", periods=100, freq="D")
ts = pd.Series(np.random.randn(100), index=dates)

# 选取 1 月数据
print(ts["2023-01"].head())

# 日期范围
print(ts["2023-01-15":"2023-01-20"])
```

### 结果输出

```text
ts['2023-01'] (前 5 行):
2023-01-01    0.496714
2023-01-02   -0.138264
2023-01-03    0.647689
2023-01-04    1.523030
2023-01-05   -0.234153
Freq: D, dtype: float64
----------------
ts['2023-01-15':'2023-01-20']:
2023-01-15   -0.463418
2023-01-16   -0.465730
2023-01-17    0.241962
2023-01-18   -1.913280
2023-01-19   -1.724918
2023-01-20   -0.562288
Freq: D, dtype: float64
```

### 理解重点

- 部分字符串索引的精度可以是年、月、日——Pandas 自动匹配范围。
- 时间切片与 `loc` 一样**包含右端点**，这与普通整数切片不同。
- 如果索引不是 DatetimeIndex，这种语法会报错。

## 4. 时间重采样

### 方法重点

- `resample` 将时间序列转换到不同频率——降采样（日→周）或升采样（日→小时）。
- 降采样需要聚合函数（`sum`、`mean` 等），升采样需要填充方法。
- `resample` 可看作 `groupby` 的时间版本。

### 参数速览（本节）

适用 API：`df.resample(rule, closed=None, label=None, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `rule` | `"W"` | 重采样频率（`W`=周，`ME`=月末） |
| `closed` | `None`（默认） | 区间哪端闭合 |
| `label` | `None`（默认） | 标签取区间哪端 |

### 示例代码

```python
dates = pd.date_range("2023-01-01", periods=30, freq="D")
df = pd.DataFrame({
    "value": np.random.randint(10, 100, 30),
    "sales": np.random.randint(100, 1000, 30),
}, index=dates)

# 按周聚合
print(df.resample("W").sum())

# 多种聚合方式
print(df["value"].resample("W").agg(["sum", "mean", "max"]))
```

### 结果输出

```text
按周聚合 sum():
            value  sales
2023-01-01     63    563
2023-01-08    401   3827
2023-01-15    349   3010
2023-01-22    396   3580
2023-01-29    327   3215
2023-01-30     12    104
----------------
多种聚合:
            sum       mean  max
2023-01-01   63  63.000000   63
2023-01-08  401  57.285714   90
...
```

### 理解重点

- `resample` 返回的是惰性对象，需要接聚合方法才能计算。
- 降采样时，`closed='left'` / `'right'` 决定区间边界。
- 可以像 `groupby` 一样使用 `agg()` 应用多种聚合函数。

## 5. 滚动窗口操作

### 方法重点

- `rolling(window)` 创建固定大小的滑动窗口。
- 常用于计算移动平均、移动标准差等。
- `ewm(span)` 提供指数加权移动平均（EWMA），对近期数据赋予更高权重。

### 参数速览（本节）

1. `ts.rolling(window, min_periods=None, center=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `window` | `3` | 窗口大小 |
| `min_periods` | `None`（默认，等于 window） | 最少有效观测数 |
| `center` | `False`（默认） | 窗口标签对齐方式 |

2. `ts.ewm(span=None, com=None, halflife=None, alpha=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `span` | `3` | 衰减跨度（等价于 `alpha = 2/(span+1)`） |

### 示例代码

```python
dates = pd.date_range("2023-01-01", periods=10, freq="D")
ts = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=dates)

# 3 日移动平均
print(ts.rolling(3).mean())

# 3 日移动求和
print(ts.rolling(3).sum())

# 指数加权移动平均
print(ts.ewm(span=3).mean())
```

### 结果输出

```text
rolling(3).mean():
2023-01-01     NaN
2023-01-02     NaN
2023-01-03     2.0
2023-01-04     3.0
2023-01-05     4.0
2023-01-06     5.0
2023-01-07     6.0
2023-01-08     7.0
2023-01-09     8.0
2023-01-10     9.0
Freq: D, dtype: float64
----------------
rolling(3).sum():
2023-01-01      NaN
2023-01-02      NaN
2023-01-03      6.0
2023-01-04      9.0
...
2023-01-10     27.0
Freq: D, dtype: float64
----------------
ewm(span=3).mean():
2023-01-01     1.000000
2023-01-02     1.500000
2023-01-03     2.142857
...
2023-01-10     8.976471
Freq: D, dtype: float64
```

### 理解重点

- 窗口大小为 3 时，前 2 个值为 `NaN`（不足 3 个数据点）。
- 设 `min_periods=1` 可以从第一个值开始计算。
- `ewm` 不产生前导 `NaN`，因为指数加权从第一个点就有效。
- 移动平均是金融分析和信号处理中最常用的平滑方法。

## 6. 时间偏移操作

### 方法重点

- `shift(n)` 将数据**向后**偏移 n 个周期（正数向后，负数向前）。
- 偏移后空出的位置填 `NaN`。
- `pct_change()` 计算相邻元素的百分比变化率。

### 参数速览（本节）

1. `ts.shift(periods=1, freq=None, fill_value=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `periods` | `1`、`-1` | 正数向后移，负数向前移 |
| `freq` | `None`（默认） | 移动索引而非数据 |

2. `ts.pct_change(periods=1)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `periods` | `1`（默认） | 与前 n 个周期比较 |

### 示例代码

```python
dates = pd.date_range("2023-01-01", periods=5, freq="D")
ts = pd.Series([10, 20, 30, 40, 50], index=dates)

# 向后偏移
print(ts.shift(1))

# 向前偏移
print(ts.shift(-1))

# 百分比变化
print(ts.pct_change())
```

### 结果输出

```text
shift(1):
2023-01-01     NaN
2023-01-02    10.0
2023-01-03    20.0
2023-01-04    30.0
2023-01-05    40.0
Freq: D, dtype: float64
----------------
shift(-1):
2023-01-01    20.0
2023-01-02    30.0
2023-01-03    40.0
2023-01-04    50.0
2023-01-05     NaN
Freq: D, dtype: float64
----------------
pct_change():
2023-01-01         NaN
2023-01-02    1.000000
2023-01-03    0.500000
2023-01-04    0.333333
2023-01-05    0.250000
Freq: D, dtype: float64
```

### 理解重点

- `shift(1)` 等价于 "昨天的值"，常用于计算环比变化。
- `pct_change()` 内部就是 `(ts - ts.shift(1)) / ts.shift(1)`。
- 在金融分析中，`pct_change()` 即为日收益率。
- `shift(freq="D")` 是移动索引而非数据，应用场景不同。

## 常见坑

1. `resample` 必须在 DatetimeIndex 上使用，普通整数索引会报错。
2. `rolling` 的窗口前 `window-1` 个值为 `NaN`，需要注意数据起始段。
3. 旧版 `freq='M'` 已废弃，应使用 `freq='ME'`（月末）。
4. `shift` 移动的是数据而非索引，容易与 `tshift`（已废弃）混淆。

## 小结

- 时间序列是 Pandas 的核心优势之一，内置了丰富的时间处理能力。
- `resample` 用于频率转换，`rolling` 用于窗口计算，`shift` 用于时间偏移——三者覆盖大部分时间序列分析需求。
- 以 DatetimeIndex 为索引后，可以解锁部分字符串切片、日期属性访问等便捷功能。
