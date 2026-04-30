---
title: Pandas 时间序列
outline: deep
---

# Pandas 时间序列

## 本章目标

1. 掌握 `to_datetime` 和 `date_range` 创建时间戳/日期范围
2. 掌握 `.dt` 访问器的日期组件提取（年/月/日/星期等）
3. 掌握 DatetimeIndex 的字符串切片与部分匹配
4. 掌握 `resample` 的降/升采样聚合
5. 掌握 `rolling` 滚动窗口计算
6. 掌握 `shift` / `diff` / `pct_change` 做时间偏移与变化率

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.to_datetime(...)` | 函数 | 解析日期时间为 `datetime64` 类型 |
| `pd.date_range(...)` | 函数 | 生成连续日期序列（DatetimeIndex） |
| `Series.dt.xxx` | 访问器 | 提取日期组件（年/月/日/星期/季度等） |
| `df.resample(...)` | 方法 | 按时间频率重采样（降/升采样） |
| `df.rolling(...)` | 方法 | 固定窗口大小的滚动计算 |
| `df.ewm(...)` | 方法 | 指数加权移动窗口 |
| `df.shift(...)` | 方法 | 行偏移（向前/向后平移） |
| `df.diff(...)` | 方法 | 相邻行差分 |
| `Series.pct_change(...)` | 方法 | 变化率（相对百分比） |

## 1. 时间戳创建

### `pd.to_datetime`

#### 作用

将字符串、整数、Series 解析为 `datetime64` 类型。支持自动推断格式，也可用 `format` 参数显式指定以加速。

参数表见 [ch04 类型转换 — `pd.to_datetime`](04-cleaning.md)（`errors` / `format` / `dayfirst` / `utc` / `unit`）。

#### 示例代码

```python
import pandas as pd

# 字符串列表
dates = pd.to_datetime(["2024-01-15", "2024-03-20", "2024-06-10"])
print(f"to_datetime:\n{dates}")
print(f"dtype: {dates.dtype}")
```

#### 输出

```text
to_datetime:
0   2024-01-15
1   2024-03-20
2   2024-06-10
dtype: datetime64[ns]
```

### `pd.date_range`

#### 作用

生成连续的日期时间索引（DatetimeIndex）。通过 `start`/`end`/`periods`/`freq` 控制范围和频率。

#### 重点方法

```python
pd.date_range(start=None, end=None, periods=None, freq=None, tz=None,
              normalize=False, name=None, inclusive='both')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `start` | `str`、`datetime` | 起始日期 | `"2024-01-01"` |
| `end` | `str`、`datetime` | 结束日期 | `"2024-01-10"` |
| `periods` | `int` | 生成的日期数量（与 `freq` 配合），默认为 `None` | `10` |
| `freq` | `str` | 频率字符串：`'D'` 天 / `'W'` 周 / `'M'` 月 / `'H'` 小时 / `'T'` 分钟，默认为 `None` | `"D"`、`"W-MON"`、`"M"` |
| `tz` | `str` 或 `None` | 时区 | `"Asia/Shanghai"` |
| `inclusive` | `str` | 端点包含策略：`'both'` / `'left'` / `'right'` / `'neither'`，默认为 `'both'` | `"left"` |

#### 常用频率字符串

| 频率 | 含义 | 频率 | 含义 |
|---|---|---|---|
| `"D"` | 每日 | `"W"` | 每周（周日） |
| `"W-MON"` | 每周一 | `"MS"` | 每月首日 |
| `"M"` | 每月末日 | `"Q"` | 每季度末日 |
| `"H"` | 每小时 | `"T"` / `"min"` | 每分钟 |
| `"B"` | 工作日 | `"YS"` | 每年首日 |

#### 示例代码

```python
import pandas as pd

# 天数
r1 = pd.date_range("2024-01-01", "2024-01-07", freq="D")
print(f"每日:\n{r1}")

# 工作日
r2 = pd.date_range("2024-01-01", periods=5, freq="B")
print(f"\n工作日:\n{r2}")

# 月初
r3 = pd.date_range("2024-01-01", periods=4, freq="MS")
print(f"\n每月初:\n{r3}")
```

#### 输出

```text
每日:
DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
               '2024-01-05', '2024-01-06', '2024-01-07'],
              dtype='datetime64[ns]', freq='D')

工作日:
DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
               '2024-01-05'],
              dtype='datetime64[ns]', freq='B')

每月初:
DatetimeIndex(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01'], dtype='datetime64[ns]', freq='MS')
```

#### 理解重点

- `periods` 和 `end` 互斥——指定数量时无需指定结束日期
- 频率字符串大小写敏感：`"D"` 天 / `"W"` 周 / `"M"` 月 / `"H"` 小时
- `date_range` 返回 `DatetimeIndex`——可直接作为 DataFrame 的行索引

## 2. 日期组件提取（`.dt` 访问器）

### `.dt` 访问器

#### 作用

通过 `Series.dt.xxx` 从 `datetime64` 列中提取日期组件（年/月/日/星期/季度/周数等），全部为向量化操作。

### 常用属性速览

| 属性 | 含义 | 示例输出 |
|---|---|---|
| `dt.year` | 年 | `2024` |
| `dt.month` | 月 (1-12) | `1` |
| `dt.day` | 日 (1-31) | `15` |
| `dt.hour` | 时 (0-23) | `14` |
| `dt.minute` | 分 (0-59) | `30` |
| `dt.dayofweek` | 星期几 (0=周一, 6=周日) | `0` |
| `dt.day_name()` | 星期几的名称 | `"Monday"` |
| `dt.quarter` | 季度 (1-4) | `1` |
| `dt.week` | 周数 (1-52) | `3` |
| `dt.is_month_start` | 是否月初 | `True` |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Date": pd.to_datetime(["2024-01-15", "2024-03-20", "2024-06-10",
                            "2024-09-05", "2024-12-25"]),
    "Value": [100, 200, 150, 300, 250],
})

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.day_name()
df["Quarter"] = df["Date"].dt.quarter

print(df)
```

#### 输出

```text
        Date  Value  Year  Month   DayOfWeek  Quarter
0 2024-01-15    100  2024      1      Monday        1
1 2024-03-20    200  2024      3   Wednesday        1
2 2024-06-10    150  2024      6      Monday        2
3 2024-09-05    300  2024      9    Thursday        3
4 2024-12-25    250  2024     12   Wednesday        4
```

#### 理解重点

- `.dt` 只能用于 `datetime64` 类型列——需先用 `pd.to_datetime` 转换
- `.dt` 的属性全部向量化——比循环快 100 倍以上
- `dt.dayofweek` 返回整数（0=周一），`dt.day_name()` 返回名称（`"Monday"`）

## 3. 时间序列索引与切片

#### 作用

当 DataFrame 索引为 `DatetimeIndex` 时，可用**日期字符串**直接切片——Pandas 会自动解析为时间范围。

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=10, freq="D")
df = pd.DataFrame({
    "Value": np.random.randint(10, 100, 10),
}, index=dates)

print(f"时间序列:\n{df}")

# 字符串部分匹配切片
print(f"\n2024-01-03 到 2024-01-06:\n{df['2024-01-03':'2024-01-06']}")

# 按年份切片
print(f"\n2024-01 全部:\n{df['2024-01']}")
```

#### 输出

```text
时间序列:
            Value
2024-01-01     56
2024-01-02     61
2024-01-03     88
2024-01-04     70
2024-01-05     87
2024-01-06     63
2024-01-07     94
2024-01-08     68
2024-01-09     56
2024-01-10     99

2024-01-03 到 2024-01-06:
            Value
2024-01-03     88
2024-01-04     70
2024-01-05     87
2024-01-06     63

2024-01 全部:
            Value
2024-01-01     56
2024-01-02     61
2024-01-03     88
2024-01-04     70
2024-01-05     87
2024-01-06     63
2024-01-07     94
2024-01-08     68
2024-01-09     56
2024-01-10     99
```

#### 理解重点

- 部分字符串切片极其强大：`df["2024-01"]` 匹配整个 1 月，`df["2024"]` 匹配整年——无需构造日期对象
- 只有 `DatetimeIndex` 支持字符串切片——普通 `object` 索引不行

## 4. 重采样

### `DataFrame.resample`

#### 作用

按指定频率重新聚合时间序列数据。降采样（如 日→月）做聚合，升采样（如 月→日）做插值。类似 `groupby`，返回 `Resampler` 对象，需调用聚合函数才执行。

#### 重点方法

```python
df.resample(rule, axis=0, closed=None, label=None, on=None, level=None)
# 后续必须链式调用聚合：.mean() / .sum() / .agg() 等
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `rule` | `str` | 目标频率字符串 | `"W"`、`"M"`、`"Q"`、`"H"` |
| `axis` | `int` | 采样轴，默认为 `0` | `1` |
| `closed` | `str` 或 `None` | 区间的哪端闭合：`'left'` / `'right'`，默认为 `None`（各频率有默认值） | `"left"` |
| `label` | `str` 或 `None` | 聚合结果用区间的哪端作标签，默认为 `None` | `"left"` |
| `on` | `str` 或 `None` | 用指定列而不是索引做时间轴 | `"date_col"` |

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=60, freq="D")
df = pd.DataFrame({
    "Sales": np.random.randint(100, 500, 60),
}, index=dates)

print(f"原始数据（前 5 天）:\n{df.head()}")

# 降采样：日 → 周
weekly = df.resample("W").agg(["sum", "mean"])
print(f"\n按周聚合:\n{weekly}")

# 降采样：日 → 月
monthly = df.resample("M").sum()
print(f"\n按月聚合:\n{monthly}")
```

#### 输出

```text
原始数据（前 5 天）:
            Sales
2024-01-01    288
2024-01-02    308
2024-01-03    214
2024-01-04    124
2024-01-05    127

按周聚合:
           Sales
             sum   mean
2024-01-07  1158  165.43
2024-01-14  2175  310.71
2024-01-21  1689  241.29
2024-01-28  2129  304.14
2024-02-04  2645  377.86
2024-02-11  1769  252.71
2024-02-18  2184  312.00
2024-02-25  2012  287.43
2024-03-01  1153  288.25

按月聚合:
            Sales
2024-01-31   9255
2024-02-29  10347
2024-03-31    279
```

#### 理解重点

- `resample` 的思维模型 = `groupby` + 时间窗口——分组依据是频率而不是列值
- 必须链式调用聚合函数：`resample("M")` 不执行，`resample("M").sum()` 才执行
- 频率字符串与 `date_range` 相同——`"D"` 天 / `"W"` 周 / `"M"` 月 / `"Q"` 季度

## 5. 滚动窗口

### `DataFrame.rolling`

#### 作用

在数据上滑动一个固定大小的窗口，对窗口内的值做聚合。与 `resample` 的区别：`rolling` 不改变数据频率，每行都有一个结果。

#### 重点方法

```python
df.rolling(window, min_periods=None, center=False, win_type=None,
           on=None, axis=0, closed=None)
# 后续必须链式调用：.mean() / .sum() / .std() / .apply() 等
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `window` | `int`、`str` | 窗口大小（行数）或时间偏移字符串 | `7`、`"3D"` |
| `min_periods` | `int` 或 `None` | 最少需要多少非空值才计算结果，默认为 `None`（等于窗口大小） | `1` |
| `center` | `bool` | `True` 时窗口居中（结果标签在窗口中心），默认为 `False` | `True` |
| `win_type` | `str` 或 `None` | 窗口类型（加权）：`'triang'` / `'gaussian'` 等，默认为 `None` | `"gaussian"` |
| `on` | `str` 或 `None` | 用指定列而不是索引做时间轴 | `"date_col"` |

### `DataFrame.ewm`

#### 作用

指数加权移动（EWM）：最近的数据点权重最大，按指数衰减。无固定窗口大小，所有历史值都参与。

#### 重点方法

```python
df.ewm(com=None, span=None, halflife=None, alpha=None, adjust=True)
# 后续必须链式调用：.mean() / .std() 等
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `span` | `float` 或 `None` | 跨度（约等于窗口中心），$\alpha = 2/(span+1)$ | `7` |
| `halflife` | `float` 或 `None` | 半衰期（权重衰减到一半的时间） | `3` |
| `alpha` | `float` 或 `None` | 直接指定平滑因子 $\alpha \in (0, 1]$ | `0.3` |
| `adjust` | `bool` | `True` 时用权重归一化（更准确），默认为 `True` | `False` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=20, freq="D")
df = pd.DataFrame({
    "Value": np.random.randint(100, 500, 20).astype(float),
}, index=dates)

# 滚动均值（窗口=3）
df["RollMean3"] = df["Value"].rolling(window=3).mean()

# 滚动均值（窗口=7）
df["RollMean7"] = df["Value"].rolling(window=7).mean()

# 指数加权
df["EWM_span7"] = df["Value"].ewm(span=7).mean()

print(df.head(12).round(1))
```

#### 输出

```text
            Value  RollMean3  RollMean7  EWM_span7
2024-01-01  288.0        NaN        NaN      288.0
2024-01-02  308.0        NaN        NaN      294.3
2024-01-03  214.0      270.0        NaN      268.9
2024-01-04  124.0      215.3        NaN      217.7
2024-01-05  127.0      155.0        NaN      186.1
2024-01-06  362.0      204.3        NaN      246.6
2024-01-07  185.0      224.7      229.7      226.1
2024-01-08  297.0      281.3      231.0      249.3
2024-01-09  491.0      324.3      257.1      326.3
2024-01-10  447.0      411.7      290.4      366.7
2024-01-11  412.0      450.0      331.6      382.0
2024-01-12  205.0      354.7      342.7      324.3
```

#### 理解重点

- `rolling` 窗口前 `window-1` 行结果为 NaN——因窗口内非空值不足
- `center=True` 时窗口以当前行为中心——适合事后分析，不适合实时预测
- `ewm` 无 NaN 首行——因子权重始终对第一个观测值有定义，首行等于自身
- `ewm` 比 `rolling` 更平滑——因为所有权重连续衰减而非等权

## 6. 偏移与变化率

### `shift` / `diff` / `pct_change`

#### 作用

- `shift(periods)`：将数据沿时间轴平移。正数为前移（滞后特征），负数为后移
- `diff(periods)`：相邻行的差分：$\Delta_t = x_t - x_{t-periods}$
- `pct_change(periods)`：相对变化率：$r_t = \frac{x_t - x_{t-periods}}{x_{t-periods}}$

#### 重点方法

```python
df.shift(periods=1, freq=None, axis=0)
df.diff(periods=1, axis=0)
Series.pct_change(periods=1, fill_method=None, limit=None)
```

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Price": [100, 105, 103, 108, 112, 110],
})

df["Prev"] = df["Price"].shift(1)
df["Change"] = df["Price"].diff(1)
df["ChangePct"] = df["Price"].pct_change(1)
print(df)
```

#### 输出

```text
   Price   Prev  Change  ChangePct
0    100    NaN     NaN        NaN
1    105  100.0     5.0   0.050000
2    103  105.0    -2.0  -0.019048
3    108  103.0     5.0   0.048544
4    112  108.0     4.0   0.037037
5    110  112.0    -2.0  -0.017857
```

#### 数学公式

$$
\begin{aligned}
\text{diff:}&\quad \Delta_t = x_t - x_{t-1} \\[4pt]
\text{pct\_change:}&\quad r_t = \frac{x_t - x_{t-1}}{x_{t-1}}
\end{aligned}
$$

#### 理解重点

- `shift(1)` 的第一行为 NaN——没有更早的数据
- `diff` 等价于 `x - x.shift(1)`
- `pct_change` 是金融分析中最常用的指标之一——计算收益率/涨跌幅
- `shift(-1)` 可做"前瞻"特征——但注意别在预测模型中造成数据泄露

## 常见坑

1. `resample` 返回 `Resampler` 对象——不调聚合方法就没有结果，容易忘记链式调用 `.mean()` 等
2. `rolling` 窗口前 N-1 行是 NaN——不要当成 bug，这是正常行为
3. `date_range` 的 `freq="M"` 返回月末，`freq="MS"` 返回月初——两者不同
4. `.dt` 访问器只能用于 `datetime64` 类型——非日期列调用会报 `AttributeError`
5. 时间字符串切片只在索引为 `DatetimeIndex` 时可用——普通 object 索引不支持
6. `shift` 可做滞后特征（`periods>0`）和前瞻（`periods<0`）——前瞻特征在预测中会造成数据泄露

## 小结

- 时间序列的核心索引类型是 `DatetimeIndex`——`date_range` 生成，`to_datetime` 解析
- `.dt` 访问器提取日期组件；字符串切片做灵活的时间范围过滤
- `resample`（改频率）+ `rolling`（窗口计算）+ `ewm`（指数加权）是时间序列分析的三大算子
- `shift` / `diff` / `pct_change` 是时间序列特征工程的三个基础操作
