---
title: Pandas 时间序列
outline: deep
---

# Pandas 时间序列

## 本章目标

1. 掌握时间戳 / 日期范围的创建（`to_datetime`、`date_range`）。
2. 掌握时间序列索引与切片（字符串部分匹配）。
3. 掌握重采样 `resample` 做降 / 升采样。
4. 掌握滚动窗口 `rolling` 和指数加权 `ewm`。
5. 掌握 `shift` / `pct_change` 做时间偏移与变化率计算。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.to_datetime(...)` | 函数 | 解析为 `datetime64` 类型 |
| `pd.date_range(...)` | 函数 | 生成连续日期序列 |
| `pd.Timestamp(...)` | 构造器 | 单个时间戳 |
| `pd.Timedelta(...)` | 构造器 | 时间差 |
| `dt` 访问器 | 访问器 | 按元素提取日期组件（year/month 等） |
| `ts["2023-01"]` | 语法 | 字符串部分匹配索引 |
| `ts.resample(...)` | 方法 | 按时间频率重采样 |
| `ts.rolling(...)` | 方法 | 滚动窗口 |
| `ts.ewm(...)` | 方法 | 指数加权窗口 |
| `ts.shift(...)` | 方法 | 沿时间轴偏移 |
| `ts.pct_change(...)` | 方法 | 百分比变化 |
| `ts.diff(...)` | 方法 | 差分 |

## 时间戳与日期范围

### `pd.to_datetime`

#### 作用

将字符串、数字或混合类型转换为 `datetime64` 类型。详细参数见 [04-cleaning.md](./04-cleaning.md)。

#### 重点方法

```python
pd.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False,
               utc=False, format=None, unit=None, origin='unix')
```

### `pd.date_range`

#### 作用

生成**等间隔**的日期 / 时间序列，是构造时间索引的标准方法。

#### 重点方法

```python
pd.date_range(start=None, end=None, periods=None, freq=None,
              tz=None, normalize=False, name=None, inclusive='both')
```

#### 参数

| 参数名     | 本例取值                            | 说明                                                                |
| ---------- | ----------------------------------- | ------------------------------------------------------------------- |
| `start`    | `"2023-01-01"`                      | 起始时间                                                            |
| `end`      | `"2023-12-31"`                      | 终止时间                                                            |
| `periods`  | `5`、`10`                            | 时间点个数；`start`、`end`、`periods` 四选三                        |
| `freq`     | `'D'`、`'H'`、`'W'`、`'ME'`、`'B'`    | 频率（见下表）                                                      |
| `tz`       | `None`、`'Asia/Shanghai'`            | 时区                                                                |
| `inclusive`| `'both'`（默认）、`'left'`、`'right'`、`'neither'` | 端点包含方式（Pandas 1.4+）                           |

### 常用 `freq` 频率字符串

| `freq`           | 含义                                        |
| ---------------- | ------------------------------------------- |
| `'D'`            | 日历日                                      |
| `'B'`            | 工作日（跳过周末）                          |
| `'H'` / `'h'`    | 小时                                        |
| `'T'` / `'min'`  | 分钟                                        |
| `'S'`            | 秒                                          |
| `'W'`            | 周                                          |
| `'W-MON'`        | 每周一                                      |
| `'ME'`           | 月末（Pandas 2.2+，旧版本用 `'M'`）          |
| `'MS'`           | 月初                                        |
| `'QE'`           | 季末                                        |
| `'YE'`           | 年末                                        |
| `'Y'` / `'A'`    | 年                                          |

### 综合示例

#### 示例代码

```python
import pandas as pd

dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
print(f"解析:\n{dates}")

print(f"\n连续 5 天:\n{pd.date_range('2023-01-01', periods=5, freq='D')}")
print(f"\n按小时:\n{pd.date_range('2023-01-01', periods=3, freq='H')}")
print(f"\n按周:\n{pd.date_range('2023-01-01', periods=3, freq='W')}")
print(f"\n工作日:\n{pd.date_range('2023-01-01', periods=5, freq='B')}")
```

#### 输出

```text
解析:
DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[ns]', freq=None)

连续 5 天:
DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
               '2023-01-05'],
              dtype='datetime64[ns]', freq='D')

按小时:
DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 01:00:00',
               '2023-01-01 02:00:00'],
              dtype='datetime64[ns]', freq='H')

按周:
DatetimeIndex(['2023-01-01', '2023-01-08', '2023-01-15'], dtype='datetime64[ns]', freq='W-SUN')

工作日:
DatetimeIndex(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
               '2023-01-06'],
              dtype='datetime64[ns]', freq='B')
```

## 时间索引与切片

### `DatetimeIndex` 日期属性

当索引是 `DatetimeIndex` 时，可直接访问每个时间点的组件。

| 属性                                     | 含义                                |
| ---------------------------------------- | ----------------------------------- |
| `.year` / `.month` / `.day`              | 年 / 月 / 日                        |
| `.hour` / `.minute` / `.second`          | 时 / 分 / 秒                        |
| `.dayofweek` / `.day_of_week`            | 星期几（0=周一）                    |
| `.dayofyear`                             | 年中的第几天                        |
| `.weekofyear` / `.isocalendar().week`    | 年中的第几周                        |
| `.quarter`                               | 所在季度                            |
| `.is_month_start` / `.is_month_end`      | 是否月初 / 月末                     |

对于 Series 的 `datetime` 列，用 `.dt.xxx` 访问器（不是 `.index.xxx`）。

### 字符串部分匹配切片

当索引是 `DatetimeIndex`（已排序）时，可用**字符串前缀**做切片：

```python
ts["2023"]                       # 2023 年所有数据
ts["2023-01"]                    # 2023 年 1 月
ts["2023-01-15":"2023-01-20"]    # 区间（闭区间）
```

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

dates = pd.date_range("2023-01-01", periods=10, freq="D")
ts = pd.Series(np.random.randn(10), index=dates)

print(f"year: {ts.index.year.tolist()}")
print(f"month: {ts.index.month.tolist()}")
print(f"dayofweek: {ts.index.dayofweek.tolist()}")

ts_long = pd.Series(np.random.randn(100),
                    index=pd.date_range("2023-01-01", periods=100, freq="D"))
print(f"\nts['2023-01'] 前 3 行:\n{ts_long['2023-01'].head(3)}")
print(f"\nts['2023-01-15':'2023-01-20']:\n{ts_long['2023-01-15':'2023-01-20']}")
```

### 理解重点

- 字符串切片的**终点是包含的**（闭区间）——与 `loc` 一致，与普通切片相反。
- 时间索引要先**排序**才能切片：`ts.sort_index()`。
- 对非索引的 datetime 列，用 `df['date'].dt.year` 等访问。

## 重采样

### `Series.resample` / `DataFrame.resample`

#### 作用

按新频率重新分组时间序列数据。是时间序列版本的 `groupby`。

- **降采样**（downsample）：高频 → 低频（如日 → 月），需要聚合
- **升采样**（upsample）：低频 → 高频（如日 → 小时），需要插值 / 填充

#### 重点方法

```python
ts.resample(rule, axis=0, closed=None, label=None, convention='start',
            kind=None, on=None, level=None, origin='start_day', offset=None)
```

#### 参数

| 参数名    | 本例取值                                           | 说明                                                           |
| --------- | -------------------------------------------------- | -------------------------------------------------------------- |
| `rule`    | `'W'`、`'ME'`、`'5T'`、`'Q'`                        | 目标频率                                                       |
| `closed`  | `None`（默认）、`'left'`、`'right'`                 | 区间闭合端；默认按频率不同而定                                 |
| `label`   | `None`（默认）、`'left'`、`'right'`                 | 结果用哪个端点作为标签                                         |
| `on`      | `None`、列名                                       | 当时间列不是索引时用这个参数指定                               |
| `origin`  | `'start_day'`（默认）、`'epoch'`、时间戳            | 窗口起点参考                                                   |

#### 常用聚合

`resample(...)` 返回 `Resampler` 对象，再链式调用聚合函数：`.mean()` / `.sum()` / `.agg([...])` / `.ohlc()` 等。

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=30, freq="D")
df = pd.DataFrame({
    "value": np.random.randint(10, 100, 30),
    "sales": np.random.randint(100, 1000, 30),
}, index=dates)

print(f"按周求和:\n{df.resample('W').sum()}")
print(f"\n多聚合:\n{df['value'].resample('W').agg(['sum', 'mean', 'max'])}")
```

### 输出

```text
按周求和:
            value  sales
2023-01-01     10    103
2023-01-08    378   3619
2023-01-15    373   4023
2023-01-22    395   3902
2023-01-29    418   3649
2023-02-05    107    801

多聚合:
            sum       mean  max
2023-01-01   10  10.000000   10
2023-01-08  378  54.000000   93
2023-01-15  373  53.285714   91
2023-01-22  395  56.428571   98
2023-01-29  418  59.714286   94
2023-02-05  107  53.500000   70
```

### 理解重点

- **降采样**需要指定聚合函数；**升采样**需要插值 / 填充（`ffill()` / `interpolate()`）。
- `closed` 和 `label` 决定区间端点归属；金融数据常用 `label='right'`。
- `resample(...).ohlc()` 对股价数据自动生成 open / high / low / close 四列。

## 滚动窗口

### `Series.rolling`

#### 作用

创建**滑动窗口**视图，可再调用聚合方法计算移动平均、移动和等。不会降低数据频率（输出与输入同长度）。

#### 重点方法

```python
ts.rolling(window, min_periods=None, center=False, win_type=None,
           on=None, axis=0, closed=None, step=None, method='single')
```

#### 参数

| 参数名        | 本例取值                            | 说明                                                              |
| ------------- | ----------------------------------- | ----------------------------------------------------------------- |
| `window`      | `3`、`7`、`'7D'`                     | 窗口大小（整数或时间字符串）                                      |
| `min_periods` | `None`、`1`                          | 窗口内所需的最小有效观察数；不足则返回 `NaN`                      |
| `center`      | `False`（默认）、`True`              | `True` 时窗口居中（结果标签在窗口中点）                           |
| `win_type`    | `None`（默认）、`'gaussian'`、`'hamming'` 等 | 窗口函数类型，支持加权窗口                                  |
| `closed`      | `None`、`'right'`、`'left'`、`'both'`、`'neither'` | 区间闭合端（时间窗口才需要）                          |

### `Series.ewm`

#### 作用

创建**指数加权**窗口。新数据权重更大，对近期波动更敏感。

#### 重点方法

```python
ts.ewm(com=None, span=None, halflife=None, alpha=None, min_periods=0,
       adjust=True, ignore_na=False, axis=0)
```

#### 参数

| 参数名        | 本例取值        | 说明                                                                 |
| ------------- | --------------- | -------------------------------------------------------------------- |
| `com`         | `None`          | 质心（center of mass）：`α = 1/(1+com)`                              |
| `span`        | `3`、`10`       | 窗口长度：`α = 2/(span+1)`                                           |
| `halflife`    | `None`、`'7D'`  | 半衰期：`α = 1 - exp(log(0.5)/halflife)`                             |
| `alpha`       | `None`、`0.3`   | 直接指定衰减因子 `α ∈ (0, 1]`                                        |
| `min_periods` | `0`（默认）     | 最小观察数                                                           |
| `adjust`      | `True`（默认）  | 是否做偏差修正；`False` 时用纯递推公式                               |

### 综合示例

#### 示例代码

```python
import pandas as pd

dates = pd.date_range("2023-01-01", periods=10, freq="D")
ts = pd.Series(range(1, 11), index=dates)

print(f"3 日移动平均:\n{ts.rolling(3).mean()}")
print(f"\n3 日移动求和:\n{ts.rolling(3).sum()}")
print(f"\nEWM span=3:\n{ts.ewm(span=3).mean()}")
```

### 输出

```text
3 日移动平均:
2023-01-01    NaN
2023-01-02    NaN
2023-01-03    2.0
2023-01-04    3.0
2023-01-05    4.0
2023-01-06    5.0
2023-01-07    6.0
2023-01-08    7.0
2023-01-09    8.0
2023-01-10    9.0
dtype: float64

EWM span=3:
2023-01-01    1.000000
2023-01-02    1.666667
2023-01-03    2.428571
2023-01-04    3.266667
2023-01-05    4.161290
2023-01-06    5.095238
2023-01-07    6.056942
2023-01-08    7.033936
2023-01-09    8.019569
2023-01-10    9.011070
dtype: float64
```

### 理解重点

- `rolling(3)` 前两行是 `NaN`（窗口不足）；用 `min_periods=1` 可从第一个点开始有值。
- 时间窗口 `rolling('7D')` 按**实际时间跨度**而非行数，适合不等间距数据。
- `ewm` 无需丢弃早期数据，适合平滑。

## 时间偏移与变化

### `Series.shift`

#### 作用

沿时间轴**偏移**数据。正数向后（滞后），负数向前（超前）。典型用途：构造"上一期" / "下一期"特征、计算差分。

#### 重点方法

```python
ts.shift(periods=1, freq=None, axis=0, fill_value=<no value>)
```

#### 参数

| 参数名       | 本例取值                    | 说明                                                         |
| ------------ | --------------------------- | ------------------------------------------------------------ |
| `periods`    | `1`、`-1`、`7`               | 偏移步数；正 = 滞后（值往下移），负 = 超前                   |
| `freq`       | `None`、`'D'`、`'M'`         | 偏移时间频率；指定后**索引**跟着移，**值**不动               |
| `fill_value` | `<no value>`、`0`、`np.nan`  | 边界填充值                                                   |

### `Series.pct_change` / `Series.diff`

- `pct_change()`：百分比变化 `(x_t - x_{t-1}) / x_{t-1}`
- `diff()`：差分 `x_t - x_{t-1}`

### 示例代码

```python
import pandas as pd

dates = pd.date_range("2023-01-01", periods=5, freq="D")
ts = pd.Series([10, 20, 30, 40, 50], index=dates)

print(f"原数据:\n{ts}")
print(f"\nshift(1):\n{ts.shift(1)}")
print(f"\nshift(-1):\n{ts.shift(-1)}")
print(f"\npct_change:\n{ts.pct_change()}")
print(f"\ndiff:\n{ts.diff()}")
```

### 输出

```text
原数据:
2023-01-01    10
2023-01-02    20
2023-01-03    30
2023-01-04    40
2023-01-05    50
Freq: D, dtype: int64

shift(1):
2023-01-01     NaN
2023-01-02    10.0
2023-01-03    20.0
2023-01-04    30.0
2023-01-05    40.0
Freq: D, dtype: float64

shift(-1):
2023-01-01    20.0
2023-01-02    30.0
2023-01-03    40.0
2023-01-04    50.0
2023-01-05     NaN
Freq: D, dtype: float64

pct_change:
2023-01-01         NaN
2023-01-02    1.000000
2023-01-03    0.500000
2023-01-04    0.333333
2023-01-05    0.250000
Freq: D, dtype: float64

diff:
2023-01-01     NaN
2023-01-02    10.0
2023-01-03    10.0
2023-01-04    10.0
2023-01-05    10.0
Freq: D, dtype: float64
```

### 理解重点

- `shift(1)` **数据向下移**（滞后 1 期）；`shift(-1)` 向上移（超前 1 期）。
- `pct_change()` = `shift(1)` 配合 `(new/old - 1)` 的快捷写法。
- 计算收益率用 `pct_change`；计算变化量用 `diff`。

## 常见坑

1. `pd.date_range(freq='M')` 在 Pandas 2.2+ 已弃用，改用 `'ME'`（月末）或 `'MS'`（月初）。
2. `resample` 的 `closed` 和 `label` 容易混淆：**日度以上默认 `closed='left'`**（含左端点），月度 / 年度默认 `closed='right'`。
3. 字符串切片 `ts['2023-01-15':'2023-01-20']` 终点**包含**，与普通切片不同。
4. `shift(1)` 数据往**下**移，不是往上；想象成"向未来推迟 1 期"更直观。
5. `rolling` 前 `window-1` 行是 `NaN`；不想要 `NaN` 用 `min_periods=1`。
6. 时间索引必须**排序**才能做切片；否则抛 `KeyError`。
7. 时区：`tz_localize` 和 `tz_convert` 不可混用——`localize` 给无时区打标签，`convert` 转换已有时区。

## 小结

- 时间序列的核心是 `DatetimeIndex`：支持字符串切片、`.dt` 访问器、`resample` / `rolling` 等专属操作。
- **降采样**（`resample`）聚合，**滚动**（`rolling`）不降频，**EWM** 是时间衰减平滑。
- `shift` / `pct_change` / `diff` 是构造时序特征的三件套。
- 记住频率字符串（`'D'` / `'W'` / `'ME'` / `'B'`）——它们贯穿整个时间序列 API。
