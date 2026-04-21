---
title: Pandas 图
outline: deep
---

# Pandas 图

> 对应脚本：`Basic/Visualization/04_pandas_viz.py`
> 运行方式：`python -m Basic.Visualization.04_pandas_viz`（仓库根目录）

## 导航

- [库生态总览](/foundations/overview)

## 本章目标

1. 掌握 `DataFrame.plot` 与 `Series.plot` 的常见图形类型与参数。
2. 理解 Pandas 绘图与 Matplotlib `Axes` 之间的协作关系。
3. 学会通过分组聚合结果快速绘制业务对比图。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `DataFrame.plot(...)` | 对多列时间序列或表格数据快速绘图 | `demo_df_plot` |
| `Series.plot(...)` | 对单变量序列快速绘图 | `demo_series_plot` |
| `DataFrame.groupby(...).mean()` | 分组聚合后绘图 | `demo_groupby_plot` |
| `plt.subplots(...)` | 组织多图布局 | `demo_df_plot` / `demo_series_plot` |

## 1. DataFrame.plot()

### 方法重点

- Pandas 绘图默认基于 Matplotlib，适合快速原型分析。
- 同一个 DataFrame 可用 `kind` 参数切换线图、面积图、箱线图等。
- 复杂布局建议先用 `plt.subplots` 创建 `Axes`，再把图绑定到指定子图。

### 参数速览（本节）

1. `pandas.DataFrame.plot(kind='line', ax=None, title=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `'line'` | 折线图 |
| `ax` | `axes[0, 0]` | 绘图目标坐标轴 |
| `title` | `'Line Plot'` | 子图标题 |
| 返回值 | `Axes` | Matplotlib 坐标轴对象 |

2. `pandas.DataFrame.plot(kind='area', ax=None, alpha=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `'area'` | 面积图 |
| `ax` | `axes[0, 1]` | 绘图目标坐标轴 |
| `alpha` | `0.5` | 透明度 |
| 返回值 | `Axes` | 坐标轴对象 |

3. `pandas.DataFrame.plot(kind='box', ax=None, title=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `'box'` | 箱线图 |
| `ax` | `axes[1, 1]` | 绘图目标坐标轴 |
| `title` | `'Box Plot'` | 子图标题 |
| 返回值 | `Axes` | 坐标轴对象 |

### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dates = pd.date_range("2023-01-01", periods=30, freq="D")
df = pd.DataFrame({
	"A": np.cumsum(np.random.randn(30)),
	"B": np.cumsum(np.random.randn(30)),
	"C": np.cumsum(np.random.randn(30)),
}, index=dates)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
df.plot(ax=axes[0, 0], title="Line Plot")
df.plot(kind="box", ax=axes[1, 1], title="Box Plot")
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/04_df_plot.png
----------------
图像内容: 线图、面积图、条形图、箱线图四宫格对比
```

### 理解重点

- `DataFrame.plot` 适合快速探索，不必每次手写 Matplotlib 底层语句。
- 当图形语义复杂时，可以混合使用 Pandas 与 Matplotlib API。

## 2. Series.plot()

### 方法重点

- `Series.plot` 是单变量分析最便捷入口。
- 通过 `kind='hist'` 可快速切换到分布视角。
- 同一序列可并行展示趋势图与分布图，互相校验。

### 参数速览（本节）

1. `pandas.Series.plot(kind='line', ax=None, title=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `'line'` | 折线图 |
| `ax` | `axes[0]` | 目标坐标轴 |
| `title` | `'Line Plot'` | 子图标题 |
| 返回值 | `Axes` | 坐标轴对象 |

2. `pandas.Series.plot(kind='hist', bins=None, ax=None, title=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `'hist'` | 直方图 |
| `bins` | `20` | 分箱数 |
| `ax` | `axes[1]` | 目标坐标轴 |
| `title` | `'Histogram'` | 子图标题 |
| 返回值 | `Axes` | 坐标轴对象 |

### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s = pd.Series(np.random.randn(100).cumsum())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
s.plot(ax=axes[0], title="Line Plot")
s.plot(kind="hist", bins=20, ax=axes[1], title="Histogram")
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/04_series_plot.png
----------------
左图展示累计走势，右图展示取值分布
```

### 理解重点

- 趋势图回答“怎么变化”，直方图回答“分布在哪里”。
- 单变量分析阶段建议两个视角同时保留。

## 3. GroupBy 绘图

### 方法重点

- 分组聚合是业务分析中最常见的数据预处理步骤。
- 先 `groupby` 再 `mean` 可压缩噪声，强调组间差异。
- 聚合结果是 Series，可直接使用 `plot(kind='bar')` 绘制。

### 参数速览（本节）

1. `pandas.DataFrame.groupby(by)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `by` | `'Category'` | 分组字段 |
| 返回值 | `DataFrameGroupBy` | 分组对象 |

2. `pandas.core.groupby.SeriesGroupBy.mean()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `Series` | 每个类别的均值结果 |

3. `pandas.Series.plot(kind='bar', ax=None, color=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `'bar'` | 条形图 |
| `ax` | `ax` | 目标坐标轴 |
| `color` | `['red', 'green', 'blue']` | 每个柱的颜色 |
| 返回值 | `Axes` | 坐标轴对象 |

### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
	"Category": np.repeat(["A", "B", "C"], 20),
	"Value": np.random.randn(60),
})

fig, ax = plt.subplots(figsize=(8, 6))
df.groupby("Category")["Value"].mean().plot(kind="bar", ax=ax, color=["red", "green", "blue"])
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/04_groupby.png
----------------
图像内容: A/B/C 三个类别的均值对比柱状图
```

### 理解重点

- 先聚合后绘图能显著降低噪音干扰。
- 分组统计应同时配合样本量信息，避免均值误导。

