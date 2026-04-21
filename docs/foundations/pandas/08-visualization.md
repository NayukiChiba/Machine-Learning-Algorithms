---
title: Pandas 数据可视化
outline: deep
---

# Pandas 数据可视化

## 本章目标

1. 掌握 `df.plot` 的统一入口与 `kind` 参数选择图表类型。
2. 掌握折线图、柱状图、直方图、散点图、箱线图、饼图的写法。
3. 了解 Pandas 绘图与 Matplotlib 的关系（底层调用）。
4. 学会将图保存到文件。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.plot(...)` | 方法 | 统一绘图入口；默认绘制折线图 |
| `df.plot.line(...)` / `df.plot(kind='line')` | 方法 | 折线图 |
| `df.plot.bar(...)` / `df.plot(kind='bar')` | 方法 | 竖柱状图 |
| `df.plot.barh(...)` | 方法 | 横柱状图 |
| `df.plot.hist(...)` / `df.plot(kind='hist')` | 方法 | 直方图 |
| `df.plot.scatter(x, y, ...)` | 方法 | 散点图 |
| `df.plot.box(...)` | 方法 | 箱线图 |
| `df.plot.pie(...)` | 方法 | 饼图 |
| `df.plot.area(...)` | 方法 | 面积图 |
| `df.plot.kde(...)` / `density()` | 方法 | 核密度估计图 |
| `df.plot.hexbin(x, y, ...)` | 方法 | 六边形分箱图（高密度散点替代） |
| `plt.savefig(...)` | 函数 | 保存图像到文件 |

## 统一绘图接口

### `DataFrame.plot`

#### 作用

Pandas 提供的**统一绘图接口**，底层调用 Matplotlib。通过 `kind` 参数切换不同图表。支持链式访问：`df.plot.bar(...)` 等价于 `df.plot(kind='bar', ...)`。

#### 重点方法

```python
df.plot(x=None, y=None, kind='line', ax=None, subplots=False,
        sharex=None, sharey=False, layout=None, figsize=None,
        use_index=True, title=None, grid=None, legend=True,
        style=None, logx=False, logy=False, loglog=False,
        xticks=None, yticks=None, xlim=None, ylim=None,
        rot=None, fontsize=None, colormap=None, table=False,
        yerr=None, xerr=None, secondary_y=False, sort_columns=False,
        **kwds)
```

#### 关键参数

| 参数名       | 本例取值                                     | 说明                                                                 |
| ------------ | -------------------------------------------- | -------------------------------------------------------------------- |
| `x` / `y`    | 列名                                         | 指定用哪列作为横 / 纵轴                                              |
| `kind`       | `'line'`（默认）、`'bar'`、`'barh'`、`'hist'`、`'box'`、`'scatter'`、`'pie'`、`'area'`、`'kde'`、`'hexbin'` | 图表类型 |
| `ax`         | `None`、Matplotlib Axes                       | 在已有轴上绘制（便于组合布局）                                       |
| `subplots`   | `False`（默认）、`True`                      | `True` 时每列一个子图                                                |
| `figsize`    | `None`、`(8, 5)`                              | 画布尺寸（英寸）                                                     |
| `title`      | `None`、`"Sales"`                             | 图表标题                                                             |
| `legend`     | `True`（默认）                                | 是否显示图例                                                         |
| `grid`       | `None`（默认）、`True`                        | 是否显示网格                                                         |
| `rot`        | `None`（默认）                                | x 轴标签旋转角度                                                     |
| `colormap`   | `None`、`'viridis'`、`'coolwarm'`             | 颜色映射                                                             |
| `secondary_y`| `False`（默认）、列名列表                     | 指定某些列用次 y 轴                                                  |
| `logy` / `logx` | `False`（默认）                           | 坐标轴取对数                                                         |

## 折线图

### `df.plot.line`

#### 作用

按索引绘制折线图，多列自动叠加显示。最适合展示时间序列趋势。

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=30, freq="D")
df = pd.DataFrame({
    "Sales": np.cumsum(np.random.randn(30)) + 100,
    "Profit": np.cumsum(np.random.randn(30)) + 50,
}, index=dates)

fig, ax = plt.subplots(figsize=(10, 5))
df.plot(ax=ax, title="Sales and Profit Over Time")
plt.tight_layout()
plt.savefig("outputs/pandas/pandas_line_plot.png", dpi=100)
plt.close()
```

#### 输出

```text
图表已保存到 outputs/pandas/pandas_line_plot.png
```

![Line Plot](../../outputs/pandas/pandas_line_plot.png)

## 柱状图

### `df.plot.bar` / `df.plot.barh`

#### 作用

- `bar`：垂直柱状图（x = 类别，y = 数值）
- `barh`：水平柱状图（适合类别名较长时）

#### 重点方法

```python
df.plot.bar(x=None, y=None, stacked=False, **kwargs)
df.plot.barh(x=None, y=None, stacked=False, **kwargs)
```

#### 参数

| 参数名     | 本例取值        | 说明                                               |
| ---------- | --------------- | -------------------------------------------------- |
| `x` / `y`  | 列名            | 横 / 纵轴的列；省略时用索引作 x                    |
| `stacked`  | `False`（默认） | `True` 时堆叠柱状图                                |

#### 示例代码

```python
df = pd.DataFrame({
    "Product": ["A", "B", "C", "D"],
    "Sales": [150, 200, 180, 220],
}).set_index("Product")

fig, ax = plt.subplots(figsize=(8, 5))
df.plot(kind="bar", ax=ax, title="Product Sales")
plt.tight_layout()
plt.savefig("outputs/pandas/pandas_bar_plot.png", dpi=100)
plt.close()
```

#### 输出

```text
图表已保存到 outputs/pandas/pandas_bar_plot.png
```

![Bar Plot](../../outputs/pandas/pandas_bar_plot.png)

## 直方图

### `df.plot.hist`

#### 作用

绘制数据分布直方图。

#### 重点方法

```python
df.plot.hist(by=None, bins=10, **kwargs)
```

#### 参数

| 参数名 | 本例取值        | 说明                                                     |
| ------ | --------------- | -------------------------------------------------------- |
| `bins` | `10`（默认）、`30` | 区间数量或边界数组                                   |
| `by`   | `None`          | 分组列；为每组单独画一个直方图                           |

#### 示例代码

```python
data = pd.Series(np.random.randn(1000))
fig, ax = plt.subplots(figsize=(8, 5))
data.plot(kind="hist", bins=30, ax=ax, title="Distribution")
plt.tight_layout()
plt.savefig("outputs/pandas/pandas_histogram.png", dpi=100)
plt.close()
```

#### 输出

```text
图表已保存到 outputs/pandas/pandas_histogram.png
```

![Histogram](../../outputs/pandas/pandas_histogram.png)

## 散点图

### `df.plot.scatter`

#### 作用

绘制两列数据的散点图。可用第三列作颜色 / 大小维度，实现气泡图效果。

#### 重点方法

```python
df.plot.scatter(x, y, s=None, c=None, colormap=None, **kwargs)
```

#### 参数

| 参数名     | 本例取值             | 说明                                               |
| ---------- | -------------------- | -------------------------------------------------- |
| `x` / `y`  | 列名                 | 必需，分别作为横 / 纵轴                            |
| `s`        | `None`、标量、列名   | 点大小；可为列名映射为每点不同大小（气泡图）       |
| `c`        | `None`、颜色、列名   | 点颜色；可为列名映射为渐变色                       |
| `colormap` | `None`、`'viridis'`  | 颜色映射，与 `c` 配合                              |

#### 示例代码

```python
df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
fig, ax = plt.subplots(figsize=(8, 6))
df.plot(kind="scatter", x="x", y="y", ax=ax, title="Scatter Plot")
plt.tight_layout()
plt.savefig("outputs/pandas/pandas_scatter.png", dpi=100)
plt.close()
```

#### 输出

```text
图表已保存到 outputs/pandas/pandas_scatter.png
```

![Scatter Plot](../../outputs/pandas/pandas_scatter.png)

## 箱线图

### `df.plot.box`

#### 作用

绘制箱线图（四分位数 + 异常值），用于比较多个分布。

#### 重点方法

```python
df.plot.box(by=None, **kwargs)
```

#### 参数

| 参数名 | 本例取值 | 说明                         |
| ------ | -------- | ---------------------------- |
| `by`   | `None`、列名 | 按此列分组，每组画一个箱线 |

#### 示例代码

```python
df = pd.DataFrame({
    "A": np.random.normal(50, 10, 100),
    "B": np.random.normal(55, 15, 100),
})
fig, ax = plt.subplots(figsize=(8, 6))
df.plot(kind="box", ax=ax, title="Box Plot")
plt.tight_layout()
plt.savefig("outputs/pandas/pandas_boxplot.png", dpi=100)
plt.close()
```

#### 输出

```text
图表已保存到 outputs/pandas/pandas_boxplot.png
```

![Box Plot](../../outputs/pandas/pandas_boxplot.png)

## 饼图

### `df.plot.pie`

#### 作用

绘制饼图，展示组成比例。只能用于 Series 或 DataFrame 的单列。

#### 重点方法

```python
df.plot.pie(y=None, autopct=None, **kwargs)
```

#### 参数

| 参数名    | 本例取值       | 说明                                  |
| --------- | -------------- | ------------------------------------- |
| `y`       | 列名           | DataFrame 时必须指定列                |
| `autopct` | `None`、`'%1.1f%%'` | 百分比文本格式                   |

#### 示例代码

```python
data = pd.Series([35, 25, 20, 15, 5], index=["A", "B", "C", "D", "E"])
fig, ax = plt.subplots(figsize=(8, 8))
data.plot(kind="pie", ax=ax, autopct="%1.1f%%", title="Pie Chart")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("outputs/pandas/pandas_pie.png", dpi=100)
plt.close()
```

#### 输出

```text
图表已保存到 outputs/pandas/pandas_pie.png
```

![Pie Chart](../../outputs/pandas/pandas_pie.png)

## 其他常用图

| 图类型 | `kind` 值 | 用途 |
|---|---|---|
| 面积图 | `'area'` | 堆叠时间序列占比 |
| 核密度估计 | `'kde'` / `'density'` | 平滑版直方图 |
| 六边形分箱图 | `'hexbin'`（需要 `x`/`y`） | 大数据量散点替代 |

## 常见坑

1. `df.plot` **默认画折线**；数据不适合折线（如类别数据）要显式指定 `kind`。
2. 保存图前**必须先 `plt.tight_layout()`**，否则标签可能被裁剪。
3. `plt.savefig` 之后要调用 `plt.close()` 关闭图，否则多次循环会积累内存。
4. 中文标题 / 标签需配置 Matplotlib 字体：`plt.rcParams['font.sans-serif'] = ['SimHei']`。
5. DataFrame 多列画饼图会报错；饼图只能用单列 Series 或指定 `y='col'`。
6. 数据规模太大时直接 `plot` 很慢，用 `hexbin` 或 `sample` 降采样后再画。

## 小结

- Pandas 的 `.plot` 是 Matplotlib 的**便捷封装**，适合快速可视化。
- 通过 `kind` 切换图表类型，参数几乎与 Matplotlib 同名。
- 精细控制布局仍需要直接用 Matplotlib（`fig, ax = plt.subplots(...)` 再传 `ax` 给 `plot`）。
- 复杂统计图（热力图、小提琴图、配对图）建议用 Seaborn 或 Matplotlib 原生接口。

