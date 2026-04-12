---
title: Pandas 数据可视化
outline: deep
---

# Pandas 数据可视化

> 对应脚本：`Basic/Pandas/08_visualization.py`  
> 运行方式：`python Basic/Pandas/08_visualization.py`（仓库根目录）

## 本章目标

1. 掌握 Pandas 内置绑定图的基本用法（`df.plot()`）。
2. 学会绘制折线图、柱状图、直方图、散点图、箱线图、饼图。
3. 理解 Pandas plot 与 Matplotlib 的关系。
4. 了解图表保存的基本流程。

## 重点方法速览

| 方法 | 图表类型 | 本章位置 |
|---|---|---|
| `df.plot()` | 折线图（默认） | `demo_line_plot` |
| `df.plot(kind="bar")` | 柱状图 | `demo_bar_plot` |
| `s.plot(kind="hist")` | 直方图 | `demo_histogram` |
| `df.plot(kind="scatter")` | 散点图 | `demo_scatter` |
| `df.plot(kind="box")` | 箱线图 | `demo_boxplot` |
| `s.plot(kind="pie")` | 饼图 | `demo_pie` |

## 前置说明

Pandas 的 `plot()` 方法底层调用 Matplotlib，因此：
- 需要 `import matplotlib.pyplot as plt`。
- 通过 `ax` 参数可与 Matplotlib 的 `Figure/Axes` 体系集成。
- 图表保存使用 `plt.savefig()`，交互显示使用 `plt.show()`。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## 1. 折线图

### 方法重点

- `df.plot()` 默认绘制折线图，每列一条线。
- 适合展示时间序列趋势。
- `title` 参数直接设置图标标题。

### 参数速览（本节）

适用 API：`df.plot(ax=None, kind='line', title=None, figsize=None, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `ax` | `ax`（Matplotlib Axes 对象） | 绑定到指定画布 |
| `kind` | `'line'`（默认） | 图表类型 |
| `title` | `"Sales and Profit Over Time"` | 图表标题 |

### 示例代码

```python
dates = pd.date_range("2023-01-01", periods=30, freq="D")
df = pd.DataFrame({
    "Sales": np.cumsum(np.random.randn(30)) + 100,
    "Profit": np.cumsum(np.random.randn(30)) + 50,
}, index=dates)

fig, ax = plt.subplots(figsize=(10, 5))
df.plot(ax=ax, title="Sales and Profit Over Time")
plt.tight_layout()
plt.savefig("pandas_line_plot.png", dpi=100)
plt.close()
```

### 理解重点

- DataFrame 的每一列自动成为一条折线，图例自动生成。
- `plt.tight_layout()` 避免标签被截断。
- `plt.close()` 在脚本中释放内存，避免图形叠加。

## 2. 柱状图

### 方法重点

- `kind="bar"` 绘制垂直柱状图，`kind="barh"` 绘制水平柱状图。
- 索引作为 x 轴标签，适合分类数据对比。

### 参数速览（本节）

适用 API：`df.plot(kind="bar", ax=None, title=None, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `"bar"` | 垂直柱状图 |
| `title` | `"Product Sales"` | 图表标题 |

### 示例代码

```python
df = pd.DataFrame({
    "Product": ["A", "B", "C", "D"],
    "Sales": [150, 200, 180, 220],
}).set_index("Product")

fig, ax = plt.subplots(figsize=(8, 5))
df.plot(kind="bar", ax=ax, title="Product Sales")
plt.tight_layout()
plt.savefig("pandas_bar_plot.png", dpi=100)
plt.close()
```

### 理解重点

- `set_index()` 将分类列设为索引，自动成为 x 轴标签。
- 多列 DataFrame 会自动绘制分组柱状图。
- `kind="barh"` 水平柱状图适合标签较长的情况。

## 3. 直方图

### 方法重点

- `kind="hist"` 展示数值分布。
- `bins` 参数控制分组数量，影响分布的粗细粒度。

### 参数速览（本节）

适用 API：`s.plot(kind="hist", bins=10, ax=None, title=None, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `"hist"` | 直方图 |
| `bins` | `30` | 分组数量 |
| `title` | `"Distribution"` | 图表标题 |

### 示例代码

```python
data = pd.Series(np.random.randn(1000))

fig, ax = plt.subplots(figsize=(8, 5))
data.plot(kind="hist", bins=30, ax=ax, title="Distribution")
plt.tight_layout()
plt.savefig("pandas_histogram.png", dpi=100)
plt.close()
```

### 理解重点

- `bins` 太少看不到细节，太多噪声过大——通常 20-50 是合理范围。
- `density=True` 可以将 y 轴改为概率密度。
- 直方图是数据探索的第一步，快速了解分布是否正态、有无偏斜。

## 4. 散点图

### 方法重点

- `kind="scatter"` 展示两个数值变量之间的关系。
- 需要明确指定 `x` 和 `y` 列名。

### 参数速览（本节）

适用 API：`df.plot(kind="scatter", x=..., y=..., ax=None, title=None, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `"scatter"` | 散点图 |
| `x` | `"x"` | x 轴对应的列名 |
| `y` | `"y"` | y 轴对应的列名 |
| `title` | `"Scatter Plot"` | 图表标题 |

### 示例代码

```python
df = pd.DataFrame({
    "x": np.random.randn(50),
    "y": np.random.randn(50),
})

fig, ax = plt.subplots(figsize=(8, 6))
df.plot(kind="scatter", x="x", y="y", ax=ax, title="Scatter Plot")
plt.tight_layout()
plt.savefig("pandas_scatter.png", dpi=100)
plt.close()
```

### 理解重点

- 散点图是发现变量相关性的快速工具。
- `c` 参数可以指定颜色列，`s` 参数可以指定点大小列——实现气泡图效果。
- 大数据集下点太多会重叠，可用 `alpha=0.5` 增加透明度。

## 5. 箱线图

### 方法重点

- `kind="box"` 展示数据的分位数分布和异常值。
- 箱体范围为 Q1-Q3（四分位距 IQR），须线延伸至 1.5 倍 IQR。
- 超出须线的点被标记为异常值。

### 参数速览（本节）

适用 API：`df.plot(kind="box", ax=None, title=None, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `"box"` | 箱线图 |
| `title` | `"Box Plot"` | 图表标题 |

### 示例代码

```python
df = pd.DataFrame({
    "A": np.random.normal(50, 10, 100),
    "B": np.random.normal(55, 15, 100),
})

fig, ax = plt.subplots(figsize=(8, 6))
df.plot(kind="box", ax=ax, title="Box Plot")
plt.tight_layout()
plt.savefig("pandas_boxplot.png", dpi=100)
plt.close()
```

### 理解重点

- 箱线图的五个要素：最小值、Q1、中位数（Q2）、Q3、最大值。
- 异常值检测的经典方法就是基于 IQR 的箱线图规则。
- 多列 DataFrame 自动并排绘制，适合对比不同变量的分布。

## 6. 饼图

### 方法重点

- `kind="pie"` 展示各部分占总体的比例。
- `autopct` 参数控制百分比标注的格式。
- 饼图适合类别较少（3-7 类）的场景。

### 参数速览（本节）

适用 API：`s.plot(kind="pie", ax=None, autopct=None, title=None, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `"pie"` | 饼图 |
| `autopct` | `"%1.1f%%"` | 百分比格式字符串 |
| `title` | `"Pie Chart"` | 图表标题 |

### 示例代码

```python
data = pd.Series([35, 25, 20, 15, 5], index=["A", "B", "C", "D", "E"])

fig, ax = plt.subplots(figsize=(8, 8))
data.plot(kind="pie", ax=ax, autopct="%1.1f%%", title="Pie Chart")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("pandas_pie.png", dpi=100)
plt.close()
```

### 理解重点

- `ax.set_ylabel("")` 去除默认的 y 轴标签，饼图不需要。
- `startangle=90` 可以从 12 点钟位置开始。
- 类别超过 7 个时，饼图可读性急剧下降，建议改用柱状图。

## 常见坑

1. 脚本中不调用 `plt.close()` 会导致多次绘图叠加。
2. `kind="scatter"` 必须指定 `x` 和 `y` 列名，否则报错。
3. 中文标签显示为方框——需要设置中文字体：`plt.rcParams['font.sans-serif'] = ['SimHei']`。
4. 饼图的 `autopct` 格式字符串中 `%%` 表示字面百分号。

## 小结

- Pandas 的 `plot()` 方法是快速数据可视化的入口，底层依赖 Matplotlib。
- 六种基本图表覆盖了趋势（折线）、对比（柱状）、分布（直方/箱线）、关系（散点）、占比（饼图）等常见需求。
- 需要精细定制时，应直接使用 Matplotlib 或 Seaborn。
