---
title: 图表
outline: deep
---

# 图表

> 对应脚本：`Basic/Visualization/02_matplotlib_charts.py`
> 运行方式：`python Basic/Visualization/02_matplotlib_charts.py`（仓库根目录）

## 导航

- [库生态总览](/foundations/overview)

## 本章目标

1. 掌握柱状图、散点图、直方图、饼图、箱线图的典型绘制流程。
2. 学会针对不同图表类型配置关键参数以提升可读性。
3. 理解统计分布与类别对比场景下的图表选型逻辑。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `ax.bar(...)` / `ax.barh(...)` | 展示类别间数值对比 | `demo_bar` |
| `ax.scatter(...)` | 展示变量关系与离散分布 | `demo_scatter` |
| `ax.hist(...)` | 展示数据频率分布 | `demo_histogram` |
| `ax.pie(...)` | 展示整体组成比例 | `demo_pie` |
| `ax.boxplot(...)` | 展示中位数、分位区间和异常值 | `demo_boxplot` |

## 1. 柱状图

### 方法重点

- `bar` 适合“类别-数值”比较，`barh` 适合类别标签较长的场景。
- 为柱子添加边框（`edgecolor`）可提升打印和投影场景可读性。
- 同一指标可用纵向和横向两种视角辅助解释。

### 参数速览（本节）

1. `matplotlib.axes.Axes.bar(x, height, color=None, edgecolor=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `['A', 'B', 'C', 'D', 'E']` | 类别标签 |
| `height` | `[23, 45, 56, 78, 32]` | 柱高（数值） |
| `color` | `'steelblue'` | 填充颜色 |
| `edgecolor` | `'black'` | 边框颜色 |
| 返回值 | `BarContainer` | 柱对象容器 |

2. `matplotlib.axes.Axes.barh(y, width, color=None, edgecolor=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y` | `['A', 'B', 'C', 'D', 'E']` | 类别标签 |
| `width` | `[23, 45, 56, 78, 32]` | 柱长度 |
| `color` | `'coral'` | 填充颜色 |
| `edgecolor` | `'black'` | 边框颜色 |
| 返回值 | `BarContainer` | 横向柱对象容器 |

### 示例代码

```python
import matplotlib.pyplot as plt

categories = ["A", "B", "C", "D", "E"]
values = [23, 45, 56, 78, 32]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(categories, values, color="steelblue", edgecolor="black")
axes[1].barh(categories, values, color="coral", edgecolor="black")
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/02_bar.png
----------------
左图为垂直柱状图，右图为水平柱状图
```

### 理解重点

- 类别比较优先柱状图，趋势比较优先折线图。
- 横向柱状图对长文本标签更友好。

## 2. 散点图

### 方法重点

- 散点图可同时编码位置、颜色、大小三个维度信息。
- `alpha` 可降低点重叠遮挡，适合高密度数据。
- 配合 colorbar 能把颜色映射转化为可解释变量。

### 参数速览（本节）

1. `matplotlib.axes.Axes.scatter(x, y, c=None, s=None, alpha=None, cmap=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `np.random.randn(100)` | 横轴数据 |
| `y` | `x + 噪声` | 纵轴数据 |
| `c` | `np.random.rand(100)` | 颜色映射值 |
| `s` | `abs(np.random.randn(100)) * 200` | 点大小 |
| `alpha` | `0.6` | 透明度 |
| `cmap` | `'viridis'` | colormap |
| 返回值 | `PathCollection` | 散点对象 |

2. `matplotlib.pyplot.colorbar(mappable, ax=None, label=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `mappable` | `scatter` | 要映射颜色的对象 |
| `ax` | `ax` | 附着的坐标轴 |
| `label` | `'Color Value'` | 颜色条标题 |
| 返回值 | `Colorbar` | 颜色条对象 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5
colors = np.random.rand(100)
sizes = np.abs(np.random.randn(100)) * 200

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap="viridis")
plt.colorbar(sc, ax=ax, label="Color Value")
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/02_scatter.png
----------------
图像内容: 点的颜色和大小分别编码额外变量
```

### 理解重点

- 当点重叠严重时，`alpha` 与采样策略要一起调整。
- 不同变量的编码优先级建议固定，避免读图歧义。

## 3. 直方图

### 方法重点

- 直方图用于观察分布形态、偏度和离散程度。
- `bins` 影响分辨率，过小会丢失细节，过大则噪声明显。
- 叠加均值线可快速定位中心位置。

### 参数速览（本节）

1. `matplotlib.axes.Axes.hist(x, bins=None, edgecolor=None, alpha=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `np.random.randn(1000)` | 输入样本 |
| `bins` | `30` | 直方图箱数 |
| `edgecolor` | `'black'` | 箱体边框颜色 |
| `alpha` | `0.7` | 透明度 |
| 返回值 | `n, bins, patches` | 频数、边界与图形对象 |

2. `matplotlib.axes.Axes.axvline(x, color=None, linestyle=None, label=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `data.mean()` | 竖线位置 |
| `color` | `'red'` | 线颜色 |
| `linestyle` | `'--'` | 线型 |
| `label` | `'Mean: ...'` | 图例名称 |
| 返回值 | `Line2D` | 竖线对象 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.randn(1000)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.2f}")
ax.legend()
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/02_histogram.png
----------------
图像内容: 正态近似分布并标注均值位置
```

### 理解重点

- 直方图不是概率密度，除非额外进行归一化。
- 结合均值、中位数线可更好判断偏态与异常值影响。

## 4. 饼图

### 方法重点

- 饼图适合少类别的占比表达，不适合精确比较接近比例。
- `explode` 可强调关键类别。
- `autopct` 能直接显示百分比，提升报告阅读效率。

### 参数速览（本节）

1. `matplotlib.axes.Axes.pie(x, labels=None, explode=None, autopct=None, startangle=None, colors=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `[35, 30, 20, 15]` | 各类别占比 |
| `labels` | `['Product A', ...]` | 类别标签 |
| `explode` | `(0.05, 0, 0, 0)` | 扇区偏移 |
| `autopct` | `'%1.1f%%'` | 百分比格式 |
| `startangle` | `90` | 起始角度 |
| `colors` | 自定义颜色列表 | 扇区颜色 |
| 返回值 | `wedges, texts, autotexts` | 饼图对象三元组 |

### 示例代码

```python
import matplotlib.pyplot as plt

labels = ["Product A", "Product B", "Product C", "Product D"]
sizes = [35, 30, 20, 15]
explode = (0.05, 0, 0, 0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, labels=labels, explode=explode, autopct="%1.1f%%", startangle=90)
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/02_pie.png
----------------
图像内容: Product A 扇区被突出显示
```

### 理解重点

- 类别超过 5~6 个时建议改用条形图。
- 比例对比不明显时应避免仅依赖角度感知。

## 5. 箱线图

### 方法重点

- 箱线图直接展示中位数、四分位区间与异常值。
- 多组箱线图适合比较组间波动差异。
- `patch_artist=True` 后可对箱体填色，增强分组辨识度。

### 参数速览（本节）

1. `matplotlib.axes.Axes.boxplot(x, patch_artist=False, labels=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | 4 组正态样本列表 | 输入样本（可多组） |
| `patch_artist` | `True` | 是否允许箱体填充颜色 |
| `labels` | `['σ=1', 'σ=2', 'σ=3', 'σ=4']` | 组标签 |
| 返回值 | `dict` | 各图元对象字典（boxes、whiskers 等） |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot(data, patch_artist=True)
for patch, color in zip(bp["boxes"], ["lightblue", "lightgreen", "lightyellow", "lightcoral"]):
	patch.set_facecolor(color)
ax.set_xticklabels(["σ=1", "σ=2", "σ=3", "σ=4"])
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/02_boxplot.png
----------------
图像内容: 四组不同标准差分布的中位数和离散度对比
```

### 理解重点

- 箱线图适合稳健比较，不依赖分布假设。
- 与直方图结合使用可同时获得整体形态与统计摘要。

