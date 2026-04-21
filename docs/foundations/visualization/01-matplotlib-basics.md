---
title: Matplotlib 基础
outline: deep
---

# Matplotlib 基础

> 对应脚本：`Basic/Visualization/01_matplotlib_basics.py`
> 运行方式：`python -m Basic.Visualization.01_matplotlib_basics`（仓库根目录）

## 导航

- [库生态总览](/foundations/overview)

## 本章目标

1. 理解 Figure、Axes、Axis 三层对象结构以及创建方式。
2. 掌握 `plot` 的线型、标记、颜色等高频可视化参数。
3. 学会使用 `subplots` 快速构建多图布局并保存输出。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `plt.subplots(...)` | 创建 Figure 与 Axes 容器 | `demo_figure_axes` / `demo_subplots` |
| `ax.plot(...)` | 绘制折线并配置线型、颜色、标记 | `demo_figure_axes` / `demo_line_styles` / `demo_markers` |
| `ax.legend(...)` | 管理图例展示 | `demo_figure_axes` / `demo_markers` |
| `plt.tight_layout()` | 自动调整子图间距 | 全章节 |
| `plt.savefig(...)` | 将图表写入输出目录 | 全章节 |

## 1. Figure 和 Axes

### 方法重点

- `plt.subplots` 返回 `(fig, ax)`，其中 `fig` 是画布，`ax` 是绘图区域。
- 多条曲线可在同一个 `Axes` 上叠加，配合图例便于对比。
- 轴标签、标题、网格属于最基础的读图语义信息，应显式设置。

### 参数速览（本节）

1. `matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=None, **fig_kw)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `nrows` | `1` | 子图行数 |
| `ncols` | `1` | 子图列数 |
| `figsize` | `(8, 5)` | 画布尺寸（英寸） |
| 返回值 | `(Figure, Axes)` | 画布对象与单个坐标轴对象 |

2. `matplotlib.axes.Axes.plot(x, y, label=None, color=None, linestyle='-')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `np.linspace(0, 10, 100)` | 横轴序列 |
| `y` | `np.sin(x)` / `np.cos(x)` | 纵轴序列 |
| `label` | `"sin(x)"` / `"cos(x)"` | 图例名称 |
| 返回值 | `Line2D` 列表 | 绘制出的线对象列表 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Basic Plot")
ax.legend()
ax.grid(True)
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/01_basic.png
----------------
图像内容: 同一坐标轴上展示 sin(x) 与 cos(x) 两条曲线
```

### 理解重点

- 把 `Figure` 理解为“画布”，`Axes` 理解为“具体图表区域”。
- 任何复杂布局都可以拆解成多个 `Axes` 的组合。

## 2. 线条样式

### 方法重点

- `plot` 支持通过格式字符串快速指定颜色和线型（如 `"r-"`、`"g--"`）。
- `linewidth` 可以显著改善可读性，建议在对比图中统一设置。
- 线型差异是彩色和灰阶打印都可区分的重要编码方式。

### 参数速览（本节）

1. `matplotlib.axes.Axes.plot(x, y, fmt, linewidth=1.5, label=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `fmt` | `"r-"`、`"g--"`、`"b:"`、`"m-."` | 颜色与线型快捷写法 |
| `linewidth` | `2` | 线宽 |
| `label` | `"solid"` 等 | 图例名称 |
| 返回值 | `Line2D` 列表 | 当前线条对象 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 50)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), "r-", linewidth=2, label="solid")
ax.plot(x, np.sin(x + 0.5), "g--", linewidth=2, label="dashed")
ax.plot(x, np.sin(x + 1.0), "b:", linewidth=2, label="dotted")
ax.plot(x, np.sin(x + 1.5), "m-.", linewidth=2, label="dashdot")
ax.legend()
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/01_line_styles.png
----------------
常用线型: - / -- / : / -.
```

### 理解重点

- 线型应优先用于“系列区分”，颜色用于“语义强调”。
- 同时设置 `label` 与 `legend` 是对比图最小闭环。

## 3. 标记符号

### 方法重点

- 标记（marker）可以突出离散采样点，适合小样本展示。
- `markersize` 决定视觉密度，过大容易遮挡趋势。
- 多序列情况下，图例列数可通过 `legend(ncol=...)` 控制紧凑布局。

### 参数速览（本节）

1. `matplotlib.axes.Axes.plot(x, y, marker=None, markersize=None, label=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `marker` | `"o"`、`"s"`、`"^"`、`"D"` 等 | 标记符号样式 |
| `markersize` | `8` | 标记大小 |
| `label` | `"'o'"` 等 | 图例中显示的系列名 |
| 返回值 | `Line2D` 列表 | 线与标记对象 |

2. `matplotlib.axes.Axes.legend(ncol=1, **kwargs)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `ncol` | `4` | 图例列数 |
| 返回值 | `Legend` | 图例对象 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 10)
markers = ["o", "s", "^", "D", "v", "p", "*", "x"]

fig, ax = plt.subplots(figsize=(10, 6))
for i, m in enumerate(markers):
		ax.plot(x, np.sin(x) + i * 0.5, marker=m, label=f"'{m}'", markersize=8)
ax.legend(ncol=4)
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/01_markers.png
----------------
图像内容: 8 种 marker 在同一图内对比
```

### 理解重点

- 标记是离散信息编码，不应替代颜色和线型的主语义。
- 数据点很多时建议降低 `alpha` 或减少 marker 使用。

## 4. 颜色设置

### 方法重点

- Matplotlib 支持单字符、颜色名、十六进制、RGB 元组与 colormap 多种写法。
- 在团队协作中建议固定调色板，避免每张图配色风格漂移。
- 顺序型数据优先使用连续 colormap，类别型数据优先使用离散色板。

### 参数速览（本节）

1. `matplotlib.axes.Axes.plot(x, y, color=None, label=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `color` | `'r'`、`'red'`、`'#FF5733'`、`(0.1, 0.2, 0.5)` | 颜色指定方式 |
| `label` | `"series name"` | 图例名称 |
| 返回值 | `Line2D` 列表 | 绘图对象 |

2. `matplotlib.cm.get_cmap(name)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `name` | `'viridis'` | colormap 名称 |
| 返回值 | `Colormap` | 颜色映射对象 |

### 示例代码

```python
import matplotlib.pyplot as plt

print("单字符:", "r", "g", "b")
print("颜色名:", "red", "green", "blue")
print("十六进制:", "#FF5733")
print("RGB元组:", (0.1, 0.2, 0.5))
print("Colormap:", plt.cm.viridis)
```

### 结果输出（示例）

```text
颜色指定方式:
	单字符: 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
----------------
	名称/十六进制/RGB/Colormap 均可用于统一配色
```

### 理解重点

- 颜色不是装饰，而是编码变量和强调信息的工具。
- 在深浅背景切换时，优先验证颜色对比度是否足够。

## 5. 子图布局

### 方法重点

- `subplots(2, 2)` 可一次性创建网格布局，适合多指标对照。
- `axes[i, j]` 访问单个子图，配置方式与普通 `ax` 完全一致。
- 保存前调用 `tight_layout` 可以避免标题和坐标标签重叠。

### 参数速览（本节）

1. `matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=None, **fig_kw)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `nrows` | `2` | 子图行数 |
| `ncols` | `2` | 子图列数 |
| `figsize` | `(10, 8)` | 画布尺寸 |
| 返回值 | `(Figure, ndarray[Axes])` | 画布对象与坐标轴数组 |

2. `matplotlib.pyplot.tight_layout()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `None` | 原地调整子图边距 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, np.sin(x)); axes[0, 0].set_title("sin(x)")
axes[0, 1].plot(x, np.cos(x)); axes[0, 1].set_title("cos(x)")
axes[1, 0].plot(x, np.exp(-x / 5) * np.sin(x)); axes[1, 0].set_title("Damped")
axes[1, 1].plot(x, x**2); axes[1, 1].set_title("x²")
plt.tight_layout()
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/01_subplots.png
----------------
图像内容: 2x2 子图分别展示不同函数形态
```

### 理解重点

- 子图布局是“同一视图比较”最有效的表达方式。
- 建议保持统一配色和字体，避免多图布局视觉噪音。

