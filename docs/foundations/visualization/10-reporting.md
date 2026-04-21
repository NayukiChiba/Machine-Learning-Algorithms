---
title: 报告
outline: deep
---

# 报告

> 对应脚本：`Basic/Visualization/10_reporting.py`
> 运行方式：`python -m Basic.Visualization.10_reporting`（仓库根目录）

## 导航

- [库生态总览](/foundations/overview)

## 本章目标

1. 掌握专业报告图的样式统一、布局设计和输出规范。
2. 学会使用 GridSpec 构建复杂多面板可视化版式。
3. 理解导出参数与配色体系对交付质量的影响。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `plt.style.use(...)` | 统一图表风格模板 | `demo_professional_style` |
| `fig.add_gridspec(...)` | 创建复杂网格布局 | `demo_multi_panel` |
| `fig.add_subplot(...)` | 在布局中添加子图 | `demo_multi_panel` |
| `plt.savefig(...)` | 导出高分辨率图像 | `demo_professional_style` / `demo_multi_panel` / `demo_export` |
| `plt.cm.*` | 使用内置色图管理配色 | `demo_color_palettes` |

## 1. 专业样式设置

### 方法重点

- 报告图首要目标是可读性一致，而非单图视觉炫技。
- `style.use` 可统一网格线、字体、背景等全局风格。
- 标题、轴标签、图例应形成固定层级规范。

### 参数速览（本节）

1. `matplotlib.pyplot.style.use(style)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `style` | `'seaborn-v0_8-whitegrid'` | 全局样式模板 |
| 返回值 | `None` | 修改全局 rc 配置 |

2. `matplotlib.axes.Axes.set_title(label, fontsize=None, fontweight=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `label` | `'Professional Style Chart'` | 标题文本 |
| `fontsize` | `14` | 标题字号 |
| `fontweight` | `'bold'` | 标题字重 |
| 返回值 | `Text` | 标题对象 |

3. `matplotlib.figure.Figure.savefig(fname, dpi=None, bbox_inches=None, transparent=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `fname` | `output_dir / '10_professional.png'` | 输出文件路径 |
| `dpi` | `150` | 分辨率 |
| 返回值 | `None` | 写出图像文件 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), linewidth=2, label="sin(x)")
ax.plot(x, np.cos(x), linewidth=2, label="cos(x)")
ax.set_title("Professional Style Chart", fontsize=14, fontweight="bold")
ax.legend(frameon=True, fancybox=True, shadow=True)
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/10_professional.png
----------------
图像内容: 统一网格风格、标题层级和图例外观
```

### 理解重点

- 风格一致性比单图复杂度更能提升报告专业感。
- 建议把常用样式配置固化为团队模板。

## 2. 多面板布局

### 方法重点

- GridSpec 支持不规则布局，适合仪表盘和报告页组合图。
- 一个主图配多个辅助图是最常见的讲故事结构。
- 布局阶段就应确定主次关系与读图顺序。

### 参数速览（本节）

1. `matplotlib.figure.Figure.add_gridspec(nrows, ncols, hspace=None, wspace=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `nrows` | `2` | 行数 |
| `ncols` | `3` | 列数 |
| `hspace` | `0.3` | 行间距 |
| `wspace` | `0.3` | 列间距 |
| 返回值 | `GridSpec` | 网格规格对象 |

2. `matplotlib.figure.Figure.add_subplot(*args)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `*args` | `gs[0, :2]` 等 | 子图所在网格切片 |
| 返回值 | `Axes` | 新建坐标轴对象 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/10_multipanel.png
----------------
图像内容: 上方主图 + 右上分布图 + 下方三图组合布局
```

### 理解重点

- 复杂布局先画草图再编码，效率更高。
- 主图面积通常应大于辅助图，避免重点分散。

## 3. 导出选项

### 方法重点

- 不同交付场景需要不同导出格式和分辨率策略。
- PNG 适合网页，PDF/SVG 适合矢量打印与论文。
- `bbox_inches='tight'` 能有效减少多余留白。

### 参数速览（本节）

1. `matplotlib.pyplot.savefig(fname, dpi=None, bbox_inches=None, transparent=False, facecolor='auto')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `fname` | `'fig.png'` / `'fig.pdf'` / `'fig.svg'` | 输出路径与格式 |
| `dpi` | `300` | 分辨率（位图格式有效） |
| `bbox_inches` | `'tight'` | 紧凑边界 |
| `transparent` | `True` / `False` | 是否透明背景 |
| `facecolor` | `'white'` | 背景色 |
| 返回值 | `None` | 写出文件 |

### 示例代码

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

plt.savefig("fig.png", dpi=300, bbox_inches="tight")
plt.savefig("fig.pdf", bbox_inches="tight")
plt.savefig("fig.svg", transparent=True)
```

### 结果输出（示例）

```text
输出格式: PNG / PDF / SVG / EPS
----------------
推荐策略: 报告预览用 PNG，正式发布优先 PDF 或 SVG
```

### 理解重点

- 导出前先确认下游使用场景，避免重复返工。
- 位图和矢量格式应并行保留，兼顾兼容性和质量。

## 4. 配色方案

### 方法重点

- 配色应服务信息层级，而不是追求“颜色多”。
- 连续变量、发散变量、类别变量应使用不同色图类别。
- 团队报告建议固定主色板与强调色，保证视觉一致。

### 参数速览（本节）

1. `matplotlib.axes.Axes.scatter(x, y, c=None, cmap=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x, y` | 数据序列 | 点坐标 |
| `c` | `values` | 颜色映射值 |
| `cmap` | `'viridis'` | 连续色图 |
| 返回值 | `PathCollection` | 散点对象 |

2. `matplotlib.cm.get_cmap(name, lut=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `name` | `'Set1'` / `'viridis'` | 色图名称 |
| `lut` | 默认 | 采样级数 |
| 返回值 | `Colormap` | 色图对象 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

print("顺序色图: viridis, plasma, magma, cividis")
print("发散色图: coolwarm, RdBu, seismic")
print("定性色图: Set1, Set2, tab10, Pastel1")

colors = plt.cm.Set1(np.linspace(0, 1, 10))
```

### 结果输出（示例）

```text
色图类型:
	顺序型用于数值大小编码
----------------
	定性型用于类别区分
```

### 理解重点

- 颜色体系应与业务语义绑定，例如红色表示风险、绿色表示健康。
- 建议做色盲友好检查，避免关键信息仅靠颜色传达。

