---
title: 交互
outline: deep
---

# 交互

> 对应脚本：`Basic/Visualization/09_interactive.py`
> 运行方式：`python Basic/Visualization/09_interactive.py`（仓库根目录）

## 导航

- [库生态总览](/foundations/overview)

## 本章目标

1. 理解 Plotly 交互式图表的核心 API 与工作流。
2. 掌握常见交互图类型（折线、散点、柱状、3D）的构建方法。
3. 学会将交互图导出为 HTML 或静态图片用于报告交付。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `px.scatter(...)` | 快速构建交互散点图 | `demo_plotly_basics` / `demo_interactive_chart` |
| `px.line(...)` | 构建交互折线图 | `demo_interactive_chart` |
| `px.bar(...)` | 构建交互柱状图 | `demo_interactive_chart` |
| `px.scatter_3d(...)` | 构建三维交互散点图 | `demo_interactive_chart` |
| `fig.update_layout(...)` | 统一图表布局样式 | `demo_plotly_tips` |
| `fig.write_html(...)` / `fig.write_image(...)` | 导出图表文件 | `demo_plotly_tips` |

## 1. Plotly 基础

### 方法重点

- Plotly 图表默认支持缩放、平移、悬停提示等交互动作。
- Plotly Express 适合快速构建，Graph Objects 适合精细控制。
- 在 Notebook 与 Web 报告中，Plotly 的交互优势明显。

### 参数速览（本节）

1. `plotly.express.scatter(data_frame, x=None, y=None, color=None, size=None, title=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data_frame` | `df` | 输入 DataFrame |
| `x` | `'x'` | 横轴字段 |
| `y` | `'y'` | 纵轴字段 |
| `color` | `'category'` | 颜色映射字段 |
| `size` | 可选 | 点大小映射字段 |
| `title` | 可选 | 图标题 |
| 返回值 | `plotly.graph_objs._figure.Figure` | Plotly 图对象 |

2. `plotly.graph_objects.Figure.show()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `None` | 在当前环境渲染交互图 |

### 示例代码

```python
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
	"x": [1, 2, 3, 4],
	"y": [2, 3, 2, 5],
	"category": ["A", "A", "B", "B"],
})

fig = px.scatter(df, x="x", y="y", color="category", title="Plotly Scatter")
fig.show()
```

### 结果输出（示例）

```text
交互能力: 鼠标悬停显示点信息，滚轮缩放坐标轴
----------------
运行结果: 浏览器或 Notebook 渲染可交互散点图
```

### 理解重点

- Plotly 的“可交互默认值”降低了前端开发成本。
- 先用 Express 快速验证，再按需下沉到 Graph Objects。

## 2. 交互式图表实例

### 方法重点

- 折线、散点、柱状、3D 散点是最常见的业务展示组合。
- 相同数据在不同图形中关注重点不同，应按问题选图。
- 交互图允许读者自己探索局部细节，提升分析透明度。

### 参数速览（本节）

1. `plotly.express.line(data_frame, x=None, y=None, color=None, title=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data_frame` | `df` | 输入 DataFrame |
| `x` | `'date'` | 时间轴字段 |
| `y` | `'value'` | 数值字段 |
| `title` | `'Time Series'` | 图标题 |
| 返回值 | `Figure` | Plotly 图对象 |

2. `plotly.express.bar(data_frame, x=None, y=None, color=None, title=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data_frame` | `df` | 输入 DataFrame |
| `x` | `'category'` | 类别字段 |
| `y` | `'value'` | 数值字段 |
| `color` | `'group'` | 分组上色字段 |
| `title` | 可选 | 图标题 |
| 返回值 | `Figure` | Plotly 图对象 |

3. `plotly.express.scatter_3d(data_frame, x=None, y=None, z=None, color=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data_frame` | `df` | 输入 DataFrame |
| `x, y, z` | `'x'`, `'y'`, `'z'` | 三维坐标字段 |
| `color` | `'label'` | 分类字段 |
| 返回值 | `Figure` | 三维图对象 |

### 示例代码

```python
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
	"date": pd.date_range("2024-01-01", periods=6),
	"value": [10, 12, 9, 15, 18, 17],
	"category": ["A", "B", "A", "B", "A", "B"],
})

fig_line = px.line(df, x="date", y="value", title="Time Series")
fig_bar = px.bar(df, x="category", y="value", color="category", title="Category Value")
```

### 结果输出（示例）

```text
折线图: 可缩放时间区间并查看局部波动
----------------
柱状图: 可点击图例切换类别显示
```

### 理解重点

- 交互式图表适合面向业务方的自助探索场景。
- 图形越多越要统一颜色和命名，降低认知负担。

## 3. Plotly 实用技巧

### 方法重点

- 导出 HTML 可保留完整交互能力，适合分享与归档。
- 导出静态图片适合论文、报告与邮件场景。
- 统一布局配置是构建图表风格系统的关键步骤。

### 参数速览（本节）

1. `plotly.graph_objects.Figure.update_layout(**kwargs)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `title` | `'Chart Title'` | 图标题 |
| `xaxis_title` | `'X Axis'` | X 轴标题 |
| `yaxis_title` | `'Y Axis'` | Y 轴标题 |
| `template` | `'plotly_dark'` | 全局主题模板 |
| 返回值 | `Figure` | 更新后的图对象 |

2. `plotly.graph_objects.Figure.write_html(file)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `file` | `'chart.html'` | 导出文件路径 |
| 返回值 | `None` | 写出 HTML 文件 |

3. `plotly.graph_objects.Figure.write_image(file)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `file` | `'chart.png'` | 导出图片路径 |
| 返回值 | `None` | 写出静态图片 |

### 示例代码

```python
import pandas as pd
import plotly.express as px

df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 1, 4]})
fig = px.line(df, x="x", y="y", title="Demo")
fig.update_layout(title="Chart Title", xaxis_title="X Axis", yaxis_title="Y Axis", template="plotly_dark")
fig.write_html("chart.html")
```

### 结果输出（示例）

```text
导出结果: chart.html 可在浏览器独立打开
----------------
可选导出: chart.png 适合静态文档嵌入
```

### 理解重点

- 导出策略取决于读者是否需要交互能力。
- 统一模板和标题规范能显著提升团队交付质量。


