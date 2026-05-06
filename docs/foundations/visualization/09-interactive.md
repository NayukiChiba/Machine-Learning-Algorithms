---
title: Plotly 交互式图表
outline: deep
---

# Plotly 交互式图表

## 本章目标

1. 理解 Plotly 交互式图表的核心 API 与工作流
2. 掌握常见交互图类型（折线、散点、柱状、3D）的构建方法
3. 学会将交互图导出为 HTML 或静态图片用于报告交付

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `px.scatter(df, x, y)` | 函数 | 快速构建交互散点图 |
| `px.line(df, x, y)` | 函数 | 构建交互折线图 |
| `px.bar(df, x, y)` | 函数 | 构建交互柱状图 |
| `px.scatter_3d(df, x, y, z)` | 函数 | 构建三维交互散点图 |
| `fig.update_layout(**kwargs)` | 方法 | 统一图表布局样式 |
| `fig.write_html(file)` | 方法 | 导出 HTML 文件（保留交互） |
| `fig.write_image(file)` | 方法 | 导出静态图片 |

## 1. Plotly 基础

### `plotly.express.scatter`

#### 作用

Plotly 图表默认支持缩放、平移、悬停提示等交互动作。Plotly Express 适合快速构建，Graph Objects 适合精细控制。在 Notebook 与 Web 报告中，Plotly 的交互优势明显——无需编写 JavaScript。

#### 重点方法

```python
px.scatter(data_frame=None, *, x=None, y=None, color=None, size=None,
           title=None)
fig.show()         # 在当前环境渲染交互图
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data_frame` | `DataFrame` | 输入数据 | `df` |
| `x` | `str` | 横轴字段 | `"x"` |
| `y` | `str` | 纵轴字段 | `"y"` |
| `color` | `str` | 颜色映射字段 | `"category"` |
| `size` | `str` | 点大小映射字段 | `"value"` |
| `title` | `str` | 图标题 | `"Plotly Scatter"` |

#### 示例代码

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

#### 输出

```text
交互能力: 鼠标悬停显示点信息，滚轮缩放坐标轴
运行结果: 浏览器或 Notebook 渲染可交互散点图
```

#### 理解重点

- Plotly 的"可交互默认值"降低了前端开发成本
- 先用 Express 快速验证，再按需下沉到 Graph Objects
- `fig.show()` 在脚本中会打开浏览器，在 Notebook 中内嵌渲染

## 2. 交互式图表实例

### `px.line` / `px.bar` / `px.scatter_3d`

#### 作用

折线、散点、柱状、3D 散点是最常见的业务展示组合。相同数据在不同图形中关注重点不同，应按问题选图。交互图允许读者自己探索局部细节，提升分析透明度。

#### 重点方法

```python
px.line(data_frame, *, x=None, y=None, color=None, title=None)
px.bar(data_frame, *, x=None, y=None, color=None, title=None)
px.scatter_3d(data_frame, *, x=None, y=None, z=None, color=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data_frame` | `DataFrame` | 输入数据 | `df` |
| `x` / `y` | `str` | 横/纵轴字段 | `"date"`, `"value"` |
| `color` | `str` | 分组上色字段 | `"group"` |
| `z` | `str` | 3D 图的第三轴字段 | `"z"` |
| `title` | `str` | 图标题 | `"Time Series"` |

#### 示例代码

```python
import pandas as pd
import plotly.express as px
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=30, freq="D"),
    "value": np.cumsum(np.random.randn(30)),
    "category": np.random.choice(["A", "B"], 30),
})

figLine = px.line(df, x="date", y="value", title="Time Series")
figBar = px.bar(df, x="category", y="value", color="category",
                title="Category Value")
```

#### 输出

```text
折线图: 可缩放时间区间并查看局部波动
柱状图: 可点击图例切换类别显示
```

#### 理解重点

- 交互式图表适合面向业务方的自助探索场景
- 图形越多越要统一颜色和命名，降低认知负担
- 3D 散点图适合展示聚类或降维结果——可旋转视角

## 3. Plotly 实用技巧

### 布局与导出

#### 作用

导出 HTML 可保留完整交互能力，适合分享与归档。导出静态图片适合论文、报告与邮件场景。统一布局配置是构建图表风格系统的关键步骤。

#### 重点方法

```python
fig.update_layout(*, title=None, xaxis_title=None, yaxis_title=None,
                  template=None)
fig.write_html(file)           # 导出交互式 HTML
fig.write_image(file)          # 导出静态图片（需 kaleido）
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `title` | `str` | 图标题 | `"Chart Title"` |
| `xaxis_title` / `yaxis_title` | `str` | 轴标题 | `"X Axis"`, `"Y Axis"` |
| `template` | `str` | 全局主题：`"plotly"` / `"plotly_dark"` / `"ggplot2"` / `"seaborn"` 等 | `"plotly_dark"` |
| `file` | `str` | 导出文件路径 | `"chart.html"` |

#### 示例代码

```python
import pandas as pd
import plotly.express as px

df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 1, 4]})
fig = px.line(df, x="x", y="y", title="Demo")
fig.update_layout(
    title="Styled Chart",
    xaxis_title="X Axis",
    yaxis_title="Y Axis",
    template="plotly_dark",
)
fig.write_html("chart.html")
print("已导出: chart.html")
```

#### 输出

```text
导出结果: chart.html 可在浏览器独立打开
可选导出: fig.write_image("chart.png") 适合静态文档嵌入
```

#### 理解重点

- 导出策略取决于读者是否需要交互能力
- 统一模板和标题规范能显著提升团队交付质量
- `write_image` 需要安装 kaleido：`pip install kaleido`
- Plotly Express 返回的 `Figure` 对象与 Graph Objects 完全兼容

## 常见坑

1. `fig.show()` 在纯脚本中可能打开空浏览器窗口——Notebook 中使用更佳
2. `write_image` 未安装 kaleido 导致导出失败
3. 3D 图默认视角可能遮挡关键数据——需手动设置 `camera` 参数

## 小结

- Plotly Express 是构建交互图的最快路径——一行代码出图且默认可交互
- 交互图适合探索和演示——静态 Matplotlib 图更适合印刷报告
- 导出 HTML 保留交互能力，导出 PNG/PDF 适合固定交付
- `update_layout` 统一设置标题和主题是专业交付的基本要求
