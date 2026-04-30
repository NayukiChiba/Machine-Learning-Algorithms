---
title: Pandas 数据可视化
outline: deep
---

# Pandas 数据可视化

## 本章目标

1. 掌握 `df.plot` 的统一绘图入口与 `kind` 参数
2. 掌握折线图、柱状图、直方图、散点图、箱线图、饼图的 Pandas 写法
3. 理解 Pandas 绘图与 Matplotlib 的关系（底层调用 matplotlib）
4. 学会将图表保存到文件

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.plot(...)` | 方法 | 统一绘图入口；默认 `kind='line'` |
| `df.plot.line(...)` | 方法 | 折线图 |
| `df.plot.area(...)` | 方法 | 面积图（堆叠） |
| `df.plot.bar(...)` / `df.plot.barh(...)` | 方法 | 垂直/水平柱状图 |
| `df.plot.hist(...)` / `df.plot.kde(...)` | 方法 | 直方图 / 核密度估计 |
| `df.plot.scatter(...)` | 方法 | 散点图 |
| `df.plot.box(...)` | 方法 | 箱线图 |
| `df.plot.pie(...)` | 方法 | 饼图 |

Pandas 绘图底层调用 Matplotlib，所有 `df.plot.*` 方法返回 `matplotlib.axes.Axes` 对象，可通过 Matplotlib API 进一步定制。

## 1. 统一绘图接口

### `DataFrame.plot`

#### 作用

Pandas 的一站式绘图入口。通过 `kind` 参数切换图表类型，底层自动调用对应的 `plot.<kind>()` 方法。返回 `matplotlib.axes.Axes` 对象，可链式调用 Matplotlib 的 API 做进一步定制。

#### 重点方法

```python
df.plot(*args, kind='line', ax=None, figsize=None, title=None,
        x=None, y=None, legend=True, grid=None, xlabel=None, ylabel=None,
        xticks=None, yticks=None, rot=None, fontsize=None, colormap=None,
        table=False, subplots=False, sharex=False, sharey=False,
        logx=False, logy=False, mark_right=True, style=None, **kwargs)
```

#### 参数（核心 12 个）

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `kind` | `str` | 图表类型，下见表，默认为 `'line'` | `"bar"`、`"hist"`、`"scatter"` |
| `x` | `str` 或 `None` | X 轴列名 | `"Date"` |
| `y` | `str`、`list[str]` 或 `None` | Y 轴列名（单列或多列） | `"Sales"`、`["A", "B"]` |
| `ax` | `matplotlib.axes.Axes` 或 `None` | 绘制到已有 axes（子图组合用） | `ax1` |
| `figsize` | `tuple[float, float]` | 图表尺寸（英寸），默认为 `(6.4, 4.8)` | `(10, 6)` |
| `title` | `str` | 图表标题 | `"销售趋势"` |
| `legend` | `bool` | 是否显示图例，默认为 `True` | `False` |
| `grid` | `bool` | 是否显示网格，默认为 `None` | `True` |
| `xlabel` / `ylabel` | `str` 或 `None` | 自定义轴标签 | `"月份"` |
| `rot` | `int` | X 轴刻度标签旋转角度 | `45` |
| `fontsize` | `int` | 字体大小 | `12` |
| `colormap` | `str` | 颜色映射方案 | `"viridis"`、`"tab10"` |

#### `kind` 图表类型速览

| `kind` | 等价方法 | 适用场景 |
|---|---|---|
| `'line'`（默认） | `df.plot.line()` | 趋势、时间序列 |
| `'area'` | `df.plot.area()` | 占比累积趋势 |
| `'bar'` | `df.plot.bar()` | 分类对比（垂直） |
| `'barh'` | `df.plot.barh()` | 分类对比（水平，适合长标签） |
| `'hist'` | `df.plot.hist()` | 单变量分布 |
| `'kde'` / `'density'` | `df.plot.kde()` | 概率密度估计 |
| `'scatter'` | `df.plot.scatter()` | 双变量关系 |
| `'box'` | `df.plot.box()` | 四分位分布 + 离群点 |
| `'pie'` | `df.plot.pie()` | 占比组成（仅单列 Series） |

#### 理解重点

- `kind` 和 `plot.xxx()` 写法等价：`df.plot(kind='bar')` = `df.plot.bar()`
- 返回 `Axes` 对象——可以用 Matplotlib API 继续修改：`.set_title(...)`、`.legend()`、`.set_xlabel(...)`
- Pandas 绘图是快速探索工具——报告级图表建议直接使用 Matplotlib / Seaborn

## 2. 折线图与面积图

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=12, freq="MS")
df = pd.DataFrame({
    "ProductA": np.random.randint(100, 300, 12),
    "ProductB": np.random.randint(80, 250, 12),
    "ProductC": np.random.randint(50, 200, 12),
}, index=dates)

import matplotlib.pyplot as plt

# 折线图：多列趋势
ax = df.plot.line(figsize=(10, 5), title="产品月度销售趋势",
                  ylabel="销售额", grid=True, marker="o")
# 可视化输出见下方图表：折线图显示三条趋势线，各自带圆形标记
plt.close()

# 面积图：堆叠面积
ax = df.plot.area(figsize=(10, 5), title="产品月度销售占比",
                  ylabel="销售额", alpha=0.7)
# 可视化输出见下方图表：堆叠面积图显示三类产品各自贡献的累积趋势
plt.close()
```

#### 输出

可视化输出见下方图表（Pandas 调用 Matplotlib 渲染）。折线图中三条趋势线分别代表 ProductA/ProductB/ProductC，X 轴为月份，Y 轴为销售额。面积图以堆叠方式显示各产品对总销售额的贡献变化。

### 理解重点

- 多列 DataFrame 直接 `plot.line()` 会为每列生成一条折线——自动区分颜色
- 面积图 `plot.area()` 默认 `stacked=True`——展示总量的组成部分

## 3. 柱状图

### 综合示例

#### 示例代码

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Sales": [150, 220, 180, 260, 190],
    "Profit": [30, 55, 40, 70, 45],
}, index=["北京", "上海", "广州", "深圳", "杭州"])

# 分组柱状图
ax = df.plot.bar(figsize=(10, 5), title="各城市销售与利润对比",
                 ylabel="金额（万元）", rot=0)
# 可视化输出见下方图表：分组柱状图，每城市两个柱子（Sales/Profit）

# 水平柱状图（适合长标签）
ax = df["Sales"].sort_values().plot.barh(figsize=(8, 5), title="各城市销售额",
                                          xlabel="金额（万元）", color="steelblue")
# 可视化输出见下方图表：水平柱状图，城市名作为 Y 轴标签
plt.close()
```

#### 输出

可视化输出见下方图表。垂直柱状图为每城市显示 Sales 和 Profit 两组柱子，便于横向对比。水平柱状图按销售额排序，城市名在 Y 轴排列清晰。

### 理解重点

- 柱状图自动按索引分组——`df.plot.bar()` 每列一组柱子
- `barh` 比 `bar` 更适合中文/长标签——垂直排列文字不被截断
- 堆叠柱状图：`df.plot.bar(stacked=True)`——显示总量+各组成部分

## 4. 直方图与密度图

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    "Score": np.random.normal(70, 15, 500).clip(0, 100),
})

# 直方图
ax = df["Score"].plot.hist(bins=20, figsize=(10, 5),
                            title="成绩分布直方图", alpha=0.7, edgecolor="black")
# 可视化输出见下方图表：直方图显示成绩分布，20 个区间
plt.close()

# 多列直方图
df2 = pd.DataFrame({
    "Class A": np.random.normal(75, 10, 200).clip(0, 100),
    "Class B": np.random.normal(65, 15, 200).clip(0, 100),
})
ax = df2.plot.hist(bins=20, alpha=0.5, figsize=(10, 5),
                    title="两班成绩分布对比")
# 可视化输出见下方图表：两个半透明直方图叠加，便于分布对比
plt.close()

# 核密度估计
ax = df2.plot.kde(figsize=(10, 5), title="两班成绩密度曲线", linewidth=2)
# 可视化输出见下方图表：两条平滑密度曲线
plt.close()
```

#### 输出

可视化输出见下方图表。直方图以柱形高度表示各分数段的人数，核密度图以平滑曲线展示概率密度分布——后者不受分箱数影响，更适合对比。

### 理解重点

- 直方图的 `bins` 控制柱子数量——过多过少都不好，20~30 是常用范围
- `alpha` 透明度对叠加直方图至关重要——否则后面的柱子会盖住前面的
- `kde` 是直方图的平滑替代——不受分箱边界影响

## 5. 散点图

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    "Height": np.random.normal(170, 10, 100),
    "Weight": np.random.normal(65, 12, 100),
    "Category": np.random.choice(["A", "B", "C"], 100),
})

# 基本散点图
ax = df.plot.scatter(x="Height", y="Weight", figsize=(8, 6),
                      title="身高-体重关系", alpha=0.6)
# 可视化输出见下方图表：散点图展示身高与体重的正相关关系
plt.close()

# 按类别着色
colors = {"A": "red", "B": "green", "C": "blue"}
ax = df.plot.scatter(x="Height", y="Weight", figsize=(8, 6),
                      title="身高-体重（按类别着色）",
                      c=df["Category"].map(colors), alpha=0.6)
# 可视化输出见下方图表：三色散点图，每组用不同颜色
plt.close()
```

#### 输出

可视化输出见下方图表。散点图每个点代表一个样本，X 轴为身高、Y 轴为体重。按类别着色后，不同颜色的点代表不同分组，可观察各组之间的分布差异。

### 理解重点

- `scatter` 必须指定 `x` 和 `y` 列名——与其他 `plot.*` 不同（它们从索引推断 X 轴）
- 散点图是发现变量间关系的最直观工具——正/负相关、聚类、离群点一目了然

## 6. 箱线图与饼图

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    "A": np.random.normal(70, 10, 100),
    "B": np.random.normal(65, 15, 100),
    "C": np.random.normal(75, 8, 100),
})

# 箱线图
ax = df.plot.box(figsize=(8, 5), title="三组分数分布对比",
                  ylabel="分数", grid=True)
# 可视化输出见下方图表：箱线图显示中位数、四分位数、范围、离群点

# 饼图
sizes = pd.Series({"产品A": 45, "产品B": 30, "产品C": 15, "其他": 10})
ax = sizes.plot.pie(figsize=(7, 7), title="产品销售占比",
                     autopct="%1.1f%%", startangle=90)
# 可视化输出见下方图表：饼图显示各类别占比，自动标注百分比
plt.close()
```

#### 输出

可视化输出见下方图表。箱线图展示每组的 Q1/中位数/Q3/须线/离群点，适合多组分布的横向对比。饼图以扇形角度表示各类别占比，`autopct` 自动标注百分比。

### 理解重点

- 箱线图的箱子 = IQR（Q1 到 Q3），须线延伸到 1.5×IQR 范围——超出的是离群点
- 饼图只适用于 Series：`series.plot.pie()`——DataFrame 需先用 `df["col"]` 提取
- `autopct="%1.1f%%"` 表示保留 1 位小数的百分比标注

## 7. 保存图表

Pandas 绘图底层返回 Matplotlib 对象，保存用 `plt.savefig()`：

```python
import matplotlib.pyplot as plt

ax = df.plot.line(title="示例")
fig = ax.get_figure()
fig.savefig("output.png", dpi=150, bbox_inches="tight")
```

或更简单的：

```python
df.plot.line(title="示例")
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```

## 常见坑

1. Pandas 绘图需先导入 `import matplotlib.pyplot as plt`——否则可能无法显示图表
2. `df.plot.scatter()` 必须显式指定 `x` 和 `y`——与其他 `plot.*()` 的默认行为不同
3. 饼图 `df.plot.pie()` 要求 `y` 参数或 `subplots=True`——对多列 DataFrame 需用 `subplots=True` 或提取 Series
4. Jupyter Notebook 中需 `%matplotlib inline` 才能在单元格内显示图表
5. Pandas 绘图返回 `Axes` 对象，`plt.savefig()` 要在图表显示前调用——`plt.show()` 后再保存会得到空白图片
6. 中文显示为方框时需设置中文字体：`plt.rcParams["font.sans-serif"] = ["SimHei"]`

## 小结

- `df.plot(kind=...)` 是快速探索的统一入口——10 种图表类型一行切换
- Pandas 绘图 = Matplotlib 的便捷封装——复杂定制用 Matplotlib/Seaborn
- 时间序列首选 `line`；分类对比首选 `bar`；分布分析首选 `hist` / `kde` / `box`
- 绘图后 `plt.savefig("out.png", dpi=150)` 保存——在 `plt.show()` 之前调用
