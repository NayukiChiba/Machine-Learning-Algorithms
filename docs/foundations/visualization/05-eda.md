---
title: EDA
outline: deep
---

# EDA

> 对应脚本：`Basic/Visualization/05_eda.py`
> 运行方式：`python Basic/Visualization/05_eda.py`（仓库根目录）

## 导航

- [库生态总览](/foundations/overview)

## 本章目标

1. 掌握连续变量分布、相关关系和分类变量分析的可视化套路。
2. 学会在 EDA 中组合 Seaborn 与 Pandas API 完成快速验证。
3. 建立“先分布、再相关、后分组”的分析顺序意识。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `sns.histplot(...)` | 查看连续变量频率分布 | `demo_distribution_analysis` |
| `ax.axvline(...)` | 在分布图中标注统计线 | `demo_distribution_analysis` |
| `DataFrame.corr()` | 计算变量相关系数矩阵 | `demo_correlation_analysis` |
| `sns.heatmap(...)` | 可视化相关矩阵 | `demo_correlation_analysis` |
| `value_counts()` | 统计类别频次 | `demo_categorical_analysis` |
| `sns.boxplot(...)` | 比较分类变量下的数值分布 | `demo_categorical_analysis` |

## 1. 分布分析

### 方法重点

- 分布图用于观察偏态、离群和集中趋势。
- 同时标注均值和中位数有助于识别偏态分布。
- 多变量分布对比应使用统一刻度范围，减少视觉偏差。

### 参数速览（本节）

1. `seaborn.histplot(data=None, kde=False, ax=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data` | `df[col]` | 某一列数值数据 |
| `kde` | `True` | 同时绘制核密度曲线 |
| `ax` | `axes[i]` | 目标坐标轴 |
| 返回值 | `Axes` | 绘图坐标轴对象 |

2. `matplotlib.axes.Axes.axvline(x, color=None, linestyle=None, label=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `mean` / `median` | 竖线位置 |
| `color` | `'red'` / `'green'` | 线颜色 |
| `linestyle` | `'--'` | 线型 |
| `label` | `'Mean'` / `'Median'` | 图例名称 |
| 返回值 | `Line2D` | 竖线对象 |

### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
df = pd.DataFrame({
	"age": np.random.normal(35, 10, 200).astype(int),
	"income": np.random.exponential(50000, 200),
	"score": np.random.beta(2, 5, 200) * 100,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, df.columns):
	sns.histplot(df[col], kde=True, ax=ax)
	ax.axvline(df[col].mean(), color="red", linestyle="--", label="Mean")
	ax.axvline(df[col].median(), color="green", linestyle="--", label="Median")
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/viz_05_distribution.png
----------------
图像内容: age、income、score 三个变量的分布与均值/中位数标记
```

### 理解重点

- 均值和中位数差距较大通常意味着偏态或异常值影响。
- EDA 第一张图建议优先看分布，而不是直接建模。

## 2. 相关性分析

### 方法重点

- 相关矩阵可快速定位强相关与弱相关变量。
- 热力图适合表达相关强度与方向（正负相关）。
- 相关不等于因果，仍需结合业务逻辑验证。

### 参数速览（本节）

1. `pandas.DataFrame.corr(method='pearson', numeric_only=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `method` | 默认 `'pearson'` | 相关系数计算方法 |
| 返回值 | `DataFrame` | 相关系数矩阵 |

2. `seaborn.heatmap(data, annot=None, cmap=None, center=None, ax=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data` | `corr` | 相关矩阵 |
| `annot` | `True` | 显示数值标签 |
| `cmap` | `'coolwarm'` | 发散色板 |
| `center` | `0` | 颜色中心点 |
| `ax` | `axes[0]` | 目标坐标轴 |
| 返回值 | `Axes` | 绘图坐标轴对象 |

### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 100
x = np.random.randn(n)
df = pd.DataFrame({
	"x": x,
	"y_strong": x + np.random.randn(n) * 0.3,
	"y_weak": x + np.random.randn(n) * 2,
	"y_none": np.random.randn(n),
})

corr = df.corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/viz_05_correlation.png
----------------
图像内容: x 与 y_strong 相关性最高，x 与 y_none 接近无关
```

### 理解重点

- 强相关特征在建模前应考虑共线性处理策略。
- 分析相关性时要同步关注样本规模与异常值敏感性。

## 3. 分类变量分析

### 方法重点

- 类别频数图回答“每类有多少”，箱线图回答“每类分布如何”。
- `value_counts` 与 `boxplot` 组合可以兼顾规模与质量两个维度。
- 分类变量分析是异常组识别和分层建模的重要入口。

### 参数速览（本节）

1. `pandas.Series.value_counts(normalize=False, dropna=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `normalize` | `False` | 返回频数而非比例 |
| `dropna` | `True` | 是否忽略缺失值 |
| 返回值 | `Series` | 各类别频数 |

2. `pandas.Series.plot(kind='bar', ax=None, color=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `'bar'` | 柱状图 |
| `ax` | `axes[0]` | 目标坐标轴 |
| `color` | `'steelblue'` | 柱颜色 |
| 返回值 | `Axes` | 坐标轴对象 |

3. `seaborn.boxplot(data=None, x=None, y=None, ax=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data` | `df` | 输入 DataFrame |
| `x` | `'category'` | 分类字段 |
| `y` | `'value'` | 数值字段 |
| `ax` | `axes[1]` | 目标坐标轴 |
| 返回值 | `Axes` | 绘图坐标轴对象 |

### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
df = pd.DataFrame({
	"category": np.random.choice(["A", "B", "C", "D"], 200),
	"value": np.random.randn(200),
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df["category"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
sns.boxplot(x="category", y="value", data=df, ax=axes[1])
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/viz_05_categorical.png
----------------
左图展示类别频数，右图展示各类别数值分布与离群点
```

### 理解重点

- 类别不平衡会直接影响模型评估，需在 EDA 阶段尽早识别。
- 同一类别中离散度明显更大时，建议追查数据来源和采样口径。


