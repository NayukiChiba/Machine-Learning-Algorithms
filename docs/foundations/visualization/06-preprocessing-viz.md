---
title: 预处理可视化
outline: deep
---

# 预处理可视化

> 对应脚本：`Basic/Visualization/06_preprocessing_viz.py`
> 运行方式：`python Basic/Visualization/06_preprocessing_viz.py`（仓库根目录）

## 导航

- [库生态总览](/foundations/overview)

## 本章目标

1. 学会可视化缺失值分布、比例与列间差异。
2. 掌握异常值识别中箱线图与 IQR 边界的组合表达。
3. 对比常见数值变换对分布形态的影响并形成直觉。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `df.isnull()` | 生成缺失值布尔矩阵 | `demo_missing_viz` |
| `sns.heatmap(...)` | 可视化缺失值模式 | `demo_missing_viz` |
| `ax.boxplot(...)` | 发现异常值和四分位范围 | `demo_outlier_viz` |
| `np.percentile(...)` | 计算 IQR 阈值 | `demo_outlier_viz` |
| `np.log1p(...)` / `np.sqrt(...)` | 变换偏态分布 | `demo_transform_viz` |

## 1. 缺失值可视化

### 方法重点

- 缺失值热力图可以快速定位“哪几列、哪些样本段”缺失集中。
- 缺失比例柱状图可用于排序优先级，决定填补或删除策略。
- 缺失分析应在建模前完成，避免隐式数据泄露和偏差。

### 参数速览（本节）

1. `pandas.DataFrame.isnull()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `DataFrame[bool]` | 与原表同形状的缺失标记矩阵 |

2. `seaborn.heatmap(data, cbar=True, cmap=None, ax=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data` | `df.isnull()` | 缺失布尔矩阵 |
| `cbar` | `True` | 显示颜色条 |
| `cmap` | `'YlOrRd'` | 颜色映射 |
| `ax` | `axes[0]` | 目标坐标轴 |
| 返回值 | `Axes` | 绘图坐标轴对象 |

3. `pandas.Series.plot(kind='bar', ax=None, color=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `kind` | `'bar'` | 柱状图 |
| `ax` | `axes[1]` | 目标坐标轴 |
| `color` | `'coral'` | 柱颜色 |
| 返回值 | `Axes` | 坐标轴对象 |

### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
df = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])
for col in df.columns:
	mask = np.random.rand(len(df)) < 0.1
	df.loc[mask, col] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=True, ax=axes[0], cmap="YlOrRd")
(df.isnull().mean() * 100).plot(kind="bar", ax=axes[1], color="coral")
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/viz_06_missing.png
----------------
左图展示缺失位置，右图展示各列缺失百分比
```

### 理解重点

- 缺失模式随机与否会决定后续填补方法选择。
- 某一列缺失比例过高时应优先评估业务可用性。

## 2. 异常值可视化

### 方法重点

- 箱线图是异常值检测最常用的统计图形。
- IQR 阈值线可直观标注“正常区间”边界。
- 异常值处理前建议先可视化再决策，避免误删有效样本。

### 参数速览（本节）

1. `matplotlib.axes.Axes.boxplot(x, patch_artist=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `data` | 输入样本 |
| `patch_artist` | 默认 `False` | 是否启用箱体填色 |
| 返回值 | `dict` | 箱线图图元字典 |

2. `numpy.percentile(a, q)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` | `data` | 输入样本 |
| `q` | `[25, 75]` | 目标分位点 |
| 返回值 | `ndarray` | 指定分位数值 |

3. `matplotlib.axes.Axes.axvline(x, color=None, linestyle=None, label=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `lower` / `upper` | IQR 下界与上界 |
| `color` | `'red'` | 线颜色 |
| `linestyle` | `'--'` | 线型 |
| `label` | `'Lower: ...'` / `'Upper: ...'` | 图例名称 |
| 返回值 | `Line2D` | 竖线对象 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.randn(100)
data = np.append(data, [5, -5, 6, -6])

q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot(data)
axes[1].hist(data, bins=20, edgecolor="black", alpha=0.7)
axes[1].axvline(lower, color="red", linestyle="--")
axes[1].axvline(upper, color="red", linestyle="--")
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/viz_06_outlier.png
----------------
箱线图显示离群点，直方图标出 IQR 上下界
```

### 理解重点

- IQR 规则稳健但并非适用于所有分布。
- 异常值处理需结合业务含义，不宜机械截断。

## 3. 特征变换可视化

### 方法重点

- 同一变量在不同变换下的分布形态可显著变化。
- 对数变换适合右偏分布，标准化适合尺度统一。
- 变换选择应以模型需求与解释性目标共同决定。

### 参数速览（本节）

1. `numpy.log1p(x)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `data` | 原始正值数据 |
| 返回值 | `ndarray` | `log(1+x)` 结果 |

2. `numpy.sqrt(x)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `data` | 原始正值数据 |
| 返回值 | `ndarray` | 平方根变换结果 |

3. `matplotlib.axes.Axes.hist(x, bins=None, edgecolor=None, alpha=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | 原始/变换后数据 | 每个子图输入样本 |
| `bins` | `30` | 分箱数 |
| `edgecolor` | `'black'` | 边框颜色 |
| `alpha` | `0.7` | 透明度 |
| 返回值 | `n, bins, patches` | 直方图对象三元组 |

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.exponential(5, 1000)
log_data = np.log1p(data)
sqrt_data = np.sqrt(data)
std_data = (data - data.mean()) / data.std()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].hist(data, bins=30, edgecolor="black", alpha=0.7)
axes[0, 1].hist(log_data, bins=30, edgecolor="black", alpha=0.7)
axes[1, 0].hist(sqrt_data, bins=30, edgecolor="black", alpha=0.7)
axes[1, 1].hist(std_data, bins=30, edgecolor="black", alpha=0.7)
```

### 结果输出（示例）

```text
控制台提示: 图表已保存到 outputs/visualization/viz_06_transform.png
----------------
四宫格对比原始分布、对数变换、平方根变换和标准化结果
```

### 理解重点

- 变换不是为了“好看”，而是为了改善建模稳定性。
- 变换后应重新评估可解释性与业务阈值含义。


