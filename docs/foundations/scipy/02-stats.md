---
title: SciPy 统计
outline: deep
---

# SciPy 统计

> 对应脚本：`Basic/Scipy/02_stats.py`
> 运行方式：`python Basic/Scipy/02_stats.py`（仓库根目录）

## 本章目标

1. 掌握 `scipy.stats` 中常见概率分布的使用方法（正态、二项、泊松）。
2. 学会使用 SciPy 进行描述性统计分析（集中趋势、离散程度、分布形态）。
3. 理解百分位数与四分位距的计算方法。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `stats.norm(loc, scale)` | 正态分布对象 | `demo_distributions` |
| `stats.binom(n, p)` | 二项分布对象 | `demo_distributions` |
| `stats.poisson(mu)` | 泊松分布对象 | `demo_distributions` |
| `.pdf(x)` / `.pmf(k)` | 概率密度/质量函数 | `demo_distributions` |
| `.cdf(x)` / `.ppf(q)` | 累积分布/分位数函数 | `demo_distributions` |
| `stats.variation(a)` | 变异系数 | `demo_descriptive_stats` |
| `stats.skew(a)` | 偏度 | `demo_descriptive_stats` |
| `stats.kurtosis(a)` | 峰度 | `demo_descriptive_stats` |
| `stats.mode(a)` | 众数 | `demo_descriptive_stats` |
| `np.percentile(a, q)` | 百分位数 | `demo_percentiles` |

## 1. 概率分布

### 方法重点

- `scipy.stats` 提供 100+ 种概率分布，每种分布都是"冻结分布"对象。
- 连续分布使用 `.pdf(x)` 计算概率密度，离散分布使用 `.pmf(k)` 计算概率质量。
- `.cdf(x)` 返回累积概率 P(X ≤ x)，`.ppf(q)` 是 CDF 的逆函数。
- `.rvs(size)` 生成随机样本，`.mean()` / `.var()` 获取分布的理论均值/方差。

### 参数速览（本节）

适用 API（分项）：

1. `stats.norm(loc=0, scale=1)` — 正态分布
2. `stats.binom(n, p)` — 二项分布
3. `stats.poisson(mu)` — 泊松分布

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `loc` | `0` | 正态分布的均值 μ |
| `scale` | `1` | 正态分布的标准差 σ |
| `n` | `10` | 二项分布的试验次数 |
| `p` | `0.5` | 二项分布的成功概率 |
| `mu` | `3` | 泊松分布的期望值 λ |

### 示例代码

```python
from scipy import stats

# 正态分布
norm = stats.norm(loc=0, scale=1)
print(f"PDF at x=0: {norm.pdf(0):.4f}")
print(f"CDF at x=0: {norm.cdf(0):.4f}")
print(f"PPF at p=0.95: {norm.ppf(0.95):.4f}")
print(f"随机样本: {norm.rvs(size=5)}")

# 二项分布
binom = stats.binom(n=10, p=0.5)
print(f"P(X=5): {binom.pmf(5):.4f}")
print(f"P(X≤5): {binom.cdf(5):.4f}")
print(f"期望值: {binom.mean()}")
print(f"方差: {binom.var()}")

# 泊松分布
poisson = stats.poisson(mu=3)
print(f"P(X=3): {poisson.pmf(3):.4f}")
print(f"期望值: {poisson.mean()}")
```

### 结果输出

```text
正态分布 (μ=0, σ=1):
  PDF at x=0: 0.3989
  CDF at x=0: 0.5000
  PPF at p=0.95: 1.6449
  随机样本: [随机值...]

二项分布 (n=10, p=0.5):
  P(X=5): 0.2461
  P(X≤5): 0.6230
  期望值: 5.0
  方差: 2.5

泊松分布 (λ=3):
  P(X=3): 0.2240
  期望值: 3.0
```

### 理解重点

- `norm.pdf(0) = 0.3989` 是标准正态分布在 x=0 处的密度值，即 1/√(2π)。
- `norm.cdf(0) = 0.5` 说明标准正态分布关于 0 对称，左半部分概率恰好 50%。
- `norm.ppf(0.95) = 1.6449` 是常用的 95% 分位数，用于置信区间计算。
- 二项分布 B(10, 0.5) 的期望值 = np = 5，方差 = np(1-p) = 2.5。
- 泊松分布的期望值和方差都等于 λ。

## 2. 描述性统计

### 方法重点

- 描述性统计从三个角度刻画数据：集中趋势、离散程度、分布形态。
- `stats.variation` 计算变异系数 = 标准差 / 均值，用于比较不同量纲数据的离散程度。
- `stats.skew` 计算偏度：0 为对称，正值右偏，负值左偏。
- `stats.kurtosis` 计算峰度：0 为正态，正值尖峰，负值平坦（Fisher 定义）。
- `stats.norm.fit(data)` 可以从数据拟合正态分布参数。

### 参数速览（本节）

适用 API（分项）：

1. `stats.mode(a, keepdims=True)`
2. `stats.variation(a)`
3. `stats.skew(a)`
4. `stats.kurtosis(a)`
5. `stats.norm.fit(data)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` / `data` | `np.random.normal(100, 15, 100)` | 输入数据数组 |
| `keepdims` | `True` | 保持输出维度（mode 所需） |

### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)
data = np.random.normal(100, 15, 100)

# 集中趋势
print(f"均值: {np.mean(data):.2f}")
print(f"中位数: {np.median(data):.2f}")
print(f"众数: {stats.mode(data.astype(int), keepdims=True)[0][0]}")

# 离散程度
print(f"方差: {np.var(data):.2f}")
print(f"标准差: {np.std(data):.2f}")
print(f"变异系数: {stats.variation(data):.4f}")

# 分布形态
print(f"偏度: {stats.skew(data):.4f}")
print(f"峰度: {stats.kurtosis(data):.4f}")

# 正态拟合
mu, std = stats.norm.fit(data)
print(f"正态拟合: N({mu:.1f}, {std:.1f}²)")
```

### 结果输出

```text
集中趋势:
  均值: 99.73
  中位数: 100.19
  众数: 86

离散程度:
  方差: 196.72
  标准差: 14.03
  变异系数: 0.1406

分布形态:
  偏度: -0.1442
  峰度: -0.2058
```

### 理解重点

- 均值 ≈ 99.73，接近真实参数 μ=100，因为样本量 100 已经足够。
- 变异系数 ≈ 0.14 表示标准差约为均值的 14%，离散程度适中。
- 偏度 ≈ -0.14 接近 0，分布近似对称。
- 峰度 ≈ -0.21 接近 0，分布形态接近正态（Fisher 定义下正态峰度为 0）。
- `stats.norm.fit(data)` 返回最大似然估计的 μ 和 σ。

## 3. 百分位数

### 方法重点

- `np.percentile(data, q)` 计算数据的第 q 百分位数。
- 四分位距 IQR = Q3 - Q1，衡量数据的中间 50% 的离散程度。
- IQR 不受极端值影响，比标准差更稳健。
- 百分位数用于箱线图绘制和异常值检测。

### 参数速览（本节）

适用 API：`np.percentile(a, q)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` | `np.random.normal(100, 15, 100)` | 输入数据数组 |
| `q` | `[25, 50, 75, 90, 95]` | 百分位数（0-100） |

### 示例代码

```python
import numpy as np

np.random.seed(42)
data = np.random.normal(100, 15, 100)

# 百分位数
for p in [25, 50, 75, 90, 95]:
    print(f"P{p}: {np.percentile(data, p):.2f}")

# 四分位距
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
print(f"四分位距 (IQR): {iqr:.2f}")
```

### 结果输出

```text
百分位数:
  P25: 90.47
  P50: 100.19
  P75: 109.07
  P90: 117.35
  P95: 123.64

四分位距 (IQR): 18.60
```

### 理解重点

- P50 = 中位数 ≈ 100.19，接近正态分布的理论中位数 μ=100。
- IQR ≈ 18.60，对于 N(100, 15) 分布，理论 IQR = 2 × 0.6745 × σ ≈ 20.24，样本值合理。
- P90 和 P95 常用于风险分析（如 VaR），表示"90%/95% 的数据不超过此值"。
- 异常值检测常用规则：小于 Q1 - 1.5×IQR 或大于 Q3 + 1.5×IQR 的点视为异常。

## 常见坑

| 坑 | 说明 |
|---|---|
| `pdf` vs `pmf` 混用 | 连续分布用 `.pdf()`，离散分布用 `.pmf()`，调错会报错 |
| `stats.mode` 需要 `keepdims` | 新版 SciPy 要求显式传 `keepdims=True`，否则会有 DeprecationWarning |
| `kurtosis` 的定义 | SciPy 默认使用 Fisher 定义（正态=0），Pearson 定义需传 `fisher=False`（正态=3） |
| `norm.fit` 返回顺序 | 返回 `(loc, scale)` 即 `(μ, σ)`，不是 `(μ, σ²)` |
| 随机种子影响结果 | `rvs()` 每次调用结果不同，复现需设置 `random_state` 或 `np.random.seed` |

## 小结

- `scipy.stats` 提供丰富的概率分布对象，统一接口（pdf/cdf/ppf/rvs）。
- 描述性统计从集中趋势（均值/中位数/众数）、离散程度（方差/标准差/变异系数）、分布形态（偏度/峰度）三个维度描述数据。
- 百分位数和四分位距是稳健的统计量，不受极端值影响，广泛用于数据分析和异常检测。
