---
title: SciPy 统计
outline: deep
---

# SciPy 统计

## 本章目标

1. 掌握 `scipy.stats` 中常见概率分布的创建与使用（正态、二项、泊松）
2. 理解分布对象的统一接口：`.pdf` / `.cdf` / `.ppf` / `.rvs`
3. 掌握描述性统计函数（变异系数、偏度、峰度、众数）
4. 理解百分位数与四分位距的计算方法

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `stats.norm(loc, scale)` | 构造器 | 创建正态分布对象 |
| `stats.binom(n, p)` | 构造器 | 创建二项分布对象 |
| `stats.poisson(mu)` | 构造器 | 创建泊松分布对象 |
| `dist.pdf(x)` / `dist.pmf(k)` | 方法 | 概率密度/质量函数 |
| `dist.cdf(x)` / `dist.ppf(q)` | 方法 | 累积分布/分位数函数 |
| `dist.rvs(size)` | 方法 | 生成随机样本 |
| `stats.variation(a)` | 函数 | 变异系数 CV = $\sigma / \mu$ |
| `stats.skew(a)` | 函数 | 偏度 |
| `stats.kurtosis(a)` | 函数 | 峰度（Fisher 定义） |
| `stats.mode(a)` | 函数 | 众数 |
| `stats.norm.fit(data)` | 函数 | 最大似然估计拟合正态分布参数 |

## 1. 概率分布对象

SciPy 的 `stats` 模块提供 100+ 种概率分布，每种都是"冻结分布"对象。创建时固定参数，随后通过统一接口（`.pdf` / `.cdf` / `.ppf` / `.rvs`）操作。

### `stats.norm`

#### 作用

创建正态（高斯）分布对象。概率密度函数：

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

#### 重点方法

```python
stats.norm(loc=0, scale=1)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `loc` | `float` | 均值 $\mu$，默认为 `0` | `100` |
| `scale` | `float` | 标准差 $\sigma$，默认为 `1` | `15` |

### `stats.binom`

#### 作用

创建二项分布对象。$n$ 次独立 Bernoulli 试验中成功次数的分布。概率质量函数：

$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

#### 重点方法

```python
stats.binom(n, p)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n` | `int` | 试验次数 | `10` |
| `p` | `float` | 每次试验的成功概率 | `0.5` |

### `stats.poisson`

#### 作用

创建泊松分布对象。描述单位时间内随机事件发生次数的分布。概率质量函数：

$$
P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

#### 重点方法

```python
stats.poisson(mu)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `mu` | `float` | 期望值 $\lambda$ | `3` |

### 分布对象的统一接口

创建分布后，以下方法对所有分布统一可用：

| 方法 | 类型 | 含义 |
|---|---|---|
| `dist.pdf(x)` | 连续 | 概率密度函数 $f(x)$ |
| `dist.pmf(k)` | 离散 | 概率质量函数 $P(X=k)$ |
| `dist.cdf(x)` | 全部 | 累积分布 $P(X \le x)$ |
| `dist.ppf(q)` | 全部 | 分位数函数（CDF 逆函数） |
| `dist.rvs(size)` | 全部 | 生成随机样本 |
| `dist.mean()` | 全部 | 理论均值 |
| `dist.var()` | 全部 | 理论方差 |

### 综合示例

#### 示例代码

```python
from scipy import stats

# 正态分布
norm = stats.norm(loc=0, scale=1)
print(f"pdf(0): {norm.pdf(0):.4f}")
print(f"cdf(0): {norm.cdf(0):.4f}")
print(f"ppf(0.95): {norm.ppf(0.95):.4f}")
print(f"rvs(5): {norm.rvs(size=5)}")

# 二项分布
binom = stats.binom(n=10, p=0.5)
print(f"\npmf(5): {binom.pmf(5):.4f}")
print(f"cdf(5): {binom.cdf(5):.4f}")
print(f"mean: {binom.mean()}, var: {binom.var()}")

# 泊松分布
poisson = stats.poisson(mu=3)
print(f"\npmf(3): {poisson.pmf(3):.4f}")
print(f"mean: {poisson.mean()}")
```

#### 输出

```text
pdf(0): 0.3989
cdf(0): 0.5000
ppf(0.95): 1.6449
rvs(5): [ 0.4967 -0.1383  0.6477  1.5230 -0.2342]

pmf(5): 0.2461
cdf(5): 0.6230
mean: 5.0, var: 2.5

pmf(3): 0.2240
mean: 3.0
```

#### 理解重点

- `norm.pdf(0) = 0.3989 = 1/\sqrt{2\pi}$——标准正态在均值处的密度值
- `norm.ppf(0.95) = 1.6449` 是 95% 单侧置信区间的常用分位数
- 二项分布 $B(10, 0.5)$ 的期望值 = $np = 5$，方差 = $np(1-p) = 2.5$
- 泊松分布的期望值和方差都等于 $\lambda$
- 连续分布用 `.pdf()`，离散分布用 `.pmf()`——不可混用

## 2. 描述性统计

### `stats.variation`

#### 作用

计算变异系数 $CV = \sigma / \mu$，消除量纲影响后比较不同数据集的离散程度。

#### 重点方法

```python
stats.variation(a)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数据 | `np.random.normal(100, 15, 100)` |

### `stats.skew`

#### 作用

计算偏度——衡量分布的不对称程度和方向。0 为对称，正值右偏（右尾更长），负值左偏。

#### 重点方法

```python
stats.skew(a)
```

### `stats.kurtosis`

#### 作用

计算峰度——衡量分布的"尖峭"程度。默认使用 Fisher 定义（正态分布峰度 = 0），正值比正态更尖峭，负值更平坦。

#### 重点方法

```python
stats.kurtosis(a, fisher=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数据 | `data` |
| `fisher` | `bool` | `True` Fisher 定义（正态=0），`False` Pearson 定义（正态=3），默认为 `True` | `False` |

### `stats.mode`

#### 作用

计算数据中出现频率最高的值（众数）。需指定 `keepdims=True`（新版 SciPy 要求）。

#### 重点方法

```python
stats.mode(a, keepdims=True)
```

### `stats.norm.fit`

#### 作用

从数据中通过最大似然估计拟合正态分布参数，返回 $(\hat{\mu}, \hat{\sigma})$。

#### 重点方法

```python
stats.norm.fit(data)
```

### 综合示例

#### 示例代码

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
print(f"标准差: {np.std(data):.2f}")
print(f"变异系数: {stats.variation(data):.4f}")

# 分布形态
print(f"偏度: {stats.skew(data):.4f}")
print(f"峰度: {stats.kurtosis(data):.4f}")

# MLE 拟合
muHat, sigmaHat = stats.norm.fit(data)
print(f"MLE 拟合: N({muHat:.1f}, {sigmaHat:.1f}²)")
```

#### 输出

```text
均值: 99.73
中位数: 100.19
众数: 86
标准差: 14.03
变异系数: 0.1406
偏度: -0.1442
峰度: -0.2058
MLE 拟合: N(99.7, 14.0²)
```

#### 理解重点

- 变异系数 0.14 表示标准差约为均值的 14%——无量纲，可跨数据集比较
- 偏度 ≈ -0.14 接近 0——分布近似对称，与正态假设一致
- 峰度 ≈ -0.21 接近 0——分布形态接近正态（Fisher 定义下正态峰度=0）
- `stats.norm.fit(data)` 返回 $(\hat{\mu}, \hat{\sigma})$，不是 $(\mu, \sigma^2)$

## 3. 百分位数与四分位距

### `np.percentile`

#### 作用

计算数据的第 $q$ 百分位数。四分位距 $IQR = Q3 - Q1$ 是稳健的离散度指标（不受极端值影响）。

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数据 | `data` |
| `q` | `float`、`list[float]` | 百分位数 (0-100) | `[25, 50, 75, 90, 95]` |

#### 示例代码

```python
import numpy as np

np.random.seed(42)
data = np.random.normal(100, 15, 100)

for p in [25, 50, 75, 90, 95]:
    print(f"P{p}: {np.percentile(data, p):.2f}")

q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
print(f"\nQ1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
```

#### 输出

```text
P25: 90.47
P50: 100.19
P75: 109.07
P90: 117.35
P95: 123.64

Q1=90.47, Q3=109.07, IQR=18.60
```

#### 理解重点

- P50 = 中位数 ≈ 100.19，接近理论值 $\mu=100$
- IQR ≈ 18.60，理论值 $2 \times 0.6745\sigma \approx 20.24$，样本值合理
- 异常值检测规则：$< Q1 - 1.5 \times IQR$ 或 $> Q3 + 1.5 \times IQR$ 的点视为异常
- 百分位数比均值/标准差更稳健——常用于箱线图和风险分析（VaR）

## 常见坑

1. 连续分布用 `.pdf()`，离散分布用 `.pmf()`——调错会报 `AttributeError`
2. `stats.mode` 新版要求显式传 `keepdims=True`——否则会有 `DeprecationWarning`
3. `stats.kurtosis` 默认 Fisher 定义（正态=0），Pearson 定义需 `fisher=False`（正态=3）
4. `stats.norm.fit` 返回 `(μ, σ)` 不是 `(μ, σ²)`——标准差不是方差
5. `dist.rvs()` 每次调用结果不同——复现需提前 `np.random.seed(seed)`

## 小结

- `scipy.stats` 提供 100+ 概率分布，统一接口 `pdf/pmf` → `cdf` → `ppf` → `rvs`
- 描述性统计从三个维度刻画数据：集中趋势、离散程度、分布形态
- 百分位数和 IQR 是稳健统计量——不受极端值影响，广泛用于数据分析和异常检测
