---
title: SciPy 概览
outline: deep
---

# SciPy 概览

## 本章目标

1. 了解 SciPy 的整体模块结构与各子模块的功能定位
2. 掌握 `scipy.constants` 中物理常数与单位换算的使用
3. 掌握 `scipy.special` 中常用特殊函数（阶乘、组合数、伽马、贝塞尔）
4. 会查询 SciPy 与 NumPy 的版本信息

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `scipy.constants.pi` / `.c` / `.h` / `.k` / `.N_A` / `.e` | 常量 | 数学与物理常数 |
| `scipy.constants.mile` / `.inch` / `.pound` | 常量 | 单位换算因子 |
| `special.factorial(...)` | 函数 | 阶乘 $n!$ |
| `special.comb(...)` | 函数 | 组合数 $C(n, k)$ |
| `special.perm(...)` | 函数 | 排列数 $P(n, k)$ |
| `special.gamma(...)` | 函数 | 伽马函数 $\Gamma(z)$ |
| `special.jv(...)` | 函数 | 第一类贝塞尔函数 $J_v(x)$ |
| `scipy.__version__` | 属性 | SciPy 版本号 |

## 1. SciPy 模块总览

SciPy 不是单一模块，而是**子模块集合**——按需导入，各子模块相对独立：

| 子模块 | 功能定位 | 对应章节 |
|---|---|---|
| `scipy.constants` | 数学/物理常数、单位换算 | ch01（本章） |
| `scipy.special` | 特殊数学函数（伽马、贝塞尔等） | ch01（本章） |
| `scipy.stats` | 概率分布、描述统计、假设检验 | ch02、ch03 |
| `scipy.optimize` | 曲线拟合、求根、最优化、线性规划 | ch04 |
| `scipy.interpolate` | 一维/多维插值、样条、RBF | ch05 |
| `scipy.integrate` | 数值积分、常微分方程 | ch06 |
| `scipy.linalg` | 线性代数（LU/QR/SVD/特征值） | ch07 |
| `scipy.signal` | 信号处理（滤波、卷积、FFT） | ch08 |
| `scipy.sparse` | 稀疏矩阵（CSR/CSC/COO） | ch09 |
| `scipy.spatial` | 空间数据（KDTree/凸包/Voronoi） | ch10 |

#### 理解重点

- 导入子模块而非顶层：`from scipy import optimize`，不要 `import scipy; scipy.optimize.minimize(...)`
- `scipy.linalg` 比 `numpy.linalg` 更全面，部分函数效率更高
- 各子模块依赖 NumPy 但不互相依赖——只学需要的部分即可

## 2. 物理常数与单位换算

### `scipy.constants` — 物理常数

#### 作用

提供 CODATA 标准物理常数和数学常数。所有常数都是**标量浮点数**，可直接参与 NumPy 运算。

#### 常用物理常数

| 名称 | 含义 | 数值（SI 单位） |
|---|---|---|
| `constants.pi` | 圆周率 $\pi$ | `3.141592653589793` |
| `constants.c` | 真空光速 | `299792458.0` m/s |
| `constants.h` | 普朗克常数 | `6.62607015e-34` J·s |
| `constants.hbar` | 约化普朗克常数 $\hbar$ | `h / (2\pi)` |
| `constants.k` | 玻尔兹曼常数 | `1.380649e-23` J/K |
| `constants.N_A` | 阿伏伽德罗常数 | `6.02214076e+23` |
| `constants.G` | 引力常数 | `6.67430e-11` |
| `constants.g` | 重力加速度 | `9.80665` m/s² |
| `constants.e` | 基本电荷（**非欧拉数！**） | `1.602176634e-19` C |
| `constants.R` | 理想气体常数 | `8.31446261815324` |

### `scipy.constants` — 单位换算

#### 常用单位→SI 换算因子

| 名称 | 含义 | 换算值 |
|---|---|---|
| `constants.mile` | 英里 → 米 | `1609.344` |
| `constants.inch` | 英寸 → 米 | `0.0254` |
| `constants.foot` | 英尺 → 米 | `0.3048` |
| `constants.pound` | 磅 → 千克 | `0.45359237` |
| `constants.minute` | 分钟 → 秒 | `60.0` |
| `constants.hour` | 小时 → 秒 | `3600.0` |
| `constants.degree` | 度 → 弧度 | $\pi / 180$ |

### 综合示例

#### 示例代码

```python
from scipy import constants

print(f"π = {constants.pi}")
print(f"光速 c = {constants.c} m/s")
print(f"普朗克 h = {constants.h} J·s")
print(f"玻尔兹曼 k = {constants.k} J/K")
print(f"阿伏伽德罗 N_A = {constants.N_A}")

print(f"\n1 英里 = {constants.mile} 米")
print(f"1 英寸 = {constants.inch} 米")
print(f"1 磅 = {constants.pound} 千克")

# 搜索与查找
print(f"\n查找 Planck 相关常数:")
for key in constants.find("Planck"):
    print(f"  {key}: {constants.value(key)}")
```

#### 输出

```text
π = 3.141592653589793
光速 c = 299792458.0 m/s
普朗克 h = 6.62607015e-34 J·s
玻尔兹曼 k = 1.380649e-23 J/K
阿伏伽德罗 N_A = 6.02214076e+23

1 英里 = 1609.344 米
1 英寸 = 0.0254 米
1 磅 = 0.45359237 千克

查找 Planck 相关常数:
  Planck length: 1.616255e-35
  Planck mass: 2.176434e-08
  Planck temperature: 1.416784e+32
  Planck time: 5.391247e-44
```

#### 理解重点

- 全部常数按 **SI 单位制**给出——物理计算可直接相乘除
- `constants.find('planck')` 搜索相关常数；`constants.value('speed of light in vacuum')` 按 CODATA 标准名查找
- **`constants.e` 是基本电荷，不是欧拉数！** 欧拉数用 `math.e` 或 `numpy.e`

## 3. 特殊函数

### `special.factorial`

#### 作用

计算阶乘 $n! = 1 \times 2 \times \cdots \times n$。`exact=True` 返回 Python 整数精确值，默认返回浮点数（大 n 会溢出）。

#### 重点方法

```python
special.factorial(n, exact=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n` | `int`、`array_like` | 非负整数 | `5`、`[1, 2, 3]` |
| `exact` | `bool` | `True` 返回精确整数，`False` 返回浮点数，默认为 `False` | `True` |

### `special.comb`

#### 作用

计算组合数 $C(N, k) = \binom{N}{k} = \frac{N!}{k!(N-k)!}$。支持 `exact=True` 精确整数和 `repetition=True` 允许重复选择。

#### 重点方法

```python
special.comb(N, k, exact=False, repetition=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `N` | `int`、`array_like` | 总元素数 | `10` |
| `k` | `int`、`array_like` | 选取数 | `3` |
| `exact` | `bool` | `True` 返回精确整数，默认为 `False` | `True` |
| `repetition` | `bool` | `True` 允许重复选择（$C(N+k-1, k)$），默认为 `False` | `True` |

### `special.perm`

#### 作用

计算排列数 $P(N, k) = \frac{N!}{(N-k)!}$。

#### 重点方法

```python
special.perm(N, k, exact=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `N` | `int`、`array_like` | 总元素数 | `10` |
| `k` | `int`、`array_like` | 选取数 | `3` |
| `exact` | `bool` | `True` 返回精确整数，默认为 `False` | `True` |

### `special.gamma`

#### 作用

计算伽马函数 $\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt$。对正整数 $n$ 有 $\Gamma(n) = (n-1)!$。支持复数输入。

#### 重点方法

```python
special.gamma(z)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `z` | `array_like` | 实数或复数输入 | `5`、`0.5`、`[1, 2, 3]` |

### `special.jv`

#### 作用

计算第一类贝塞尔函数 $J_v(x)$，是贝塞尔微分方程的正则解。$v$ 为阶数，$x$ 为自变量（可为复数）。

#### 重点方法

```python
special.jv(v, x)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `v` | `array_like` | 贝塞尔函数的阶数（可为浮点） | `0`、`1.5` |
| `x` | `array_like` | 自变量（实数或复数） | `1.0`、`[0, 1, 2]` |

### 其他常用特殊函数速览

| 函数 | 含义 | 示例 |
|---|---|---|
| `special.gammaln(z)` | $\ln\Gamma(z)$，避免溢出 | `special.gammaln(1000)` |
| `special.beta(a, b)` | 贝塔函数 $B(a, b)$ | `special.beta(2, 3)` |
| `special.yv(v, x)` | 第二类贝塞尔函数 $Y_v(x)$ | `special.yv(0, 1)` |
| `special.erf(x)` | 误差函数 $\operatorname{erf}(x)$ | `special.erf(1)` |
| `special.expit(x)` | sigmoid 函数 $\frac{1}{1+e^{-x}}$ | `special.expit(0)` |
| `special.logit(x)` | sigmoid 的逆函数 | `special.logit(0.5)` |
| `special.softmax(x)` | softmax 激活，数值稳定 | `special.softmax([1, 2, 3])` |

### 综合示例

#### 示例代码

```python
from scipy import special
import math

# 阶乘与组合数
print(f"5! = {special.factorial(5)}")
print(f"5! (精确) = {special.factorial(5, exact=True)}")
print(f"C(10, 3) = {special.comb(10, 3)}")
print(f"P(10, 3) = {special.perm(10, 3)}")

# 伽马函数
print(f"\nΓ(5) (=4!) = {special.gamma(5)}")
print(f"Γ(0.5) (=√π) = {special.gamma(0.5)}")
print(f"对比 √π = {math.sqrt(math.pi)}")

# 贝塞尔函数
print(f"\nJ_0(1) = {special.jv(0, 1):.6f}")
print(f"J_1(1) = {special.jv(1, 1):.6f}")

# 机器学习常用
print(f"\nexpit(0) = {special.expit(0)}")
print(f"softmax([1,2,3]) = {special.softmax([1, 2, 3])}")
print(f"logit(0.5) = {special.logit(0.5)}")
```

#### 输出

```text
5! = 120.0
5! (精确) = 120
C(10, 3) = 120.0
P(10, 3) = 720.0

Γ(5) (=4!) = 24.0
Γ(0.5) (=√π) = 1.7724538509055159
对比 √π = 1.7724538509055159

J_0(1) = 0.765198
J_1(1) = 0.440051

expit(0) = 0.5
softmax([1,2,3]) = [0.09003057 0.24472847 0.66524096]
logit(0.5) = 0.0
```

#### 理解重点

- `factorial(n, exact=True)` 返回整数精确值——大 n 用 `gammaln(n+1)` 避免溢出
- `special.expit` 是 sigmoid 的数值稳定实现，`special.softmax` 自动处理 overflow——机器学习中优先使用
- 贝塞尔函数 $J_v(x)$ 和 $Y_v(x)$ 是振动问题的标准解——$v$ 可为非整数

## 4. 版本查询

### `scipy.__version__`

#### 作用

SciPy 版本号字符串。配合 `numpy.__version__` 做环境诊断。

#### 示例代码

```python
import scipy
import numpy as np

print(f"SciPy 版本: {scipy.__version__}")
print(f"NumPy 版本: {np.__version__}")
```

#### 输出

```text
SciPy 版本: 1.11.4
NumPy 版本: 1.26.2
```

#### 理解重点

- SciPy 与 NumPy 有版本兼容矩阵——升级前查官方说明
- 版本号遵循 `major.minor.patch` 语义

## 常见坑

1. 不要写 `import scipy` 然后 `scipy.optimize.minimize(...)`——旧版顶层不自动导入子模块，需 `from scipy import optimize`
2. `scipy.constants.e` 是**基本电荷**（$1.6 \times 10^{-19}$ C），不是欧拉数——欧拉数用 `math.e` 或 `numpy.e`
3. `special.factorial(n)` 对大 n 会溢出（`exact=False` 时）——改用 `gammaln(n+1)` 取对数
4. `special.comb(N, k)` 同理——大数用 `exact=True` 或 `gammaln`
5. `exact=True` 返回 Python `int` 而非 NumPy 数组——批量计算时注意类型不一致

## 小结

- SciPy 是**建立在 NumPy 之上**的科学计算生态，按子模块组织——按需导入
- 常数（`constants`）+ 特殊函数（`special`）是最轻量的入口——物理计算和组合数学的基础
- 后续章节逐一深入核心子模块：统计、优化、插值、积分、线代、信号、稀疏、空间
