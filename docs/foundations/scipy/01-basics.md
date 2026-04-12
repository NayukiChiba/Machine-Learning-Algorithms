---
title: SciPy 概览
outline: deep
---

# SciPy 概览

> 对应脚本：`Basic/Scipy/01_basics.py`
> 运行方式：`python Basic/Scipy/01_basics.py`（仓库根目录）

## 本章目标

1. 了解 SciPy 的整体模块结构与各子模块的功能定位。
2. 掌握 `scipy.constants` 中物理常数与单位换算的使用。
3. 认识 `scipy.special` 中常用特殊函数。
4. 确认 SciPy 与 NumPy 版本信息。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `scipy.constants.c` / `.h` / `.k` 等 | 物理常数 | `demo_constants` |
| `scipy.constants.mile` / `.inch` 等 | 单位换算 | `demo_constants` |
| `special.factorial(n)` | 阶乘 | `demo_special_functions` |
| `special.comb(N, k)` | 组合数 | `demo_special_functions` |
| `special.perm(N, k)` | 排列数 | `demo_special_functions` |
| `special.gamma(z)` | 伽马函数 | `demo_special_functions` |
| `special.jv(v, z)` | 贝塞尔函数 | `demo_special_functions` |

## 1. SciPy 模块总览

### 方法重点

- SciPy 是基于 NumPy 的科学计算库，包含 10+ 个专业子模块。
- 每个子模块专注于一个领域：常数、特殊函数、统计、优化、插值、积分、线性代数、信号处理、稀疏矩阵、空间数据。
- 使用时按需导入子模块（如 `from scipy import optimize`），而非 `import scipy`。

### 示例代码

```python
import scipy

# SciPy 子模块一览
modules = {
    "constants": "物理常数和单位换算",
    "special":   "特殊数学函数",
    "integrate": "数值积分和常微分方程",
    "optimize":  "优化和求根",
    "interpolate": "插值",
    "linalg":    "线性代数",
    "signal":    "信号处理",
    "sparse":    "稀疏矩阵",
    "stats":     "统计分布和检验",
    "spatial":   "空间数据结构和算法",
}

for name, desc in modules.items():
    print(f"  scipy.{name:15s} - {desc}")
```

### 结果输出

```text
  scipy.constants       - 物理常数和单位换算
  scipy.special         - 特殊数学函数
  scipy.integrate       - 数值积分和常微分方程
  scipy.optimize        - 优化和求根
  scipy.interpolate     - 插值
  scipy.linalg          - 线性代数
  scipy.signal          - 信号处理
  scipy.sparse          - 稀疏矩阵
  scipy.stats           - 统计分布和检验
  scipy.spatial         - 空间数据结构和算法
```

### 理解重点

- SciPy 不是一个单一模块，而是子模块集合。
- 各子模块之间相对独立，可以只学习需要的部分。
- 后续章节会逐一深入每个子模块。

## 2. 物理常数与单位换算

### 方法重点

- `scipy.constants` 提供 CODATA 推荐的物理常数值。
- 常数以模块属性形式访问，如 `constants.c`（光速）。
- 单位换算常量将指定单位转换为 SI 单位（如 `constants.mile` 返回一英里对应的米数）。

### 参数速览（本节）

适用 API：`scipy.constants` 模块属性（非函数调用）

| 常数名 | 值 | 说明 |
|---|---|---|
| `constants.pi` | 3.141593 | 圆周率 |
| `constants.c` | 299792458.0 | 光速 (m/s) |
| `constants.h` | 6.626e-34 | 普朗克常数 (J·s) |
| `constants.k` | 1.381e-23 | 玻尔兹曼常数 (J/K) |
| `constants.N_A` | 6.022e+23 | 阿伏伽德罗常数 (1/mol) |
| `constants.e` | 1.602e-19 | 基本电荷 (C) |

单位换算（返回对应 SI 值）：

| 常量名 | 值 | 说明 |
|---|---|---|
| `constants.mile` | 1609.344 | 1 英里 = 1609.344 米 |
| `constants.inch` | 0.0254 | 1 英寸 = 0.0254 米 |
| `constants.pound` | 0.45359... | 1 磅 ≈ 0.4536 千克 |

### 示例代码

```python
from scipy import constants

# 物理常数
print(f"圆周率 π = {constants.pi}")
print(f"光速 c = {constants.c} m/s")
print(f"普朗克常数 h = {constants.h} J·s")
print(f"玻尔兹曼常数 k = {constants.k} J/K")
print(f"阿伏伽德罗常数 N_A = {constants.N_A} 1/mol")
print(f"基本电荷 e = {constants.e} C")

# 单位换算
print(f"\n1 英里 = {constants.mile} 米")
print(f"1 英寸 = {constants.inch} 米")
print(f"1 磅 = {constants.pound} 千克")
```

### 结果输出

```text
圆周率 π = 3.141592653589793
光速 c = 299792458.0 m/s
普朗克常数 h = 6.62607015e-34 J·s
玻尔兹曼常数 k = 1.380649e-23 J/K
阿伏伽德罗常数 N_A = 6.02214076e+23 1/mol
基本电荷 e = 1.602176634e-19 C

1 英里 = 1609.3439999999998 米
1 英寸 = 0.0254 米
1 磅 = 0.45359236999999997 千克
```

### 理解重点

- 所有常数值均为最新 CODATA 推荐值，可直接用于科学计算。
- 单位换算：`constants.mile` 的含义是"1 英里等于多少米"，乘以数量即可换算。
- 不需要手动记忆常数值，直接引用即可保证精度。

## 3. 特殊函数

### 方法重点

- `scipy.special` 包含数学中常用的特殊函数。
- 阶乘 / 组合 / 排列用于组合数学。
- 伽马函数是阶乘在实数域的推广：`Γ(n) = (n-1)!`。
- 贝塞尔函数 `jv(v, z)` 在物理学（波动、热传导）中广泛使用。

### 参数速览（本节）

适用 API（分项）：

1. `special.factorial(n, exact=False)`
2. `special.comb(N, k, exact=False)`
3. `special.perm(N, k, exact=False)`
4. `special.gamma(z)`
5. `special.jv(v, z)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n` / `N` | `5` / `10` | 整数参数 |
| `k` | `3` | 选取数量 |
| `exact` | `False`（默认） | `True` 返回精确整数，`False` 返回浮点数 |
| `z` | `5` / `0.5` / `1` | 伽马函数或贝塞尔函数的自变量 |
| `v` | `0` / `1` | 贝塞尔函数的阶数 |

### 示例代码

```python
from scipy import special

# 阶乘、组合、排列
print(f"5! = {special.factorial(5, exact=True)}")
print(f"C(10,3) = {special.comb(10, 3, exact=True)}")
print(f"P(10,3) = {special.perm(10, 3, exact=True)}")

# 伽马函数
print(f"\nΓ(5) = {special.gamma(5)}")     # = 4! = 24
print(f"Γ(0.5) = {special.gamma(0.5)}")   # = √π

# 贝塞尔函数
print(f"\nJ₀(1) = {special.jv(0, 1):.6f}")
print(f"J₁(1) = {special.jv(1, 1):.6f}")
```

### 结果输出

```text
5! = 120
C(10,3) = 120
P(10,3) = 720

Γ(5) = 24.0
Γ(0.5) = 1.7724538509055159

J₀(1) = 0.765198
J₁(1) = 0.440051
```

### 理解重点

- `factorial(5, exact=True)` 返回 Python 整数 `120`，`exact=False` 返回浮点数 `120.0`。
- `Γ(n) = (n-1)!`，所以 `Γ(5) = 4! = 24`。
- `Γ(0.5) = √π ≈ 1.7725`，这是一个经典数学结论。
- 贝塞尔函数 `jv(v, z)` 中 `v` 是阶数，`z` 是自变量。

## 4. 版本信息

### 方法重点

- 通过 `scipy.__version__` 和 `np.__version__` 确认当前环境版本。
- 版本号在排查 API 行为差异时非常有用。

### 示例代码

```python
import scipy
import numpy as np

print(f"SciPy 版本: {scipy.__version__}")
print(f"NumPy 版本: {np.__version__}")
```

### 结果输出

```text
SciPy 版本: 1.15.2
NumPy 版本: 2.2.4
```

### 理解重点

- SciPy 依赖 NumPy，版本之间存在兼容性要求。
- 升级 SciPy 时需要注意 NumPy 最低版本要求。

## 常见坑

| 坑 | 说明 |
|---|---|
| `import scipy` 不会导入子模块 | 必须 `from scipy import optimize` 显式导入 |
| `factorial` 默认返回浮点数 | 需要精确整数时传 `exact=True` |
| 常数 `e` 是基本电荷 | 不是自然常数 e ≈ 2.718，自然常数用 `np.e` |
| 单位换算方向 | `constants.mile` 是"1 英里 = ? 米"，不是反过来 |

## 小结

- SciPy 是 NumPy 之上的科学计算工具箱，由 10+ 个专业子模块组成。
- `constants` 模块提供物理常数和单位换算，无需手动定义。
- `special` 模块提供阶乘、组合数、伽马函数、贝塞尔函数等特殊数学函数。
- 使用 SciPy 前先确认版本，不同版本的 API 可能有变化。
