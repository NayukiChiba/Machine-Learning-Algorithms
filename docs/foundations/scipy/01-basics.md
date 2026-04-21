---
title: SciPy 概览
outline: deep
---

# SciPy 概览

## 本章目标

1. 了解 SciPy 的整体模块结构与各子模块的功能定位。
2. 掌握 `scipy.constants` 中物理常数与单位换算的使用。
3. 认识 `scipy.special` 中常用特殊函数（阶乘、组合数、伽马、贝塞尔）。
4. 会查询 SciPy 与 NumPy 的版本信息。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `scipy.constants.pi` / `.c` / `.h` / `.k` / `.N_A` / `.e` | 常量 | 数学与物理常数 |
| `scipy.constants.mile` / `.inch` / `.pound` | 常量 | 单位换算因子 |
| `special.factorial(...)` | 函数 | 阶乘 |
| `special.comb(...)` | 函数 | 组合数 `C(n, k)` |
| `special.perm(...)` | 函数 | 排列数 `P(n, k)` |
| `special.gamma(...)` | 函数 | 伽马函数 Γ(x) |
| `special.jv(...)` | 函数 | 第一类贝塞尔函数 |
| `scipy.__version__` | 属性 | SciPy 版本号 |

## SciPy 模块总览

### 子模块职能

| 模块               | 功能定位                         |
| ------------------ | -------------------------------- |
| `scipy.constants`  | 数学与物理常数、单位换算         |
| `scipy.special`    | 特殊数学函数（伽马、贝塞尔等）   |
| `scipy.integrate`  | 数值积分、常微分方程             |
| `scipy.optimize`   | 优化与求根                       |
| `scipy.interpolate`| 插值                             |
| `scipy.linalg`     | 线性代数（比 `numpy.linalg` 更全） |
| `scipy.signal`     | 信号处理                         |
| `scipy.sparse`     | 稀疏矩阵                         |
| `scipy.stats`      | 统计分布与检验                   |
| `scipy.spatial`    | 空间数据结构（KDTree、凸包等）   |
| `scipy.fft`        | 快速傅里叶变换                   |
| `scipy.ndimage`    | 多维图像处理                     |
| `scipy.io`         | 读写 MATLAB、WAV 等文件          |

### 示例代码

```python
import scipy

modules = {
    "constants":   "物理常数和单位换算",
    "special":     "特殊数学函数",
    "integrate":   "数值积分和常微分方程",
    "optimize":    "优化和求根",
    "interpolate": "插值",
    "linalg":      "线性代数",
    "signal":      "信号处理",
    "sparse":      "稀疏矩阵",
    "stats":       "统计分布和检验",
    "spatial":     "空间数据结构和算法",
}
for name, desc in modules.items():
    print(f"  scipy.{name:15s} - {desc}")
```

### 输出

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

- SciPy 不是一个单一模块，而是子模块集合；按需导入：`from scipy import optimize`。
- 各子模块相对独立，可只学习需要的部分。
- `scipy.linalg` 比 `numpy.linalg` 更全面，部分函数效率更高。

## 物理常数与单位换算

### `scipy.constants`

#### 作用

提供标准的数学 / 物理常数以及常用单位到 SI 的换算因子。常数都是**标量浮点数**，可直接参与 NumPy 运算。

#### 常用常数

| 名称           | 含义                        | 值（近似）                          |
| -------------- | --------------------------- | ----------------------------------- |
| `pi`           | 圆周率 π                    | `3.141592653589793`                 |
| `e`（模块级）  | 自然对数底 e                | `2.718281828459045`                 |
| `c`            | 真空光速（m/s）             | `299792458.0`                       |
| `h`            | 普朗克常数（J·s）           | `6.62607015e-34`                    |
| `hbar`         | 约化普朗克常数 ħ            | `h / (2π)`                          |
| `k`            | 玻尔兹曼常数（J/K）         | `1.380649e-23`                      |
| `N_A`          | 阿伏伽德罗常数              | `6.02214076e23`                     |
| `G`            | 引力常数                    | `6.67430e-11`                       |
| `g`            | 重力加速度（m/s²）          | `9.80665`                           |
| `elementary_charge` / `e` | 基本电荷（C）    | `1.602176634e-19`                   |
| `R`            | 理想气体常数                | `8.31446261815324`                  |

#### 常用单位换算（到 SI）

| 名称     | 含义                    | 值                    |
| -------- | ----------------------- | --------------------- |
| `mile`   | 1 英里 = ? 米           | `1609.344`            |
| `inch`   | 1 英寸 = ? 米           | `0.0254`              |
| `foot`   | 1 英尺 = ? 米           | `0.3048`              |
| `pound`  | 1 磅 = ? 千克           | `0.45359237`          |
| `minute` | 1 分钟 = ? 秒           | `60.0`                |
| `hour`   | 1 小时 = ? 秒           | `3600.0`              |
| `degree` | 1 度 = ? 弧度           | `π / 180`             |

### 示例代码

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
```

### 输出

```text
π = 3.141592653589793
光速 c = 299792458.0 m/s
普朗克 h = 6.62607015e-34 J·s
玻尔兹曼 k = 1.380649e-23 J/K
阿伏伽德罗 N_A = 6.02214076e+23

1 英里 = 1609.344 米
1 英寸 = 0.0254 米
1 磅 = 0.45359237 千克
```

### 理解重点

- 全部常数按 **SI 单位制**给出；做物理计算时可直接相乘除。
- `constants.value('speed of light in vacuum')` 可按 CODATA 标准名查找；`constants.find('planck')` 搜索相关常数。

## 特殊函数

### `scipy.special` 速览

| 函数                       | 含义                      |
| -------------------------- | ------------------------- |
| `factorial(n, exact=True)` | 阶乘 `n!`                 |
| `comb(N, k, exact=False)`  | 组合数 `C(N, k)`          |
| `perm(N, k, exact=False)`  | 排列数 `P(N, k)`          |
| `gamma(z)`                 | 伽马函数 `Γ(z)`           |
| `gammaln(z)`               | `log(Γ(z))`，避免溢出     |
| `beta(a, b)`               | 贝塔函数                  |
| `jv(v, x)`                 | 第一类贝塞尔函数 `J_v(x)` |
| `yv(v, x)`                 | 第二类贝塞尔函数 `Y_v(x)` |
| `erf(x)`                   | 误差函数                  |
| `expit(x)`                 | sigmoid 函数              |
| `logit(x)`                 | sigmoid 的逆函数          |
| `softmax(x)`               | softmax 激活              |

### 示例代码

```python
from scipy import special
import math

# 阶乘与组合数
print(f"5! = {special.factorial(5)}")
print(f"C(10, 3) = {special.comb(10, 3)}")
print(f"P(10, 3) = {special.perm(10, 3)}")

# 伽马函数
print(f"Γ(5) (=4!) = {special.gamma(5)}")
print(f"Γ(0.5) (=√π) = {special.gamma(0.5)}")
print(f"对比 √π = {math.sqrt(math.pi)}")

# 贝塞尔函数
print(f"J_0(1) = {special.jv(0, 1):.6f}")
print(f"J_1(1) = {special.jv(1, 1):.6f}")
```

### 输出

```text
5! = 120.0
C(10, 3) = 120.0
P(10, 3) = 720.0
Γ(5) (=4!) = 24.0
Γ(0.5) (=√π) = 1.7724538509055159
对比 √π = 1.7724538509055159
J_0(1) = 0.765198
J_1(1) = 0.440051
```

### 理解重点

- `factorial(n, exact=True)` 返回整数精确值；默认 `exact=False` 返回浮点。
- **大数阶乘 / 组合数易溢出**：用 `gammaln` / `comb(..., exact=True)` 更稳。
- 机器学习中常用：`special.expit` 是 sigmoid 的数值稳定实现、`special.softmax` 避免 overflow。

## 版本查询

### 示例代码

```python
import scipy
import numpy as np

print(f"SciPy 版本: {scipy.__version__}")
print(f"NumPy 版本: {np.__version__}")
```

### 输出

```text
SciPy 版本: 1.11.4
NumPy 版本: 1.26.2
```

## 常见坑

1. 不要写 `import scipy` 然后 `scipy.optimize.minimize(...)`——旧版 `scipy` 顶层不自动导入子模块，需要 `from scipy import optimize`。
2. `scipy.constants.e` 是**基本电荷**（不是欧拉数 e！），欧拉数用 `math.e` 或 `numpy.e`。
3. `special.factorial(n)` 对大 `n` 会溢出（尤其 `exact=False` 时），改用 `gammaln(n+1)`。
4. `special.comb(N, k)` 同理，大数用 `exact=True` 或 `gammaln`。
5. SciPy 与 NumPy 版本有兼容矩阵，升级前查官方说明。

## 小结

- SciPy 是**建立在 NumPy 之上**的科学计算生态，按子模块组织。
- 常数 + 特殊函数是最轻量、最通用的入口，本章作为后续章节的起点。
- 后续章节会逐一深入核心子模块：`stats`、`optimize`、`interpolate`、`integrate`、`linalg`、`signal`、`sparse`、`spatial`。
