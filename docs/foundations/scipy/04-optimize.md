---
title: SciPy 优化
outline: deep
---

# SciPy 优化

## 本章目标

1. 掌握 `curve_fit` 进行非线性曲线拟合
2. 学会使用 `brentq` 和 `fsolve` 求解方程与方程组
3. 理解一维和多维最小化方法（`minimize_scalar` / `minimize`）
4. 了解线性规划 `linprog` 的问题建模与求解

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `optimize.curve_fit(f, xdata, ydata)` | 函数 | 非线性最小二乘曲线拟合 |
| `optimize.brentq(f, a, b)` | 函数 | 区间求根（Brent 方法，一维） |
| `optimize.fsolve(func, x0)` | 函数 | 方程组求根（多维） |
| `optimize.minimize_scalar(fun)` | 函数 | 一维标量函数最小化 |
| `optimize.minimize(fun, x0, method)` | 函数 | 多维函数最小化 |
| `optimize.linprog(c, A_ub, b_ub)` | 函数 | 线性规划 |

## 1. 曲线拟合

### `optimize.curve_fit`

#### 作用

使用非线性最小二乘法将自定义模型函数拟合到数据。返回 `(popt, pcov)`：最优参数和协方差矩阵。参数标准误 = `np.sqrt(np.diag(pcov))`。

模型函数的第一个参数必须是自变量 x，后续参数为待拟合参数。

#### 重点方法

```python
optimize.curve_fit(f, xdata, ydata, p0=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `f` | `callable` | 模型函数，签名为 `f(x, *params)` | `lambda x, a, b, c: a*x**2 + b*x + c` |
| `xdata` | `array_like` | 自变量数据 | `np.linspace(0, 10, 50)` |
| `ydata` | `array_like` | 因变量数据 | 含噪声的二次函数值 |
| `p0` | `array_like` 或 `None` | 参数初始猜测；`None` 时默认为全 1 | `[1, 1, 1]` |

#### 示例代码

```python
import numpy as np
from scipy import optimize

def model(x, a, b, c):
    return a * x**2 + b * x + c

np.random.seed(42)
xData = np.linspace(0, 10, 50)
yData = 2 * xData**2 + 3 * xData + 5 + np.random.normal(0, 5, 50)

params, cov = optimize.curve_fit(model, xData, yData)
print(f"真实参数: a=2, b=3, c=5")
print(f"拟合参数: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}")
print(f"标准误: {np.sqrt(np.diag(cov))}")
```

#### 输出

```text
真实参数: a=2, b=3, c=5
拟合参数: a=2.0144, b=2.8485, c=5.4753
标准误: [0.0417 0.3987 0.7717]
```

#### 理解重点

- 拟合参数 a≈2.01, b≈2.85, c≈5.48 接近真实值 (2, 3, 5)
- 标准误反映参数估计的不确定性：a 的标准误最小（0.04），c 最大（0.77）
- 高次项系数估计更精确——其对 y 的影响更大
- 复杂模型需提供合理的 `p0`，否则可能收敛到局部最优

## 2. 求根算法

### `optimize.brentq` / `optimize.fsolve`

#### 作用

- **brentq**：在区间 [a, b] 内求根，要求 f(a)·f(b) < 0，保证收敛，仅限一维
- **fsolve**：从初始点 x0 出发用牛顿类方法求根，支持多元方程组

#### 重点方法

```python
optimize.brentq(f, a, b)
optimize.fsolve(func, x0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `f` | `callable` | 一维目标函数 f(x)，须 f(a)·f(b) < 0 | `lambda x: x**2 - 4` |
| `a` | `float` | 搜索区间左端点 | `0` |
| `b` | `float` | 搜索区间右端点 | `3` |
| `func` | `callable` | 返回残差向量；多元时返回 `[r1, r2, ...]` | `lambda p: [p[0]+p[1]-3, p[0]-p[1]-1]` |
| `x0` | `array_like` | 初始猜测（一维传标量，多维传 list） | `[0, 0]` |

#### 示例代码

```python
from scipy import optimize

# 一维求根：f(x) = x² - 4
def f(x):
    return x**2 - 4

root = optimize.brentq(f, 0, 3)
print(f"brentq 求根 [0,3]: x = {root:.6f}, f(x) = {f(root):.2e}")

# fsolve 求根
root2 = optimize.fsolve(f, x0=1)[0]
print(f"fsolve 求根 (x0=1): x = {root2:.6f}")

# 多元方程组：x+y=3, x-y=1
def equations(p):
    x, y = p
    return [x + y - 3, x - y - 1]

sol = optimize.fsolve(equations, x0=[0, 0])
print(f"方程组解: x={sol[0]:.1f}, y={sol[1]:.1f}")
```

#### 输出

```text
brentq 求根 [0,3]: x = 2.000000, f(x) = 0.00e+00
fsolve 求根 (x0=1): x = 2.000000
方程组解: x=2.0, y=1.0
```

#### 理解重点

- `brentq` 精确找到 x=2（x²−4=0 的正根），残差达机器精度
- `fsolve` 不同 x0 可能找不同根——x0=−1 会找到 x=−2
- 方程组 x+y=3, x−y=1 的解为 (2, 1)
- `brentq` 适合一维且已知根区间；`fsolve` 适合多维或不知区间

## 3. 最小化

### `optimize.minimize_scalar` / `optimize.minimize`

#### 作用

- **minimize_scalar**：一维标量函数最小化，无需提供梯度
- **minimize**：多维函数最小化，需指定初始点 `x0` 和优化方法

常用方法：`'BFGS'`（拟牛顿）、`'Nelder-Mead'`（单纯形）、`'L-BFGS-B'`（支持边界约束）。

#### 重点方法

```python
optimize.minimize_scalar(fun)
optimize.minimize(fun, x0, method='BFGS')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fun` | `callable` | 目标函数 | `lambda x: (x-3)**2 + 2` |
| `x0` | `array_like` | 初始点（多维时） | `[0.0, 0.0]` |
| `method` | `str` | 优化算法：`'BFGS'` / `'Nelder-Mead'` / `'L-BFGS-B'` 等 | `'BFGS'` |

#### 示例代码

```python
import numpy as np
from scipy import optimize

# 一维最小化：f(x) = (x-3)² + 2
def f(x):
    return (x - 3)**2 + 2

r1 = optimize.minimize_scalar(f)
print(f"一维: x={r1.x:.6f}, f(x)={r1.fun:.6f}")

# 多维最小化：Rosenbrock 函数
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

r2 = optimize.minimize(rosenbrock, np.array([0.0, 0.0]), method='BFGS')
print(f"Rosenbrock: x={r2.x}, f={r2.fun:.6f}, 迭代={r2.nit}")
```

#### 输出

```text
一维: x=3.000000, f(x)=2.000000
Rosenbrock: x=[1. 1.], f=0.000000, 迭代=34
```

#### 理解重点

- (x−3)²+2 的最小值点为 x=3，最小值=2——`minimize_scalar` 精确找到
- Rosenbrock 函数的全局最小值在 (1, 1)，值为 0——其"香蕉形"山谷使优化困难
- BFGS 算法约 34 次迭代找到最优解
- 不同方法适用于不同问题：无约束用 BFGS，有界约束用 L-BFGS-B

## 4. 线性规划

### `optimize.linprog`

#### 作用

求解标准形式的线性规划问题：$\min \mathbf{c}^T \mathbf{x}$，约束 $A_{ub}\mathbf{x} \leq b_{ub}$，$A_{eq}\mathbf{x} = b_{eq}$，$x \geq 0$。

最大化问题需对目标系数取负，结果也需取负。

#### 重点方法

```python
optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                 bounds=None, method='highs')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `c` | `array_like` | 目标函数系数（最小化） | `[-2, -3]`（取负以最大化） |
| `A_ub` | `array_like` | 不等式约束左侧矩阵 | `[[1, 1], [1, 0], [0, 1]]` |
| `b_ub` | `array_like` | 不等式约束右侧向量 | `[4, 2, 3]` |
| `bounds` | `tuple` 或 `None` | 变量边界；`None` 表示 $x \geq 0$ | `(0, None)` |
| `method` | `str` | 求解算法，默认为 `'highs'` | `'highs'` |

#### 示例代码

```python
from scipy import optimize

# 最大化 z = 2x + 3y
# 约束: x + y ≤ 4, x ≤ 2, y ≤ 3, x,y ≥ 0
# linprog 求最小化，目标取负

c = [-2, -3]
A_ub = [[1, 1], [1, 0], [0, 1]]
b_ub = [4, 2, 3]

result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
print(f"最优解: x={result.x[0]:.1f}, y={result.x[1]:.1f}")
print(f"最大值: z={-result.fun:.1f}")
```

#### 输出

```text
最优解: x=1.0, y=3.0
最大值: z=11.0
```

#### 理解重点

- 最优解 (1, 3)，最大值 z = 2×1 + 3×3 = 11
- 最优解一定出现在可行域的顶点上（线性规划基本定理）
- `linprog` 返回的 `result.fun` 是最小化结果（即 −11），取负得最大值 11
- 可行域顶点: (0,0), (2,0), (2,2), (1,3), (0,3)——逐个计算 z 可验证 (1,3) 最优

## 常见坑

1. `curve_fit` 对复杂模型需提供合理的 `p0`——否则可能不收敛或收敛到局部最优
2. `brentq` 要求区间端点异号——f(a)·f(b) < 0，否则报错
3. `fsolve` 不同 `x0` 可能找到不同的根——多根情况需多次尝试不同初始值
4. `linprog` 是最小化——最大化问题必须对目标系数取负，最终结果也取负
5. `minimize` 方法选择：无约束用 BFGS，有界约束用 L-BFGS-B，非光滑用 Nelder-Mead

## 小结

- `curve_fit` 通过非线性最小二乘拟合自定义模型，返回最优参数和协方差矩阵
- `brentq`（区间法）和 `fsolve`（牛顿法）用于求解方程和方程组——各有适用场景
- `minimize_scalar` / `minimize` 覆盖从一维到多维的函数最小化——`method` 参数选择关键
- `linprog` 求解线性规划问题——关键是将实际问题转化为标准形式
- 优化问题的核心：选择合适方法 → 提供好的初始值 → 理解约束条件
