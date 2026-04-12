---
title: 优化
outline: deep
---

# 优化

> 对应脚本：`Basic/Scipy/04_optimize.py`
> 运行方式：`python Basic/Scipy/04_optimize.py`（仓库根目录）

## 本章目标

1. 掌握 `curve_fit` 进行非线性曲线拟合。
2. 学会使用 `brentq` 和 `fsolve` 求解方程和方程组。
3. 理解一维和多维最小化方法（`minimize_scalar` / `minimize`）。
4. 了解线性规划 `linprog` 的问题建模与求解。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `optimize.curve_fit(f, xdata, ydata)` | 非线性最小二乘曲线拟合 | `demo_curve_fit` |
| `optimize.brentq(f, a, b)` | 区间求根（Brent 方法） | `demo_root_finding` |
| `optimize.fsolve(func, x0)` | 求根 / 方程组求解 | `demo_root_finding` |
| `optimize.minimize_scalar(fun)` | 一维函数最小化 | `demo_minimize` |
| `optimize.minimize(fun, x0, method)` | 多维函数最小化 | `demo_minimize` |
| `optimize.linprog(c, A_ub, b_ub)` | 线性规划 | `demo_linear_programming` |

## 1. 曲线拟合

### 方法重点

- `curve_fit` 使用非线性最小二乘法，将自定义模型函数拟合到数据。
- 返回 `(popt, pcov)`：最优参数和协方差矩阵。
- 参数标准误 = `np.sqrt(np.diag(pcov))`。
- 模型函数的第一个参数必须是自变量 x，后续参数是待拟合参数。

### 参数速览（本节）

适用 API：`optimize.curve_fit(f, xdata, ydata, p0=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `f` | `model(x, a, b, c)` | 模型函数，`y = ax² + bx + c` |
| `xdata` | `np.linspace(0, 10, 50)` | 自变量数据 |
| `ydata` | 含噪声的二次函数数据 | 因变量数据 |
| `p0` | `None`（默认全 1） | 参数初始猜测 |

### 示例代码

```python
import numpy as np
from scipy import optimize

def model(x, a, b, c):
    return a * x**2 + b * x + c

# 生成带噪声的数据
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_data = 2 * x_data**2 + 3 * x_data + 5 + np.random.normal(0, 5, 50)

# 拟合
params, covariance = optimize.curve_fit(model, x_data, y_data)

print(f"真实参数: a=2, b=3, c=5")
print(f"拟合参数: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}")
print(f"参数标准误: {np.sqrt(np.diag(covariance))}")
```

### 结果输出

```text
真实参数: a=2, b=3, c=5
拟合参数: a=2.0144, b=2.8485, c=5.4753
参数标准误: [0.0417 0.3987 0.7717]
```

### 理解重点

- 拟合参数 a≈2.01, b≈2.85, c≈5.48 非常接近真实值 (2, 3, 5)。
- 标准误反映参数估计的不确定性：a 的标准误最小（0.04），c 的最大（0.77）。
- 高次项系数通常估计更精确，因为其对 y 的影响更大。
- 残差应随机分布在零线两侧，呈现模式则说明模型选择不当。

## 2. 求根算法

### 方法重点

- `brentq(f, a, b)` 在区间 [a, b] 内求根，要求 f(a) 和 f(b) 异号，保证收敛。
- `fsolve(func, x0)` 使用牛顿法类算法，从初始点 x0 出发寻找根。
- `fsolve` 可以求解多元方程组，`func` 返回残差向量。
- `brentq` 更可靠但仅限一维；`fsolve` 更灵活但依赖初始值。

### 参数速览（本节）

适用 API（分项）：

1. `optimize.brentq(f, a, b)`
2. `optimize.fsolve(func, x0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `f` | `lambda x: x**2 - 4` | 目标函数 |
| `a`, `b` | `0`, `3` | 搜索区间（需 f(a)·f(b) < 0） |
| `func` | 方程组函数 | 返回残差向量 |
| `x0` | `[0, 0]` | 初始猜测 |

### 示例代码

```python
from scipy import optimize

# 一元求根: f(x) = x^2 - 4
def f(x):
    return x**2 - 4

root = optimize.brentq(f, 0, 3)
print(f"brentq 求根: x = {root:.6f}")
print(f"验证 f({root:.6f}) = {f(root):.10f}")

# fsolve 求根
root_fsolve = optimize.fsolve(f, x0=1)[0]
print(f"fsolve 求根: x = {root_fsolve:.6f}")

# 多元方程组: x+y=3, x-y=1
def equations(p):
    x, y = p
    return [x + y - 3, x - y - 1]

solution = optimize.fsolve(equations, x0=[0, 0])
print(f"解: x = {solution[0]:.4f}, y = {solution[1]:.4f}")
```

### 结果输出

```text
f(x) = x^2 - 4
  brentq 求根 [0, 3]: x = 2.000000
  验证 f(2.000000) = 0.0000000000

  fsolve 求根 (x0=1): x = 2.000000

多元方程组求解:
  x + y = 3
  x - y = 1
  解: x = 2.0000, y = 1.0000
```

### 理解重点

- `brentq` 精确找到 x=2（x²-4=0 的正根），残差为机器精度级别。
- `fsolve` 从 x0=1 出发也找到 x=2，但不同的 x0 可能找到不同的根（如 x0=-1 会找到 x=-2）。
- 方程组 x+y=3, x-y=1 的解为 (2, 1)，`fsolve` 正确求解。
- `brentq` 适合一维且已知根的大致区间；`fsolve` 适合多维或不知区间的情况。

## 3. 最小化

### 方法重点

- `minimize_scalar` 用于一维标量函数的最小化，无需提供初始点。
- `minimize` 用于多维函数最小化，需指定初始点 `x0` 和优化方法。
- 常用方法：`'BFGS'`（拟牛顿法）、`'Nelder-Mead'`（单纯形法）、`'L-BFGS-B'`（支持约束）。
- `callback` 参数可记录优化路径，用于可视化和调试。

### 参数速览（本节）

适用 API（分项）：

1. `optimize.minimize_scalar(fun)`
2. `optimize.minimize(fun, x0, method, callback)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `fun` | `(x-3)² + 2` / Rosenbrock | 目标函数 |
| `x0` | `[0.0, 0.0]` | 初始点（多维） |
| `method` | `'BFGS'` | 优化算法 |
| `callback` | 记录路径的函数 | 每次迭代后调用 |

### 示例代码

```python
import numpy as np
from scipy import optimize

# 一维最小化
def f(x):
    return (x - 3) ** 2 + 2

result1 = optimize.minimize_scalar(f)
print(f"最小值点: x = {result1.x:.6f}")
print(f"最小值: f(x) = {result1.fun:.6f}")

# 多维最小化 - Rosenbrock 函数
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

x0 = np.array([0.0, 0.0])
result2 = optimize.minimize(rosenbrock, x0, method='BFGS')

print(f"最优解: {result2.x}")
print(f"最小值: {result2.fun:.6f}")
print(f"迭代次数: {result2.nit}")
```

### 结果输出

```text
一维最小化 f(x) = (x-3)^2 + 2:
  最小值点: x = 3.000000
  最小值: f(x) = 2.000000

Rosenbrock 函数最小化:
  最优解: [1. 1.]
  最小值: 0.000000
  迭代次数: 34
```

### 理解重点

- 一维函数 (x-3)²+2 的最小值点为 x=3，最小值为 2，`minimize_scalar` 精确找到。
- Rosenbrock 函数 f(x,y) = (1-x)² + 100(y-x²)² 的全局最小值在 (1,1)，值为 0。
- Rosenbrock 函数是优化算法的经典测试函数，其"香蕉形"山谷使优化困难。
- BFGS 算法经过约 34 次迭代找到最优解，路径沿山谷逐步逼近。

## 4. 线性规划

### 方法重点

- `linprog` 求解**最小化**问题，最大化问题需对目标函数取负。
- 标准形式：min c^T x, s.t. A_ub·x ≤ b_ub, A_eq·x = b_eq, bounds。
- 默认所有变量 x ≥ 0（非负约束）。
- `method='highs'` 是默认求解器，适用于大多数线性规划问题。

### 参数速览（本节）

适用 API：`optimize.linprog(c, A_ub, b_ub, method='highs')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `c` | `[-2, -3]` | 目标函数系数（取负以最大化） |
| `A_ub` | `[[1,1],[1,0],[0,1]]` | 不等式约束左侧矩阵 |
| `b_ub` | `[4, 2, 3]` | 不等式约束右侧向量 |
| `method` | `'highs'` | 求解算法 |

### 示例代码

```python
from scipy import optimize

# 最大化 z = 2x + 3y
# 约束: x + y ≤ 4, x ≤ 2, y ≤ 3, x,y ≥ 0
# linprog 求最小化，所以目标取负

c = [-2, -3]
A_ub = [[1, 1], [1, 0], [0, 1]]
b_ub = [4, 2, 3]

result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

print(f"最优解: x = {result.x[0]:.4f}, y = {result.x[1]:.4f}")
print(f"最大值: z = {-result.fun:.4f}")
```

### 结果输出

```text
线性规划问题:
  max z = 2x + 3y
  s.t. x + y ≤ 4, x ≤ 2, y ≤ 3, x,y ≥ 0
  最优解: x = 1.0000, y = 3.0000
  最大值: z = 11.0000
```

### 理解重点

- 最优解 (1, 3)，最大值 z = 2×1 + 3×3 = 11。
- 最优解一定出现在可行域的顶点上（线性规划基本定理）。
- 可行域顶点为 (0,0), (2,0), (2,2), (1,3), (0,3)，逐个计算 z 值可验证 (1,3) 最优。
- `linprog` 返回的 `result.fun` 是最小化结果（即 -11），取负得到最大值 11。

## 常见坑

| 坑 | 说明 |
|---|---|
| `curve_fit` 初始值敏感 | 复杂模型需提供合理的 `p0`，否则可能收敛到局部最优或不收敛 |
| `brentq` 要求异号 | 区间端点的函数值必须异号，否则报错 |
| `fsolve` 依赖初始值 | 不同 `x0` 可能找到不同的根，多根情况需多次尝试 |
| `linprog` 是最小化 | 最大化问题必须对目标系数取负，最终结果也要取负 |
| `minimize` 方法选择 | 不同方法适用于不同问题：无约束用 BFGS，有界约束用 L-BFGS-B |

## 小结

- `curve_fit` 通过非线性最小二乘拟合自定义模型，返回最优参数和不确定性。
- `brentq`（区间法）和 `fsolve`（牛顿法）用于求解方程和方程组。
- `minimize_scalar` / `minimize` 用于函数最小化，支持多种优化算法。
- `linprog` 求解线性规划问题，最优解一定在可行域顶点上。
- 优化问题的核心是：选择合适的方法、提供好的初始值、理解约束条件。
