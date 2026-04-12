---
title: 积分
outline: deep
---

# 积分

> 对应脚本：`Basic/Scipy/06_integrate.py`
> 运行方式：`python Basic/Scipy/06_integrate.py`（仓库根目录）

## 本章目标

1. 掌握 `quad` 计算定积分（有限区间与无穷区间）。
2. 学会使用 `dblquad` 计算二重积分。
3. 理解 `odeint` 求解一阶常微分方程。
4. 了解 `odeint` 求解 ODE 方程组（Lotka-Volterra 模型）。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `integrate.quad(func, a, b)` | 一维定积分 | `demo_quad` |
| `integrate.dblquad(func, a, b, gfun, hfun)` | 二重积分 | `demo_dblquad` |
| `integrate.odeint(func, y0, t)` | 常微分方程数值求解 | `demo_odeint` |
| `integrate.odeint(func, y0, t)` | ODE 方程组求解 | `demo_ode_system` |

## 1. 定积分

### 方法重点

- `quad` 使用自适应 Gauss 求积法计算定积分，返回 `(result, error)`。
- 支持有限区间 `[a, b]` 和无穷区间（`a=-np.inf`、`b=np.inf`）。
- 误差估计 `error` 表示数值积分的绝对误差上界。
- 被积函数是一个可调用对象 `func(x)`，可以是 lambda 或命名函数。

### 参数速览（本节）

适用 API：`integrate.quad(func, a, b)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `func` | `lambda x: x**2` / `np.sin` / 高斯函数 | 被积函数 |
| `a` | `0` / `0` / `-np.inf` | 积分下限 |
| `b` | `1` / `np.pi` / `np.inf` | 积分上限 |

### 示例代码

```python
import numpy as np
from scipy import integrate

# 积分 ∫x^2 dx, 从0到1
result1, error1 = integrate.quad(lambda x: x**2, 0, 1)
print(f"∫₀¹ x² dx = {result1:.6f} (误差: {error1:.2e})")
print(f"解析解: 1/3 = {1/3:.6f}")

# 积分 ∫sin(x) dx, 从0到π
result2, error2 = integrate.quad(np.sin, 0, np.pi)
print(f"∫₀π sin(x) dx = {result2:.6f} (误差: {error2:.2e})")
print("解析解: 2")

# 无穷积分 ∫e^(-x^2) dx, 从-inf到+inf
result3, error3 = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(f"∫_(-inf)^(+inf) e^(-x²) dx = {result3:.6f}")
print(f"解析解: √π = {np.sqrt(np.pi):.6f}")
```

### 结果输出

```text
∫₀¹ x² dx = 0.333333 (误差: 3.70e-15)
解析解: 1/3 = 0.333333

∫₀π sin(x) dx = 2.000000 (误差: 2.22e-14)
解析解: 2

∫_(-inf)^(+inf) e^(-x²) dx = 1.772454
解析解: √π = 1.772454
```

### 理解重点

- 三个积分的数值解与解析解完全一致，误差在 10⁻¹⁴ ~ 10⁻¹⁵ 量级（机器精度）。
- `quad` 能自动处理无穷区间，通过变量替换将无穷积分转化为有限区间。
- 高斯积分 ∫e^(-x²)dx = √π 是概率论和统计学中的基础结果。
- 误差估计值远小于结果，说明数值积分的可靠性很高。

## 2. 二重积分

### 方法重点

- `dblquad` 计算 ∫∫f(y,x) dy dx 形式的二重积分。
- **注意**：被积函数的参数顺序是 `func(y, x)`，不是 `func(x, y)`。
- y 的积分范围可以是 x 的函数，用于处理非矩形区域。
- `gfun(x)` 和 `hfun(x)` 分别是 y 的下限和上限函数。

### 参数速览（本节）

适用 API：`integrate.dblquad(func, a, b, gfun, hfun)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `func` | `lambda y, x: x*y` / `lambda y, x: 1` | 被积函数（注意 y 在前） |
| `a`, `b` | `0`, `1` / `-1`, `1` | x 的积分范围 |
| `gfun` | `lambda x: 0` / `lambda x: -sqrt(1-x²)` | y 的下限函数 |
| `hfun` | `lambda x: 2` / `lambda x: sqrt(1-x²)` | y 的上限函数 |

### 示例代码

```python
import numpy as np
from scipy import integrate

# 矩形区域: ∫∫xy dA, [0,1]×[0,2]
result1, error1 = integrate.dblquad(
    lambda y, x: x * y,
    0, 1,              # x 的范围
    lambda x: 0,       # y 的下限
    lambda x: 2        # y 的上限
)
print(f"∫₀¹∫₀² xy dy dx = {result1:.6f}")
print("解析解: 1")

# 圆形区域: 计算单位圆面积
result2, error2 = integrate.dblquad(
    lambda y, x: 1,
    -1, 1,
    lambda x: -np.sqrt(1 - x**2),
    lambda x: np.sqrt(1 - x**2)
)
print(f"单位圆面积 = {result2:.6f}")
print(f"解析解: π = {np.pi:.6f}")
```

### 结果输出

```text
∫₀¹∫₀² xy dy dx = 1.000000
解析解: 1

单位圆面积 = 3.141593
解析解: π = 3.141593
```

### 理解重点

- 矩形区域积分 ∫₀¹∫₀² xy dy dx = [x²/2]₀¹ × [y²/2]₀² = 1/2 × 2 = 1。
- 圆形区域通过变积分上下限（y = ±√(1-x²)）实现非矩形区域积分。
- `dblquad` 的参数顺序容易混淆：`func(y, x)` 中 y 是内层积分变量。
- 单位圆面积 = π 验证了积分的正确性。

## 3. 常微分方程 (ODE)

### 方法重点

- `odeint` 使用 LSODA 算法求解初值问题 dy/dt = f(y, t)。
- 函数签名 `func(y, t)` 返回 dy/dt 的值。
- `y0` 是初始条件，`t` 是求解的时间点数组。
- 返回的解数组形状为 `(len(t), len(y0))`。

### 参数速览（本节）

适用 API：`integrate.odeint(func, y0, t)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `func` | `lambda y, t: -y` | 微分方程右端函数 |
| `y0` | `1` | 初始条件 y(0) = 1 |
| `t` | `np.linspace(0, 5, 6)` | 求解时间点 |

### 示例代码

```python
import numpy as np
from scipy import integrate

# 一阶ODE: dy/dt = -y, y(0) = 1
# 解析解: y = e^(-t)

def dydt(y, t):
    return -y

t = np.linspace(0, 5, 6)
y0 = 1
y = integrate.odeint(dydt, y0, t)

print("t\t数值解\t\t解析解")
for ti, yi in zip(t, y.flatten()):
    print(f"{ti:.1f}\t{yi:.6f}\t{np.exp(-ti):.6f}")
```

### 结果输出

```text
一阶ODE: dy/dt = -y, y(0) = 1

t       数值解          解析解
0.0     1.000000        1.000000
1.0     0.367879        0.367879
2.0     0.135335        0.135335
3.0     0.049787        0.049787
4.0     0.018316        0.018316
5.0     0.006738        0.006738
```

### 理解重点

- dy/dt = -y 描述指数衰减过程，解析解 y = e^(-t)。
- 数值解与解析解在所有时间点完全一致（6位有效数字内），精度极高。
- `odeint` 内部自适应步长，即使用户只指定 6 个时间点，算法也会在中间加密计算。
- LSODA 算法自动在刚性和非刚性方法之间切换，适应性强。

## 4. ODE 方程组

### 方法重点

- `odeint` 也可以求解方程组，`func` 返回向量，`y0` 是向量。
- Lotka-Volterra 方程是经典的捕食者-猎物模型。
- 相空间轨迹（相图）展示系统的周期性行为。
- 参数 α, β, δ, γ 控制种群动力学特征。

### 参数速览（本节）

适用 API：`integrate.odeint(func, y0, t)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `func` | Lotka-Volterra 方程 | dx/dt = αx - βxy, dy/dt = δxy - γy |
| `y0` | `[10, 5]` | 初始条件：猎物=10, 捕食者=5 |
| `t` | `np.linspace(0, 40, 500)` | 求解时间范围 |
| `α, β, δ, γ` | `1.0, 0.1, 0.075, 1.5` | 模型参数 |

### 示例代码

```python
import numpy as np
from scipy import integrate

# Lotka-Volterra 方程 (捕食者-猎物模型)
# dx/dt = αx - βxy  (猎物)
# dy/dt = δxy - γy  (捕食者)
alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5

def lotka_volterra(state, t):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

t = np.linspace(0, 40, 500)
state0 = [10, 5]

solution = integrate.odeint(lotka_volterra, state0, t)

print("Lotka-Volterra 模型:")
print(f"  参数: α={alpha}, β={beta}, δ={delta}, γ={gamma}")
print(f"  初始条件: 猎物={state0[0]}, 捕食者={state0[1]}")
```

### 结果输出

```text
Lotka-Volterra 模型:
  参数: α=1.0, β=0.1, δ=0.075, γ=1.5
  初始条件: 猎物=10, 捕食者=5
```

### 理解重点

- Lotka-Volterra 模型描述了猎物与捕食者之间的周期性波动。
- 猎物增多 → 捕食者增多 → 猎物减少 → 捕食者减少 → 猎物增多（循环）。
- 相空间轨迹呈闭合环，说明系统具有周期性（守恒量存在）。
- `odeint` 返回形状为 `(500, 2)` 的数组，`solution[:, 0]` 是猎物、`solution[:, 1]` 是捕食者。

## 常见坑

| 坑 | 说明 |
|---|---|
| `dblquad` 参数顺序 | 被积函数是 `func(y, x)` 不是 `func(x, y)`，内层积分变量在前 |
| `odeint` 的 `func` 签名 | 参数顺序是 `func(y, t)`，不是 `func(t, y)`（`solve_ivp` 则相反） |
| 无穷积分的收敛性 | `quad` 对不收敛的积分可能返回错误结果，需要检查 error 值 |
| `odeint` 刚性问题 | 对于刚性 ODE，LSODA 会自动切换方法，但极端情况可能需要调整容差 |
| 时间点密度 | `odeint` 的 `t` 数组只影响输出点，不影响内部计算步长 |

## 小结

- `quad` 高精度计算一维定积分，支持有限区间和无穷区间。
- `dblquad` 计算二重积分，支持非矩形区域（y 的范围可以是 x 的函数）。
- `odeint` 使用 LSODA 算法求解常微分方程初值问题，自适应步长和方法切换。
- ODE 方程组通过向量化的 `func` 和 `y0` 实现，适用于物理、生态等多维动力系统。
- 数值积分的核心是：理解函数签名的参数顺序、检查误差估计、选择合适的求解器。
