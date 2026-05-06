---
title: SciPy 积分
outline: deep
---

# SciPy 积分

## 本章目标

1. 掌握 `quad` 计算定积分（有限区间与无穷区间）
2. 学会使用 `dblquad` 计算二重积分
3. 理解 `odeint` 求解一阶常微分方程
4. 了解 `odeint` 求解 ODE 方程组（Lotka-Volterra 模型）

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `integrate.quad(func, a, b)` | 函数 | 一维定积分（自适应 Gauss 求积法） |
| `integrate.dblquad(func, a, b, gfun, hfun)` | 函数 | 二重积分 |
| `integrate.odeint(func, y0, t)` | 函数 | 常微分方程数值求解 |
| `integrate.solve_ivp(fun, t_span, y0)` | 函数 | ODE 初值问题（新版推荐） |

`quad` 返回 `(result, error)`，`odeint` 返回解数组形状为 `(len(t), len(y0))`。

## 1. 定积分

### `integrate.quad`

#### 作用

使用自适应 Gauss 求积法计算定积分。支持有限区间 $[a, b]$ 和无穷区间（`a=-np.inf`、`b=np.inf`）。返回 `(result, error)`，`error` 为绝对误差上界。

#### 重点方法

```python
integrate.quad(func, a, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `callable` | 被积函数 `func(x)` | `lambda x: x**2` |
| `a` | `float` | 积分下限 | `0`、`-np.inf` |
| `b` | `float` | 积分上限 | `1`、`np.inf` |

#### 示例代码

```python
import numpy as np
from scipy import integrate

# ∫₀¹ x² dx
r1, e1 = integrate.quad(lambda x: x**2, 0, 1)
print(f"∫₀¹ x² dx = {r1:.6f} (误差: {e1:.2e})")
print(f"解析解: 1/3 = {1/3:.6f}")

# ∫₀ᵠ sin(x) dx
r2, e2 = integrate.quad(np.sin, 0, np.pi)
print(f"∫₀ᵠ sin(x) dx = {r2:.6f} (解析解: 2)")

# 无穷积分 ∫e^(-x²) dx
r3, e3 = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(f"∫e^(-x²) dx = {r3:.6f} (解析解: √π = {np.sqrt(np.pi):.6f})")
```

#### 输出

```text
∫₀¹ x² dx = 0.333333 (误差: 3.70e-15)
解析解: 1/3 = 0.333333
∫₀ᵠ sin(x) dx = 2.000000 (解析解: 2)
∫e^(-x²) dx = 1.772454 (解析解: √π = 1.772454)
```

#### 理解重点

- 三个积分的数值解与解析解完全一致——误差在 $10^{-14} \sim 10^{-15}$ 量级
- `quad` 能自动处理无穷区间——通过变量替换将无穷积分转化为有限区间
- 高斯积分 $\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$ 是概率论的基础结果
- 误差估计值远小于结果——说明数值积分的可靠性很高

## 2. 二重积分

### `integrate.dblquad`

#### 作用

计算 $\iint f(y, x) \,dy\,dx$ 形式的二重积分。**注意**：被积函数的参数顺序是 `func(y, x)`（内层积分变量在前）。y 的积分范围可以是 x 的函数，用于处理非矩形区域。

#### 重点方法

```python
integrate.dblquad(func, a, b, gfun, hfun)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `callable` | 被积函数 `func(y, x)`——注意 y 在前 | `lambda y, x: x*y` |
| `a` | `float` | x 的积分下限 | `0` |
| `b` | `float` | x 的积分上限 | `1` |
| `gfun` | `callable` | y 的下限函数 `gfun(x)` | `lambda x: 0` |
| `hfun` | `callable` | y 的上限函数 `hfun(x)` | `lambda x: 2` |

#### 示例代码

```python
import numpy as np
from scipy import integrate

# 矩形区域: ∬xy dA, [0,1]×[0,2]
r1, e1 = integrate.dblquad(
    lambda y, x: x * y, 0, 1,
    lambda x: 0, lambda x: 2
)
print(f"∬xy dA = {r1:.6f} (解析解: 1)")

# 圆形区域: 单位圆面积
r2, e2 = integrate.dblquad(
    lambda y, x: 1, -1, 1,
    lambda x: -np.sqrt(1 - x**2),
    lambda x: np.sqrt(1 - x**2)
)
print(f"单位圆面积 = {r2:.6f} (解析解: π = {np.pi:.6f})")
```

#### 输出

```text
∬xy dA = 1.000000 (解析解: 1)
单位圆面积 = 3.141593 (解析解: π = 3.141593)
```

#### 理解重点

- 矩形区域积分 $\int_0^1\int_0^2 xy\,dy\,dx = \frac{1}{2} \times 2 = 1$
- 圆形区域通过变积分上下限（$y = \pm\sqrt{1-x^2}$）实现非矩形区域积分
- `dblquad` 的参数顺序容易混淆：`func(y, x)` 中 y 是内层积分变量
- 单位圆面积 = π——验证了积分的正确性

## 3. 常微分方程（ODE）

### `integrate.odeint`

#### 作用

使用 LSODA 算法求解初值问题 $dy/dt = f(y, t)$。内部自适应步长，自动在刚性和非刚性方法之间切换。函数签名为 `func(y, t)`（注意 y 在前，与 `solve_ivp` 相反）。

#### 重点方法

```python
integrate.odeint(func, y0, t)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `callable` | 微分方程右端函数 `func(y, t)`，返回 `dy/dt` | `lambda y, t: -y` |
| `y0` | `array_like` | 初始条件 | `1` 或 `[10, 5]` |
| `t` | `array_like` | 求解的时间点数组——只影响输出点，不影响内部步长 | `np.linspace(0, 5, 100)` |

#### 示例代码

```python
import numpy as np
from scipy import integrate

# 一阶 ODE: dy/dt = -y, y(0) = 1  →  解析解: y = e^(-t)
def dydt(y, t):
    return -y

t = np.linspace(0, 5, 6)
y = integrate.odeint(dydt, 1, t)

print("t      数值解      解析解")
for ti, yi in zip(t, y.flatten()):
    print(f"{ti:.1f}    {yi:.6f}    {np.exp(-ti):.6f}")
```

#### 输出

```text
t      数值解      解析解
0.0    1.000000    1.000000
1.0    0.367879    0.367879
2.0    0.135335    0.135335
3.0    0.049787    0.049787
4.0    0.018316    0.018316
5.0    0.006738    0.006738
```

#### 理解重点

- dy/dt = −y 描述指数衰减——解析解 $y = e^{-t}$，数值解与解析解完全一致
- `odeint` 内部自适应步长——用户指定的 `t` 只影响输出点密度
- LSODA 算法自动在刚性和非刚性方法间切换——适应性极强
- 函数签名是 `func(y, t)`，新版 `solve_ivp` 的签名是 `fun(t, y)`——两者相反

## 4. ODE 方程组

### `integrate.odeint`（向量化）

#### 作用

`odeint` 同样可求解方程组——`func` 返回向量，`y0` 为向量。Lotka-Volterra 方程是经典的捕食者-猎物模型。

$$
\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta xy \quad \text{(猎物)} \\[4pt]
\frac{dy}{dt} &= \delta xy - \gamma y \quad \text{(捕食者)}
\end{aligned}
$$

#### 示例代码

```python
import numpy as np
from scipy import integrate

alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5

def lotkaVolterra(state, t):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

t = np.linspace(0, 40, 500)
state0 = [10, 5]
solution = integrate.odeint(lotkaVolterra, state0, t)

print(f"Lotka-Volterra 模型 (α={alpha}, β={beta}, δ={delta}, γ={gamma})")
print(f"初始: 猎物={state0[0]}, 捕食者={state0[1]}")
print(f"解形状: {solution.shape}")
print(f"猎物 [t=0,20,40]: {solution[0,0]:.1f}, {solution[250,0]:.1f}, {solution[-1,0]:.1f}")
```

#### 输出

```text
Lotka-Volterra 模型 (α=1.0, β=0.1, δ=0.075, γ=1.5)
初始: 猎物=10, 捕食者=5
解形状: (500, 2)
猎物 [t=0,20,40]: 10.0, 8.5, 10.0
```

#### 理解重点

- 猎物增多 → 捕食者增多 → 猎物减少 → 捕食者减少 → 猎物增多（循环）
- 相空间轨迹呈闭合环——系统具有周期性（守恒量存在）
- `odeint` 返回形状 `(500, 2)`：`solution[:, 0]` 是猎物，`solution[:, 1]` 是捕食者
- 方程组通过向量化的 `func` 和 `y0` 实现——扩展到更高维同样简单

## 常见坑

1. `dblquad` 参数顺序：被积函数是 `func(y, x)` 不是 `func(x, y)`——内层积分变量在前
2. `odeint` 的 `func` 签名：参数顺序是 `func(y, t)`，`solve_ivp` 则是 `fun(t, y)`
3. 无穷积分的收敛性：`quad` 对不收敛积分可能返回错误结果——需检查 error 值
4. `odeint` 刚性问题：极端情况可能需要调整容差参数
5. 时间点密度：`odeint` 的 `t` 数组只影响输出，不影响内部计算步长

## 小结

- `quad` 高精度计算一维定积分——支持有限区间和无穷区间
- `dblquad` 计算二重积分——支持非矩形区域（y 的范围可以是 x 的函数）
- `odeint` 使用 LSODA 算法求解 ODE 初值问题——自适应步长和方法切换
- ODE 方程组通过向量化的 `func` 和 `y0` 实现——适用于物理、生态等多维动力系统
- 数值积分的核心：理解函数签名的参数顺序、检查误差估计、选择合适的求解器
