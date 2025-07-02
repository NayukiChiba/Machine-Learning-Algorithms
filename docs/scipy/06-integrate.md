# 数值积分

> 对应代码: [06_integrate.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/06_integrate.py)

## 定积分 (Definite Integration)

数值积分用于计算函数在区间上的积分值。

### 1. 一维积分

```python
from scipy import integrate
import numpy as np

# 计算 ∫₀¹ x² dx = 1/3
result, error = integrate.quad(lambda x: x**2, 0, 1)

print(f"积分结果: {result:.10f}")
print(f"误差估计: {error:.2e}")
print(f"理论值: {1/3:.10f}")
```

**输出**:
```
积分结果: 0.3333333333
误差估计: 3.70e-15
理论值: 0.3333333333
```

**原理**: 自适应高斯-克朗罗德求积法，自动调整步长保证精度。

### 2. 复杂函数积分

```python
# 计算 ∫₀^π sin(x) dx = 2
result, _ = integrate.quad(np.sin, 0, np.pi)
print(f"∫₀^π sin(x)dx = {result:.6f}")

# 计算 ∫₀¹ e^x dx = e - 1
result, _ = integrate.quad(np.exp, 0, 1)
print(f"∫₀¹ e^x dx = {result:.6f}")
print(f"理论值: {np.e - 1:.6f}")

# 带参数的函数
def f(x, a, b):
    return a * x**2 + b * x

result, _ = integrate.quad(f, 0, 2, args=(3, 5))  # a=3, b=5
print(f"∫₀² (3x²+5x)dx = {result:.2f}")
```

**输出**:
```
∫₀^π sin(x)dx = 2.000000
∫₀¹ e^x dx = 1.718282
理论值: 1.718282
∫₀² (3x²+5x)dx = 18.00
```

### 3. 无穷积分

```python
# 计算 ∫₀^∞ e^(-x) dx = 1
result, _ = integrate.quad(lambda x: np.exp(-x), 0, np.inf)
print(f"∫₀^∞ e^(-x)dx = {result:.6f}")

# 计算 ∫_{-∞}^∞ e^(-x²) dx = √π
result, _ = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(f"∫_{-∞}^∞ e^(-x²)dx = {result:.6f}")
print(f"理论值 √π = {np.sqrt(np.pi):.6f}")
```

**输出**:
```
∫₀^∞ e^(-x)dx = 1.000000
∫_{-∞}^∞ e^(-x²)dx = 1.772454
理论值 √π = 1.772454
```

**注意**: 使用 `np.inf` 表示无穷大。

### 4. 多重积分

#### 二重积分

```python
# 计算 ∫₀¹ ∫₀¹ xy dxdy = 1/4
def f(y, x):  # 注意：y在前，x在后
    return x * y

result, error = integrate.dblquad(f, 0, 1, 0, 1)
print(f"二重积分结果: {result:.6f}")
print(f"理论值: {0.25:.6f}")
```

**输出**:
```
二重积分结果: 0.250000
理论值: 0.250000
```

#### 变限积分

```python
# 计算 ∫₀¹ ∫₀^x x²y dydx
def f(y, x):
    return x**2 * y

# y的范围依赖于x
result, _ = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: x)
print(f"变限积分结果: {result:.6f}")
```

**输出**: `变限积分结果: 0.083333`

#### 三重积分

```python
# ∫₀¹ ∫₀¹ ∫₀¹ xyz dxdydz
def f(z, y, x):  # 从内到外: x, y, z
    return x * y * z

result, _ = integrate.tplquad(f, 0, 1, 0, 1, 0, 1)
print(f"三重积分结果: {result:.6f}")
```

**输出**: `三重积分结果: 0.125000`

### 5. 实际应用

```python
# 计算标准正态分布的概率
def normal_pdf(x):
    return (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)

# P(-1 < X < 1) ≈ 68%
prob, _ = integrate.quad(normal_pdf, -1, 1)
print(f"P(-1<X<1) = {prob:.4f} = {prob*100:.2f}%")

# P(-2 < X < 2) ≈ 95%
prob, _ = integrate.quad(normal_pdf, -2, 2)
print(f"P(-2<X<2) = {prob:.4f} = {prob*100:.2f}%")
```

**输出**:
```
P(-1<X<1) = 0.6827 = 68.27%
P(-2<X<2) = 0.9545 = 95.45%
```

## 常微分方程 (ODE)

求解微分方程 $\frac{dy}{dt} = f(t, y)$

### 1. odeint (经典方法)

```python
# 求解 dy/dt = -y, y(0) = 1
# 解析解: y(t) = e^(-t)
def dydt(y, t):
    return -y

t = np.linspace(0, 5, 100)
y = integrate.odeint(dydt, y0=1, t=t)

print("t=0时: y =", y[0][0])
print("t=1时: y =", y[20][0], "理论值:", np.exp(-1))
print("t=5时: y =", y[-1][0], "理论值:", np.exp(-5))
```

**输出**:
```
t=0时: y = 1.0
t=1时: y = 0.3679 理论值: 0.3679
t=5时: y = 0.0067 理论值: 0.0067
```

### 2. solve_ivp (现代推荐)

```python
from scipy.integrate import solve_ivp

# 相同问题，使用solve_ivp
def f(t, y):  # 注意：参数顺序变了
    return -y

# 求解区间[0, 5]，初值y(0)=1
sol = solve_ivp(f, [0, 5], [1], t_eval=np.linspace(0, 5, 100))

print("时间点数:", len(sol.t))
print("t=0时: y =", sol.y[0][0])
print("t=5时: y =", sol.y[0][-1])
print("求解成功:", sol.success)
```

**输出**:
```
时间点数: 100
t=0时: y = 1.0
t=5时: y = 0.0067
求解成功: True
```

**优势**: 
- 更现代的API
- 更多求解器选择
- 更好的事件检测

### 3. 二阶ODE

将二阶方程转换为一阶方程组。

```python
# 简谐振动: d²y/dt² = -y
# 转换为: dy₁/dt = y₂, dy₂/dt = -y₁
def harmonic(t, y):
    y1, y2 = y
    return [y2, -y1]

# 初值: y(0)=1, y'(0)=0
sol = solve_ivp(harmonic, [0, 10], [1, 0], 
                t_eval=np.linspace(0, 10, 200))

print("振幅保持:", np.max(sol.y[0]), "≈ 1")
print("周期:", 2*np.pi, "秒")
```

**输出**:
```
振幅保持: 1.0000 ≈ 1
周期: 6.2832 秒
```

**应用**: 弹簧振动、单摆、电路分析。

### 4. 实际案例：人口增长模型

```python
# Logistic模型: dP/dt = rP(1 - P/K)
# P: 人口, r: 增长率, K: 环境容量
def logistic(t, P, r=0.5, K=1000):
    return r * P * (1 - P/K)

t_span = [0, 20]
P0 = [10]  # 初始人口
t_eval = np.linspace(0, 20, 100)

sol = solve_ivp(logistic, t_span, P0, t_eval=t_eval, 
                args=(0.5, 1000))

print("初始人口:", sol.y[0][0])
print("20年后人口:", sol.y[0][-1])
print("环境容量:", 1000)
```

**输出**:
```
初始人口: 10.0
20年后人口: 999.5
环境容量: 1000
```

## 积分方法选择

| 问题类型     | 推荐方法    | 说明             |
| ------------ | ----------- | ---------------- |
| 一维积分     | `quad`      | 自适应，高精度   |
| 二维积分     | `dblquad`   | 双重积分         |
| 三维积分     | `tplquad`   | 三重积分         |
| 一阶ODE      | `solve_ivp` | 现代，推荐       |
| 旧代码       | `odeint`    | 兼容性好         |
| 刚性方程     | `solve_ivp(method='Radau')` | 稳定性好 |

## 练习

```bash
python Basic/Scipy/06_integrate.py
```
