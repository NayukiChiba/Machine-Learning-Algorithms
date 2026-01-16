# 数值积分

> 对应代码: [06_integrate.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/06_integrate.py)

## 定积分

```python
from scipy import integrate

# 一维积分
result, error = integrate.quad(lambda x: x**2, 0, 1)

# 无穷积分
result, error = integrate.quad(f, -np.inf, np.inf)

# 二重积分
result, error = integrate.dblquad(
    f, x_min, x_max, y_min_func, y_max_func
)
```

## 常微分方程

```python
# 一阶 ODE: dy/dt = f(y, t)
def dydt(y, t):
    return -y

y = integrate.odeint(dydt, y0=1, t=t_array)
```

## solve_ivp (推荐)

```python
from scipy.integrate import solve_ivp

def f(t, y):
    return -y

sol = solve_ivp(f, [0, 5], [1], t_eval=np.linspace(0, 5, 100))
```

## 练习

```bash
python Basic/Scipy/06_integrate.py
```
