# 优化算法

> 对应代码: [04_optimize.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/04_optimize.py)

## 曲线拟合

```python
from scipy import optimize

def model(x, a, b, c):
    return a * x**2 + b * x + c

params, covariance = optimize.curve_fit(model, x_data, y_data)
```

## 求根算法

```python
# 区间法
root = optimize.brentq(f, 0, 3)

# 牛顿法
root = optimize.fsolve(f, x0=1)

# 多元方程组
def equations(p):
    x, y = p
    return [x + y - 3, x - y - 1]
solution = optimize.fsolve(equations, x0=[0, 0])
```

## 最小化

```python
# 一维最小化
result = optimize.minimize_scalar(f)

# 多维最小化
result = optimize.minimize(f, x0=[0, 0], method='BFGS')
```

## 线性规划

```python
c = [-2, -3]           # 目标函数系数
A_ub = [[1, 1], [1, 0]]  # 约束矩阵
b_ub = [4, 2]          # 约束右侧
result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub)
```

## 练习

```bash
python Basic/Scipy/04_optimize.py
```
