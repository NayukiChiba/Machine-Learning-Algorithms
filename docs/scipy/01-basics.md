# SciPy 基础入门

> 对应代码: [01_basics.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/01_basics.py)

## SciPy 模块结构

| 模块                | 功能           |
| ------------------- | -------------- |
| `scipy.constants`   | 物理和数学常数 |
| `scipy.special`     | 特殊函数       |
| `scipy.integrate`   | 数值积分       |
| `scipy.optimize`    | 优化算法       |
| `scipy.interpolate` | 插值           |
| `scipy.linalg`      | 线性代数       |
| `scipy.signal`      | 信号处理       |
| `scipy.sparse`      | 稀疏矩阵       |
| `scipy.stats`       | 统计分布       |
| `scipy.spatial`     | 空间数据       |

## 物理常数

```python
from scipy import constants

constants.pi      # 圆周率
constants.c       # 光速
constants.h       # 普朗克常数
constants.mile    # 1英里 = ? 米
```

## 特殊函数

```python
from scipy import special

special.factorial(5)   # 阶乘
special.comb(10, 3)    # 组合数
special.gamma(5)       # 伽马函数
```

## 练习

```bash
python Basic/Scipy/01_basics.py
```
