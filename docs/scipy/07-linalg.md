# 线性代数扩展

> 对应代码: [07_linalg.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/07_linalg.py)

## 矩阵分解

```python
from scipy import linalg

# LU 分解
P, L, U = linalg.lu(A)

# QR 分解
Q, R = linalg.qr(A)

# SVD 分解
U, s, Vh = linalg.svd(A)
```

## 特征值分解

```python
eigenvalues, eigenvectors = linalg.eig(A)
```

## 线性方程组

```python
# Ax = b
x = linalg.solve(A, b)
```

## scipy.linalg vs numpy.linalg

| 功能     | scipy.linalg | numpy.linalg |
| -------- | ------------ | ------------ |
| 功能覆盖 | 更全面       | 基础功能     |
| 性能     | 优化更多     | 标准         |
| LU分解   | ✓            | ✗            |
| Cholesky | ✓            | ✓            |

## 练习

```bash
python Basic/Scipy/07_linalg.py
```
