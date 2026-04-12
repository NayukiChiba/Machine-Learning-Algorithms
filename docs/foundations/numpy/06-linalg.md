---
title: NumPy 线性代数
outline: deep
---

# NumPy 线性代数

> 对应脚本：`Basic/Numpy/06_linalg.py`  
> 运行方式：`python Basic/Numpy/06_linalg.py`

## 本章目标

1. 掌握向量点积与矩阵乘法。
2. 掌握转置、行列式、逆矩阵与特征分解。
3. 会使用 `np.linalg.solve` 解线性方程组。
4. 理解常见向量/矩阵范数。

## 重点方法速览

| 方法 | 作用 |
|---|---|
| `np.dot` / `@` | 点积与矩阵乘法 |
| `A.T` / `np.transpose` | 转置 |
| `np.linalg.det` | 行列式 |
| `np.linalg.inv` | 逆矩阵 |
| `np.linalg.eig` | 特征值与特征向量 |
| `np.linalg.solve` | 解线性方程组 $Ax=b$ |
| `np.linalg.norm` | 计算范数 |

## 1. 点积与矩阵乘法

### 参数速览（本节）

适用 API/表达式（分项）：

1. `np.dot(a, b, out=None)`
2. `np.matmul(a, b, out=None)`
3. `A @ B`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` / `b`（向量点积） | `[1,2,3]` / `[4,5,6]` | 一维输入时执行向量点积 |
| `a` / `b`（矩阵乘法） | `A(2x2)` / `B(2x2)` | 二维输入时执行矩阵乘法 |
| `a` / `b`（`matmul/@`） | `A(2x2)` / `B(2x2)` | 专注矩阵乘法语义，支持更高维批量计算 |
### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)
print(np.dot(A, B))
```

### 结果输出

```text
32
----------------
[[19 22]
 [43 50]]
----------------
[[19 22]
 [43 50]]
```

## 2. 转置

### 参数速览（本节）

适用 API/属性（分项）：

1. `A.T`
2. `np.transpose(a, axes=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值（`A.T`） | `ndarray` | 返回转置后的数组视图 |
| `a` | `A(2x3)` | 对数组执行转置 |
| `axes` | `None`（默认） | 未指定时反转维度顺序 |
### 示例代码

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.T)
print(np.transpose(A))
```

### 结果输出

```text
[[1 4]
 [2 5]
 [3 6]]
----------------
[[1 4]
 [2 5]
 [3 6]]
```

## 3. 行列式与逆矩阵

### 参数速览（本节）

适用 API（分项）：

1. `np.linalg.det(a)`
2. `np.linalg.inv(a)`
3. `np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a`（`det`） | `[[4, 7], [2, 6]]` | 计算方阵行列式 |
| `a`（`inv`） | `[[4, 7], [2, 6]]` | 计算可逆方阵的逆矩阵 |
| `a` / `b`（`allclose`） | `A @ A_inv` / `np.eye(2)` | 用容差判断两数组是否近似相等 |
| `rtol` / `atol`（`allclose`） | `1e-05` / `1e-08` | 相对与绝对误差阈值 |
### 示例代码

```python
import numpy as np

A = np.array([[4, 7], [2, 6]])
det = np.linalg.det(A)
A_inv = np.linalg.inv(A)

print(det)
print(A_inv)
print(A @ A_inv)
print(np.allclose(A @ A_inv, np.eye(2)))
```

### 结果输出

```text
10.0
----------------
[[ 0.6 -0.7]
 [-0.2  0.4]]
----------------
[[ 1.  0.]
 [-0.  1.]]
----------------
True
```

## 4. 特征值与特征向量：`np.linalg.eig`

### 参数速览（本节）

适用 API：`np.linalg.eig(a)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` | `[[4, 2], [1, 3]]` | 输入方阵，计算特征分解 |
| 返回值1 | `eigenvalues` | 特征值数组 |
| 返回值2 | `eigenvectors` | 特征向量矩阵，第 `i` 列对应第 `i` 个特征值 |
### 示例代码

```python
import numpy as np

A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(eigenvalues)
print(eigenvectors)
```

### 结果输出

```text
[5. 2.]
----------------
[[ 0.89442719 -0.70710678]
 [ 0.4472136   0.70710678]]
```

### 验证关系

对每个特征对 $(\lambda, v)$，都有：

$$
Av = \lambda v
$$

脚本用 `np.allclose(left, right)` 做了逐个验证，结果均为 `True`。

## 5. 解线性方程组：`np.linalg.solve`

方程组：

$$
\begin{cases}
2x + y = 5 \\
x + 3y = 7
\end{cases}
$$

### 参数速览（本节）

适用 API：`np.linalg.solve(a, b)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` | `[[2, 1], [1, 3]]` | 方程组系数矩阵（方阵） |
| `b` | `[5, 7]` | 方程组右端向量 |
| 返回值 | `x` | 解向量，满足 `a @ x = b` |
### 示例代码

```python
import numpy as np

A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])
x = np.linalg.solve(A, b)

print(x)
print(A @ x)
```

### 结果输出

```text
[1.6 1.8]
----------------
[5. 7.]
```

## 6. 范数：`np.linalg.norm`

### 参数速览（本节）

适用 API：`np.linalg.norm(x, ord=None, axis=None, keepdims=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `v=[3,4]`、`A=[[1,2],[3,4]]` | 向量或矩阵输入 |
| `ord` | `1`、`2`、`np.inf`、`None` | 指定范数类型，`None` 时矩阵默认 Frobenius |
| `axis` | `None`（默认） | 对整体计算范数 |
| `keepdims` | `False`（默认） | 约简后不保留维度 |
### 示例代码

```python
import numpy as np

v = np.array([3, 4])
print(np.linalg.norm(v, ord=1))
print(np.linalg.norm(v, ord=2))
print(np.linalg.norm(v, ord=np.inf))

A = np.array([[1, 2], [3, 4]])
print(np.linalg.norm(A))
```

### 结果输出

```text
7.0
----------------
5.0
----------------
4.0
----------------
5.4772
```

### 理解重点

- 向量 $[3,4]$ 的 L2 范数为 $\sqrt{3^2 + 4^2}=5$。
- `np.linalg.norm(A)` 默认给出 Frobenius 范数。

## 常见坑

1. 只有方阵才能直接求逆；且接近奇异时数值不稳定。
2. 线性方程组优先 `solve`，不要先求逆再乘。
3. 比较浮点结果请用 `np.allclose`，不要直接 `==`。

## 小结

- 本章是机器学习中最常用的线代工具箱。
- 理解矩阵维度匹配，是避免大部分错误的关键。
