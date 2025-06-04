# 线性代数

> 对应代码: [06_linalg.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Numpy/06_linalg.py)

## 学习目标

- 掌握 NumPy 中的线性代数运算
- 理解矩阵运算的概念
- 学会使用 NumPy 的线性代数函数

## 重要函数 (np.linalg 模块)

| 函数                    | 说明                          |
| ----------------------- | ----------------------------- |
| `np.dot(a, b)`          | 矩阵乘法（或向量点积）        |
| `a @ b`                 | 矩阵乘法运算符（Python 3.5+） |
| `np.linalg.inv(a)`      | 矩阵求逆                      |
| `np.linalg.det(a)`      | 计算行列式                    |
| `np.linalg.eig(a)`      | 计算特征值和特征向量          |
| `np.linalg.solve(a, b)` | 解线性方程组 Ax = b           |
| `np.linalg.norm(a)`     | 计算范数                      |

## 矩阵乘法 vs 元素乘法

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 元素乘法（对应位置相乘）
A * B
# [[5, 12],
#  [21, 32]]

# 矩阵乘法（线性代数乘法）
A @ B  # 或 np.dot(A, B)
# [[19, 22],
#  [43, 50]]
```

## 向量点积

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)  # 32 = 1*4 + 2*5 + 3*6
```

## 矩阵转置

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3

A.T  # 转置为 3x2
# [[1, 4],
#  [2, 5],
#  [3, 6]]

np.transpose(A)  # 等价于 A.T
```

## 行列式和逆矩阵

```python
A = np.array([[4, 7], [2, 6]])

# 行列式
det = np.linalg.det(A)  # 10.0

# 逆矩阵
A_inv = np.linalg.inv(A)

# 验证 A @ A^(-1) = I
A @ A_inv  # 接近单位矩阵
```

> [!WARNING]
> 只有方阵（行数=列数）且行列式不为 0 才能求逆矩阵。

## 特征值和特征向量

```python
A = np.array([[4, 2], [1, 3]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 验证 A @ v = λ * v
v = eigenvectors[:, 0]  # 第一个特征向量
lam = eigenvalues[0]    # 对应的特征值

A @ v  # 应该等于 lam * v
```

## 解线性方程组

解方程组 **Ax = b**：

```
2x + y = 5
x + 3y = 7
```

```python
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])

x = np.linalg.solve(A, b)  # [1.6, 1.8]

# 验证
A @ x  # 应该等于 b
```

## 向量和矩阵范数

```python
v = np.array([3, 4])

# L1 范数（曼哈顿距离）
np.linalg.norm(v, ord=1)   # 7

# L2 范数（欧几里得距离）
np.linalg.norm(v, ord=2)   # 5 = sqrt(3² + 4²)

# 无穷范数
np.linalg.norm(v, ord=np.inf)  # 4

# 矩阵 Frobenius 范数
A = np.array([[1, 2], [3, 4]])
np.linalg.norm(A)  # sqrt(1+4+9+16) = sqrt(30)
```

## 其他线性代数函数

| 函数                       | 说明          |
| -------------------------- | ------------- |
| `np.linalg.matrix_rank(A)` | 矩阵的秩      |
| `np.linalg.svd(A)`         | 奇异值分解    |
| `np.linalg.qr(A)`          | QR 分解       |
| `np.linalg.cholesky(A)`    | Cholesky 分解 |

## 注意事项

> [!CAUTION]
> 由于浮点数精度问题，矩阵运算结果可能有微小误差。比较时使用 `np.allclose()` 而非 `==`。

```python
result = A @ np.linalg.inv(A)
np.allclose(result, np.eye(2))  # True
```

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/06_linalg.py
```
